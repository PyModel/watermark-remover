"""Bounded, shell-free execution for optional external adapters."""

from __future__ import annotations

import math
import os
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass

_READ_SIZE = 64 * 1024
_TERMINATION_GRACE = 0.5


class ExternalCommandTimeout(RuntimeError):
    """Raised after a command exceeds its configured execution deadline."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


def command_from_template(template: str, **values: str) -> tuple[str, ...]:
    """Tokenize a command template before substituting trusted string values."""
    if not isinstance(template, str) or not template.strip():
        raise ValueError("command template must not be empty")
    if any(not isinstance(value, str) for value in values.values()):
        raise TypeError("command template values must be strings")
    try:
        tokens = shlex.split(template)
        return tuple(token.format_map(values) for token in tokens)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid command template: {error}") from error


def _validate_command(
    argv: Sequence[str],
    timeout: float,
    output_limit: int,
) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments, not a command string")
    command = tuple(argv)
    if not command or any(not isinstance(argument, str) or not argument for argument in command):
        raise ValueError("argv must contain non-empty string arguments")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a finite positive number")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if isinstance(output_limit, bool) or not isinstance(output_limit, int):
        raise TypeError("output_limit must be a positive integer")
    if output_limit <= 0:
        raise ValueError("output_limit must be a positive integer")
    return command


def run_command(
    argv: Sequence[str],
    *,
    timeout: float,
    output_limit: int,
) -> CommandResult:
    """Execute argv without a shell while retaining bounded stdout/stderr tails."""
    command = _validate_command(argv, timeout, output_limit)
    tails = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    read_errors: list[BaseException] = []
    readers: list[threading.Thread] = []
    process: subprocess.Popen[bytes] | None = None
    deadline = time.monotonic() + timeout
    timed_out = False
    returncode: int | None = None

    def drain(name: str, stream) -> None:
        try:
            while chunk := stream.read(_READ_SIZE):
                tail = tails[name]
                tail.extend(chunk)
                overflow = len(tail) - output_limit
                if overflow > 0:
                    del tail[:overflow]
                    truncated[name] = True
        except (OSError, ValueError) as error:
            read_errors.append(error)

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
            start_new_session=os.name == "posix",
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("external command output pipes unavailable")

        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            reader = threading.Thread(target=drain, args=(name, stream), daemon=True)
            readers.append(reader)
            reader.start()

        try:
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True

        if not timed_out:
            drain_deadline = time.monotonic() + max(_TERMINATION_GRACE, deadline - time.monotonic())
            for reader in readers:
                reader.join(max(0.0, drain_deadline - time.monotonic()))
            timed_out = any(reader.is_alive() for reader in readers)
    finally:
        if process is not None:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                with suppress(ProcessLookupError):
                    process.kill()

            if returncode is None:
                try:
                    returncode = process.wait(timeout=_TERMINATION_GRACE)
                except subprocess.TimeoutExpired:
                    timed_out = True

            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    with suppress(OSError, ValueError):
                        stream.close()
        for reader in readers:
            with suppress(RuntimeError):
                reader.join(_TERMINATION_GRACE)

    if any(reader.is_alive() for reader in readers):
        timed_out = True
    if timed_out or returncode is None:
        raise ExternalCommandTimeout(f"external command timed out after {timeout:g} seconds")
    if read_errors:
        raise RuntimeError(f"failed to read external command output: {read_errors[0]}")
    return CommandResult(
        returncode=returncode,
        stdout=bytes(tails["stdout"]),
        stderr=bytes(tails["stderr"]),
        stdout_truncated=truncated["stdout"],
        stderr_truncated=truncated["stderr"],
    )
