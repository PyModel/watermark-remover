"""Behavioral contracts for bounded external command execution."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import external_command
from external_command import (
    ExternalCommandTimeout,
    command_from_template,
    run_command,
)


def test_template_values_remain_single_arguments() -> None:
    value = "input name; echo injected"

    command = command_from_template(
        'tool --input "{input}" --output={output}',
        input=value,
        output="result name.png",
    )

    assert command == ("tool", "--input", value, "--output=result name.png")


def test_shell_syntax_in_argument_is_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "unexpected"
    value = f"literal; touch {marker}"

    result = run_command(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", value],
        timeout=5,
        output_limit=1024,
    )

    assert result.returncode == 0
    assert result.stdout_text.strip() == value
    assert not marker.exists()


def test_output_is_bounded_to_tail_and_reports_truncation() -> None:
    result = run_command(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.stdout.write('prefix-' + 'x' * 64 + '-stdout-tail'); "
            "sys.stderr.write('prefix-' + 'y' * 64 + '-stderr-tail')",
        ],
        timeout=5,
        output_limit=24,
    )

    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert result.stdout_text.endswith("-stdout-tail")
    assert result.stderr_text.endswith("-stderr-tail")
    assert "prefix-" not in result.stdout_text
    assert "prefix-" not in result.stderr_text
    assert len(result.stdout) <= 24
    assert len(result.stderr) <= 24


def test_timeout_terminates_command_promptly() -> None:
    started = time.monotonic()

    with pytest.raises(ExternalCommandTimeout, match="timed out"):
        run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout=0.05,
            output_limit=1024,
        )

    assert time.monotonic() - started < 2


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-specific")
def test_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"
    child = (
        "import sys, time; from pathlib import Path; "
        "time.sleep(0.2); Path(sys.argv[1]).write_text('survived')"
    )
    parent = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "time.sleep(10)"
    )

    with pytest.raises(ExternalCommandTimeout):
        run_command(
            [sys.executable, "-c", parent, child, str(marker)],
            timeout=0.05,
            output_limit=1024,
        )

    time.sleep(0.4)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-specific")
def test_parent_exit_cannot_leave_inherited_output_pipes_running(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"
    child = (
        "import sys, time; from pathlib import Path; "
        "time.sleep(0.7); Path(sys.argv[1]).write_text('survived')"
    )
    parent = (
        "import subprocess, sys; subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])"
    )

    with pytest.raises(ExternalCommandTimeout):
        run_command(
            [sys.executable, "-c", parent, child, str(marker)],
            timeout=0.05,
            output_limit=1024,
        )

    # Outlast the descendant's own delay: a surviving descendant would write the
    # marker at 0.7s, so 1.0s is the earliest a pass means cleanup worked.
    time.sleep(1.0)
    assert not marker.exists()


def test_normal_exit_allows_minimum_pipe_drain_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(0.05)'])",
    ]
    calls = iter((0.0, 0.0))
    monkeypatch.setattr(
        external_command,
        "time",
        SimpleNamespace(monotonic=lambda: next(calls, 5.0)),
    )

    result = run_command(command, timeout=5.0, output_limit=4096)

    assert result.returncode == 0


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-specific")
def test_keyboard_interrupt_terminates_command_process_group(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    marker = tmp_path / "descendant-survived"
    child = (
        "import sys, time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text('ready'); "
        "time.sleep(0.4); Path(sys.argv[2]).write_text('survived')"
    )
    runner = (
        "import sys; "
        f"sys.path.insert(0, {str(SCRIPTS)!r}); "
        "from external_command import run_command; "
        "run_command("
        "[sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]], "
        "timeout=5, output_limit=1024)"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", runner, child, str(ready), str(marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 2
    while not ready.exists() and worker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert ready.exists()
        worker.send_signal(signal.SIGINT)
        assert worker.wait(timeout=2) != 0
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=2)

    time.sleep(0.6)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-specific")
def test_interrupt_during_reader_start_terminates_command_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "command-survived"

    def interrupt(_reader) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(external_command, "_start_thread", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_command(
            [
                sys.executable,
                "-c",
                "import sys, time; from pathlib import Path; "
                "time.sleep(0.3); Path(sys.argv[1]).write_text('survived')",
                str(marker),
            ],
            timeout=5,
            output_limit=1024,
        )

    time.sleep(0.5)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("timeout", "output_limit", "message"),
    [(0, 1024, "timeout"), (float("inf"), 1024, "timeout"), (1, 0, "output_limit")],
)
def test_invalid_bounds_fail_before_execution(
    timeout: float,
    output_limit: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_command(["not-executed"], timeout=timeout, output_limit=output_limit)


@pytest.mark.parametrize(
    ("argv", "timeout", "output_limit"),
    [
        ("echo hi", 1, 1024),
        (["ok"], True, 1024),
        (["ok"], "1", 1024),
        (["ok"], 1, True),
        (["ok"], 1, 1.0),
    ],
)
def test_invalid_types_raise_type_error_before_execution(argv, timeout, output_limit) -> None:
    with pytest.raises(TypeError):
        run_command(argv, timeout=timeout, output_limit=output_limit)


def test_read_error_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingReader:
        def read(self, _size: int) -> bytes:
            raise OSError("simulated read failure")

        def close(self) -> None:
            pass

    class FakePopen:
        pid = 1 << 30  # beyond any pid_max, so cleanup killpg is a no-op

        def __init__(self, *_args, **_kwargs):
            self.stdout = FailingReader()
            self.stderr = FailingReader()

        def wait(self, *, timeout: float) -> int:
            return 0

    monkeypatch.setattr(external_command.subprocess, "Popen", FakePopen)

    with pytest.raises(RuntimeError, match="failed to read external command output"):
        run_command(["ok"], timeout=5, output_limit=1024)
