"""Shared test fakes."""

from __future__ import annotations

from types import SimpleNamespace


def fake_command_result(returncode: int, *, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout.encode(),
        stderr=stderr.encode(),
        stdout_text=stdout,
        stderr_text=stderr,
        stdout_truncated=False,
        stderr_truncated=False,
    )
