"""Shared test fakes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))


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
