"""Shared test fakes."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def repo_files() -> list[Path]:
    """Every file that could be committed: tracked plus untracked-but-not-ignored.

    Tree guards scan this instead of walking the disk, so a developer's
    gitignored local dirs (research/ snapshots, .gitnexus/, .venv/) never trip
    them. Outside a git checkout (an sdist) the walk falls back to the disk.
    """
    git = shutil.which("git")
    try:
        if git is None:
            raise OSError("git not found")
        listing = subprocess.run(
            [git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return [p for p in REPO_ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    paths = (REPO_ROOT / name for name in listing.decode().split("\0") if name)
    return [p for p in paths if p.is_file()]  # a staged deletion is listed but gone


def load_toml(name: str) -> dict:
    """Parse a TOML file at the repo root; skip on Python 3.10, which lacks tomllib."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib needs Python 3.11+")
    return tomllib.loads((REPO_ROOT / name).read_text(encoding="utf-8"))


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
