"""Tests for inspect_text.py CLI threshold validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSPECT_SCRIPT = ROOT / "skills" / "remove-ai-marks" / "scripts" / "inspect_text.py"


def _run_inspect(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(INSPECT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
        input="hello world",
    )


class TestThresholdValidation:
    """--threshold must be a finite value in [0.0, 1.0]."""

    def test_valid_zero(self):
        r = _run_inspect("--threshold", "0.0", "-")
        assert r.returncode != 2, f"threshold 0.0 should be accepted: {r.stderr}"

    def test_valid_one(self):
        r = _run_inspect("--threshold", "1.0", "-")
        assert r.returncode != 2, f"threshold 1.0 should be accepted: {r.stderr}"

    def test_valid_default(self):
        r = _run_inspect("--threshold", "0.65", "-")
        assert r.returncode != 2, f"threshold 0.65 should be accepted: {r.stderr}"

    def test_reject_nan(self):
        r = _run_inspect("--threshold", "nan", "-")
        assert r.returncode == 2
        assert "threshold" in (r.stderr or "").lower()

    def test_reject_inf(self):
        r = _run_inspect("--threshold", "inf", "-")
        assert r.returncode == 2
        assert "threshold" in (r.stderr or "").lower()

    def test_reject_neg_inf(self):
        r = _run_inspect("--threshold", "-inf", "-")
        assert r.returncode == 2
        assert "threshold" in (r.stderr or "").lower()

    def test_reject_above_one(self):
        r = _run_inspect("--threshold", "1.1", "-")
        assert r.returncode == 2
        assert "threshold" in (r.stderr or "").lower()

    def test_reject_negative(self):
        r = _run_inspect("--threshold", "-0.1", "-")
        assert r.returncode == 2
        assert "threshold" in (r.stderr or "").lower()


class TestMissingInputFile:
    """A missing input path must fail cleanly, not with a traceback."""

    def test_missing_file_exits_2_without_traceback(self, tmp_path):
        r = _run_inspect(str(tmp_path / "does-not-exist.txt"))
        assert r.returncode == 2, r.stderr
        assert "Traceback" not in (r.stderr or "")
        assert "not a regular file" in (r.stderr or "")

    def test_directory_exits_2_without_traceback(self, tmp_path):
        r = _run_inspect(str(tmp_path))
        assert r.returncode == 2, r.stderr
        assert "Traceback" not in (r.stderr or "")
        assert "not a regular file" in (r.stderr or "")
