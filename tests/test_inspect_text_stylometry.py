"""Tests for inspect_text --stylometry wire-up."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_text.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_inspect_text_stylometry_flag_ai_file():
    """AI sample with --stylometry triggers exit 1 (score >= 0.65)."""
    ai_path = FIXTURES / "stylometry_ai_sample.txt"
    res = _run([str(ai_path), "--stylometry", "--json"])
    assert res.returncode == 1, res.stderr
    data = json.loads(res.stdout)
    assert "stylometry" in data
    assert data["stylometry"]["score"] >= 0.65


def test_inspect_text_stylometry_human_file():
    """Human sample with --stylometry exits 0 (score < 0.65)."""
    human_path = FIXTURES / "stylometry_human_sample.txt"
    res = _run([str(human_path), "--stylometry", "--json"])
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["stylometry"]["score"] < 0.65


def test_inspect_text_without_stylometry_omits_field():
    """Without --stylometry, output is the standard text_unicode report."""
    human_path = FIXTURES / "stylometry_human_sample.txt"
    res = _run([str(human_path), "--json"])
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert "stylometry" not in data


def test_inspect_text_stylometry_threshold_override():
    """Custom threshold of 0.99 lets even the AI sample pass."""
    ai_path = FIXTURES / "stylometry_ai_sample.txt"
    res = _run([str(ai_path), "--stylometry", "--threshold", "0.99", "--json"])
    assert res.returncode == 0, res.stderr
