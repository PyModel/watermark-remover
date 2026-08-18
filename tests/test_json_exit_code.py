"""--json must not suppress the residual-signal exit code (was: always 0).

Adapted to OURS' clean_asset()/CleanResult API (THEIRS patched clean_container).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_file
from clean_asset import CleanResult


def _result(kind, dest, residual, details=None):
    base = {}
    if kind == "container":
        base = {"format": "markdown", "actions": ["clean"], "bytes_in": 1, "bytes_out": 1}
    else:
        base = {"actions": ["strip"], "bytes_in": 1, "bytes_out": 1}
    base.update(details or {})
    return CleanResult(
        kind, Path("x.md") if kind == "container" else Path("x.png"), dest, residual, base
    )


def _run_clean(monkeypatch, tmp_path, *, json_flag, residual, degraded=False):
    src = tmp_path / "x.md"
    src.write_text("---\ngenerator: Claude\n---\nhi\n", encoding="utf-8")
    dest = tmp_path / "x.cleaned.md"
    details = {}
    if degraded:
        details["meta"] = {"degraded": True}
    monkeypatch.setattr(
        clean_file,
        "clean_asset",
        lambda *a, **k: _result("container", dest, residual, details),
    )
    argv = ["clean_file.py", str(src), "-o", str(dest)]
    if json_flag:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    return clean_file.main()


def test_clean_file_json_and_human_agree_on_residual(monkeypatch, tmp_path):
    # The bug: --json returned 0 while human mode returned 1.
    assert _run_clean(monkeypatch, tmp_path, json_flag=False, residual=True) == 1
    assert _run_clean(monkeypatch, tmp_path, json_flag=True, residual=True) == 1


def test_clean_file_json_and_human_agree_on_clean(monkeypatch, tmp_path):
    assert _run_clean(monkeypatch, tmp_path, json_flag=False, residual=False) == 0
    assert _run_clean(monkeypatch, tmp_path, json_flag=True, residual=False) == 0


def test_clean_file_json_and_human_agree_on_image_residual(monkeypatch, tmp_path):
    src = tmp_path / "x.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    dest = tmp_path / "x.cleaned.png"
    monkeypatch.setattr(
        clean_file,
        "clean_asset",
        lambda *a, **k: _result("image", dest, True),
    )
    argv = ["clean_file.py", str(src), "-o", str(dest)]
    argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    assert clean_file.main() == 1
