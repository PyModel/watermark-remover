"""Tests for multi-worker concurrency and SARIF 2.1.0 export in audit_dir."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_lib import format_sarif


def test_audit_dir_partial_scan_exit_3(monkeypatch, tmp_path, capsys):
    """Every output format reports a partial scan (skipped files) with exit 3."""
    import audit_dir

    (tmp_path / "clean.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(
        audit_dir,
        "_scan_worker",
        lambda _path, _check: (None, {"path": "x", "reason": "boom"}),
    )
    for extra in ((), ("--json",), ("--format", "sarif")):
        monkeypatch.setattr(sys, "argv", ["audit_dir.py", str(tmp_path), *extra])
        assert audit_dir.main() == 3, extra


def test_audit_dir_partial_outranks_actionable(monkeypatch, tmp_path):
    """Partial wins over actionable findings: incomplete audit is the more important signal."""
    import audit_dir

    calls = {"n": 0}

    def _worker(path, _check):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                {
                    "path": str(path),
                    "kind": "text",
                    "has_c2pa": False,
                    "has_ai_metadata": True,
                    "suspicious_total": 1,
                    "findings": ["meta: ai"],
                    "confidence": ["probable"],
                    "notes": [],
                },
                None,
            )
        return (None, {"path": str(path), "reason": "boom"})

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    monkeypatch.setattr(audit_dir, "_scan_worker", _worker)
    monkeypatch.setattr(sys, "argv", ["audit_dir.py", str(tmp_path), "--json"])
    assert audit_dir.main() == 3


def test_format_sarif_structure():
    fake_report = {
        "root": "/workspace/project",
        "files_scanned": 3,
        "files": [
            {
                "path": "/workspace/project/assets/logo.png",
                "kind": "png",
                "has_c2pa": True,
                "has_ai_metadata": True,
                "findings": ["marker:C2PA manifest", "ai:Midjourney"],
                "confidence": ["confirmed", "probable"],
            },
            {
                "path": "/workspace/project/docs/readme.md",
                "kind": "markdown",
                "has_c2pa": False,
                "has_ai_metadata": False,
                "findings": ["layer-a: U+200B ZERO WIDTH SPACE x3 (zero_width)"],
                "confidence": ["probable"],
            },
            {
                "path": "/workspace/project/clean.txt",
                "kind": "text",
                "has_c2pa": False,
                "has_ai_metadata": False,
                "findings": [],
                "confidence": [],
            },
        ],
    }

    sarif = format_sarif(fake_report)
    assert sarif["version"] == "2.1.0"
    assert "$schema" in sarif
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "watermark-remover"
    rule_ids = {r["id"] for r in driver["rules"]}
    assert "AI-WATERMARK-C2PA" in rule_ids
    assert "AI-WATERMARK-METADATA" in rule_ids
    assert "AI-WATERMARK-UNICODE-LAYER-A" in rule_ids

    results = run["results"]
    assert len(results) == 3

    c2pa_res = next(r for r in results if r["ruleId"] == "AI-WATERMARK-C2PA")
    assert c2pa_res["level"] == "error"
    assert "logo.png" in c2pa_res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
