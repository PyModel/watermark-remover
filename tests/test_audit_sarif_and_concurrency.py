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


def _item(findings: list[str], confidences: list[str], has_c2pa: bool = False) -> dict:
    return {
        "path": "/workspace/project/img.jpg",
        "kind": "jpeg",
        "has_c2pa": has_c2pa,
        "has_ai_metadata": bool(findings),
        "findings": findings,
        "confidence": confidences,
    }


def test_new_finding_classes_get_distinct_sarif_rules():
    report = {
        "root": "/workspace/project",
        "files": [
            _item(
                [
                    "JPEG trailer after EOI: Samsung Galaxy AI marker: PhotoEditor_Re_Edit_Data genAIType=1",
                    "JPEG APP1: xAI/Grok provenance marker: Signature + artist UUID pair",
                    "JPEG APP1: IPTC digital-source trainedAlgorithmicMedia (AI-generated)",
                    "JPEG COM: Higgsfield job hint marker: hf-job-id=7c9e6679-7425-40de-944b-e07fc1f90ae7 (medium confidence)",
                    "JPEG APP1: TC260 AIGC label (Label=1, ContentProducer=ByteDance)",
                ],
                ["probable", "probable", "confirmed", "probable", "probable"],
            )
        ],
    }
    sarif = format_sarif(report)
    rules = {r["id"]: r for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
    assert {
        "AI-WATERMARK-SAMSUNG",
        "AI-WATERMARK-XAI",
        "AI-WATERMARK-IPTC",
        "AI-WATERMARK-HF-JOB",
        "AI-WATERMARK-TC260",
    } <= set(rules)
    by_rule = {}
    for result in sarif["runs"][0]["results"]:
        by_rule.setdefault(result["ruleId"], []).append(result)
    assert len(by_rule["AI-WATERMARK-SAMSUNG"]) == 1
    assert len(by_rule["AI-WATERMARK-XAI"]) == 1
    assert len(by_rule["AI-WATERMARK-IPTC"]) == 1
    assert len(by_rule["AI-WATERMARK-HF-JOB"]) == 1
    assert len(by_rule["AI-WATERMARK-TC260"]) == 1
    # Confirmed digital-source semantics stays an error; hints stay warnings.
    assert by_rule["AI-WATERMARK-IPTC"][0]["level"] == "error"
    assert by_rule["AI-WATERMARK-SAMSUNG"][0]["level"] == "warning"
    assert by_rule["AI-WATERMARK-HF-JOB"][0]["level"] == "warning"
    assert by_rule["AI-WATERMARK-TC260"][0]["level"] == "warning"


def test_existing_sarif_classes_are_unchanged():
    report = {
        "root": "/workspace/project",
        "files": [
            _item(["marker:C2PA manifest"], ["confirmed"], has_c2pa=True),
            _item(["meta: ai"], ["probable"]),
            _item(["layer-a [space] U+2002 x3"], ["probable"]),
        ],
    }
    sarif = format_sarif(report)
    rules = {r["id"] for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
    assert {
        "AI-WATERMARK-C2PA",
        "AI-WATERMARK-METADATA",
        "AI-WATERMARK-UNICODE-LAYER-A",
        "AI-STYLES-HIGH-PROBABILITY",
    } <= rules
    by_rule = {}
    for result in sarif["runs"][0]["results"]:
        by_rule.setdefault(result["ruleId"], []).append(result)
    assert by_rule["AI-WATERMARK-C2PA"][0]["level"] == "error"
    assert by_rule["AI-WATERMARK-METADATA"][0]["level"] == "warning"
    assert by_rule["AI-WATERMARK-UNICODE-LAYER-A"][0]["level"] == "warning"
