"""F7: best-effort Claude/Anthropic detector-evasion risk scoring.

This module estimates how likely a *cleaned* asset would still be flagged by
Anthropic/Claude's watermark detector.  It is deliberately labeled best-effort:
Anthropic's detector thresholds are not public, so the score is a transparent
heuristic over *residual signals* we can actually measure — invisible Unicode
carriers, residual C2PA/AI metadata, SynthID spectral confidence, and visible
marks.  It never claims to guarantee evasion.

Signals -> score weights are documented inline so the estimate is auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import eprint
from image_meta import detect_format, inspect_image
from text_unicode import inspect_text

#: Human-readable, stable label for the honesty contract.
BEST_EFFORT_NOTE = (
    "best-effort estimate over residual signals; Anthropic/Claude detector "
    "thresholds are not public and are not guaranteed"
)

#: Weight of each residual-signal category (0..1 * weight).  Arbitrary but
#: documented and stable; the score is capped at 100.
_SIGNAL_WEIGHTS = {
    "invisible_unicode_carriers": 20,  # per hit kind present, cap 60
    "residual_c2pa": 40,  # provenance manifest still embedded
    "residual_ai_metadata": 35,  # AI-generation tags still embedded
    "synthid_confidence": 45,  # spectral signal above detector floor
    "visible_mark_findings": 15,  # per finding, cap 45
    "base": 5,  # any AI watermark can evade static analysis
}

_VERDICT_CUTOFFS = {"low": 30, "medium": 60}  # < low, < medium, else high


def _classify(path: Path) -> str:
    data = path.read_bytes()
    if detect_format(data) in ("png", "jpeg", "heif", "avif"):
        return "image"
    return "text"


def _assess_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    report = inspect_text(text)
    signals: list[dict] = []
    if report.suspicious_total > 0:
        signals.append(
            {
                "signal": "invisible_unicode_carriers",
                "detail": f"{report.suspicious_total} suspicious char(s) remain "
                "(edit-based carriers still present)",
                "severity": "medium" if report.suspicious_total >= 3 else "low",
            }
        )
    # Layer B (token-distribution rewrite) is not detectable from a single file;
    # it is the mitigation for Claude's statistical watermark, so note it.
    score = 0
    score += _SIGNAL_WEIGHTS["base"]
    hit_kinds = {h.kind for h in report.hits}
    if hit_kinds:
        score += min(60, _SIGNAL_WEIGHTS["invisible_unicode_carriers"] * len(hit_kinds))
    verdict = _verdict(score)
    return {
        "kind": "text",
        "signals": signals,
        "score": score,
        "verdict": verdict,
        "note": (
            BEST_EFFORT_NOTE + "; statistical (token-sampling) watermarks are invisible here — "
            "Layer B rewrite is the intended mitigation and is best-effort"
        ),
    }


def _assess_image(path: Path, synthid_dir: str | None) -> dict:
    after = inspect_image(path, synthid_dir=synthid_dir)
    signals: list[dict] = []
    score = _SIGNAL_WEIGHTS["base"]
    if after.has_c2pa:
        signals.append(
            {
                "signal": "residual_c2pa",
                "detail": "C2PA/provenance manifest still embedded",
                "severity": "high",
            }
        )
        score += _SIGNAL_WEIGHTS["residual_c2pa"]
    if after.has_ai_metadata:
        signals.append(
            {
                "signal": "residual_ai_metadata",
                "detail": "AI-generation metadata tags still embedded",
                "severity": "medium",
            }
        )
        score += _SIGNAL_WEIGHTS["residual_ai_metadata"]
    if after.synthid is not None:
        conf = after.synthid.get("confidence") if isinstance(after.synthid, dict) else None
        if conf is not None and conf >= 0.2:
            signals.append(
                {
                    "signal": "synthid_confidence",
                    "detail": f"SynthID-class spectral confidence {conf:.3f} remains "
                    "(above the ~0.1 detector floor)",
                    "severity": "medium",
                }
            )
            score += _SIGNAL_WEIGHTS["synthid_confidence"]
    for finding in after.findings:
        signals.append({"signal": "visible_mark_findings", "detail": finding, "severity": "low"})
        score += _SIGNAL_WEIGHTS["visible_mark_findings"]
    score = min(100, score)
    verdict = _verdict(score)
    return {
        "kind": "image",
        "signals": signals,
        "score": score,
        "verdict": verdict,
        "note": BEST_EFFORT_NOTE,
        "synthid_after": after.synthid,
    }


def assess_claude_risk(path: Path, *, synthid_dir: str | None = None) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    kind = _classify(path)
    if kind == "image":
        return _assess_image(path, synthid_dir)
    return _assess_text(path)


def _verdict(score: int) -> str:
    if score < _VERDICT_CUTOFFS["low"]:
        return "low"
    if score < _VERDICT_CUTOFFS["medium"]:
        return "medium"
    return "high"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="claude_risk.py",
        description="Best-effort Claude/Anthropic detector-evasion risk estimate",
    )
    p.add_argument("path", type=Path, help="cleaned file (text or image) to assess")
    p.add_argument("--synthid-dir", type=str, default=None, help="reverse-SynthID checkout root")
    p.add_argument("--json", action="store_true", help="emit JSON report")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        report = assess_claude_risk(args.path, synthid_dir=args.synthid_dir)
    except Exception as error:
        eprint(f"error: {error}")
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"{report['kind']} risk: {report['verdict']} (score {report['score']}/100)")
        for signal in report["signals"]:
            print(f"  - [{signal['severity']}] {signal['signal']}: {signal['detail']}")
        print(f"  note: {report['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
