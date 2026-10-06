#!/usr/bin/env python3
"""Unified cleaner for text, images, and document containers.

Supports single files, multiple files, and directory batches (--glob,
--recursive, --extensions). Advanced transforms are opt-in:
  --rewrite        live Layer B rewrite (paraphrase/humanize/code/... or tsapa)
  --tsapa          alias for --rewrite tsapa
  --char-perturb   character-level anti-watermark noise (intentionally after Layer A)
  --visible-*      MorphoMod mask→dilate→inpaint pipeline
  --soft-binding   detect residual soft-binding / remote-manifest risk

Option collection, plan construction, and batch preflight live in
``clean_request``; this module is the argparse front end over that seam.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEGRADE_CLI_CHOICES = ("freq-dct", "blur", "median", "jpeg", "rotate", "two-stage")
MORPHO_CLI_CHOICES = ("grid", "diagonal", "noise", "quantize")
from clean_pipeline import execute
from clean_request import (
    DEFAULT_TIMEOUT,
    DEFAULT_VISIBLE_PROMPT,
    REWRITE_CLI_CHOICES,
    CleanPlanPreflightError,
    CleanRequest,
)
from common import atomic_write_text, eprint
from morphomod import VISIBLE_CLEAN_BACKENDS
from operation import ExitCode
from perturb_text import MODES as PERTURB_MODES
from pipeline_actions import action_sentences
from rewrite_text import LIVE_REWRITE_BACKENDS, REASONING_EFFORTS

# Preserved under the old private name: external callers are not expected, but
# the rename should not be the thing that breaks an in-tree import.
_CleanPlanPreflightError = CleanPlanPreflightError


def _parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(v) for v in value.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError("box must be x,y,w,h") from e
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x,y,w,h")
    return parts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="+", type=Path, help="Input file(s) or directories")
    p.add_argument("-o", "--output", type=Path, help="Output path (single) or directory (batch)")
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--recursive", action="store_true")
    p.add_argument(
        "--glob", default="*", help="Directory glob (use with --recursive for **-style traversal)"
    )
    p.add_argument("--extensions", help="Comma-separated extension allow-list")
    p.add_argument("--nfkc", action="store_true", help="Text: NFKC normalize")
    p.add_argument("--aggressive-homoglyphs", action="store_true")
    p.add_argument(
        "--strip-semantic-format",
        action="store_true",
        help="Text: aggressively strip contextual ZWJ/variation/bidi/math controls",
    )
    p.add_argument("--keep-non-ai-metadata", action="store_true")
    p.add_argument(
        "--as", dest="force_type", choices=("auto", "text", "image", "container"), default="auto"
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Clean as text even when the bytes look like a binary container",
    )

    # Layer B: live LLM rewrite. Settings resolve explicit flag > environment
    # (WATERMARKS_REWRITE_*) > default, so an endpoint configured once in the
    # environment still works with a bare --rewrite.
    p.add_argument(
        "--rewrite",
        choices=REWRITE_CLI_CHOICES,
        default=None,
        help="Text: live Layer B rewrite with the named strength (best-effort)",
    )
    p.add_argument("--rewrite-backend", choices=LIVE_REWRITE_BACKENDS, default=None)
    p.add_argument("--rewrite-model", default=None)
    p.add_argument("--rewrite-base-url", default=None)
    p.add_argument("--rewrite-lang", default=None, help="Pivot language for backtranslate")
    p.add_argument("--rewrite-original-lang", default=None)
    p.add_argument("--rewrite-timeout", type=float, default=None)
    p.add_argument("--rewrite-temperature", type=float, default=None)
    p.add_argument(
        "--rewrite-candidates",
        type=int,
        default=None,
        help="Generate N candidates and keep the most lexically divergent",
    )
    p.add_argument("--rewrite-reasoning-effort", choices=REASONING_EFFORTS, default=None)
    p.add_argument(
        "--rewrite-disable-thinking",
        action="store_true",
        default=None,
        help="Ask the model to skip chain-of-thought output",
    )
    p.add_argument(
        "--rewrite-allow-remote",
        action="store_true",
        # Tri-state, like --rewrite-disable-thinking above: absent means "ask
        # the environment", not "deny". Without this the flag's own default of
        # False was passed through as an explicit denial, so the documented
        # WATERMARKS_REWRITE_ALLOW_REMOTE could never take effect under wm.
        default=None,
        help="Permit a non-loopback Layer B endpoint (your text leaves this machine)",
    )
    p.add_argument("--tsapa", action="store_true", help="Alias for --rewrite tsapa")
    p.add_argument("--tsapa-generations", type=int, default=5)
    p.add_argument("--tsapa-population", type=int, default=12)
    p.add_argument("--char-perturb", action="store_true")
    p.add_argument("--char-mode", choices=PERTURB_MODES, default="zero-width")
    p.add_argument("--char-strength", type=float, default=0.1)
    p.add_argument("--seed", type=int)

    visible = p.add_mutually_exclusive_group()
    visible.add_argument("--visible-mask", type=Path)
    visible.add_argument("--visible-box", type=_parse_box, metavar="X,Y,W,H")
    visible.add_argument("--detect-command", help="Detector template: {input} {mask} {prompt}")
    visible.add_argument(
        "--mark",
        default=None,
        help="Registry-driven visible-mark removal: 'auto' scans every registered "
        "mark, or name one key. Opt-in; absent, the explicit-mask behavior is "
        "unchanged. Requires the pixel extra and a PNG destination",
    )
    p.add_argument(
        "--sensitivity",
        choices=("auto", "strict"),
        default="auto",
        help="Trust policy for --mark: 'auto' relaxes a mark's gate only on "
        "same-product evidence (TC260 producer metadata or a strict sibling "
        "mark); 'strict' never relaxes. Requires --mark",
    )
    p.add_argument("--dilate", type=int, default=None, metavar="RADIUS")
    p.add_argument(
        "--visible-backend",
        choices=VISIBLE_CLEAN_BACKENDS,
        default="texture",
    )
    p.add_argument("--inpaint-command", help="Inpainter template: {input} {mask} {output} {prompt}")
    p.add_argument("--visible-prompt", default=DEFAULT_VISIBLE_PROMPT)
    p.add_argument("--soft-binding", action="store_true")

    # Quality profile, dry-run, timeout
    p.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help="Quality profile for visible cleaning (affects backend selection)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan visible cleaning without running inpainting",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Seconds to wait for each external inpaint command (default: 1800)",
    )

    # Frequency / morphological degradation (Layer V extension)
    degrade = p.add_mutually_exclusive_group()
    degrade.add_argument(
        "--degrade",
        choices=list(DEGRADE_CLI_CHOICES),
        help="Frequency-domain image degradation",
    )
    degrade.add_argument(
        "--morpho",
        choices=list(MORPHO_CLI_CHOICES),
        help="Morphological perturbation",
    )
    p.add_argument(
        "--degrade-strength",
        type=float,
        default=0.6,
        help="Degradation strength (0-1; freq-dct only)",
    )
    p.add_argument(
        "--degrade-seed", type=int, default=None, help="Deterministic seed for degradation"
    )

    # SynthID detection-evasion (Layer V)
    p.add_argument(
        "--remove-synthid",
        action="store_true",
        help="Remove SynthID-class spectral signal via DCT mid-frequency band suppression "
        "(seed-independent, best-effort; PNG output only)",
    )
    p.add_argument(
        "--synthid-strength",
        type=float,
        default=0.6,
        help="SynthID removal strength (0-1; default: 0.6)",
    )
    p.add_argument(
        "--wmct-marker",
        action="store_true",
        help="Replace removed provenance with a truthful wmCt marker (PNG output). "
        "Default: strip-without-replacement (frictionless)",
    )
    p.add_argument(
        "--humanize",
        action="store_true",
        help="Image: humanizing post-stage (film grain, chromatic aberration, unsharp, "
        "adaptive polish); off by default. Requires PNG output and the pixel extra",
    )

    # Artifact / audit-trail control
    p.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Publish .mask.pgm / .bak review artifacts (default: frictionless, no artifacts)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress/success output (silent batch op). Errors still surface.",
    )
    p.add_argument(
        "--audit",
        nargs="?",
        const="wm-audit.json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write a JSON audit report (what was removed). Optional path; "
        "default: wm-audit.json. Off by default (no audit trail).",
    )
    return p


def request_from_cli(args) -> CleanRequest:
    """Adapt parser output into the argparse-independent canonical request."""
    values = vars(args).copy()
    values["paths"] = tuple(values.pop("path", ()) or ())
    return CleanRequest(**values)


def main() -> int:
    args = _build_parser().parse_args()
    try:
        request = request_from_cli(args)
    except ValueError as error:
        eprint(f"invalid options: {error}")
        return ExitCode.USAGE_ERROR.value
    if request.in_place and request.output:
        eprint("warning: -o ignored with --in-place")
    try:
        run = execute(request)
    except CleanPlanPreflightError as error:
        result = {
            "kind": "unknown",
            "input": str(error.path),
            "output": str(error.output),
            "action_details": [
                {
                    "code": "failed",
                    "effect": "error",
                    "text": f"error: {error}",
                    "params": {},
                }
            ],
            "error": str(error),
            "exit_code": ExitCode.RESIDUAL_OR_ERROR.value,
        }
        if request.json:
            payload = {"total": 1, "results": [result]} if len(request.paths) > 1 else result
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            eprint(f"error on {error.path}: {error}")
        return ExitCode.RESIDUAL_OR_ERROR.value
    except ValueError as error:
        if request.json:
            entry = {"error": str(error), "exit_code": ExitCode.USAGE_ERROR.value}
            payload = {"total": 1, "results": [entry]} if len(request.paths) > 1 else entry
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        label = "invalid input selection" if "glob" in str(error) else "invalid output selection"
        eprint(f"{label}: {error}")
        return ExitCode.USAGE_ERROR.value

    results = [item.payload for item in run.results]
    payload: dict | list = {"total": len(results), "results": results} if run.batch else results[0]
    _write_audit(request, payload)
    if request.dry_run and not request.json and not request.quiet:
        for result in results:
            print(f"dry-run: {result['input']} -> {result['output']}")
            for action in action_sentences(result.get("action_details", ())):
                print(f"  - {action}")
    elif request.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for result in results:
            # Errors surface even with --quiet; everything else is suppressible.
            if result.get("error"):
                eprint(f"error on {result['input']}: {result['error']}")
            elif not request.quiet:
                _present_result(result)
            if result.get("skipped_text_transforms") and not request.quiet:
                eprint(
                    f"note: {Path(result['input']).name} skipped "
                    f"{', '.join(result['skipped_text_transforms'])}"
                )
        if run.batch and not request.quiet:
            errors = sum(r.get("exit_code", 0) != 0 for r in results)
            eprint(f"done: {len(results)} file(s), {errors} with warnings/errors")
    if all(r.get("exit_code", 0) == 0 for r in results):
        return ExitCode.SUCCESS.value
    return ExitCode.RESIDUAL_OR_ERROR.value


def _write_audit(request: CleanRequest, payload: dict | list) -> None:
    """Write the JSON audit report when --audit is requested.

    The audit is the same combined payload as ``--json``; it is written to a
    file so the frictionless default (no audit trail) can be overridden when a
    record of what was removed is wanted.  Uses atomic write.

    The bare ``--audit`` default resolves next to the output (or into the
    batch output directory); an explicit ``--audit PATH`` is honored verbatim.
    """
    if request.audit is None:
        return
    audit_path = Path(request.audit)
    if str(audit_path) == "wm-audit.json":
        if isinstance(payload, dict) and payload.get("output"):
            audit_path = Path(payload["output"]).resolve().parent / "wm-audit.json"
        elif request.output:
            audit_path = Path(request.output).resolve() / "wm-audit.json"
        else:
            audit_path = Path.cwd() / "wm-audit.json"
    elif audit_path.is_dir():
        audit_path = audit_path / "wm-audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        audit_path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def _present_result(payload: dict) -> None:
    """Print what one cleaned file became; a nonzero exit_code means residual risk."""
    output, residual = payload["output"], payload.get("exit_code", 0) != 0
    if payload["kind"] == "text":
        stats = payload["stats"]
        eprint(
            f"wrote {output} removed={stats['removed_count']} replaced={stats['replaced_count']}"
        )
        if residual:
            eprint("warning: residual hidden characters may remain")
        return
    if payload["kind"] == "image":
        eprint(f"wrote {output} ({payload['bytes_in']} -> {payload['bytes_out']})")
    else:
        eprint(f"wrote {output} format={payload['format']}")
    for action in action_sentences(payload.get("action_details", ())):
        eprint(f"  - {action}")
    if residual:
        eprint("warning: residual C2PA/AI/soft-binding signals may remain")
        for finding in payload.get("post_findings") or []:
            eprint(f"  ! {finding}")


if __name__ == "__main__":
    raise SystemExit(main())
