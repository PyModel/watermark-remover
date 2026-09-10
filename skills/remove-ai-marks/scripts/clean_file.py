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
from asset_kind import SUPPORTED_EXTENSIONS
from batch_inputs import select_inputs
from clean_asset import (
    DEGRADE_CLI_CHOICES,
    MORPHO_CLI_CHOICES,
    CleanPlan,
    CleanResult,
    clean_asset,
)
from clean_request import (
    REWRITE_CLI_CHOICES,
    CleanPlanPreflightError,
    CleanRequest,
    describe_dropped_text_transforms,
    dropped_text_transforms,
    plan_work,
)
from common import (
    atomic_write_text,
    cleaned_path,
    eprint,
)
from morphomod import VISIBLE_CLEAN_BACKENDS
from operation import ExitCode, OperationStatus, status_to_exit_code
from perturb_text import MODES as PERTURB_MODES
from rewrite_text import LIVE_REWRITE_BACKENDS, REASONING_EFFORTS, TokenSink, remote_warning

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
    p.add_argument("--dilate", type=int, default=None, metavar="RADIUS")
    p.add_argument(
        "--visible-backend",
        choices=VISIBLE_CLEAN_BACKENDS,
        default="texture",
    )
    p.add_argument("--inpaint-command", help="Inpainter template: {input} {mask} {output} {prompt}")
    p.add_argument("--visible-prompt", default="Remove watermark, fill with background")
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
        default=1800.0,
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


def main() -> int:
    args = _build_parser().parse_args()
    try:
        request = CleanRequest.from_args(args)
    except ValueError as error:
        eprint(f"invalid options: {error}")
        return ExitCode.USAGE_ERROR.value
    allowed = request.allowed_extensions(SUPPORTED_EXTENSIONS)
    excluded_roots = (
        (request.output,)
        if request.output
        and not request.in_place
        and any(source.is_dir() for source in request.paths)
        else ()
    )
    try:
        selection = select_inputs(
            request.paths,
            recursive=request.recursive,
            pattern=request.glob,
            extensions=allowed,
            excluded_roots=excluded_roots,
        )
    except ValueError as error:
        eprint(f"invalid input selection: {error}")
        return ExitCode.USAGE_ERROR.value
    items = selection.items
    batch = selection.batch
    if batch and (request.visible_mask or request.visible_box):
        eprint(
            "error: --visible-mask/--visible-box are single-file options; use --detect-command for batch"
        )
        return ExitCode.USAGE_ERROR.value
    if request.in_place and request.output:
        eprint("warning: -o ignored with --in-place")
    try:
        work = plan_work(items, request, batch)
    except CleanPlanPreflightError as error:
        result = _error_payload(error.path, error.output, error)
        if request.json:
            payload = {"total": 1, "results": [result]} if batch else result
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            eprint(f"error on {error.path}: {error}")
        return ExitCode.RESIDUAL_OR_ERROR.value
    except ValueError as error:
        # Preflight refusals (unrecognized format, binary-as-text, oversized
        # input, output collisions) must not vanish as a bare exit code when
        # --json is requested: emit the same structured shape a batch consumer
        # can parse, while keeping the human message and the usage-error exit.
        if request.json:
            entry = {"error": str(error), "exit_code": ExitCode.USAGE_ERROR.value}
            payload = {"total": 1, "results": [entry]} if batch else entry
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        eprint(f"invalid output selection: {error}")
        return ExitCode.USAGE_ERROR.value
    if request.dry_run:
        results = [
            dry_run_payload(item.path, output, plan, request.in_place)
            for item, output, plan in work
        ]
        payload = {"total": len(results), "results": results} if batch else results[0]
        _write_audit(request, payload)
        if request.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif not request.quiet:
            for result in results:
                print(f"dry-run: {result['input']} -> {result['output']}")
                for action in result["actions"]:
                    print(f"  - {action}")
        return ExitCode.SUCCESS.value

    if batch and request.output and not request.in_place:
        request.output.mkdir(parents=True, exist_ok=True)

    results = [run_clean_item(item.path, output, request, plan) for item, output, plan in work]

    payload: dict | list = {"total": len(results), "results": results} if batch else results[0]
    _write_audit(request, payload)
    if request.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif batch and not request.quiet:
        errors = sum(r.get("exit_code", 0) != 0 for r in results)
        eprint(f"done: {len(results)} file(s), {errors} with warnings/errors")
    if all(r.get("exit_code", 0) == 0 for r in results):
        return ExitCode.SUCCESS.value
    return ExitCode.RESIDUAL_OR_ERROR.value


def dry_run_payload(
    path: Path,
    output: Path | None,
    plan: CleanPlan,
    in_place: bool,
) -> dict:
    """Describe what a visible-mark clean would do, without touching a file.

    Shared with the TUI so its preview is the CLI's preview, not a second
    description that can drift from what actually runs.
    """
    visible = plan.visible
    if visible is None:
        raise ValueError("dry-run requires a visible cleaning plan")
    destination = path if in_place else output or cleaned_path(path)
    if visible.mask_path is not None:
        localization = f"mask:{visible.mask_path}"
    elif visible.box is not None:
        localization = f"box:{','.join(map(str, visible.box))}"
    else:
        localization = "external-detector"
    actions = [
        f"localize visible mark via {localization}",
        f"fill holes + dilate radius={visible.dilation_radius}",
        f"inpaint with {visible.backend} backend",
        "strip requested metadata",
    ]
    if plan.degrade is not None:
        actions.append(f"apply {plan.degrade.strategy} degradation")
    actions.append(f"publish mask to {visible.mask_output} and image to {destination}")
    return {
        "kind": "image",
        "status": "dry-run",
        "input": str(path),
        "output": str(destination),
        "mask": str(visible.mask_output),
        "backend": visible.backend,
        "timeout": visible.timeout,
        "actions": actions,
        "exit_code": ExitCode.SUCCESS.value,
    }


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


def _error_payload(path: Path, output: Path, error: Exception) -> dict:
    return {
        "kind": "unknown",
        "input": str(path),
        "output": str(output),
        "actions": [f"error: {error}"],
        "error": str(error),
        "exit_code": status_to_exit_code(OperationStatus.FAILED),
    }


def _present_result(result: CleanResult, payload: dict, *, quiet: bool = False) -> None:
    if quiet:
        return
    if result.kind == "text":
        stats = payload["stats"]
        eprint(
            f"wrote {result.output} removed={stats['removed_count']} "
            f"replaced={stats['replaced_count']}"
        )
        return
    if result.kind == "image":
        eprint(f"wrote {result.output} ({payload['bytes_in']} -> {payload['bytes_out']})")
        for action in payload.get("actions", []):
            eprint(f"  - {action}")
        if result.residual:
            eprint("warning: residual C2PA/AI/soft-binding signals may remain")
        return

    eprint(f"wrote {result.output} format={payload['format']}")
    for action in payload.get("actions", []):
        eprint(f"  - {action}")
    if result.residual:
        eprint("warning: residual C2PA/AI metadata remains")
        for finding in payload.get("post_findings") or []:
            eprint(f"  ! {finding}")


def run_clean_item(
    path: Path,
    output_path: Path | None,
    request: CleanRequest,
    plan: CleanPlan,
    *,
    on_token: TokenSink | None = None,
) -> dict:
    """Clean one asset and return its JSON payload.

    Shared with the TUI, which passes a request with ``json=True`` so this
    stays silent and the caller renders the payload itself, plus an *on_token*
    sink so a Layer B rewrite is visible while it runs.
    """
    dest = path if request.in_place else output_path or cleaned_path(path)
    try:
        rewrite_plan = plan.text.rewrite_plan
        if rewrite_plan is not None and (warning := remote_warning(rewrite_plan.base_url)):
            eprint(warning)
        result = clean_asset(path, dest, plan, on_token=on_token)
    except Exception as error:
        if not request.json:
            eprint(f"error on {path}: {error}")
        return _error_payload(path, dest, error)

    payload = result.to_dict()
    status = OperationStatus.VERIFIED if not result.residual else OperationStatus.RESIDUAL_RISK
    payload["exit_code"] = status_to_exit_code(status)
    # Text-only options are deliberate no-ops off the text path; say so rather
    # than let the caller assume the body was rewritten.
    dropped = dropped_text_transforms(request, result.kind)
    if dropped:
        payload["skipped_text_transforms"] = list(dropped)
    if not request.json:
        _present_result(result, payload, quiet=request.quiet)
        if dropped and not request.quiet:
            eprint(describe_dropped_text_transforms(request, result.kind, path))
    return payload


#: Preserved private aliases for in-tree callers predating the renames.
_run_clean_item = run_clean_item
_dry_run_payload = dry_run_payload


if __name__ == "__main__":
    raise SystemExit(main())
