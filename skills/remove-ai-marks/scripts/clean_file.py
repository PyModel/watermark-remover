#!/usr/bin/env python3
"""Unified cleaner for text, images, and document containers.

Supports single files, multiple files, and directory batches (--glob,
--recursive, --extensions). Advanced transforms are opt-in:
  --tsapa          evolutionary Layer B rewrite (requires a live configured backend)
  --char-perturb   character-level anti-watermark noise (intentionally after Layer A)
  --visible-*      MorphoMod mask→dilate→inpaint pipeline
  --soft-binding   detect residual soft-binding / remote-manifest risk
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_kind import SUPPORTED_EXTENSIONS, AssetKind, classify_asset
from batch_inputs import InputItem, safe_output_path, select_inputs
from clean_asset import (
    DEGRADE_CLI_CHOICES,
    MORPHO_CLI_CHOICES,
    CleanPlan,
    CleanResult,
    TextCleanPlan,
    clean_asset,
)
from common import backup_path, cleaned_path, eprint, paths_alias, validate_output_path
from morphomod import DEFAULT_DILATION_RADIUS, VISIBLE_CLEAN_BACKENDS, VisiblePlan
from operation import ExitCode, OperationStatus, status_to_exit_code
from perturb_text import MODES as PERTURB_MODES
from rewrite_text import RewritePlan, remote_warning


class _CleanPlanPreflightError(RuntimeError):
    """A per-asset policy failed before batch execution."""

    def __init__(self, path: Path, output: Path, error: Exception) -> None:
        super().__init__(str(error))
        self.path = path
        self.output = output


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

    p.add_argument("--tsapa", action="store_true", help="Text: live TSAPA evolutionary rewrite")
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
    return p


def main() -> int:
    args = _build_parser().parse_args()
    allowed = SUPPORTED_EXTENSIONS
    if args.extensions:
        allowed = {
            "." + extension.strip().lstrip(".").lower()
            for extension in args.extensions.split(",")
            if extension.strip()
        }
    excluded_roots = (
        (args.output,)
        if args.output and not args.in_place and any(source.is_dir() for source in args.path)
        else ()
    )
    try:
        selection = select_inputs(
            args.path,
            recursive=args.recursive,
            pattern=args.glob,
            extensions=allowed,
            excluded_roots=excluded_roots,
        )
    except ValueError as error:
        eprint(f"invalid input selection: {error}")
        return ExitCode.USAGE_ERROR.value
    items = selection.items
    batch = selection.batch
    if batch and (args.visible_mask or args.visible_box):
        eprint(
            "error: --visible-mask/--visible-box are single-file options; use --detect-command for batch"
        )
        return ExitCode.USAGE_ERROR.value
    if args.in_place and args.output:
        eprint("warning: -o ignored with --in-place")
    try:
        work = _plan_work(items, args, batch)
    except _CleanPlanPreflightError as error:
        result = _error_payload(error.path, error.output, error)
        if args.json:
            payload = {"total": 1, "results": [result]} if batch else result
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            eprint(f"error on {error.path}: {error}")
        return ExitCode.RESIDUAL_OR_ERROR.value
    except ValueError as error:
        eprint(f"invalid output selection: {error}")
        return ExitCode.USAGE_ERROR.value
    if batch and args.output and not args.in_place:
        args.output.mkdir(parents=True, exist_ok=True)

    results = [_run_clean_item(item.path, output, args, plan) for item, output, plan in work]

    if args.json:
        payload: dict | list = {"total": len(results), "results": results} if batch else results[0]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif batch:
        errors = sum(r.get("exit_code", 0) != 0 for r in results)
        eprint(f"done: {len(results)} file(s), {errors} with warnings/errors")
    if all(r.get("exit_code", 0) == 0 for r in results):
        return ExitCode.SUCCESS.value
    return ExitCode.RESIDUAL_OR_ERROR.value


def _plan_work(
    items: Sequence[InputItem], args, batch: bool
) -> list[tuple[InputItem, Path | None, CleanPlan]]:
    """Resolve and validate every destination and plan before the first write."""
    inputs = [item.path for item in items]
    ancillary_inputs = [candidate for candidate in (args.visible_mask,) if candidate is not None]
    for ancillary in ancillary_inputs:
        if not ancillary.is_file() or ancillary.is_symlink():
            raise ValueError(f"not a regular mask file: {ancillary}")
    all_inputs = [*inputs, *ancillary_inputs]
    destinations: list[Path] = []
    work: list[tuple[InputItem, Path | None, CleanPlan]] = []

    for item in items:
        if args.in_place:
            backup = backup_path(item.path)
            if backup.exists() or backup.is_symlink():
                raise ValueError(f"backup already exists: {backup}")
            output = None
            dest = item.path
        else:
            if args.output is None:
                output = cleaned_path(item.path)
            elif batch:
                output = safe_output_path(args.output, item.relative)
            else:
                output = args.output

            validate_output_path(item.path, output)
            for source in all_inputs:
                if paths_alias(output, source):
                    raise ValueError(f"output aliases an input: {output}")
            for existing in destinations:
                if paths_alias(output, existing):
                    raise ValueError(f"batch output collision: {output}")
            destinations.append(output)
            dest = output

        try:
            kind = classify_asset(item.path, forced_kind=args.force_type)
            plan = _build_clean_plan(args, dest, kind)
        except Exception as error:
            raise _CleanPlanPreflightError(item.path, dest, error) from error

        if plan.visible is not None:
            mask_output = plan.visible.mask_output
            if mask_output is None:
                raise ValueError("visible plan is missing a mask output path")
            if mask_output.is_symlink():
                raise ValueError(f"mask output is a symlink: {mask_output}")
            if any(paths_alias(mask_output, source) for source in all_inputs):
                raise ValueError(f"mask output aliases an input: {mask_output}")
            if any(paths_alias(mask_output, existing) for existing in destinations):
                raise ValueError(f"mask/output collision: {mask_output}")
            destinations.append(mask_output)

        work.append((item, output, plan))
    return work


def _build_clean_plan(args, dest: Path, kind: AssetKind) -> CleanPlan:
    text_plan = TextCleanPlan()
    if kind == "text":
        rewrite_plan = (
            RewritePlan.live_tsapa_from_environment(
                generations=args.tsapa_generations,
                population=args.tsapa_population,
            )
            if args.tsapa
            else None
        )
        text_plan = TextCleanPlan(
            nfkc=args.nfkc,
            aggressive_homoglyphs=args.aggressive_homoglyphs,
            preserve_semantic=not args.strip_semantic_format,
            rewrite_plan=rewrite_plan,
            perturb_mode=args.char_mode if args.char_perturb else None,
            perturb_strength=args.char_strength if args.char_perturb else 0.1,
            perturb_seed=args.seed if args.char_perturb else None,
        )

    visible_plan = None
    if kind == "image" and _visible_requested(args):
        visible_plan = VisiblePlan(
            mask_path=args.visible_mask,
            box=args.visible_box,
            detect_command=args.detect_command,
            backend=args.visible_backend,
            command=args.inpaint_command,
            dilation_radius=(args.dilate if args.dilate is not None else DEFAULT_DILATION_RADIUS),
            mask_output=dest.with_name(f"{dest.stem}.mask.pgm"),
            prompt=args.visible_prompt,
        )

    # Build degradation plan for images
    degrade_plan = None
    if kind == "image":
        from clean_asset import ImageDegradePlan

        if args.degrade:
            degrade_plan = ImageDegradePlan(
                strategy=args.degrade,
                strength=args.degrade_strength,
                seed=args.degrade_seed,
            )
        elif args.morpho:
            degrade_plan = ImageDegradePlan(
                strategy=args.morpho,
                strength=args.degrade_strength,
                seed=args.degrade_seed,
            )

    return CleanPlan(
        forced_kind=kind,
        in_place=args.in_place,
        text=text_plan,
        strip_all_metadata=not args.keep_non_ai_metadata,
        visible=visible_plan,
        inspect_soft_binding=args.soft_binding,
        degrade=degrade_plan,
    )


def _visible_requested(args) -> bool:
    return any(
        (
            args.visible_mask,
            args.visible_box,
            args.detect_command,
            args.dilate is not None,
            args.inpaint_command,
            args.visible_backend != "texture",
        )
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


def _present_result(result: CleanResult, payload: dict) -> None:
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


def _run_clean_item(
    path: Path,
    output_path: Path | None,
    args,
    plan: CleanPlan,
) -> dict:
    dest = path if args.in_place else output_path or cleaned_path(path)
    try:
        rewrite_plan = plan.text.rewrite_plan
        if rewrite_plan is not None and (warning := remote_warning(rewrite_plan.base_url)):
            eprint(warning)
        result = clean_asset(path, dest, plan)
    except Exception as error:
        if not args.json:
            eprint(f"error on {path}: {error}")
        return _error_payload(path, dest, error)

    payload = result.to_dict()
    status = OperationStatus.VERIFIED if not result.residual else OperationStatus.RESIDUAL_RISK
    payload["exit_code"] = status_to_exit_code(status)
    if not args.json:
        _present_result(result, payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
