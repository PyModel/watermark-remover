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
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_inputs import collect_inputs, safe_output_path
from common import cleaned_path, eprint
from container_meta import clean_container, detect_container_format
from image_meta import clean_image
from image_meta import detect_format as detect_image_format
from inspect_soft_binding import inspect_soft_binding
from morphomod import DEFAULT_DILATION_RADIUS, remove_visible
from perturb_text import MODES as PERTURB_MODES
from perturb_text import perturb_text
from rewrite_text import rewrite
from text_unicode import clean_text

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".avif"}
CONTAINER_EXTS = {".svg", ".pdf", ".docx", ".odt", ".html", ".htm", ".md", ".markdown", ".mdx"}
TEXT_EXTS = {
    ".txt",
    ".text",
    ".css",
    ".js",
    ".py",
    ".rs",
    ".go",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}
SUPPORTED_EXTS = IMAGE_EXTS | CONTAINER_EXTS | TEXT_EXTS


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in CONTAINER_EXTS:
        return "container"
    if ext in TEXT_EXTS:
        return "text"
    data = path.read_bytes()
    if detect_image_format(data) in ("png", "jpeg", "heif", "avif"):
        return "image"
    if detect_container_format(path, data) != "unknown":
        return "container"
    return "text"


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
    p.add_argument("--visible-backend", choices=("simple", "external"), default="simple")
    p.add_argument("--inpaint-command", help="Inpainter template: {input} {mask} {output} {prompt}")
    p.add_argument("--visible-prompt", default="Remove watermark, fill with background")
    p.add_argument("--soft-binding", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    missing = [source for source in args.path if not source.exists()]
    if missing:
        for source in missing:
            eprint(f"not a file or directory: {source}")
        return 2
    allowed = SUPPORTED_EXTS
    if args.extensions:
        allowed = {
            "." + e.strip().lstrip(".").lower() for e in args.extensions.split(",") if e.strip()
        }
    items = collect_inputs(
        args.path,
        recursive=args.recursive,
        pattern=args.glob,
        extensions=allowed,
    )
    if args.output and not args.in_place and any(source.is_dir() for source in args.path):
        output_root = args.output.resolve()
        items = [item for item in items if not item.path.resolve().is_relative_to(output_root)]
    if not items:
        eprint("no matching input files")
        return 2
    batch = len(items) > 1 or any(source.is_dir() for source in args.path)
    if batch and (args.visible_mask or args.visible_box):
        eprint(
            "error: --visible-mask/--visible-box are single-file options; use --detect-command for batch"
        )
        return 2
    if args.in_place and args.output:
        eprint("warning: -o ignored with --in-place")
    if batch and args.output and not args.in_place:
        args.output.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    output_paths: set[Path] = set()
    for item in items:
        if args.in_place or not args.output:
            out = None
        elif batch:
            try:
                out = safe_output_path(args.output, item.relative)
            except ValueError as error:
                results.append(
                    {
                        "kind": "unknown",
                        "input": str(item.path),
                        "output": str(args.output / item.relative),
                        "error": str(error),
                        "exit_code": 1,
                    }
                )
                continue
        else:
            out = args.output
        if out and out in output_paths:
            results.append(
                {
                    "kind": "unknown",
                    "input": str(item.path),
                    "output": str(out),
                    "error": "batch output collision",
                    "exit_code": 1,
                }
            )
            continue
        if out:
            output_paths.add(out)
        results.append(_clean_single_file(item.path, out, args))

    if args.json:
        payload: dict | list = {"total": len(results), "results": results} if batch else results[0]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif batch:
        errors = sum(r.get("exit_code", 0) != 0 for r in results)
        eprint(f"done: {len(results)} file(s), {errors} with warnings/errors")
    return 0 if all(r.get("exit_code", 0) == 0 for r in results) else 1


def _rewrite_tsapa_live(text: str, args) -> tuple[str, dict]:
    backend = os.environ.get("WATERMARKS_REWRITE_BACKEND", "print-prompt")
    if backend not in ("ollama", "openai-compatible"):
        raise ValueError(
            "--tsapa requires a live backend; set WATERMARKS_REWRITE_BACKEND="
            "ollama|openai-compatible"
        )
    model = os.environ.get("WATERMARKS_REWRITE_MODEL")
    if not model:
        raise ValueError("--tsapa requires WATERMARKS_REWRITE_MODEL")
    return rewrite(
        text,
        backend=backend,
        model=model,
        base_url=os.environ.get("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
        api_key=os.environ.get("WATERMARKS_REWRITE_API_KEY"),
        strength="tsapa",
        lang="French",
        original_lang="English",
        timeout=120.0,
        layer_a_after=True,
        generations=args.tsapa_generations,
        population=args.tsapa_population,
    )


def _visible_requested(args) -> bool:
    return any((args.visible_mask, args.visible_box, args.detect_command, args.dilate is not None))


def _clean_single_file(path: Path, output_path: Path | None, args) -> dict:
    try:
        kind = args.force_type if args.force_type != "auto" else classify(path)
        if args.in_place:
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_bytes(path.read_bytes())
            src, dest = backup, path
        else:
            src, dest = path, output_path or cleaned_path(path)

        if kind == "text":
            text = src.read_text(encoding="utf-8", errors="surrogateescape")
            cleaned, stats = clean_text(
                text,
                nfkc=args.nfkc,
                aggressive_homoglyphs=args.aggressive_homoglyphs,
                preserve_semantic=not args.strip_semantic_format,
            )
            if args.tsapa:
                cleaned, tsapa_info = _rewrite_tsapa_live(cleaned, args)
                stats["tsapa"] = tsapa_info.get("tsapa", tsapa_info)
            if args.char_perturb:
                cleaned, perturb_stats = perturb_text(
                    cleaned, mode=args.char_mode, strength=args.char_strength, seed=args.seed
                )
                stats["char_perturb"] = perturb_stats
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(cleaned, encoding="utf-8", errors="surrogateescape")
            if not args.json:
                eprint(
                    f"wrote {dest} removed={stats['removed_count']} replaced={stats['replaced_count']}"
                )
            return {
                "kind": "text",
                "input": str(path),
                "output": str(dest),
                "stats": stats,
                "exit_code": 0,
            }

        if kind == "image":
            soft = inspect_soft_binding(src) if args.soft_binding else None
            visible_report = None
            if _visible_requested(args):
                if not (args.visible_mask or args.visible_box or args.detect_command):
                    raise ValueError(
                        "visible removal requires --visible-mask, --visible-box, or --detect-command"
                    )
                with tempfile.TemporaryDirectory(prefix="wm-visible-") as td:
                    fmt = detect_image_format(src.read_bytes())
                    visible_dest = Path(td) / ("visible.png" if fmt == "png" else src.name)
                    visible_report = remove_visible(
                        src,
                        visible_dest,
                        mask_path=args.visible_mask,
                        box=args.visible_box,
                        detect_command=args.detect_command,
                        backend=args.visible_backend,
                        command=args.inpaint_command,
                        dilation_radius=(
                            args.dilate if args.dilate is not None else DEFAULT_DILATION_RADIUS
                        ),
                        mask_output=dest.with_name(f"{dest.stem}.mask.pgm"),
                        prompt=args.visible_prompt,
                    )
                    if visible_report["status"] != "completed":
                        raise RuntimeError("visible pipeline did not produce an output")
                    result = clean_image(
                        visible_dest, dest, strip_all_metadata=not args.keep_non_ai_metadata
                    )
            else:
                result = clean_image(src, dest, strip_all_metadata=not args.keep_non_ai_metadata)
            result["input"] = str(path)
            if visible_report:
                result["visible"] = visible_report
            if soft:
                result["soft_binding"] = soft
            soft_found = bool(soft and soft["soft_binding"]["found"])
            residual = result["still_has_c2pa"] or result["still_has_ai_metadata"] or soft_found
            if not args.json:
                eprint(f"wrote {result['output']} ({result['bytes_in']} -> {result['bytes_out']})")
                for action in result.get("actions", []):
                    eprint(f"  - {action}")
                if residual:
                    eprint("warning: residual C2PA/AI/soft-binding signals may remain")
            return {"kind": "image", **result, "exit_code": 1 if residual else 0}

        result = clean_container(src, dest)
        residual = result["still_has_c2pa"] or result["still_has_ai_metadata"]
        if not args.json:
            eprint(f"wrote {result['output']} format={result['format']}")
            for action in result.get("actions", []):
                eprint(f"  - {action}")
            if residual:
                eprint("warning: residual C2PA/AI metadata remains")
                for finding in result.get("post_findings") or []:
                    eprint(f"  ! {finding}")
        return {"kind": "container", **result, "exit_code": 1 if residual else 0}
    except Exception as error:
        if not args.json:
            eprint(f"error on {path}: {error}")
        return {
            "kind": "unknown",
            "input": str(path),
            "output": str(output_path or cleaned_path(path)),
            "actions": [f"error: {error}"],
            "error": str(error),
            "exit_code": 1,
        }


if __name__ == "__main__":
    raise SystemExit(main())
