#!/usr/bin/env python3
"""Unified inspection for text, images, and document containers.

Supports single files, multiple files, and directory batches (--glob,
--recursive). --soft-binding adds detection for remote-manifest / in-content
soft-binding risk without claiming removal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_kind import SUPPORTED_EXTENSIONS, classify_asset
from batch_inputs import select_inputs
from common import EXIT_PARTIAL, MAX_INPUT_BYTES, emit_json, eprint, read_text_input
from container_meta import inspect_container
from image_meta import inspect_image
from inspect_soft_binding import inspect_soft_binding
from text_unicode import human_report, inspect_text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="+", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--aggressive", action="store_true", help="Text: flag confusables")
    p.add_argument("--soft-binding", action="store_true")
    p.add_argument(
        "--as", dest="force_type", choices=("text", "image", "container", "auto"), default="auto"
    )
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--glob", default="*")
    args = p.parse_args()

    try:
        selection = select_inputs(
            args.path,
            recursive=args.recursive,
            pattern=args.glob,
            extensions=SUPPORTED_EXTENSIONS,
        )
    except ValueError as error:
        eprint(f"invalid input selection: {error}")
        return 2
    results = [_inspect_single(item.path, args) for item in selection.items]
    batch = selection.batch
    if args.json:
        emit_json({"total": len(results), "results": results} if batch else results[0])
    elif batch:
        eprint(f"inspected {len(results)} file(s)")
    # An incomplete audit is the more important CI signal: any input that was
    # not scanned (unrecognized or refused) outranks both clean and suspicious.
    if any(r.get("unscanned") for r in results):
        return EXIT_PARTIAL
    return 0 if all(not r.get("suspicious", False) for r in results) else 1


def _inspect_single(path: Path, args) -> dict:
    if path.stat().st_size > MAX_INPUT_BYTES:
        return {
            "kind": "refused",
            "path": str(path),
            "note": f"input larger than {MAX_INPUT_BYTES} bytes",
            "suspicious": False,
            "unscanned": True,
        }
    kind = classify_asset(path, forced_kind=args.force_type)
    if kind == "text":
        report = inspect_text(read_text_input(str(path)), aggressive=args.aggressive)
        if not args.json:
            print("Kind: text")
            print(human_report(report))
        return {
            "kind": "text",
            "path": str(path),
            **report.to_dict(),
            "suspicious": report.suspicious_total > 0,
        }

    if kind == "image":
        report = inspect_image(path)
        soft = inspect_soft_binding(path) if args.soft_binding else None
        soft_found = bool(soft and soft["soft_binding"]["found"])
        if not args.json:
            print("Kind: image")
            print(f"Path: {report.path}")
            print(f"Format: {report.format}")
            print(f"C2PA: {report.has_c2pa}")
            print(f"AI metadata: {report.has_ai_metadata}")
            for finding in report.findings:
                print(f"  - {finding}")
            if soft_found:
                print(f"Soft binding: {soft['soft_binding']['labels']}")
        result = {
            "kind": "image",
            "path": str(path),
            **report.to_dict(),
            "suspicious": report.has_c2pa or report.has_ai_metadata or soft_found,
        }
        if soft is not None:
            result["soft_binding"] = soft
        return result

    if kind == "unknown":
        note = "These bytes match no supported text, image or container format."
        if not args.json:
            print("Kind: unknown")
            print(f"Path: {path}")
            print(f"Note: {note}")
            print("Use --as text|image|container to force a pipeline.")
        return {
            "kind": "unknown",
            "path": str(path),
            "note": note,
            "suspicious": False,
            "unscanned": True,
        }

    report = inspect_container(path)
    if not args.json:
        print("Kind: container")
        print(f"Path: {report.path}")
        print(f"Format: {report.format}")
        print(f"C2PA: {report.has_c2pa}")
        print(f"AI metadata: {report.has_ai_metadata}")
        for finding in report.findings:
            print(f"  - {finding}")
    return {
        "kind": "container",
        "path": str(path),
        **report.to_dict(),
        "suspicious": report.has_c2pa or report.has_ai_metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())
