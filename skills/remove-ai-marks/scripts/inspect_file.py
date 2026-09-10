#!/usr/bin/env python3
"""Unified inspection for text, images, and document containers.

Supports single files, multiple files, and directory batches (--glob,
--recursive). --soft-binding adds detection for remote-manifest / in-content
soft-binding risk without claiming removal.

``inspect_asset`` is the data seam: it returns the report dict and prints
nothing, so the CLI, the HTTP service, and the TUI all read the same findings.
``_inspect_single`` is the CLI presenter layered on top of it.
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


def inspect_asset(
    path: Path,
    *,
    force_type: str = "auto",
    aggressive: bool = False,
    soft_binding: bool = False,
) -> dict:
    """Inspect one asset and return its report. Prints nothing.

    The returned dict always carries ``kind``, ``path`` and ``suspicious``;
    the remaining keys are the per-kind report's own fields.
    """
    return _inspect_asset(
        path,
        force_type=force_type,
        aggressive=aggressive,
        soft_binding=soft_binding,
    )[0]


def _inspect_asset(
    path: Path,
    *,
    force_type: str,
    aggressive: bool,
    soft_binding: bool,
) -> tuple[dict, object | None]:
    """Inspect once, returning the payload and the text report it came from.

    The CLI's human rendering needs the ``TextReport`` object, so it is handed
    back here rather than re-read from disk — one read, one classification, one
    set of findings for every consumer.
    """
    if path.stat().st_size > MAX_INPUT_BYTES:
        return (
            {
                "kind": "refused",
                "path": str(path),
                "note": f"input larger than {MAX_INPUT_BYTES} bytes",
                "suspicious": False,
                "unscanned": True,
            },
            None,
        )
    kind = classify_asset(path, forced_kind=force_type)

    if kind == "text":
        report = inspect_text(read_text_input(str(path)), aggressive=aggressive)
        return (
            {
                "kind": "text",
                "path": str(path),
                **report.to_dict(),
                "suspicious": report.suspicious_total > 0,
            },
            report,
        )

    if kind == "image":
        report = inspect_image(path)
        soft = inspect_soft_binding(path) if soft_binding else None
        soft_found = bool(soft and soft["soft_binding"]["found"])
        result = {
            "kind": "image",
            "path": str(path),
            **report.to_dict(),
            "suspicious": report.has_c2pa or report.has_ai_metadata or soft_found,
        }
        if soft is not None:
            result["soft_binding"] = soft
        return result, None

    if kind == "unknown":
        return (
            {
                "kind": "unknown",
                "path": str(path),
                "note": "These bytes match no supported text, image or container format.",
                "suspicious": False,
                "unscanned": True,
            },
            None,
        )

    report = inspect_container(path)
    return (
        {
            "kind": "container",
            "path": str(path),
            **report.to_dict(),
            "suspicious": report.has_c2pa or report.has_ai_metadata,
        },
        None,
    )


def _inspect_single(path: Path, args) -> dict:
    """CLI presenter: inspect, then render the human report when not --json."""
    result, text_report = _inspect_asset(
        path,
        force_type=args.force_type,
        aggressive=args.aggressive,
        soft_binding=args.soft_binding,
    )
    if args.json:
        return result

    kind = result["kind"]
    if kind == "refused":
        return result
    if kind == "text":
        print("Kind: text")
        print(human_report(text_report))
        return result
    if kind == "unknown":
        print("Kind: unknown")
        print(f"Path: {path}")
        print(f"Note: {result['note']}")
        print("Use --as text|image|container to force a pipeline.")
        return result

    print(f"Kind: {kind}")
    print(f"Path: {result['path']}")
    print(f"Format: {result.get('format')}")
    print(f"C2PA: {result.get('has_c2pa')}")
    print(f"AI metadata: {result.get('has_ai_metadata')}")
    for finding in result.get("findings", []):
        print(f"  - {finding}")
    if kind == "image" and result.get("soft_binding", {}).get("soft_binding", {}).get("found"):
        print(f"Soft binding: {result['soft_binding']['soft_binding']['labels']}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
