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

from batch_inputs import collect_inputs
from common import emit_json, eprint, read_text_input
from container_meta import detect_container_format, inspect_container
from image_meta import detect_format as detect_image_format
from image_meta import inspect_image
from inspect_soft_binding import inspect_soft_binding
from text_unicode import human_report, inspect_text

TEXT_EXTS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".mdx",
    ".html",
    ".htm",
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
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".avif"}
CONTAINER_EXTS = {".svg", ".pdf", ".docx", ".odt", ".html", ".htm", ".md", ".markdown", ".mdx"}
SUPPORTED_EXTS = TEXT_EXTS | IMAGE_EXTS | CONTAINER_EXTS


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

    missing = [source for source in args.path if not source.exists()]
    if missing:
        for source in missing:
            eprint(f"not a file or directory: {source}")
        return 2
    items = collect_inputs(
        args.path,
        recursive=args.recursive,
        pattern=args.glob,
        extensions=SUPPORTED_EXTS,
    )
    if not items:
        eprint("no matching input files")
        return 2
    results = [_inspect_single(item.path, args) for item in items]
    batch = len(items) > 1 or any(source.is_dir() for source in args.path)
    if args.json:
        emit_json({"total": len(results), "results": results} if batch else results[0])
    elif batch:
        eprint(f"inspected {len(results)} file(s)")
    return 0 if all(not r.get("suspicious", False) for r in results) else 1


def _inspect_single(path: Path, args) -> dict:
    kind = args.force_type if args.force_type != "auto" else classify(path)
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
