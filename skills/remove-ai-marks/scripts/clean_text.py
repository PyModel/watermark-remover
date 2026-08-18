#!/usr/bin/env python3
"""Strip invisible Unicode / normalize space homoglyphs (Layer A)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    TEXT_TOOL_ADVICE,
    cleaned_path,
    create_backup,
    eprint,
    read_text_input,
    validate_output_path,
    write_text_output,
)
from text_unicode import clean_text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout or *.cleaned.*)")
    p.add_argument("--nfkc", action="store_true", help="Apply Unicode NFKC after scrub")
    p.add_argument(
        "--aggressive-homoglyphs",
        action="store_true",
        help="Map Cyrillic/fullwidth Latin confusables to ASCII Latin",
    )
    p.add_argument(
        "--no-normalize-spaces",
        action="store_true",
        help="Do not rewrite exotic spaces to U+0020",
    )
    p.add_argument(
        "--strip-semantic-format",
        action="store_true",
        help="Aggressive: also strip contextual ZWJ/ZWNJ, variation selectors, math and balanced bidi controls",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Treat binary-looking input as text anyway",
    )
    p.add_argument("--stats", action="store_true", help="Print stats JSON to stderr")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file (creates .bak backup)",
    )
    args = p.parse_args()

    if args.in_place and args.path in (None, "-"):
        eprint("--in-place requires a file path")
        return 2
    if args.path not in (None, "-"):
        source = Path(args.path)
        if not source.is_file() or source.is_symlink():
            eprint(f"not a regular file: {source}")
            return 2

    text = read_text_input(args.path, allow_binary=args.force_text, advice=TEXT_TOOL_ADVICE)
    cleaned, stats = clean_text(
        text,
        nfkc=args.nfkc,
        aggressive_homoglyphs=args.aggressive_homoglyphs,
        normalize_spaces=not args.no_normalize_spaces,
        preserve_semantic=not args.strip_semantic_format,
    )

    out = args.output
    try:
        if args.in_place:
            source = Path(args.path)
            create_backup(source)
            out = str(source)
        elif args.path not in (None, "-"):
            source = Path(args.path)
            destination = Path(out) if out is not None else cleaned_path(source)
            validate_output_path(source, destination)
            out = str(destination)
        write_text_output(cleaned, out)
    except (OSError, ValueError) as error:
        eprint(f"error: {error}")
        return 1

    if args.stats:
        eprint(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"removed={stats['removed_count']} replaced={stats['replaced_count']} "
            f"preserved={stats['preserved_count']} "
            f"len {stats['input_length']}->{stats['output_length']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
