#!/usr/bin/env python3
"""Inspect text for invisible Unicode / space homoglyphs (Layer A)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TEXT_TOOL_ADVICE, emit_json, read_text_input
from text_unicode import human_report, inspect_text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Text file path, or - for stdin")
    p.add_argument("--json", action="store_true", help="JSON report")
    p.add_argument(
        "--aggressive",
        action="store_true",
        help="Also flag Latin confusable / fullwidth lookalikes",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Treat binary-looking input as text anyway",
    )
    p.add_argument(
        "--strip-emoji-glue",
        action="store_true",
        help="Strip load-bearing emoji glue even in context",
    )
    args = p.parse_args()

    text = read_text_input(args.path, allow_binary=args.force_text, advice=TEXT_TOOL_ADVICE)
    report = inspect_text(text, aggressive=args.aggressive, strip_emoji_glue=args.strip_emoji_glue)
    if args.json:
        emit_json(report.to_dict())
    else:
        print(human_report(report))
    return 0 if report.suspicious_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
