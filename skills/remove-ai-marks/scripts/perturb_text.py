#!/usr/bin/env python3
"""Character-level perturbations that disrupt statistical text watermarks (NDSS 2026 class).

WARNING — this is an opt-in *anti-detection* transform, not a hygiene pass:
modes inject the very invisible-Unicode artifacts that Layer A removes, or alter
visible characters. Do NOT run clean_text on the output (Layer A undoes it).
Best-effort only; no guarantee against any vendor detector. For text you own.

Modes:
  zero-width   inject ZWSP inside words (invisible; Layer-A reversible)
  space-swap   replace some spaces with U+00A0/U+2002/U+2003 (invisible-ish; reversible)
  confusable   swap some Latin letters for Cyrillic lookalikes (VISIBLY identical,
               NOT reversible, breaks search/copy semantics)
  case         random case flips (VISIBLE changes; only for case-insensitive contexts)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cleaned_path, eprint, read_text_input, validate_output_path, write_text_output

MODES = ("zero-width", "space-swap", "confusable", "case")

# Only ZWSP is injected here: ZWNJ/ZWJ/WORD JOINER may be semantically
# meaningful in script/math contexts and are preserved by Layer A by default.
ZW_CHARS = ("​",)
SPACE_ALTS = (" ", " ", " ")
CONFUSABLES = {
    "a": "а",
    "e": "е",
    "o": "о",
    "p": "р",
    "c": "с",
    "x": "х",
    "i": "і",
    "A": "А",
    "E": "Е",
    "O": "О",
    "P": "Р",
    "C": "С",
    "X": "Х",
    "I": "І",
}


def perturb_text(
    text: str,
    *,
    mode: str = "zero-width",
    strength: float = 0.1,
    seed: int | None = None,
) -> tuple[str, dict]:
    """Apply character-level perturbations. strength = fraction of eligible positions."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be a finite number in [0, 1]")
    rng = random.Random(seed)  # noqa: S311
    out: list[str] = []
    changed = 0

    if mode == "zero-width":
        for ch in text:
            out.append(ch)
            if ch.isalnum() and rng.random() < strength:
                out.append(rng.choice(ZW_CHARS))
                changed += 1
    elif mode == "space-swap":
        for ch in text:
            if ch == " " and rng.random() < strength:
                out.append(rng.choice(SPACE_ALTS))
                changed += 1
            else:
                out.append(ch)
    elif mode == "confusable":
        for ch in text:
            if ch in CONFUSABLES and rng.random() < strength:
                out.append(CONFUSABLES[ch])
                changed += 1
            else:
                out.append(ch)
    else:  # case
        for ch in text:
            if ch.isalpha() and rng.random() < strength:
                out.append(ch.swapcase())
                changed += 1
            else:
                out.append(ch)

    result = "".join(out)
    stats = {
        "mode": mode,
        "strength": strength,
        "seed": seed,
        "changed": changed,
        "input_chars": len(text),
        "output_chars": len(result),
        "reversible_by_layer_a": mode in ("zero-width", "space-swap"),
        "note": (
            "Best-effort anti-watermark noise; no guarantee against any vendor detector. "
            + (
                "Reversible: Layer A (clean_text) restores the original."
                if mode in ("zero-width", "space-swap")
                else "NOT reversible: output differs from the original at the character level."
            )
        ),
    }
    return result, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout)")
    p.add_argument("--mode", choices=MODES, default="zero-width")
    p.add_argument("--strength", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=None, help="Deterministic RNG seed")
    p.add_argument("--json", action="store_true", help="Stats JSON on stderr")
    args = p.parse_args()

    text = read_text_input(args.path)
    result, stats = perturb_text(text, mode=args.mode, strength=args.strength, seed=args.seed)
    out = args.output
    if out is None and args.path not in (None, "-"):
        out = str(cleaned_path(Path(args.path), suffix=".perturbed"))
    if args.path not in (None, "-") and out not in (None, "-"):
        try:
            validate_output_path(Path(args.path), Path(out))
        except ValueError as error:
            eprint(f"error: {error}")
            return 2
    write_text_output(result, out)
    if args.json:
        eprint(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"mode={stats['mode']} changed={stats['changed']} "
            f"chars {stats['input_chars']}->{stats['output_chars']} "
            f"reversible={stats['reversible_by_layer_a']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
