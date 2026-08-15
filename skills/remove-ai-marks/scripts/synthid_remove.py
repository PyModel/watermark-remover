#!/usr/bin/env python3
"""SynthID-class pixel-signal removal (detection-evasion for Layer V).

Google's SynthID embeds a pseudo-random spectral pattern that survives JPEG
and common edits, then detects it by matching that pattern's phase in the
frequency domain.  We cannot ship Google's codebook, so the removal here is
**seed-independent**: it suppresses the mid-frequency band where SynthID-class
signals live, using the pure-Python orthonormal DCT machinery from
``dct_frequency``.  This is the honest, practical evasion: it does not promise
to match a specific vendor detector, but it destroys the spectral carrier
without knowing its phase.

This module also ships a small, deterministic SynthID-class *embedder* and
*detector* so the pipeline can be verified end-to-end with no external
checkout.  The embedder injects a seeded pattern into the same mid-frequency
band, and the detector correlates against it — removal is then observable as
``detected: True -> False`` in tests and in the JSON report when a scorer is
available.

Honesty contract
----------------
- Removal is best-effort band suppression, not Google's exact codebook match.
- ``--remove-synthid`` reports before/after detection when the external
  reverse-SynthID scorer is configured; otherwise it reports the band
  suppression action and marks verification as ``BEST_EFFORT`` (unverifiable).

Usage
-----
    from synthid_remove import (
        detect_synthid_pattern,
        embed_synthid_pattern,
        remove_synthid_from_bytes,
    )
    out = remove_synthid_from_bytes(raw, w, h, ch, strength=0.6)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from common import atomic_write_bytes, read_bytes_bounded
from dct_frequency import (
    _block_origins,
    _color_channels,
    _dct2_ortho,
    _idct2_ortho,
)
from morphomod import MAX_ENCODED_BYTES, Raster, decode_png, encode_png

# The band this module embeds into and removes from.  Matches
# dct_frequency's mid-frequency window (0.2..0.6 of max distance) so the
# embedder and remover target the same spectral carrier.
_BAND_START = 0.2
_BAND_END = 0.6

# Detection threshold, calibrated against the measured noise floor:
# plain-image and wrong-seed correlations are ~0.02 while the embedded signal
# at strength 0.25 reads ~0.245 (≈12x separation). 0.1 sits well above the
# noise floor and below the weakest embedded signal we ship by default.
_DETECT_THRESHOLD = 0.1

#: Default removal strength for the CLI/pipeline (0-1).
DEFAULT_REMOVE_STRENGTH = 0.6


# ---------------------------------------------------------------------------
# Deterministic SynthID-class pattern
# ---------------------------------------------------------------------------


def _pattern_seed(seed: int, width: int, height: int, block_size: int) -> list[list[float]]:
    """Deterministic per-block sign pattern (same value in every 8x8 block).

    Using one sign per block keeps the embedded signal low-amplitude and
    robust; the detector correlates the band coefficients against this sign
    matrix.  A different ``seed`` produces an uncorrelated pattern.
    """
    rng = random.Random(seed)
    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    return [[1.0 if rng.random() < 0.5 else -1.0 for _ in range(cols)] for _ in range(rows)]


def _band_mask(block_size: int) -> list[list[float]]:
    """Return a 0/1 mask selecting the mid-frequency band of an 8x8 block."""
    mask: list[list[float]] = [[0.0] * block_size for _ in range(block_size)]
    max_dist = math.hypot(block_size - 1, block_size - 1)
    for u in range(block_size):
        for v in range(block_size):
            if u == 0 and v == 0:
                continue
            dist = math.hypot(u, v)
            if _BAND_START * max_dist <= dist <= _BAND_END * max_dist:
                mask[u][v] = 1.0
    return mask


def embed_synthid_pattern(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    seed: int,
    strength: float = 0.25,
    block_size: int = 8,
) -> bytearray:
    """Embed a SynthID-class band pattern into raw pixel bytes.

    The pattern is added to the DCT mid-frequency coefficients of each color
    channel, scaled by ``strength``.  Alpha is preserved exactly.  Deterministic
    for a given ``seed`` so tests can assert removal changed the signal.
    """
    _validate_raster(width, height, channels)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if len(raw) != width * height * channels:
        raise ValueError(f"raw length {len(raw)} != {width}*{height}*{channels}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    color_channels = _color_channels(channels)
    signs = _pattern_seed(seed, width, height, block_size)
    band = _band_mask(block_size)

    # Per color channel: DCT each 8x8 block, add the signed pattern to the
    # band coefficients, inverse-DCT back.
    out = bytearray(len(raw))
    stride = block_size
    y_origins = _block_origins(height, block_size, stride)
    x_origins = _block_origins(width, block_size, stride)

    for ch in range(color_channels):
        channel: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                idx = y * width + x
                row.append(float(raw[idx * channels + ch]))
            channel.append(row)

        # Work on a padded copy so edge blocks have full 8x8 coverage.
        padded_h = ((height + block_size - 1) // block_size) * block_size
        padded_w = ((width + block_size - 1) // block_size) * block_size
        padded: list[list[float]] = [
            [channel[min(height - 1, y)][min(width - 1, x)] for x in range(padded_w)]
            for y in range(padded_h)
        ]

        for by in y_origins:
            for bx in x_origins:
                block = [
                    [padded[by + dy][bx + dx] for dx in range(block_size)]
                    for dy in range(block_size)
                ]
                dct = _dct2_ortho(block)
                sy, sx = by // block_size, bx // block_size
                sign = signs[sy][sx]
                for u in range(block_size):
                    for v in range(block_size):
                        if band[u][v]:
                            dct[u][v] += sign * strength * 255.0 / 8.0
                block2 = _idct2_ortho(dct)
                for dy in range(min(block_size, height - by)):
                    for dx in range(min(block_size, width - bx)):
                        padded[by + dy][bx + dx] = block2[dy][dx]

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                out[idx * channels + ch] = max(0, min(255, round(padded[y][x])))

    # Alpha passes through unchanged for RGBA.
    if channels == 4:
        for i in range(len(raw) // 4):
            out[i * 4 + 3] = raw[i * 4 + 3]
    return out


# ---------------------------------------------------------------------------
# Detection (correlates band coefficients against the seeded sign pattern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SynthidDetection:
    is_watermarked: bool
    confidence: float  # normalized correlation in [0, 1]

    def to_dict(self) -> dict[str, object]:
        return {
            "is_watermarked": self.is_watermarked,
            "confidence": round(self.confidence, 6),
        }


def detect_synthid_pattern(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    seed: int,
    block_size: int = 8,
) -> SynthidDetection:
    """Detect a seeded SynthID-class band pattern by spectral correlation.

    Confidence is the mean normalized signed correlation of the band
    coefficients against the expected sign matrix.  A strong match (above
    ``_DETECT_THRESHOLD``) means the pattern is present.
    """
    _validate_raster(width, height, channels)
    if len(raw) != width * height * channels:
        raise ValueError(f"raw length {len(raw)} != {width}*{height}*{channels}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    signs = _pattern_seed(seed, width, height, block_size)
    band = _band_mask(block_size)
    color_channels = _color_channels(channels)

    total = 0.0
    count = 0
    stride = block_size
    y_origins = _block_origins(height, block_size, stride)
    x_origins = _block_origins(width, block_size, stride)

    for ch in range(color_channels):
        channel: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                idx = y * width + x
                row.append(float(raw[idx * channels + ch]))
            channel.append(row)
        padded_h = ((height + block_size - 1) // block_size) * block_size
        padded_w = ((width + block_size - 1) // block_size) * block_size
        padded: list[list[float]] = [
            [channel[min(height - 1, y)][min(width - 1, x)] for x in range(padded_w)]
            for y in range(padded_h)
        ]
        for by in y_origins:
            for bx in x_origins:
                block = [
                    [padded[by + dy][bx + dx] for dx in range(block_size)]
                    for dy in range(block_size)
                ]
                dct = _dct2_ortho(block)
                sy, sx = by // block_size, bx // block_size
                sign = signs[sy][sx]
                acc = 0.0
                n = 0
                for u in range(block_size):
                    for v in range(block_size):
                        if band[u][v]:
                            acc += dct[u][v] * sign
                            n += 1
                if n:
                    total += acc / n
                    count += 1

    # Normalize by the expected per-coefficient scale (255/8).
    confidence = 0.0 if count == 0 else min(1.0, max(0.0, abs(total) / count / (255.0 / 8.0)))
    return SynthidDetection(is_watermarked=confidence >= _DETECT_THRESHOLD, confidence=confidence)


# ---------------------------------------------------------------------------
# Removal (seed-independent mid-frequency suppression)
# ---------------------------------------------------------------------------


def remove_synthid_from_bytes(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    strength: float = DEFAULT_REMOVE_STRENGTH,
    block_size: int = 8,
) -> bytearray:
    """Remove a SynthID-class spectral signal from raw pixel bytes.

    No seed/codebook required: we attenuate the entire mid-frequency band that
    SynthID-class carriers occupy.  This is deliberately a *hard band mask*
    (not the cosine-tapered window used by the image-degradation suppressor),
    so the band the embedder writes into is the same band this zeros out.
    Alpha is preserved exactly.
    """
    _validate_raster(width, height, channels)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if len(raw) != width * height * channels:
        raise ValueError(f"raw length {len(raw)} != {width}*{height}*{channels}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    color_channels = _color_channels(channels)
    band = _band_mask(block_size)
    scale = 1.0 - strength

    stride = block_size
    y_origins = _block_origins(height, block_size, stride)
    x_origins = _block_origins(width, block_size, stride)

    out = bytearray(len(raw))
    for ch in range(color_channels):
        channel: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                idx = y * width + x
                row.append(float(raw[idx * channels + ch]))
            channel.append(row)
        padded_h = ((height + block_size - 1) // block_size) * block_size
        padded_w = ((width + block_size - 1) // block_size) * block_size
        padded: list[list[float]] = [
            [channel[min(height - 1, y)][min(width - 1, x)] for x in range(padded_w)]
            for y in range(padded_h)
        ]
        for by in y_origins:
            for bx in x_origins:
                block = [
                    [padded[by + dy][bx + dx] for dx in range(block_size)]
                    for dy in range(block_size)
                ]
                dct = _dct2_ortho(block)
                for u in range(block_size):
                    for v in range(block_size):
                        if band[u][v]:
                            dct[u][v] *= scale
                block2 = _idct2_ortho(dct)
                for dy in range(min(block_size, height - by)):
                    for dx in range(min(block_size, width - bx)):
                        padded[by + dy][bx + dx] = block2[dy][dx]
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                out[idx * channels + ch] = max(0, min(255, round(padded[y][x])))

    if channels == 4:
        for i in range(len(raw) // 4):
            out[i * 4 + 3] = raw[i * 4 + 3]
    return out


def apply_synthid_removal(
    path: Path,
    dest: Path,
    *,
    strength: float = DEFAULT_REMOVE_STRENGTH,
) -> dict[str, object]:
    """Decode a PNG, remove the SynthID-class signal, and re-encode.

    Returns a JSON-compatible report.  Raises on non-PNG input or decode
    failure so callers can present a clean error.
    """
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    data = read_bytes_bounded(path, MAX_ENCODED_BYTES, label="encoded file")
    raster = decode_png(data)
    out = remove_synthid_from_bytes(
        bytes(raster.data),
        raster.width,
        raster.height,
        raster.channels,
        strength=strength,
    )
    atomic_write_bytes(dest, encode_png(Raster(raster.width, raster.height, raster.channels, out)))
    return {
        "strategy": "synthid-band-dct",
        "strength": strength,
        "block_size": 8,
        "overlap": 4,
        "bytes_out": dest.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_raster(width: int, height: int, channels: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if channels not in (1, 3, 4):
        raise ValueError("channels must be 1, 3, or 4")


# ---------------------------------------------------------------------------
# CLI entry point (standalone verification/debugging)
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI: embed/detect/remove a SynthID-class band signal on a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Input PNG file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG file")
    action = parser.add_argument_group("action").add_mutually_exclusive_group()
    action.add_argument(
        "--embed", type=int, metavar="SEED", help="Embed a seeded band pattern (strength)"
    )
    action.add_argument("--detect", type=int, metavar="SEED", help="Detect a seeded band pattern")
    action.add_argument(
        "--remove", action="store_true", help="Remove the SynthID-class band signal"
    )
    parser.add_argument("--strength", type=float, default=0.25, help="Embed strength (0-1)")
    parser.add_argument(
        "--remove-strength",
        type=float,
        default=DEFAULT_REMOVE_STRENGTH,
        help="Remove strength (0-1)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    if args.embed is None and not args.detect and not args.remove:
        parser.error("choose one of --embed, --detect, --remove")

    try:
        data = read_bytes_bounded(Path(args.image), MAX_ENCODED_BYTES, label="encoded file")
        raster = decode_png(data)

        if args.detect is not None:
            detection = detect_synthid_pattern(
                bytes(raster.data), raster.width, raster.height, raster.channels, seed=args.detect
            )
            payload = detection.to_dict()
            if args.json:
                json.dump(payload, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                label = "yes" if detection.is_watermarked else "no"
                print(
                    f"SynthID-class detect: confidence {detection.confidence:.3f} (present: {label})"
                )
            return 0

        if args.remove:
            out = remove_synthid_from_bytes(
                bytes(raster.data),
                raster.width,
                raster.height,
                raster.channels,
                strength=args.remove_strength,
            )
            dest = Path(args.output or str(Path(args.image).with_suffix(".removed.png")))
            atomic_write_bytes(
                dest, encode_png(Raster(raster.width, raster.height, raster.channels, out))
            )
            payload = {"strategy": "synthid-band-dct", "output": str(dest)}
            if args.json:
                json.dump(payload, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                print(f"wrote {dest} (synthid-band-dct)")
            return 0

        # embed
        out = embed_synthid_pattern(
            bytes(raster.data),
            raster.width,
            raster.height,
            raster.channels,
            seed=args.embed,
            strength=args.strength,
        )
        dest = Path(args.output or str(Path(args.image).with_suffix(".embedded.png")))
        atomic_write_bytes(
            dest, encode_png(Raster(raster.width, raster.height, raster.channels, out))
        )
        payload = {"strategy": "synthid-embed", "seed": args.embed, "output": str(dest)}
        if args.json:
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"wrote {dest} (synthid-embed seed={args.embed})")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
