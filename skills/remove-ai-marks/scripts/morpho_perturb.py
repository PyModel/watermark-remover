"""Morphological perturbation for Layer C — structural noise patterns.

Enhances the existing character-level perturbation (perturb_text.py) with
image-space structural noise that disrupts pattern-based watermark detection
without obvious visual damage.

Strategies
----------
    grid      — Subtle grid overlay (breaks logo continuity)
    diagonal  — Fine diagonal scan lines
    noise     — Per-pixel Gaussian noise injection
    quantize  — Color-quantization to break fine gradients

Complexity   : O(width * height) — one pass per channel
Quality     : High (adjustable strength)
Legal risk   : Low (generic image operations)
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Per-channel morphological operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MorphoResult:
    """Result of a morphological perturbation."""

    data: bytearray
    width: int
    height: int
    channels: int
    strategy: str
    strength: float
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "strength": self.strength,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }


def _pixels_from_bytes(raw: bytes, channels: int) -> list[list[float]]:
    """Parse raw bytes into per-pixel float values."""
    pixels: list[list[float]] = []
    for i in range(0, len(raw), channels):
        pixels.append([float(raw[i + c]) for c in range(channels)])
    return pixels


def _bytes_from_pixels(pixels: list[list[float]], channels: int) -> bytearray:
    """Flatten per-pixel float values back to bytes."""
    out = bytearray(len(pixels) * channels)
    for i, row in enumerate(pixels):
        for c in range(channels):
            out[i * channels + c] = max(0, min(255, round(row[c])))
    return out


def morpho_grid(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    spacing: int = 8,
    opacity: float = 0.05,
    seed: int | None = None,
) -> MorphoResult:
    """Overlay a subtle grid pattern to break logo/overlay continuity.

    Grid lines are rendered at *spacing*-pixel intervals with alpha *opacity*.
    Higher opacity = more aggressive watermark disruption.
    """
    if spacing < 2:
        raise ValueError("spacing must be >= 2")
    rng = random.Random(seed)
    pixels = _pixels_from_bytes(raw, channels)

    for i, row in enumerate(pixels):
        y = i // width
        x = i % width
        is_grid = (x % spacing == 0) or (y % spacing == 0)
        if is_grid:
            base = rng.uniform(opacity * 1.5, opacity * 3)
            for c in range(channels):
                row[c] = row[c] * (1.0 - base) + (128 if c == 3 else 200) * base

    return MorphoResult(
        data=_bytes_from_pixels(pixels, channels),
        width=width,
        height=height,
        channels=channels,
        strategy="grid",
        strength=opacity,
        seed=seed,
    )


def morpho_diagonal(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    spacing: int = 16,
    opacity: float = 0.03,
    angle: float = 45.0,
    seed: int | None = None,
) -> MorphoResult:
    """Add fine diagonal scan lines to disrupt texture-based watermarks."""
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rng = random.Random(seed)
    pixels = _pixels_from_bytes(raw, channels)

    diag_length = math.sqrt(width * width + height * height)
    for i, row in enumerate(pixels):
        y = i // width
        x = i % width
        # Project (x, y) onto the diagonal axis
        proj = (x * cos_a + y * sin_a) / diag_length * (width + height)
        is_line = (proj % spacing) < 1.0
        if is_line:
            base = rng.uniform(opacity * 1.2, opacity * 2.5)
            for c in range(channels):
                row[c] = row[c] * (1.0 - base) + 255 * base

    return MorphoResult(
        data=_bytes_from_pixels(pixels, channels),
        width=width,
        height=height,
        channels=channels,
        strategy="diagonal",
        strength=opacity,
        seed=seed,
    )


def morpho_noise(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    sigma: float = 5.0,
    seed: int | None = None,
) -> MorphoResult:
    """Inject per-pixel Gaussian noise to disrupt frequency-domain marks."""
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    rng = random.Random(seed)
    pixels = _pixels_from_bytes(raw, channels)

    for row in pixels:
        for c in range(channels):
            noise = rng.gauss(0, sigma)
            row[c] = max(0, min(255, row[c] + noise))

    return MorphoResult(
        data=_bytes_from_pixels(pixels, channels),
        width=width,
        height=height,
        channels=channels,
        strategy="noise",
        strength=sigma,
        seed=seed,
    )


def morpho_quantize(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    levels: int = 32,
) -> MorphoResult:
    """Color quantization to break fine gradients used by some watermark detectors."""
    step = 255.0 / max(1, levels - 1)
    pixels = _pixels_from_bytes(raw, channels)

    for row in pixels:
        for c in range(channels):
            row[c] = round(row[c] / step) * step

    return MorphoResult(
        data=_bytes_from_pixels(pixels, channels),
        width=width,
        height=height,
        channels=channels,
        strategy="quantize",
        strength=1.0 / max(1, levels - 1),
        seed=None,
    )


def morpho_perturb(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    strategy: str = "grid",
    **kwargs: Any,
) -> MorphoResult:
    """Apply one morphological perturbation strategy to raw pixel bytes.

    Strategies
    ----------
    grid      — Subtle grid overlay (default)
    diagonal  — Fine diagonal scan lines
    noise     — Per-pixel Gaussian noise
    quantize  — Color quantization
    """
    if strategy == "grid":
        return morpho_grid(raw, width, height, channels, **kwargs)
    if strategy == "diagonal":
        return morpho_diagonal(raw, width, height, channels, **kwargs)
    if strategy == "noise":
        return morpho_noise(raw, width, height, channels, **kwargs)
    if strategy == "quantize":
        return morpho_quantize(raw, width, height, channels, **kwargs)
    raise ValueError(f"unknown morphological strategy: {strategy}")


# ---------------------------------------------------------------------------
# Combined morphological attack
# ---------------------------------------------------------------------------


def combined_morpho(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    strategies: tuple[str, ...] = ("grid", "noise"),
    seed: int | None = None,
    **kwargs: Any,
) -> MorphoResult:
    """Chain multiple morphological strategies sequentially.

    Example: grid + noise disrupts both pattern-based and frequency-domain marks.
    """
    current = bytearray(raw)
    active_seed = seed
    last_result = None

    for strat in strategies:
        result = morpho_perturb(
            bytes(current),
            width,
            height,
            channels,
            strategy=strat,
            seed=active_seed,
            **kwargs,
        )
        current = result.data
        active_seed = (result.seed or 0) + 1 if active_seed else seed
        last_result = result

    if last_result is None:
        raise ValueError("no strategies applied")

    return MorphoResult(
        data=current,
        width=width,
        height=height,
        channels=channels,
        strategy="+".join(strategies),
        strength=last_result.strength,
        seed=last_result.seed,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI: apply morphological perturbation to a PNG image."""
    import argparse

    from morphomod import decode_png, encode_png

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Input PNG file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG file")
    parser.add_argument(
        "--strategy",
        choices=["grid", "diagonal", "noise", "quantize"],
        default="grid",
        help="Perturbation strategy",
    )
    parser.add_argument("--opacity", type=float, default=0.05, help="Grid/diagonal opacity")
    parser.add_argument("--spacing", type=int, default=8, help="Grid spacing in pixels")
    parser.add_argument("--sigma", type=float, default=5.0, help="Noise sigma")
    parser.add_argument("--levels", type=int, default=32, help="Quantization levels")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    try:
        raw = open(args.image, "rb").read()  # noqa: SIM115
        raster = decode_png(raw)

        result = morpho_perturb(
            bytes(raster.data),
            raster.width,
            raster.height,
            raster.channels,
            strategy=args.strategy,
            opacity=args.opacity,
            spacing=args.spacing,
            sigma=args.sigma,
            levels=args.levels,
            seed=args.seed,
        )

        if args.output is None:
            args.output = str(Path(args.image).with_suffix(".perturbed.png"))

        out_raster = type(
            "Raster",
            (),
            {
                "width": result.width,
                "height": result.height,
                "channels": result.channels,
                "data": result.data,
            },
        )()
        with open(args.output, "wb") as f:
            f.write(encode_png(out_raster))

        report = result.to_dict()
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"wrote {args.output} strategy={result.strategy}")
        return 0
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1


if __name__ == "__main__":
    import json
    from pathlib import Path

    raise SystemExit(main())
