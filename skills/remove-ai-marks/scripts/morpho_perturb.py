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

The alpha channel of RGBA input is never modified: perturbing opacity would
change compositing, not watermark structure.

Complexity   : O(width * height) — one pass per channel
Quality     : High (adjustable strength)
Legal risk   : Low (generic image operations)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Strategies this module dispatches, mapped to the keyword arguments each one
# accepts. ``morpho_perturb`` rejects any keyword outside this catalog.
MORPHO_STRATEGY_KWARGS: dict[str, tuple[str, ...]] = {
    "grid": ("spacing", "opacity", "seed"),
    "diagonal": ("spacing", "opacity", "angle", "seed"),
    "noise": ("sigma", "seed"),
    "quantize": ("levels",),
}
MORPHO_STRATEGIES: tuple[str, ...] = tuple(MORPHO_STRATEGY_KWARGS)


def _strategy_kwargs(strategy: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the caller kwargs allowed for ``strategy``, rejecting the rest.

    Rejecting unexpected keywords catches typos and keeps callers honest:
    silently dropping ``opacit=0.1`` would apply a default the caller never chose.
    """
    allowed = MORPHO_STRATEGY_KWARGS.get(strategy)
    if allowed is None:
        raise ValueError(f"unknown morphological strategy: {strategy}")
    unexpected = sorted(set(kwargs) - set(allowed))
    if unexpected:
        names = ", ".join(unexpected)
        raise TypeError(f"unexpected keyword argument(s) for strategy {strategy!r}: {names}")
    return {name: kwargs[name] for name in allowed if name in kwargs}


def _validate_input(raw: bytes, width: int, height: int, channels: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if channels not in (1, 3, 4):
        raise ValueError("channels must be 1, 3, or 4")
    expected = width * height * channels
    if len(raw) != expected:
        raise ValueError(f"raw length {len(raw)} != {width}*{height}*{channels}")


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


def _color_channels(channels: int) -> int:
    """Channels to perturb: alpha is always left untouched for RGBA input."""
    return channels - 1 if channels == 4 else channels


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
    Higher opacity = more aggressive watermark disruption. The alpha channel
    of RGBA input is preserved exactly.
    """
    _validate_input(raw, width, height, channels)
    if spacing < 2:
        raise ValueError("spacing must be >= 2")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")
    rng = random.Random(seed)
    pixels = _pixels_from_bytes(raw, channels)
    color_channels = _color_channels(channels)

    for i, row in enumerate(pixels):
        y = i // width
        x = i % width
        is_grid = (x % spacing == 0) or (y % spacing == 0)
        if is_grid:
            base = rng.uniform(opacity * 1.5, opacity * 3)
            for c in range(color_channels):
                row[c] = row[c] * (1.0 - base) + 200 * base

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
    _validate_input(raw, width, height, channels)
    if spacing < 2:
        raise ValueError("spacing must be >= 2")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rng = random.Random(seed)
    pixels = _pixels_from_bytes(raw, channels)
    color_channels = _color_channels(channels)

    diag_length = math.sqrt(width * width + height * height)
    for i, row in enumerate(pixels):
        y = i // width
        x = i % width
        # Project (x, y) onto the diagonal axis
        proj = (x * cos_a + y * sin_a) / diag_length * (width + height)
        is_line = (proj % spacing) < 1.0
        if is_line:
            base = rng.uniform(opacity * 1.2, opacity * 2.5)
            for c in range(color_channels):
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
    _validate_input(raw, width, height, channels)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    rng = random.Random(seed)
    pixels = _pixels_from_bytes(raw, channels)
    color_channels = _color_channels(channels)

    for row in pixels:
        for c in range(color_channels):
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
    _validate_input(raw, width, height, channels)
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = 255.0 / (levels - 1)
    pixels = _pixels_from_bytes(raw, channels)
    color_channels = _color_channels(channels)

    for row in pixels:
        for c in range(color_channels):
            row[c] = round(row[c] / step) * step

    return MorphoResult(
        data=_bytes_from_pixels(pixels, channels),
        width=width,
        height=height,
        channels=channels,
        strategy="quantize",
        strength=1.0 / (levels - 1),
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

    Only the keywords listed in ``MORPHO_STRATEGY_KWARGS`` are accepted for
    each strategy; any other keyword raises ``TypeError`` so typos cannot
    silently apply defaults.
    """
    filtered = _strategy_kwargs(strategy, kwargs)
    if strategy == "grid":
        return morpho_grid(raw, width, height, channels, **filtered)
    if strategy == "diagonal":
        return morpho_diagonal(raw, width, height, channels, **filtered)
    if strategy == "noise":
        return morpho_noise(raw, width, height, channels, **filtered)
    if strategy == "quantize":
        return morpho_quantize(raw, width, height, channels, **filtered)
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

    When ``seed`` is provided, each strategy in the chain receives a distinct
    derived seed (seed, seed + 1, ...) so every stage stays reproducible.
    Strategies that take no seed (quantize) are left deterministic.
    """
    current = bytearray(raw)
    active_seed = seed
    last_result: MorphoResult | None = None

    for strat in strategies:
        stage_kwargs = dict(kwargs)
        if active_seed is not None and "seed" in MORPHO_STRATEGY_KWARGS[strat]:
            stage_kwargs["seed"] = active_seed
        result = morpho_perturb(
            bytes(current),
            width,
            height,
            channels,
            strategy=strat,
            **stage_kwargs,
        )
        current = result.data
        if active_seed is not None:
            active_seed += 1
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

    from common import atomic_write_bytes, read_bytes_bounded
    from morphomod import MAX_ENCODED_BYTES, Raster, decode_png, encode_png
    from structured_log import init_logger

    logger = init_logger()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Input PNG file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG file")
    parser.add_argument(
        "--strategy",
        choices=list(MORPHO_STRATEGIES),
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

    # Forward only the keywords the selected strategy accepts.
    all_kwargs = {
        "spacing": args.spacing,
        "opacity": args.opacity,
        "sigma": args.sigma,
        "levels": args.levels,
        "seed": args.seed,
    }
    kwargs = {
        name: all_kwargs[name]
        for name in MORPHO_STRATEGY_KWARGS[args.strategy]
        if name in all_kwargs and all_kwargs[name] is not None
    }

    try:
        raw = read_bytes_bounded(Path(args.image), MAX_ENCODED_BYTES, label="encoded file")
        raster = decode_png(raw)

        result = morpho_perturb(
            bytes(raster.data),
            raster.width,
            raster.height,
            raster.channels,
            strategy=args.strategy,
            **kwargs,
        )

        if args.output is None:
            args.output = str(Path(args.image).with_suffix(".perturbed.png"))

        out_raster = Raster(result.width, result.height, result.channels, result.data)
        atomic_write_bytes(Path(args.output), encode_png(out_raster))

        report = result.to_dict()
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"wrote {args.output} strategy={result.strategy}")
        return 0
    except Exception as error:
        logger.error(f"error: {error}", module="morpho_perturb")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
