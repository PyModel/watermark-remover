"""Synthetic benchmark generator for watermark removal.

Creates deterministic watermarked images from clean sources for benchmarking
inpaint backends.  Generates images with varying watermark properties
(opacity, font, position, rotation, scale, color, shadow, outline, JPEG quality).

Usage
-----
    from benchmark import generate_synthetic_benchmark, BenchmarkConfig

    config = BenchmarkConfig(
        output_dir=Path("tests/fixtures/golden"),
        count=4,
        seed=42,
    )
    for item in generate_synthetic_benchmark(config):
        print(f"  {item.name}: mask={item.mask_path} watermark={item.watermark_path}")
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optional_deps


@dataclass(frozen=True, slots=True)
class BenchmarkImage:
    """One generated benchmark fixture."""

    name: str
    source_path: Path  # clean source
    watermark_path: Path  # watermarked image
    mask_path: Path  # exact ground-truth mask
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for synthetic benchmark generation."""

    output_dir: Path = field(default_factory=lambda: Path("tests/fixtures/golden"))
    count: int = 4
    seed: int = 42
    sizes: tuple[int, ...] = (128, 256)  # small enough for git
    watermark_opacities: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
    jpeg_qualities: tuple[int, ...] = (95, 75)


def generate_synthetic_benchmark(config: BenchmarkConfig) -> list[BenchmarkImage]:
    """Generate synthetic benchmark images.

    Creates a corpus of clean images, applies synthetic watermarks with
    known masks, and optionally compresses with JPEG to simulate real photos.

    Returns a list of BenchmarkImage entries.
    """
    rng = random.Random(config.seed)
    images: list[BenchmarkImage] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Only generate if we have Pillow for image creation.
    pil = optional_deps._import_safe("PIL.Image")
    if pil is None:
        raise RuntimeError("Pillow is required; install watermark-remover[visible]")

    for size in config.sizes:
        for idx in range(config.count):
            name = f"synth_{size}x{size}_{idx}"

            # Generate random clean source image
            source = _generate_source_image(pil, size, rng)
            source_path = config.output_dir / f"{name}.source.png"
            source.save(str(source_path), "PNG")

            # Apply watermark
            watermark, mask_data = _apply_synthetic_watermark(
                source, size, idx, rng, config.watermark_opacities
            )
            # Apply JPEG compression to some and persist the compressed bytes at
            # a path whose extension matches the encoded format.
            jpeg_q = config.jpeg_qualities[idx % len(config.jpeg_qualities)]
            extension = ".jpg" if jpeg_q < 100 else ".png"
            watermark_path = config.output_dir / f"{name}.watermarked{extension}"
            if jpeg_q < 100:
                watermark.convert("RGB").save(str(watermark_path), "JPEG", quality=jpeg_q)
            else:
                watermark.save(str(watermark_path), "PNG")

            # Write ground-truth mask as PGM
            mask_path = config.output_dir / f"{name}.mask.pgm"
            _write_pgm_mask(mask_path, mask_data, size, size)

            images.append(
                BenchmarkImage(
                    name=name,
                    source_path=source_path,
                    watermark_path=watermark_path,
                    mask_path=mask_path,
                    config={
                        "size": size,
                        "jpeg_quality": jpeg_q,
                        "seed": config.seed,
                    },
                )
            )

    return images


def _generate_source_image(pil: Any, size: int, rng: random.Random) -> Any:
    """Generate a synthetic clean source image."""
    # Create a gradient background with some random color patches.
    img = pil.new("RGB", (size, size), color=(128, 128, 128))
    pixels = img.load()

    for y in range(size):
        for x in range(size):
            # Simple gradient
            r = int(128 + 127 * math.sin(x / size * math.pi))
            g = int(128 + 127 * math.cos(y / size * math.pi))
            b = int(128 + 63 * math.sin((x + y) / size * math.pi * 2))
            pixels[x, y] = (r & 255, g & 255, b & 255)

    # Add random color patches for texture
    for _ in range(rng.randint(2, 5)):
        x = rng.randint(0, size - 20)
        y = rng.randint(0, size - 20)
        w = rng.randint(10, 40)
        h = rng.randint(10, 40)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        for py in range(y, min(y + h, size)):
            for px in range(x, min(x + w, size)):
                pixels[px, py] = color

    return img


def _apply_synthetic_watermark(
    img: Any,
    size: int,
    seed: int,
    rng: random.Random,
    opacities: tuple[float, ...],
) -> tuple[Any, bytearray]:
    """Apply a synthetic watermark and return the watermarked image + mask."""
    pil = optional_deps._import_safe("PIL.Image")
    pil_draw = optional_deps._import_safe("PIL.ImageDraw")
    if pil is None or pil_draw is None:
        raise RuntimeError("Pillow is required; install watermark-remover[visible]")

    watermark = img.convert("RGBA")
    overlay = pil.new("RGBA", watermark.size, (0, 0, 0, 0))
    draw = pil_draw.Draw(overlay)

    # Create a text-like watermark shape
    opacity = opacities[seed % len(opacities)]
    color = (
        int(255 * (0.5 + 0.5 * math.sin(seed))),
        int(255 * (0.5 + 0.5 * math.cos(seed))),
        int(255 * (0.5 + 0.5 * math.sin(seed * 1.5))),
        int(255 * opacity),
    )

    # Create a rectangular watermark region with text-like appearance
    margin = max(10, size // 10)
    x = rng.randint(margin, size - margin * 2)
    y = rng.randint(margin, size - margin * 2)
    w = rng.randint(size // 4, size // 2)
    h = rng.randint(20, size // 4)

    # Pillow rectangle bounds are inclusive, so subtract one to match the
    # width/height convention used by the exact ground-truth mask below.
    bounds = [x, y, x + w - 1, y + h - 1]
    draw.rectangle(bounds, fill=color)
    draw.rectangle(
        bounds,
        outline=(255, 255, 255, int(128 * opacity)),
    )
    watermark = pil.alpha_composite(watermark, overlay).convert(img.mode)

    # Create ground-truth mask: marked (255) where watermark is
    mask_data = bytearray(size * size)
    for py in range(y, min(y + h, size)):
        for px in range(x, min(x + w, size)):
            mask_data[py * size + px] = 255

    return watermark, mask_data


def _write_pgm_mask(path: Path, data: bytearray, width: int, height: int) -> None:
    """Write a binary PGM mask file."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(bytes(data))
