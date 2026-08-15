"""Frequency-domain watermark degradation attacks (DCT/FFT).

T1 Tier 1 classical attack — targets mid-frequency bands where spatial
watermarks and soft logos typically live.  Works on any RGBA/RGB/grayscale
image via a pure-Python orthonormal DCT-II implementation.

Complexity   : O(width * height) per block with an 8x8 DCT kernel; the pure
               Python DCT is expensive, so DCT-based strategies enforce a
               per-image pixel cap and fail loudly above it.
Quality     : Fair–Good (noticeable banding at aggressive settings)
Legal risk   : High (specific intent to remove frequency-domain marks)

Strategies
----------
    freq-dct   — DCT mid-frequency suppression (default)
    blur       — Gaussian blur (separable)
    median     — Median filter
    jpeg       — Simulated JPEG compression
    rotate     — Nearest-neighbor rotation
    two-stage  — blur → jpeg → median (95–98% ASR paper-reported)

The alpha channel of RGBA input is preserved exactly by every strategy:
degrading opacity would change compositing, not watermark structure.

Usage
-----
    from dct_frequency import degrade_image, frequency_suppress_from_bytes
    result = degrade_image(raw_bytes, w, h, ch, strategy="freq-dct", suppress=0.7)
    cleaned = result.data          # bytearray in original pixel layout
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Strategies this module dispatches, mapped to the keyword arguments each one
# accepts. ``degrade_image`` rejects any keyword outside this catalog.
DEGRADE_STRATEGY_KWARGS: dict[str, tuple[str, ...]] = {
    "freq-dct": ("suppress", "block_size", "overlap"),
    "blur": ("sigma",),
    "median": ("kernel_size",),
    "jpeg": ("quality",),
    "rotate": ("angle_deg",),
    "two-stage": ("blur_sigma", "quality"),
}
DEGRADE_STRATEGIES: tuple[str, ...] = tuple(DEGRADE_STRATEGY_KWARGS)

# The pure-Python DCT kernels cost ~8k float ops per 8x8 block, so DCT-based
# strategies bound the total pixel count instead of letting a 40 MP input run
# for hours. O(width*height) strategies (blur/median/rotate) have no cap here;
# callers bound them with their own image limits.
MAX_FREQ_DCT_PIXELS = 65_536  # 256x256
MAX_JPEG_PIXELS = 262_144  # 512x512


# ---------------------------------------------------------------------------
# 2D orthonormal DCT-II
# ---------------------------------------------------------------------------


def _cos_table(n: int) -> list[list[float]]:
    """Precompute cos(pi * k * (2 * i + 1) / (2 * n)) for i, k in [0, n)."""
    return [[math.cos(math.pi * k * (2 * i + 1) / (2 * n)) for i in range(n)] for k in range(n)]


_COS_CACHE: dict[int, list[list[float]]] = {}
_SCALE_CACHE: dict[int, list[float]] = {}


def _cos_lookup(n: int) -> list[list[float]]:
    table = _COS_CACHE.get(n)
    if table is None:
        table = _cos_table(n)
        _COS_CACHE[n] = table
    return table


def _scales(n: int) -> list[float]:
    factors = _SCALE_CACHE.get(n)
    if factors is None:
        factors = [math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n) for k in range(n)]
        _SCALE_CACHE[n] = factors
    return factors


def _dct2_ortho(block: list[list[float]]) -> list[list[float]]:
    """Compute the orthonormal 2-D DCT-II of a block (cached cosine tables)."""
    h = len(block)
    w = len(block[0]) if h else 0
    cos_h = _cos_lookup(h)  # cos_h[k][i]
    cos_w = _cos_lookup(w)
    au_factors = _scales(h)
    av_factors = _scales(w)
    result: list[list[float]] = [[0.0] * w for _ in range(h)]
    for u in range(h):
        au = au_factors[u]
        row_u = cos_h[u]
        out_row = result[u]
        for v in range(w):
            av = av_factors[v]
            col_v = cos_w[v]
            s = 0.0
            for x in range(h):
                in_row = block[x]
                cux = row_u[x]
                for y in range(w):
                    s += in_row[y] * cux * col_v[y]
            out_row[v] = au * av * s
    return result


def _idct2_ortho(block: list[list[float]]) -> list[list[float]]:
    """Inverse orthonormal 2-D DCT-II (cached cosine tables)."""
    h = len(block)
    w = len(block[0]) if h else 0
    cos_h = _cos_lookup(h)  # cos_h[k][i]
    cos_w = _cos_lookup(w)
    au_factors = _scales(h)
    av_factors = _scales(w)
    result: list[list[float]] = [[0.0] * w for _ in range(h)]
    for x in range(h):
        out_row = result[x]
        for y in range(w):
            s = 0.0
            for u in range(h):
                row_u = block[u]
                cux = cos_h[u][x]
                au = au_factors[u]
                for v in range(w):
                    s += au * av_factors[v] * row_u[v] * cux * cos_w[v][y]
            out_row[y] = s
    return result


def _dct_block(block: list[list[float]], suppress: float) -> list[list[float]]:
    """DCT → suppress mid-frequencies → IDCT.

    Parameters
    ----------
    block : 8 × 8 pixel float block
    suppress : 0.0 (keep all) → 1.0 (aggressive mid-freq removal)
    """
    dct_block = _dct2_ortho(block)
    h = len(dct_block)
    w = len(dct_block[0]) if h else 0
    for u in range(h):
        for v in range(w):
            if u == 0 and v == 0:
                continue  # DC component
            dist = math.hypot(u, v)
            max_dist = math.hypot(h - 1, w - 1)
            mid_start = max_dist * 0.2
            mid_end = max_dist * 0.6
            if mid_start <= dist <= mid_end:
                window = 1.0 - 0.5 * (
                    1.0 + math.cos(math.pi * (dist - mid_start) / (mid_end - mid_start))
                )
                dct_block[u][v] *= 1.0 - suppress * window
    return _idct2_ortho(dct_block)


# ---------------------------------------------------------------------------
# Keyword dispatch
# ---------------------------------------------------------------------------


def _strategy_kwargs(strategy: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the caller kwargs allowed for ``strategy``, rejecting the rest.

    Rejecting unexpected keywords catches typos and keeps callers honest:
    silently dropping ``sigm=2.0`` would apply a default the caller never chose.
    """
    allowed = DEGRADE_STRATEGY_KWARGS.get(strategy)
    if allowed is None:
        raise ValueError(f"unknown strategy: {strategy}")
    unexpected = sorted(set(kwargs) - set(allowed))
    if unexpected:
        names = ", ".join(unexpected)
        raise TypeError(f"unexpected keyword argument(s) for strategy {strategy!r}: {names}")
    return {name: kwargs[name] for name in allowed if name in kwargs}


def _validate_raster(width: int, height: int, channels: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if channels not in (1, 3, 4):
        raise ValueError("channels must be 1, 3, or 4")


def _color_channels(channels: int) -> int:
    """Channels to degrade: alpha is always left untouched for RGBA input."""
    return channels - 1 if channels == 4 else channels


def _append_alpha(out: list[list[float]], pixels: list[list[float]], channels: int) -> None:
    """Copy the original alpha channel through after a color-only transform."""
    if channels == 4:
        for i in range(len(out)):
            out[i].append(pixels[i][3])


def _block_origins(length: int, block_size: int, stride: int) -> list[int]:
    """Return starts that cover an axis, including a final anchored block."""
    last = max(0, length - block_size)
    origins = list(range(0, last + 1, stride))
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


# ---------------------------------------------------------------------------
# Public attack functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrequencyResult:
    """Result of a frequency-domain attack."""

    data: bytearray
    width: int
    height: int
    channels: int
    strategy: str
    suppress_ratio: float
    block_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "suppress_ratio": self.suppress_ratio,
            "block_size": self.block_size,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }


def frequency_suppress(
    pixels: list[list[float]],
    width: int,
    height: int,
    channels: int,
    *,
    suppress: float = 0.6,
    block_size: int = 8,
    overlap: int = 4,
) -> list[list[float]]:
    """Apply frequency-domain suppression to per-pixel float data.

    Operates independently on each channel using overlapping DCT blocks.
    """
    _validate_raster(width, height, channels)
    if not 0.0 <= suppress <= 1.0:
        raise ValueError("suppress must be in [0, 1]")
    if block_size < 2:
        raise ValueError("block_size must be >= 2")
    if not isinstance(overlap, int) or not 0 <= overlap < block_size:
        raise ValueError("overlap must be an integer in [0, block_size)")

    # Build per-channel channel data
    color_channels = _color_channels(channels)
    channel_data: list[list[list[float]]] = []
    for ch in range(color_channels):
        channel: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                idx = y * width + x
                row.append(pixels[idx][ch])
            channel.append(row)
        channel_data.append(channel)

    # Apply frequency suppression per channel. Edge-replicated padding ensures
    # sub-block images and right/bottom remainders are transformed too.
    result: list[list[list[float]]] = []
    stride = block_size - overlap
    y_origins = _block_origins(height, block_size, stride)
    x_origins = _block_origins(width, block_size, stride)
    for ch in range(color_channels):
        channel = channel_data[ch]
        temp: list[list[float]] = [[0.0] * width for _ in range(height)]
        count: list[list[float]] = [[0.0] * width for _ in range(height)]
        for by in y_origins:
            for bx in x_origins:
                block = [
                    [
                        channel[min(height - 1, by + dy)][min(width - 1, bx + dx)]
                        for dx in range(block_size)
                    ]
                    for dy in range(block_size)
                ]
                dct_out = _dct_block(block, suppress)
                for dy in range(min(block_size, height - by)):
                    for dx in range(min(block_size, width - bx)):
                        temp[by + dy][bx + dx] += dct_out[dy][dx]
                        count[by + dy][bx + dx] += 1.0

        channel_out: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                if count[y][x] > 0:
                    row.append(temp[y][x] / count[y][x])
                else:
                    row.append(channel[y][x])
            channel_out.append(row)
        result.append(channel_out)

    flat: list[list[float]] = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            flat.append([result[ch][y][x] for ch in range(color_channels)])
    _append_alpha(flat, pixels, channels)
    return flat


def frequency_suppress_from_bytes(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    suppress: float = 0.6,
    block_size: int = 8,
    overlap: int = 4,
) -> bytearray:
    """Apply frequency-domain suppression to raw pixel bytes.

    Enforces the same pixel cap as ``degrade_image`` so the pure-Python DCT
    cannot be asked to churn for hours on a large raster.
    """
    _validate_raster(width, height, channels)
    if len(raw) != width * height * channels:
        raise ValueError(f"raw length {len(raw)} != {width}*{height}*{channels}")
    if width * height > MAX_FREQ_DCT_PIXELS:
        raise ValueError(
            f"freq-dct supports at most {MAX_FREQ_DCT_PIXELS:,} pixels "
            f"({width * height:,} given); downscale the image first"
        )

    pixels: list[list[float]] = []
    for i in range(0, len(raw), channels):
        pixels.append([float(raw[i + c]) for c in range(channels)])

    result = frequency_suppress(
        pixels, width, height, channels, suppress=suppress, block_size=block_size, overlap=overlap
    )

    out = bytearray(len(raw))
    for i, row in enumerate(result):
        for c in range(channels):
            out[i * channels + c] = max(0, min(255, round(row[c])))
    return out


# ---------------------------------------------------------------------------
# Additional signal-domain degradations
# ---------------------------------------------------------------------------


def gaussian_blur_2d(
    pixels: list[list[float]],
    width: int,
    height: int,
    channels: int,
    *,
    sigma: float = 1.0,
) -> list[list[float]]:
    """Apply a Gaussian blur (separable 1-D passes) to each channel."""
    _validate_raster(width, height, channels)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return [list(row) for row in pixels]

    radius = max(1, int(3 * sigma))
    kernel: list[float] = []
    kernel_sum = 0.0
    for i in range(-radius, radius + 1):
        val = math.exp(-i * i / (2 * sigma * sigma))
        kernel.append(val)
        kernel_sum += val
    kernel = [v / kernel_sum for v in kernel]

    def _blur_row(row: list[float]) -> list[float]:
        out: list[float] = [0.0] * len(row)
        for x in range(len(row)):
            total = 0.0
            used_weight = 0.0
            for j, offset in enumerate(range(-radius, radius + 1)):
                nx = x + offset
                if 0 <= nx < len(row):
                    total += row[nx] * kernel[j]
                    used_weight += kernel[j]
            out[x] = total / used_weight
        return out

    def _blur_column(col_vals: list[float]) -> list[float]:
        out: list[float] = [0.0] * len(col_vals)
        for y in range(len(col_vals)):
            total = 0.0
            used_weight = 0.0
            for j, offset in enumerate(range(-radius, radius + 1)):
                ny = y + offset
                if 0 <= ny < len(col_vals):
                    total += col_vals[ny] * kernel[j]
                    used_weight += kernel[j]
            out[y] = total / used_weight
        return out

    out: list[list[float]] = [[] for _ in range(height * width)]
    for ch in range(_color_channels(channels)):
        col_data: list[list[float]] = [[0.0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                col_data[y][x] = pixels[y * width + x][ch]
        row_pass: list[list[float]] = [[0.0] * width for _ in range(height)]
        for y in range(height):
            row_pass[y] = _blur_row(col_data[y])
        for x in range(width):
            col_vals = [row_pass[y][x] for y in range(height)]
            blurred = _blur_column(col_vals)
            for y in range(height):
                out[y * width + x].append(blurred[y])
    _append_alpha(out, pixels, channels)
    return out


def median_filter_2d(
    pixels: list[list[float]],
    width: int,
    height: int,
    channels: int,
    *,
    kernel_size: int = 3,
) -> list[list[float]]:
    """Apply a per-channel median filter."""
    _validate_raster(width, height, channels)
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    half = kernel_size // 2
    out: list[list[float]] = [[] for _ in range(height * width)]
    for ch in range(_color_channels(channels)):
        vals: list[list[float]] = [[0.0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                vals[y][x] = pixels[y * width + x][ch]
        for y in range(height):
            for x in range(width):
                samples: list[float] = []
                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            samples.append(vals[ny][nx])
                samples.sort()
                mid = len(samples) // 2
                out[y * width + x].append(samples[mid])
    _append_alpha(out, pixels, channels)
    return out


def jpeg_compress_sim(
    pixels: list[list[float]],
    width: int,
    height: int,
    channels: int,
    *,
    quality: int = 40,
) -> list[list[float]]:
    """Simulate JPEG compression via quantized DCT + rounding."""
    _validate_raster(width, height, channels)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")
    standard = [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    qm = [[max(1, min(255, int((entry * scale + 50) // 100))) for entry in row] for row in standard]

    out: list[list[float]] = [[] for _ in range(height * width)]
    for ch in range(_color_channels(channels)):
        block_data: list[list[float]] = [[0.0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                block_data[y][x] = pixels[y * width + x][ch]

        for by in range(0, height, 8):
            for bx in range(0, width, 8):
                blk = [
                    [
                        block_data[min(height - 1, by + dy)][min(width - 1, bx + dx)]
                        for dx in range(8)
                    ]
                    for dy in range(8)
                ]
                dct_blk = _dct2_ortho(blk)
                for u in range(8):
                    for v in range(8):
                        dct_blk[u][v] = round(dct_blk[u][v] / qm[u][v]) * qm[u][v]
                out_blk = _idct2_ortho(dct_blk)
                for dy in range(min(8, height - by)):
                    for dx in range(min(8, width - bx)):
                        block_data[by + dy][bx + dx] = out_blk[dy][dx]

        for y in range(height):
            for x in range(width):
                out[y * width + x].append(block_data[y][x])

    _append_alpha(out, pixels, channels)
    return out


def rotate_image(
    pixels: list[list[float]],
    width: int,
    height: int,
    channels: int,
    *,
    angle_deg: float = 3.0,
) -> list[list[float]]:
    """Rotate pixels by angle (degrees) using nearest-neighbor sampling."""
    _validate_raster(width, height, channels)
    if angle_deg == 0:
        return [list(row) for row in pixels]

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    cx, cy = width / 2, height / 2
    color_channels = _color_channels(channels)
    out: list[list[float]] = [[0.0] * color_channels for _ in range(height * width)]

    for y in range(height):
        for x in range(width):
            i = y * width + x
            rx = cos_a * (x - cx) + sin_a * (y - cy) + cx
            ry = -sin_a * (x - cx) + cos_a * (y - cy) + cy
            sx, sy = round(rx), round(ry)
            if 0 <= sx < width and 0 <= sy < height:
                out[i] = [pixels[sy * width + sx][c] for c in range(color_channels)]
            # Out-of-bounds pixels stay [0.0, ...]; alpha is copied below so
            # rotation never manufactures transparency.
            if channels == 4:
                out[i].append(pixels[i][3])
    return out


def two_stage_attack(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    blur_sigma: float = 1.0,
    quality: int = 40,
) -> FrequencyResult:
    """Two-stage degradation + soft-restore.

    Stage 1: degrade (blur + low-quality JPEG)
    Stage 2: soft-restore (median filter to recover edges)

    Paper-reported success rate: 95–98 % for robust spatial marks.
    """
    pixels: list[list[float]] = []
    for i in range(0, len(raw), channels):
        pixels.append([float(raw[i + c]) for c in range(channels)])

    blurred = gaussian_blur_2d(pixels, width, height, channels, sigma=blur_sigma)
    jpeg = jpeg_compress_sim(blurred, width, height, channels, quality=quality)
    restored = median_filter_2d(jpeg, width, height, channels, kernel_size=3)

    out = bytearray(len(raw))
    for i, row in enumerate(restored):
        for c in range(channels):
            out[i * channels + c] = max(0, min(255, round(row[c])))

    return FrequencyResult(
        data=out,
        width=width,
        height=height,
        channels=channels,
        strategy="two-stage",
        suppress_ratio=0.0,
        block_size=8,
    )


def degrade_image(
    raw: bytes,
    width: int,
    height: int,
    channels: int,
    *,
    strategy: str = "freq-dct",
    **kwargs: Any,
) -> FrequencyResult:
    """Apply one degradation strategy to raw pixel bytes.

    Strategies
    ----------
    freq-dct   — frequency suppression via DCT (default)
    blur       — Gaussian blur
    median     — median filter
    jpeg       — Simulated JPEG compression
    rotate     — Nearest-neighbor rotation
    two-stage  — blur → jpeg → median (95–98% ASR paper-reported)

    Only the keywords listed in ``DEGRADE_STRATEGY_KWARGS`` are accepted for
    each strategy; any other keyword raises ``TypeError`` so typos cannot
    silently apply defaults.
    """
    _validate_raster(width, height, channels)
    if len(raw) != width * height * channels:
        raise ValueError(f"raw length {len(raw)} != {width}*{height}*{channels}")
    pixel_count = width * height
    if strategy == "freq-dct" and pixel_count > MAX_FREQ_DCT_PIXELS:
        raise ValueError(
            f"freq-dct supports at most {MAX_FREQ_DCT_PIXELS:,} pixels "
            f"({pixel_count:,} given); downscale the image first"
        )
    if strategy in ("jpeg", "two-stage") and pixel_count > MAX_JPEG_PIXELS:
        raise ValueError(
            f"{strategy} supports at most {MAX_JPEG_PIXELS:,} pixels "
            f"({pixel_count:,} given); downscale the image first"
        )

    filtered = _strategy_kwargs(strategy, kwargs)
    pixels: list[list[float]] = []
    for i in range(0, len(raw), channels):
        pixels.append([float(raw[i + c]) for c in range(channels)])

    def _bytes_from_result(result_pixels: list[list[float]]) -> bytearray:
        out = bytearray(len(raw))
        for i, row in enumerate(result_pixels):
            for c in range(channels):
                out[i * channels + c] = max(0, min(255, round(row[c])))
        return out

    if strategy == "freq-dct":
        result = frequency_suppress(pixels, width, height, channels, **filtered)
        return FrequencyResult(
            data=_bytes_from_result(result),
            width=width,
            height=height,
            channels=channels,
            strategy="freq-dct",
            suppress_ratio=filtered.get("suppress", 0.6),
            block_size=filtered.get("block_size", 8),
        )

    if strategy == "blur":
        result = gaussian_blur_2d(pixels, width, height, channels, **filtered)
        return FrequencyResult(
            data=_bytes_from_result(result),
            width=width,
            height=height,
            channels=channels,
            strategy="blur",
            suppress_ratio=0.0,
            block_size=8,
        )

    if strategy == "median":
        result = median_filter_2d(pixels, width, height, channels, **filtered)
        return FrequencyResult(
            data=_bytes_from_result(result),
            width=width,
            height=height,
            channels=channels,
            strategy="median",
            suppress_ratio=0.0,
            block_size=8,
        )

    if strategy == "jpeg":
        result = jpeg_compress_sim(pixels, width, height, channels, **filtered)
        return FrequencyResult(
            data=_bytes_from_result(result),
            width=width,
            height=height,
            channels=channels,
            strategy="jpeg",
            suppress_ratio=0.0,
            block_size=8,
        )

    if strategy == "rotate":
        result = rotate_image(pixels, width, height, channels, **filtered)
        return FrequencyResult(
            data=_bytes_from_result(result),
            width=width,
            height=height,
            channels=channels,
            strategy="rotate",
            suppress_ratio=0.0,
            block_size=8,
        )

    if strategy == "two-stage":
        return two_stage_attack(raw, width, height, channels, **filtered)

    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI: degrade a PNG image through a frequency attack."""
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
        choices=list(DEGRADE_STRATEGIES),
        default="freq-dct",
        help="Degradation strategy",
    )
    parser.add_argument("--suppress", type=float, default=0.6, help="DCT suppression ratio (0-1)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Blur sigma")
    parser.add_argument("--angle", type=float, default=3.0, help="Rotation angle in degrees")
    parser.add_argument("--quality", type=int, default=40, help="JPEG quality (1-100)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    # Forward only the keywords the selected strategy accepts.
    all_kwargs = {
        "suppress": args.suppress,
        "sigma": args.sigma,
        "blur_sigma": args.sigma,
        "angle_deg": args.angle,
        "quality": args.quality,
    }
    kwargs = {
        name: all_kwargs[name]
        for name in DEGRADE_STRATEGY_KWARGS[args.strategy]
        if name in all_kwargs
    }

    try:
        raw = read_bytes_bounded(Path(args.image), MAX_ENCODED_BYTES, label="encoded file")
        raster = decode_png(raw)

        result = degrade_image(
            bytes(raster.data),
            raster.width,
            raster.height,
            raster.channels,
            strategy=args.strategy,
            **kwargs,
        )

        if args.output is None:
            args.output = str(Path(args.image).with_suffix(".degraded.png"))

        out_raster = Raster(result.width, result.height, result.channels, result.data)
        atomic_write_bytes(Path(args.output), encode_png(out_raster))

        report = result.to_dict()
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"wrote {args.output} strategy={result.strategy}")
        return 0
    except Exception as error:
        logger.error(f"error: {error}", module="dct_frequency")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
