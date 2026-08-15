"""Post-inpaint verification for visible watermark removal."""

from __future__ import annotations

import math
from typing import Any


def _validate_inputs(source: Any, final: Any, mask: Any) -> int:
    """Validate exact raster/mask compatibility and return channel count."""
    if (source.width, source.height) != (final.width, final.height):
        raise ValueError("source/final dimensions differ")
    if source.channels != final.channels:
        raise ValueError("source/final channel counts differ")
    if (mask.width, mask.height) != (source.width, source.height):
        raise ValueError("mask/raster dimensions differ")
    return source.channels


def verify_outside_mask(source: Any, final: Any, effective_mask: Any) -> tuple[bool, int]:
    """Return whether every channel outside the effective mask is unchanged."""
    channels = _validate_inputs(source, final, effective_mask)
    diff_count = 0
    for index, marked in enumerate(effective_mask.data):
        if marked:
            continue
        offset = index * channels
        for channel in range(channels):
            if source.data[offset + channel] != final.data[offset + channel]:
                diff_count += 1
    return diff_count == 0, diff_count


def verify_boundary_seam(source: Any, final: Any, effective_mask: Any) -> float:
    """Score discontinuity along the effective-mask boundary ring."""
    channels = _validate_inputs(source, final, effective_mask)
    eroded = _erode_simple(effective_mask)
    boundary_indices = [
        index
        for index, marked in enumerate(effective_mask.data)
        if marked and not eroded.data[index]
    ]
    if not boundary_indices:
        return 0.0

    total_error = 0.0
    count = 0
    for index in boundary_indices:
        x, y = index % effective_mask.width, index // effective_mask.width
        neighbor = _find_boundary_neighbor(effective_mask, x, y)
        if neighbor is None:
            continue
        neighbor_index = (neighbor[1] * effective_mask.width + neighbor[0]) * channels
        offset = index * channels
        for channel in range(channels):
            difference = final.data[offset + channel] - final.data[neighbor_index + channel]
            total_error += difference * difference
            count += 1
    return min(1.0, math.sqrt(total_error / count) / 25.5) if count else 0.0


def verify_halo(source: Any, final: Any, original_mask: Any) -> tuple[float, tuple[str, ...]]:
    """Score residual structure in the one-pixel ring around the original mask."""
    channels = _validate_inputs(source, final, original_mask)
    dilated = _dilate_simple(original_mask, 1)
    halo_indices = [
        index
        for index, marked in enumerate(dilated.data)
        if marked and not original_mask.data[index]
    ]
    if not halo_indices:
        return 0.0, ()

    total_diff = 0.0
    edge_count = 0
    high_contrast_count = 0
    for index in halo_indices:
        x, y = index % original_mask.width, index // original_mask.width
        offset = index * channels
        inner = _find_inner_neighbor(original_mask, x, y)
        if inner is not None:
            inner_offset = (inner[1] * original_mask.width + inner[0]) * channels
            for channel in range(channels):
                difference = abs(final.data[offset + channel] - final.data[inner_offset + channel])
                total_diff += difference
                if difference > 60:
                    high_contrast_count += 1

        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < original_mask.width and 0 <= ny < original_mask.height):
                continue
            neighbor = ny * original_mask.width + nx
            if original_mask.data[neighbor]:
                continue
            neighbor_offset = neighbor * channels
            for channel in range(channels):
                if abs(final.data[offset + channel] - final.data[neighbor_offset + channel]) > 40:
                    edge_count += 1

    denominator = max(1, len(halo_indices))
    score = (total_diff / denominator) / 25.5 + (edge_count / denominator) * 0.1
    warnings: list[str] = []
    if high_contrast_count > len(halo_indices) * 0.1:
        warnings.append("strong edge remnants detected in halo ring")
    if score > 0.5:
        warnings.append("abnormal local contrast in halo region")
    return min(1.0, score), tuple(warnings)


def _dilate_simple(mask: Any, radius: int) -> Any:
    from morphomod import dilate

    return dilate(mask, radius)


def _erode_simple(mask: Any) -> Any:
    width, height = mask.width, mask.height
    data = bytearray(width * height)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            index = y * width + x
            if (
                mask.data[index]
                and mask.data[index - width]
                and mask.data[index + width]
                and mask.data[index - 1]
                and mask.data[index + 1]
            ):
                data[index] = 255
    return type(mask)(width, height, data)


def _find_boundary_neighbor(mask: Any, x: int, y: int) -> tuple[int, int] | None:
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < mask.width and 0 <= ny < mask.height and not mask.data[ny * mask.width + nx]:
            return nx, ny
    return None


def _find_inner_neighbor(mask: Any, x: int, y: int) -> tuple[int, int] | None:
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < mask.width and 0 <= ny < mask.height and mask.data[ny * mask.width + nx]:
            return nx, ny
    return None
