"""Sliding-window tiled execution for large images.

Ported from wiltodelta/remove-ai-watermarks ``_internal/tiling.py``, Apache-2.0;
modified: tiles are numpy ``HxWxC`` arrays instead of PIL images (the clean
pipeline has no PIL dependency), ``generate_tile`` receives the :class:`Tile`
so a mask can be shifted into tile coordinates, the blend rounds with
``numpy.rint`` before the uint8 cast (upstream ``unsharp_mask`` precedent), and
the auto-tile threshold policy lives here.

Upstream rationale, kept: the whole-image pass OOMs above ~2K pixels per side
(upstream issue #10). Tiling splits the image into overlapping tiles — each
kept near the ~1024 working size — regenerates each tile independently, and
feather-blends the overlaps. The result retains the input's native dimensions.

Same-result guarantee (stated): :func:`run_tiled` is *exact* whenever the
per-tile results agree wherever tiles overlap — in particular an identity or
locally-deterministic ``generate_tile`` reproduces the input byte-for-byte,
because every blend of equal values rounds back to that value. A
context-sensitive fill that reads past its tile's bounds can disagree across
tile seams; that is precisely why tiled execution always reports its
engagement (tile grid, overlap) instead of staying silent.

The geometry (``plan_tiles``) and the blend weighting (``feather_weights``)
are pure functions. ``run_tiled`` is the orchestration loop; it takes a
``generate_tile`` callable so it stays decoupled from the pipeline internals.

numpy is imported lazily inside the functions so importing this module loads
no third-party code (the stdlib core import graph is guarded by test); a call
without numpy installed raises the labeled
:class:`~image_io.PixelDependencyUnavailable` error naming
``watermark-remover[pixel]``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, NamedTuple

from image_io import PixelDependencyUnavailable, _import_pixel

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

#: Strictly-positive floor for the accumulated blend weights so a region
#: covered by a single feathered tile edge (an image corner, no neighbour to
#: blend with) never divides by zero.
_WEIGHT_EPS = 1e-3

#: Auto-tiling engages when the longest side exceeds this many pixels
#: (upstream: the global pass OOMs above ~2K). Overridable through
#: ``WATERMARKS_AUTO_TILE_MAX_DIMENSION`` (the ``WATERMARKS_MAX_INPUT_BYTES``
#: environment precedent) so tests and constrained hosts can move the line.
AUTO_TILE_MAX_DIMENSION = 2048

#: Working tile size and feather overlap for the automatic path (upstream
#: CLI defaults: ``tile_size=1024``, ``tile_overlap=128``).
AUTO_TILE_SIZE = 1024
AUTO_TILE_OVERLAP = 128


class Tile(NamedTuple):
    """A tile crop box in the source image: top-left ``(x, y)`` + ``width``/``height``."""

    x: int
    y: int
    width: int
    height: int


def auto_tile_threshold() -> int:
    """The effective auto-tile dimension threshold for this run."""
    return int(os.environ.get("WATERMARKS_AUTO_TILE_MAX_DIMENSION", str(AUTO_TILE_MAX_DIMENSION)))


def should_auto_tile(width: int, height: int, threshold: int | None = None) -> bool:
    """Whether a ``width`` x ``height`` image takes the tiled path."""
    limit = auto_tile_threshold() if threshold is None else threshold
    return max(width, height) > limit


def _axis_positions(length: int, tile: int, overlap: int) -> list[int]:
    """Tile start offsets along one axis, last tile flush to the far edge.

    Every interior tile is exactly ``tile`` long; the final tile is pulled back
    to ``length - tile`` so it ends exactly at the edge (it simply overlaps its
    predecessor a little more). Keeping all tiles the same size is what lets a
    per-tile pass run at one preferred dimension on every tile.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if length <= tile:
        return [0]
    # Guarantee forward progress even on a pathological overlap >= tile.
    overlap = min(max(overlap, 0), tile - 1)
    step = tile - overlap
    positions = list(range(0, length - tile + 1, step))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def plan_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Lay out a grid of overlapping tiles covering ``width`` x ``height``.

    All tiles are ``min(tile_size, width)`` x ``min(tile_size, height)`` (uniform
    size; the image itself when it fits in one tile). Returned in row-major order.
    """
    xs = _axis_positions(width, tile_size, overlap)
    ys = _axis_positions(height, tile_size, overlap)
    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    return [Tile(x, y, tile_w, tile_h) for y in ys for x in xs]


def feather_weights(width: int, height: int, overlap: int) -> NDArray[Any]:
    """A 2D blend window: ~1 in the interior, ramping down toward each edge.

    Separable linear taper over ``overlap`` pixels from every edge (capped at
    half the tile so short tiles still taper symmetrically). Strictly positive
    everywhere, so the normalized blend is well-defined even at an image corner
    that only one tile covers.
    """
    np = _import_pixel("numpy")

    def ramp(n: int) -> NDArray[Any]:
        w = np.ones(n, dtype=np.float32)
        if overlap > 0 and n > 1:
            ramp_len = min(overlap, max(1, n // 2))
            taper = (np.arange(ramp_len, dtype=np.float32) + 1.0) / (ramp_len + 1.0)
            w[:ramp_len] = taper
            w[n - ramp_len :] = taper[::-1]
        return w

    weights = np.outer(ramp(height), ramp(width))
    np.maximum(weights, _WEIGHT_EPS, out=weights)
    return weights


def run_tiled(
    generate_tile: Callable[[Tile, NDArray[Any]], NDArray[Any]],
    image: NDArray[Any],
    tile_size: int,
    overlap: int,
    set_progress: Callable[[str], None] | None = None,
) -> NDArray[Any]:
    """Tile ``image``, run ``generate_tile`` per tile, and feather-blend the result.

    ``image`` is a ``HxWxC`` uint8 array; ``generate_tile`` receives the
    :class:`Tile` (so a caller can shift masks/boxes into tile coordinates) and
    the crop, and returns the regenerated tile at the same shape. Upstream
    resized a dimension-rounded diffusion output back before blending; our
    callers regenerate in place, so a shape mismatch is a caller bug and
    raises.

    The blend is exact for agreeing per-tile results: equal values blend back
    to themselves (see the module docstring's same-result guarantee), because
    the normalized sum rounds through ``numpy.rint`` before the uint8 cast.
    """
    np = _import_pixel("numpy")
    if image.ndim < 2:
        raise ValueError(f"expected a HxWxC image array, got shape {image.shape}")
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    tiles = plan_tiles(width, height, tile_size, overlap)
    accum = np.zeros((height, width, channels), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    # All tiles share one size (plan_tiles is uniform), so the feather window is
    # loop-invariant -- compute it once.
    weights = feather_weights(tiles[0].width, tiles[0].height, overlap)[:, :, None]

    total = len(tiles)
    for index, tile in enumerate(tiles, start=1):
        if set_progress is not None:
            set_progress(
                f"Tiled pass: tile {index}/{total} at ({tile.x},{tile.y}) "
                f"{tile.width}x{tile.height}..."
            )
        crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        result = generate_tile(tile, crop)
        if result.shape != crop.shape:
            raise ValueError(
                f"generate_tile returned shape {result.shape} for tile {tile}; "
                f"expected {crop.shape}"
            )
        arr = result.astype(np.float32)
        accum[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += arr * weights
        weight_sum[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += weights

    blended = accum / np.maximum(weight_sum, _WEIGHT_EPS)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


__all__ = [
    "AUTO_TILE_MAX_DIMENSION",
    "AUTO_TILE_OVERLAP",
    "AUTO_TILE_SIZE",
    "PixelDependencyUnavailable",
    "Tile",
    "auto_tile_threshold",
    "feather_weights",
    "plan_tiles",
    "run_tiled",
    "should_auto_tile",
]
