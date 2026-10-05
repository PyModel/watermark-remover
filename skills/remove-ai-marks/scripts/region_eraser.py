"""Universal region eraser: remove anything inside user-given boxes via inpainting.

Position- and content-agnostic. You supply the rectangle(s); the eraser inpaints
whatever is inside, so it removes any visible logo / watermark / object regardless
of color, style, or location. Localization is the user's responsibility (pass the
box); restoration runs on CPU. This is the universal fallback for marks the
registered visible detectors do not cover.

Backends:
  - ``cv2`` (default): ``cv2.inpaint`` (Telea / Navier-Stokes). Instant, no extra
    dependencies beyond the ``pixel`` extra, lower quality on large or textured
    regions. ``cv2_method`` picks telea or ns; the registry exposes the same
    fills as the ``cv2-telea`` / ``cv2-ns`` backends.
  - ``migan`` (optional, extra ``migan``): MI-GAN via onnxruntime
    (``andraniksargsyan/migan``, MIT). CPU, ~28 MB model. Model downloaded on
    first use. Like ``lama`` it crops a padded region around the mask before
    inference (at native resolution -- MI-GAN takes arbitrary dims), so peak RAM
    is bounded by the mark size rather than scaling with the image.
  - ``lama`` (optional, extra ``lama``): big-LaMa via onnxruntime
    (``Carve/LaMa-ONNX``, Apache-2.0). CPU, resolution-robust, best quality on
    texture but ~200 MB model and ~4.7 GB peak RAM (too heavy for a small host).
    The model is downloaded on first use and cached by huggingface_hub; it is
    never bundled in this repo.

``FILL_BACKENDS`` states every caller-facing fill choice including ``auto``.
Registry-level resolution (auto = lama > migan > cv2 > texture/simple, skipping
tiers that are not installed) lives in ``inpaint_backends.resolve_backend``;
inside this module an ``erase(backend="auto")`` call degrades to the classical
cv2 fill, matching upstream semantics.

This module is on the pixel path: it imports numpy and OpenCV at import time
(like ``image_io``'s callers), so only pixel-extra code should import it --
the stdlib core import graph never reaches here. onnxruntime and
huggingface_hub stay lazy: they load only inside the learned-backend probes
and session loaders.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Backend = Literal["cv2", "lama", "migan"]

# Every fill backend this module can execute, plus the caller-facing `auto`. Kept as a
# literal tuple (not derived through an import) so the registry and the CLI can state
# their choices without importing this cv2-loading module at their import time --
# a test pins the copies in sync.
FILL_BACKENDS: tuple[str, ...] = ("auto", "cv2", "migan", "lama")


def _full_scale(image: NDArray[Any]) -> float:
    """Full-scale value of an integer image dtype: 255 for uint8, 65535 for uint16."""
    if image.dtype.kind not in "ui":
        raise RuntimeError(
            f"Unsupported image depth {image.dtype}: the fill backends take integer "
            "images (8- or 16-bit). Convert the source before erasing."
        )
    return float(np.iinfo(image.dtype).max)


def _erase_via_uint8(
    backend: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]],
    image_bgr: NDArray[Any],
    mask: NDArray[Any],
) -> NDArray[Any]:
    """Run a uint8-only backend on a deeper image without flattening the whole frame.

    ``cv2.inpaint`` takes 8-bit 3-channel colour (its 16-bit support is
    single-channel only), and the shipped MI-GAN ONNX declares a uint8 input
    tensor. A 16-bit source therefore has to be narrowed for the fill itself --
    but only for the fill. The filled pixels are written back into the original
    array, so everything outside the mask stays bit-exact at its original depth,
    which is the same guarantee the learned backends already make by pasting only
    masked pixels.

    LaMa is the exception and does not come through here: its model is float32, so
    ``erase_lama`` carries the source depth end to end.
    """
    scale = _full_scale(image_bgr)
    source = image_bgr[..., :3] if image_bgr.ndim == 3 and image_bgr.shape[2] == 4 else image_bgr
    work = cv2.convertScaleAbs(source, alpha=255.0 / scale)
    filled = backend(work, mask)
    result = image_bgr.copy()
    destination = result[..., :3] if image_bgr.ndim == 3 and image_bgr.shape[2] == 4 else result
    _paste_scaled_uint8_fill(destination, filled, mask, scale, image_bgr.dtype)
    return result


def _paste_scaled_uint8_fill(
    destination: NDArray[Any],
    filled: NDArray[Any],
    mask: NDArray[Any],
    scale: float,
    dtype: np.dtype[Any],
) -> None:
    """Scale uint8 fill pixels into a deeper destination, only inside ``mask``."""
    hole = mask > 127
    destination[hole] = np.clip(filled[hole].astype(np.float32) * (scale / 255.0), 0, scale).astype(
        dtype
    )


def boxes_to_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    dilate: int = 3,
) -> NDArray[Any]:
    """Build a uint8 mask (255 inside boxes) from ``(x, y, w, h)`` rectangles.

    Boxes are clipped to the image bounds (a partly-outside box must not raise).
    ``dilate`` is clamped to the image diagonal: any radius at least that large
    saturates the whole frame, so the clamp changes no result while bounding the
    ``(2r+1)``-square kernel allocation for absurd inputs (untrusted radii are a
    tier-1 concern; the kernel would otherwise allocate ``(2r+1)**2`` bytes).
    """
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if dilate > 0 and mask.any():
        radius = min(int(dilate), int(math.hypot(h, w)) + 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, k)
    return mask


def _padded_crop_box(
    mask: NDArray[Any], h: int, w: int, *, pad_frac: float, pad_min: int
) -> tuple[int, int, int, int] | None:
    """Bounding box of the set mask pixels, padded and clamped to the image.

    Returns ``(x0, y0, x1, y1)`` or ``None`` when the mask is empty. Both learned
    backends crop to this box so the ONNX working set is bounded by the mark size
    rather than the whole image; ``pad_frac``/``pad_min`` tune how much surrounding
    context the inpainter sees (LaMa then resizes the crop to its fixed square,
    MI-GAN feeds it at native resolution).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = max(pad_min, int(pad_frac * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def erase_cv2(
    image_bgr: NDArray[Any],
    mask: NDArray[Any],
    *,
    method: Literal["telea", "ns"] = "telea",
    radius: int = 6,
) -> NDArray[Any]:
    """Inpaint ``mask`` with classical cv2 inpainting (CPU, no extra deps).

    Accepts 1-/3-channel BGR (passed straight to ``cv2.inpaint``) and 4-channel
    BGRA: ``cv2.inpaint`` rejects 4 channels, so the alpha plane is split off,
    the BGR is inpainted, and alpha is re-attached unchanged.

    A deeper-than-8-bit source is filled through :func:`_erase_via_uint8`: OpenCV's
    inpaint takes 16-bit only as a single channel, so 16-bit colour would otherwise
    raise a bare ``icvInpaint`` "Unsupported format" error.
    """
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    if image_bgr.dtype == np.uint16 and image_bgr.ndim == 2:
        return cv2.inpaint(image_bgr, mask, radius, flag)
    if image_bgr.dtype != np.uint8:
        return _erase_via_uint8(
            lambda img, m: erase_cv2(img, m, method=method, radius=radius), image_bgr, mask
        )
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        bgr = cv2.inpaint(image_bgr[:, :, :3], mask, radius, flag)
        return np.dstack([bgr, image_bgr[:, :, 3]])
    return cv2.inpaint(image_bgr, mask, radius, flag)


def erase(
    image_bgr: NDArray[Any],
    *,
    boxes: list[tuple[int, int, int, int]] | None = None,
    mask: NDArray[Any] | None = None,
    backend: str = "cv2",
    dilate: int = 3,
    cv2_method: Literal["telea", "ns"] = "telea",
    cv2_radius: int = 6,
) -> NDArray[Any]:
    """Erase the given boxes (or mask) via the chosen inpainting backend.

    Provide either ``boxes`` (list of ``(x, y, w, h)``) or a precomputed ``mask``
    (uint8, 255 = erase). Returns an unmodified copy when nothing is selected.

    ``auto`` (and anything unrecognized) degrades to the classical cv2 fill rather
    than raising -- registry-level auto resolution across all tiers (including the
    pure backends) is ``inpaint_backends.resolve_backend``.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if mask is None:
        if not boxes:
            return image_bgr.copy()
        mask = boxes_to_mask(image_bgr.shape[:2], boxes, dilate=dilate)
    if not mask.any():
        return image_bgr.copy()

    return erase_cv2(image_bgr, mask, method=cv2_method, radius=cv2_radius)
