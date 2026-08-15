"""OpenCV inpainting backends (Telea + Navier-Stokes).

Adds two new algorithms behind the InpaintBackend protocol.  Both are
pure adapters around ``cv2.inpaint()`` and share conversion/validation
infrastructure.

Requirements
------------
    pip install opencv-python-headless   # or: watermark-remover[visible]

Usage
-----
    from inpaint_backends import cv2_telea_backend, cv2_navier_stokes_backend
    from inpaint_backends import get_opencv_backends

    for backend in get_opencv_backends():
        if backend.available:
            result = backend.inpaint(request)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optional_deps
from domain_types import InpaintRequest, InpaintResult
from inpaint_backends import InpaintBackend

# ---------------------------------------------------------------------------
# Shared OpenCV conversion utilities
# ---------------------------------------------------------------------------


def _raster_to_cv2(raster: Any, mask: Any) -> tuple[Any, Any]:
    """Convert a morphomod Raster + Mask to OpenCV format (uint8).

    Returns (cv_image, cv_mask) where cv_image is a numpy array and
    cv_mask is a uint8 binary mask.
    """
    import numpy as np

    _ = optional_deps.cv2

    w, h = raster.width, raster.height
    ch = raster.channels

    # Build numpy array from raster data
    arr = np.frombuffer(bytes(raster.data), dtype=np.uint8).reshape(h, w, ch)

    # Convert RGBA → RGB if needed (cv2.inpaint doesn't handle alpha)
    if ch == 4:
        arr = arr[:, :, :3]  # drop alpha channel
        ch = 3

    # Convert grayscale → RGB for consistent handling
    if ch == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
        ch = 3

    # Convert mask: 0→keep, 255→remove → uint8 binary
    cv_mask = np.frombuffer(bytes(mask.data), dtype=np.uint8).reshape(h, w)

    return arr, cv_mask


def _cv2_to_raster(cv_image: Any, original: Any) -> Any:
    """Convert a CV2 image back to morphomod.Raster.

    Preserves the original's channel count and dimensions.
    """
    import numpy as np
    from morphomod import Raster

    h, w = cv_image.shape[:2]
    # If original had alpha, extend back to 4 channels
    if original.channels == 4:
        alpha = np.frombuffer(bytes(original.data), dtype=np.uint8).reshape(h, w, 4)[:, :, 3:4]
        cv_image = np.concatenate([cv_image, alpha], axis=-1)
    elif original.channels == 1:
        # Keep single channel
        cv_image = cv_image[:, :, 0:1]

    data = bytearray(cv_image.tobytes())
    return Raster(width=w, height=h, channels=original.channels, data=data)


def _validate_request(request: InpaintRequest) -> None:
    """Validate that request parameters are suitable for cv2.inpaint."""
    if request.source.width * request.source.height > 40_000_000:
        raise ValueError("image exceeds 40 MP safety limit for OpenCV inpainting")
    if request.mask.marked == 0:
        raise ValueError("mask contains no marked pixels")
    if request.mask.marked >= request.source.width * request.source.height:
        raise ValueError("mask covers entire image")


# ---------------------------------------------------------------------------
# Telea backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cv2TeleaBackend:
    """cv2.INPAINT_TELEA — fast method based on Navier-Stokes principle."""

    name = "opencv-telea"
    radius: int = 3

    @property
    def available(self) -> bool:
        return optional_deps.has_visible()

    def supports(self, request: InpaintRequest) -> bool:
        # Telea excels at thin/small masks
        mask_ratio = request.mask.marked / max(1, request.source.width * request.source.height)
        return mask_ratio < 0.3 and request.source.channels in (1, 3, 4)

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        _validate_request(request)
        cv_image, cv_mask = _raster_to_cv2(request.source, request.mask)
        radius = request.params.get("radius", self.radius)
        radius = max(1, min(9, radius))  # safe bounds

        cv = optional_deps.cv2
        restored = cv.inpaint(cv_image, cv_mask, radius, cv.INPAINT_TELEA)
        return InpaintResult(
            inpainted=_cv2_to_raster(restored, request.source),
            backend_name=self.name,
            quality_estimate=0.72,
            diagnostics={"radius": radius, "method": "TELEA"},
        )


# ---------------------------------------------------------------------------
# Navier-Stokes backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cv2NavierStokesBackend:
    """cv2.INPAINT_NS — Navier-Stokes based inpainting."""

    name = "opencv-ns"
    radius: int = 3

    @property
    def available(self) -> bool:
        return optional_deps.has_visible()

    def supports(self, request: InpaintRequest) -> bool:
        # NS works well for small-to-medium structured masks
        mask_ratio = request.mask.marked / max(1, request.source.width * request.source.height)
        return mask_ratio < 0.4 and request.source.channels in (1, 3, 4)

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        _validate_request(request)
        cv_image, cv_mask = _raster_to_cv2(request.source, request.mask)
        radius = request.params.get("radius", self.radius)
        radius = max(1, min(9, radius))

        cv = optional_deps.cv2
        restored = cv.inpaint(cv_image, cv_mask, radius, cv.INPAINT_NS)
        return InpaintResult(
            inpainted=_cv2_to_raster(restored, request.source),
            backend_name=self.name,
            quality_estimate=0.70,
            diagnostics={"radius": radius, "method": "NS"},
        )


# ---------------------------------------------------------------------------
# OpenCV backend registry helper
# ---------------------------------------------------------------------------


def get_opencv_backends() -> list[InpaintBackend]:
    """Return available OpenCV backends (Telea + Navier-Stokes).

    Both backends share the same availability gate (cv2 present).
    """
    backends: list[InpaintBackend] = []
    if optional_deps.cv2 is not None:
        backends.append(Cv2TeleaBackend())
        backends.append(Cv2NavierStokesBackend())
    return backends
