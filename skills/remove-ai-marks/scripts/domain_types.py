"""Domain types shared across inpainting backends and verification.

These dataclasses fill the gap between the low-level Mask/Raster types in
morphomod.py and the high-level orchestration in clean_asset.py.  They are
frozen so callers can safely pass them through the verification pipeline
without mutation surprises.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# ---------------------------------------------------------------------------
# Mask result — replaces bare Morphomod Mask with richer semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaskResult:
    """Result of mask acquisition + refinement.

    Wraps the original input mask and the effective mask used for inpainting,
    plus source and confidence metadata so verification can attribute failures.
    """

    #: The raw mask exactly as provided (PGM, PNG, or generated from box).
    original_mask: Any  # morphomod.Mask
    #: Mask after hole-fill, component filtering, and dilation.
    effective_mask: Any  # morphomod.Mask
    #: Human-readable origin: ``"mask:/path"`` or ``"box:..."`` etc.
    source: str
    #: Detection confidence in [0, 1], if an external detector generated this mask.
    confidence: float | None = None
    #: Non-empty when refinement dropped or added unexpected regions.
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be a finite number in [0, 1] or None")
        object.__setattr__(self, "warnings", tuple(self.warnings))


# ---------------------------------------------------------------------------
# Inpaint request / result — abstracts away backend specifics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InpaintRequest:
    """Parameters passed to an InpaintBackend.

    The backend decides *how* to fill the mask; the request carries the
    *what* and *where*.
    """

    #: The original image raster (RGBA/RGB/grayscale).
    source: Any  # morphomod.Raster
    #: The effective mask defining the region to inpaint.
    mask: Any  # morphomod.Mask
    #: Immutable backend-specific tuning (radius, seed, etc.).
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class InpaintResult:
    """Output produced by an InpaintBackend.

    A backend may return BEST_EFFORT even when successful because it cannot
    guarantee the restored region is identical to what a human would draw.
    """

    #: The inpainted raster (same dimensions as request.source).
    inpainted: Any  # morphomod.Raster
    #: Backend name that produced this result.
    backend_name: str
    #: Confidence in the restoration quality (finite, from 0.0 through 1.0).
    quality_estimate: float = 0.0
    #: Immutable diagnostic info for logs / JSON report.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.quality_estimate, bool)
            or not isinstance(self.quality_estimate, (int, float))
            or not math.isfinite(self.quality_estimate)
            or not 0.0 <= self.quality_estimate <= 1.0
        ):
            raise ValueError("quality_estimate must be a finite number in [0, 1]")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


# ---------------------------------------------------------------------------
# Quality result — structured verification output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityResult:
    """Aggregated verification metrics for one visible-clean run.

    Built from independent verifiers (outside-mask, boundary, halo) so the
    JSON report can expose each signal separately.
    """

    outside_mask_preserved: bool
    outside_mask_difference_count: int = 0
    boundary_score: float = 0.0  # continuity estimate, 0 = perfect
    halo_score: float = 0.0  # residual edge strength, 0 = clean
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.outside_mask_preserved, bool):
            raise TypeError("outside_mask_preserved must be a bool")
        if (
            isinstance(self.outside_mask_difference_count, bool)
            or not isinstance(self.outside_mask_difference_count, int)
            or self.outside_mask_difference_count < 0
        ):
            raise ValueError("outside_mask_difference_count must be a non-negative integer")
        for name, value in (
            ("boundary_score", self.boundary_score),
            ("halo_score", self.halo_score),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outside_mask_preserved": self.outside_mask_preserved,
            "outside_mask_difference_count": self.outside_mask_difference_count,
            "boundary_score": self.boundary_score,
            "halo_score": self.halo_score,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Verification status mapping — replaces ad-hoc status strings
# ---------------------------------------------------------------------------


def map_quality_to_status(
    backend_failed: bool = False,
    backend_unavailable: bool = False,
    outside_mask_modified: bool = False,
    quality_uncertain: bool = False,
) -> str:
    """Map verification signals to an OperationStatus-like string.

    Ordering (best → worst):
        VERIFIED > BEST_EFFORT > RESIDUAL_RISK > UNSUPPORTED > FAILED
    """
    if backend_failed:
        return "FAILED"
    if backend_unavailable:
        return "UNSUPPORTED"
    if outside_mask_modified:
        return "FAILED"
    if quality_uncertain:
        return "BEST_EFFORT"
    return "VERIFIED"
