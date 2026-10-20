# Ported from wiltodelta/remove-ai-watermarks
# src/remove_ai_watermarks/_internal/source_spectral.py; modified: relocated to
# this repo's flat scripts directory under the claimed source_spectral.py path
# (the upstream module lives in its _internal package), and the numpy/Pillow
# imports are lazy and fail-soft instead of hard module-level requirements — an
# environment without the numeric stack abstains (``None`` /
# ``is_available() == False``) rather than raising at import or call time,
# because an absent dependency is unknown evidence, never a zero vector, and
# feature_from_image normalizes its input to RGB before the canonical resize
# (upstream assumes the caller already did). The feature math itself is
# verbatim.
"""Phase-free spectral features for source/export-pipeline classification.

Evidence-only input for the abstaining source classifiers: the representation
deliberately discards FFT phase, so it cannot reconstruct or decode any
watermark payload — it measures radial residual texture structure only.
Requires the ``pixel``-tier numeric stack (numpy + Pillow); guard with
:func:`is_available`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image


@dataclass(frozen=True)
class SpectralGeometry:
    """Precomputed coordinates for one phase-free spectral representation."""

    size: int
    grid: int
    radius: NDArray[Any]
    radius_counts: NDArray[Any]
    passband: NDArray[Any]


def is_available() -> bool:
    """True when the numeric stack the features need is importable."""
    from optional_deps import module_available

    return module_available("numpy", "PIL")


def _lazy(name: str) -> Any:
    """Import a numeric dependency or abstain: missing deps are unknown, not zeros."""
    from optional_deps import module_available

    if not module_available(name):
        return None
    import importlib

    return importlib.import_module(name)


@lru_cache(maxsize=4)
def make_geometry(size: int = 256, grid: int = 16) -> SpectralGeometry | None:
    """Build radial detrending and pooling geometry (``None`` without numpy)."""
    np = _lazy("numpy")
    if np is None:
        return None
    if size < 16 or grid < 2 or size % grid:
        raise ValueError("size must be at least 16 and divisible by grid")
    yy, xx = np.mgrid[:size, :size]
    radius = np.rint(np.hypot(yy - size // 2, xx - size // 2)).astype(np.int32)
    radius_counts = np.bincount(radius.ravel())
    passband = (radius >= 4) & (radius <= size * 0.48)
    return SpectralGeometry(size, grid, radius, radius_counts, passband)


def phase_free_log_spectrum(channel: NDArray[Any]) -> NDArray[Any] | None:
    """Return centered log FFT magnitude, deliberately discarding phase.

    ``None`` when numpy is absent; :class:`ValueError` for a non-2D channel.
    """
    np = _lazy("numpy")
    if np is None:
        return None
    if channel.ndim != 2:
        raise ValueError("channel must be a two-dimensional array")
    centered = channel - float(np.mean(channel))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))))


def spectral_feature(
    pixels: NDArray[Any], geometry: SpectralGeometry | None
) -> NDArray[Any] | None:
    """Extract pooled radial-residual spectra from luma and opponent color.

    ``None`` when the numeric stack or the geometry is unavailable.
    """
    np = _lazy("numpy")
    if np is None or geometry is None:
        return None
    expected = (geometry.size, geometry.size, 3)
    if pixels.shape != expected:
        raise ValueError(f"pixels must have shape {expected}, got {pixels.shape}")
    red, green, blue = np.moveaxis(pixels, 2, 0)
    channels = (
        0.299 * red + 0.587 * green + 0.114 * blue,
        red - green,
        blue - (red + green) / 2.0,
    )
    block = geometry.size // geometry.grid
    features: list[NDArray[Any]] = []
    for channel in channels:
        spectrum = phase_free_log_spectrum(channel)
        if spectrum is None:
            return None
        radial_sum = np.bincount(geometry.radius.ravel(), weights=spectrum.ravel())
        radial_mean = radial_sum / geometry.radius_counts
        residual = np.where(
            geometry.passband,
            spectrum - radial_mean[geometry.radius],
            0.0,
        )
        pooled = residual.reshape(geometry.grid, block, geometry.grid, block).mean(axis=(1, 3))
        features.append(pooled.ravel())
    return np.concatenate(features)


def feature_from_image(
    image: Image.Image, geometry: SpectralGeometry | None
) -> NDArray[Any] | None:
    """Resize an RGB image canonically and extract its phase-free feature.

    The canonical 256px (``geometry.size``) LANCZOS resize is the fixed compute
    bound; ``None`` when the numeric stack or the geometry is unavailable.
    """
    np = _lazy("numpy")
    if np is None or geometry is None:
        return None
    from PIL import Image as _image_module

    canonical = image.convert("RGB").resize(
        (geometry.size, geometry.size), _image_module.Resampling.LANCZOS
    )
    pixels = np.asarray(canonical, dtype=np.float64) / 255.0
    return spectral_feature(pixels, geometry)
