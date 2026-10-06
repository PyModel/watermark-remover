"""Tests for the 124-d forensic feature extractor (``forensic_124d``).

Ported from wiltodelta/remove-ai-watermarks tests/test_forensic_124d.py; the
adapted vectors keep the upstream shape/determinism/abstain pins and add the
adaptation this port requires: the OpenCV dependency is lazy and fail-soft, so
an environment without cv2 abstains (``None``) instead of raising — an absent
dependency is unknown evidence, never a zero vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _noise(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size[0], size[1], 3), dtype=np.uint8)


def test_a_full_patch_returns_124_floats() -> None:
    from forensic_124d import FEATURE_WIDTH, PATCH, image_features

    vector = image_features(_noise((PATCH, PATCH)))

    assert vector is not None
    assert vector.shape == (FEATURE_WIDTH,)
    assert FEATURE_WIDTH == 124
    assert vector.dtype == np.float64


def test_a_larger_image_still_returns_124_floats() -> None:
    from forensic_124d import FEATURE_WIDTH, image_features

    vector = image_features(_noise((512, 640), seed=1))

    assert vector is not None
    assert vector.shape == (FEATURE_WIDTH,)


def test_the_same_pixels_yield_the_same_vector() -> None:
    """Determinism is the pin that makes vectors comparable to the frozen
    provider heads: identical pixels, identical 124 floats, byte for byte."""
    from forensic_124d import image_features

    pixels = _noise((300, 300), seed=2)

    first, second = image_features(pixels), image_features(pixels)

    assert first is not None and second is not None
    assert np.array_equal(first, second)


def test_different_pixels_yield_a_different_vector() -> None:
    from forensic_124d import image_features

    one = image_features(_noise((256, 256), seed=3))
    two = image_features(_noise((256, 256), seed=4))

    assert one is not None and two is not None
    assert not np.array_equal(one, two)


def test_an_undersized_image_returns_none() -> None:
    from forensic_124d import PATCH, image_features

    assert image_features(np.zeros((PATCH - 1, PATCH, 3), dtype=np.uint8)) is None


def test_non_rgb_returns_none() -> None:
    from forensic_124d import image_features

    assert image_features(np.zeros((256, 256), dtype=np.uint8)) is None


def test_a_flat_image_returns_none() -> None:
    """A zero-energy plane has no spectrum to describe: the honest answer is
    abstention, not a vector of zeros."""
    from forensic_124d import image_features

    assert image_features(np.zeros((256, 256, 3), dtype=np.uint8)) is None


def test_patch_cap_bounds_the_compute() -> None:
    """At most MAX_PATCHES 256px patches are ever measured, so a huge image
    costs bounded work (12 patches x 3 channels), not work proportional to
    the pixel count."""
    from forensic_124d import MAX_PATCHES, _patch_origins, image_features

    height, width = 2048, 4096
    origins = _patch_origins(height, width)

    assert len(origins) == MAX_PATCHES == 12
    assert all(0 <= row <= height - 256 and 0 <= col <= width - 256 for row, col in origins)
    assert image_features(_noise((height, width), seed=5)) is not None


def test_missing_cv2_abstains_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fail-soft adaptation: without OpenCV the extractor cannot blur, and
    an absent dependency is unknown evidence -- ``is_available`` says so and
    ``image_features`` abstains to ``None``, rather than raising ImportError or
    fabricating numbers."""
    import forensic_124d
    import optional_deps

    real_module_available = optional_deps.module_available

    def without_cv2(*names: str) -> bool:
        """The real probe, except cv2 now fails to import."""
        if "cv2" in names:
            return False
        return real_module_available(*names)

    monkeypatch.setattr(optional_deps, "module_available", without_cv2)

    assert forensic_124d.is_available() is False
    assert forensic_124d.image_features(_noise((256, 256), seed=6)) is None

    # sanity: the gate is the only change -- with the probe restored, the same
    # pixels produce a vector again.
    monkeypatch.setattr(optional_deps, "module_available", real_module_available)
    assert forensic_124d.image_features(_noise((256, 256), seed=6)) is not None
