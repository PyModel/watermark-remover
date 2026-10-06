# Tests for the E4 spectral internals (image_math.py, source_spectral.py) and
# the E2 source-export classifier (source_classify.py), plus the identify
# signal adapters and the extra-gated identify collector. No network: the HF
# download path is monkeypatched and weights are synthetic in-repo npz files
# built to the modules' own schemas.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# E4a: image_math.dct_matrix — the pure orthonormal DCT-II basis factory
# ---------------------------------------------------------------------------


def test_dct_matrix_size_two_is_the_known_hadamard_like_basis() -> None:
    """A hand-computed literal: the 2-point orthonormal DCT-II basis."""
    import image_math
    import numpy as np

    matrix = image_math.dct_matrix(np, 2)
    inverse_root_two = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            [inverse_root_two, inverse_root_two],
            [inverse_root_two, inverse_root_two * -1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_dct_matrix_is_orthonormal_for_the_classifier_geometry() -> None:
    """M @ M.T is the identity at the 8-point and 16-point sizes."""
    import image_math
    import numpy as np

    for size in (8, 16):
        matrix = image_math.dct_matrix(np, size)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(size), atol=1e-12)


def test_dct_matrix_diagonalizes_a_constant_signal_into_dc_only() -> None:
    """A constant input has energy only in the DC (first) coefficient."""
    import image_math
    import numpy as np

    matrix = image_math.dct_matrix(np, 8)
    coefficients = matrix @ np.full(8, 3.0)
    np.testing.assert_allclose(coefficients[1:], 0.0, atol=1e-12)
    assert abs(coefficients[0]) > 0.0


# ---------------------------------------------------------------------------
# E4b: source_spectral — phase-free log-spectrum features
# ---------------------------------------------------------------------------


def test_make_geometry_covers_every_pixel_and_validates_arguments() -> None:

    import source_spectral

    geometry = source_spectral.make_geometry(size=256, grid=16)
    assert geometry is not None
    assert geometry.size == 256 and geometry.grid == 16
    assert int(geometry.radius_counts.sum()) == 256 * 256
    assert geometry.passband.any()
    with pytest.raises(ValueError):
        source_spectral.make_geometry(size=255, grid=16)
    with pytest.raises(ValueError):
        source_spectral.make_geometry(size=8, grid=2)
    with pytest.raises(ValueError):
        source_spectral.make_geometry(size=256, grid=1)


def test_phase_free_log_spectrum_is_invariant_to_sign_and_centered() -> None:
    import numpy as np
    import source_spectral

    rng = np.random.default_rng(20260922)
    channel = rng.standard_normal((32, 32))
    spectrum = source_spectral.phase_free_log_spectrum(channel)
    assert spectrum.shape == (32, 32)
    # Phase-free: negating the input flips every FFT phase, not any magnitude.
    np.testing.assert_allclose(
        source_spectral.phase_free_log_spectrum(-channel), spectrum, atol=1e-12
    )
    # A constant channel centers to zero, so the log magnitude is exactly zero.
    np.testing.assert_allclose(
        source_spectral.phase_free_log_spectrum(np.full((32, 32), 5.0)), 0.0, atol=1e-12
    )
    with pytest.raises(ValueError):
        source_spectral.phase_free_log_spectrum(np.zeros((2, 2, 2)))


def test_spectral_feature_shape_and_constant_image_zero_vector() -> None:
    import numpy as np
    import source_spectral

    geometry = source_spectral.make_geometry(size=64, grid=16)
    assert geometry is not None
    constant = np.full((64, 64, 3), 0.5)
    features = source_spectral.spectral_feature(constant, geometry)
    assert features.shape == (3 * 16 * 16,)
    np.testing.assert_allclose(features, 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        source_spectral.spectral_feature(np.zeros((8, 8, 3)), geometry)


def test_feature_from_image_is_deterministic_for_a_fixed_synthetic() -> None:
    import numpy as np
    import source_spectral
    from PIL import Image

    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    geometry = source_spectral.make_geometry()
    assert geometry is not None
    first = source_spectral.feature_from_image(image, geometry)
    assert first is not None and first.shape == (768,)
    second = source_spectral.feature_from_image(image, geometry)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_spectral_stack_abstains_when_numpy_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dep-absence is abstention (None), never zeros and never a raise."""
    import source_spectral
    from PIL import Image

    # make_geometry is process-wide cached; drop entries from earlier tests so
    # the absent-stack path is actually exercised, and again on the way out.
    source_spectral.make_geometry.cache_clear()
    try:
        monkeypatch.setitem(sys.modules, "numpy", None)
        assert not source_spectral.is_available()
        assert source_spectral.make_geometry() is None
        image = Image.new("RGB", (64, 64))
        # No geometry (and no numeric stack) means no feature: abstain, never zeros.
        assert source_spectral.feature_from_image(image, None) is None
    finally:
        source_spectral.make_geometry.cache_clear()
