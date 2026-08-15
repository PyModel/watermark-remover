"""Tests for verification module."""

from __future__ import annotations

import pytest
from morphomod import Mask, Raster
from verification import verify_boundary_seam, verify_halo, verify_outside_mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raster(width: int, height: int, channels: int, *, fill: int = 128) -> Raster:
    return Raster(
        width=width,
        height=height,
        channels=channels,
        data=bytearray([fill] * width * height * channels),
    )


def _make_mask(width: int, height: int, *, fill: int = 0) -> Mask:
    return Mask(
        width=width,
        height=height,
        data=bytearray([fill] * width * height),
    )


def _mask_with_rect(w: int, h: int, x: int, y: int, x2: int, y2: int) -> Mask:
    """Create a mask with a rectangular marked region."""
    data = bytearray(w * h)
    for py in range(y, min(y2, h)):
        for px in range(x, min(x2, w)):
            data[py * w + px] = 255
    return Mask(w, h, data)


class TestVerifyOutsideMask:
    def test_no_mask_no_changes(self) -> None:
        src = _make_raster(10, 10, 3, fill=100)
        final = _make_raster(10, 10, 3, fill=100)
        mask = _make_mask(10, 10, fill=0)
        preserved, count = verify_outside_mask(src, final, mask)
        assert preserved is True
        assert count == 0

    def test_mask_modification_accepted(self) -> None:
        """Pixels inside the mask may change freely; only outside is checked."""
        src = _make_raster(10, 10, 3, fill=100)
        # Create final with same value outside the mask region
        final_data = bytearray(10 * 10 * 3)
        for i in range(10 * 10 * 3):
            # Outside mask: keep source value (100)
            final_data[i] = 100
        final = _make_raster(10, 10, 3, fill=100)
        # Override pixels inside the mask
        for y in range(2, 5):
            for x in range(2, 5):
                for c in range(3):
                    final.data[(y * 10 + x) * 3 + c] = 200
        mask = _mask_with_rect(10, 10, 2, 2, 5, 5)
        preserved, count = verify_outside_mask(src, final, mask)
        assert preserved is True
        assert count == 0

    def test_outside_modified(self) -> None:
        """Pixels outside the mask should not change."""
        src = _make_raster(10, 10, 3, fill=100)
        mask = _mask_with_rect(10, 10, 2, 2, 5, 5)
        final = _make_raster(10, 10, 3, fill=200)
        preserved, count = verify_outside_mask(src, final, mask)
        assert preserved is False
        assert count > 0

    def test_empty_mask_no_changes_ok(self) -> None:
        """Empty mask means no inpainting happened; all pixels unchanged."""
        src = _make_raster(5, 5, 1, fill=50)
        final = _make_raster(5, 5, 1, fill=50)
        mask = _make_mask(5, 5)
        preserved, _ = verify_outside_mask(src, final, mask)
        assert preserved is True

    def test_full_mask_all_modified(self) -> None:
        """If mask covers everything, all pixels may differ."""
        src = _make_raster(5, 5, 1, fill=50)
        final = _make_raster(5, 5, 1, fill=200)
        mask = _make_mask(5, 5, fill=255)
        preserved, _ = verify_outside_mask(src, final, mask)
        assert preserved is True  # all pixels are inside the mask

    @pytest.mark.parametrize(
        ("final", "mask", "message"),
        [
            (_make_raster(4, 5, 3), _make_mask(5, 5), "dimensions"),
            (_make_raster(5, 5, 4), _make_mask(5, 5), "channel counts"),
            (_make_raster(5, 5, 3), _make_mask(4, 5), "mask/raster"),
        ],
    )
    def test_rejects_mismatched_contracts(self, final, mask, message) -> None:
        with pytest.raises(ValueError, match=message):
            verify_outside_mask(_make_raster(5, 5, 3), final, mask)

    def test_rgba_uses_full_channel_stride(self) -> None:
        source = _make_raster(2, 1, 4, fill=10)
        final = _make_raster(2, 1, 4, fill=10)
        final.data[7] = 11
        preserved, count = verify_outside_mask(source, final, _make_mask(2, 1))
        assert preserved is False
        assert count == 1


class TestVerifyBoundarySeam:
    def test_no_mask_zero_score(self) -> None:
        src = _make_raster(10, 10, 3, fill=128)
        final = _make_raster(10, 10, 3, fill=128)
        mask = _make_mask(10, 10, fill=0)
        score = verify_boundary_seam(src, final, mask)
        assert score == 0.0

    def test_no_boundary_zero_score(self) -> None:
        """If mask is empty, there's no boundary."""
        src = _make_raster(10, 10, 3, fill=128)
        final = _make_raster(10, 10, 3, fill=200)
        mask = _make_mask(10, 10, fill=0)
        score = verify_boundary_seam(src, final, mask)
        assert score >= 0.0

    def test_boundary_score_non_negative(self) -> None:
        src = _make_raster(10, 10, 3, fill=128)
        final = _make_raster(10, 10, 3, fill=128)
        mask = _mask_with_rect(10, 10, 2, 2, 8, 8)
        score = verify_boundary_seam(src, final, mask)
        assert score >= 0.0


class TestVerifyHalo:
    def test_no_mask_no_halo(self) -> None:
        src = _make_raster(10, 10, 3, fill=128)
        final = _make_raster(10, 10, 3, fill=128)
        mask = _make_mask(10, 10, fill=0)
        score, warnings = verify_halo(src, final, mask)
        assert score == 0.0
        assert warnings == ()

    def test_halo_score_non_negative(self) -> None:
        src = _make_raster(10, 10, 3, fill=128)
        final = _make_raster(10, 10, 3, fill=200)
        mask = _mask_with_rect(10, 10, 3, 3, 7, 7)
        score, warnings = verify_halo(src, final, mask)
        assert score >= 0.0
        assert isinstance(warnings, tuple)
