"""Tests for connected-component filtering, closing, and feather_blend in morphomod."""

from __future__ import annotations

import pytest
from morphomod import Mask, Raster, closing, feather_blend, filter_components


def _make_mask(w: int = 20, h: int = 20, *, fill: int = 0) -> Mask:
    return Mask(width=w, height=h, data=bytearray([fill] * w * h))


def _make_raster(w: int = 20, h: int = 20, ch: int = 3, fill: int = 128) -> Raster:
    return Raster(width=w, height=h, channels=ch, data=bytearray([fill] * w * h * ch))


# ---------------------------------------------------------------------------
# filter_components
# ---------------------------------------------------------------------------


class TestFilterComponents:
    def test_removes_small_components(self) -> None:
        """Components smaller than min_size should be removed."""
        mask = _make_mask(20, 20)
        # Component 1: 5x5 square at top-left (25 pixels)
        for y in range(5):
            for x in range(5):
                mask.data[y * 20 + x] = 255
        # Component 2: 3x3 square at bottom-right (9 pixels)
        for y in range(17, 20):
            for x in range(17, 20):
                mask.data[y * 20 + x] = 255

        filtered = filter_components(mask, min_size=10)
        # Only the 5x5 component should remain
        marked = sum(1 for v in filtered.data if v != 0)
        assert marked == 25

    def test_removes_all_below_threshold(self) -> None:
        mask = _make_mask(10, 10)
        # Two small components
        mask.data[0] = 255  # 1 pixel
        mask.data[99] = 255  # 1 pixel
        filtered = filter_components(mask, min_size=10)
        assert filtered.marked == 0

    def test_keeps_all_above_threshold(self) -> None:
        mask = _make_mask(20, 20)
        # Large component: 10x10 square (100 pixels)
        for y in range(10):
            for x in range(10):
                mask.data[y * 20 + x] = 255
        filtered = filter_components(mask, min_size=10)
        assert filtered.marked == 100

    def test_max_components_limit(self) -> None:
        mask = _make_mask(20, 20)
        # Create 6 components: 3 large (>=10) and 3 small (<10)
        # Large component 1: 4x4 = 16 pixels at (0,0)
        for dy in range(4):
            for dx in range(4):
                mask.data[dy * 20 + dx] = 255
        # Large component 2: 4x4 = 16 pixels at (0,8)
        for dy in range(4):
            for dx in range(4):
                mask.data[dy * 20 + 8 + dx] = 255
        # Large component 3: 4x4 = 16 pixels at (8,0)
        for dy in range(4):
            for dx in range(4):
                mask.data[(8 + dy) * 20 + dx] = 255
        # Small component 1: 2x2 = 4 pixels
        mask.data[12 * 20 + 12] = 255
        mask.data[12 * 20 + 13] = 255
        mask.data[13 * 20 + 12] = 255
        mask.data[13 * 20 + 13] = 255
        # Small component 2: 2 pixels
        mask.data[15 * 20 + 15] = 255
        mask.data[15 * 20 + 16] = 255

        # Keep only 2 components
        filtered = filter_components(mask, min_size=5, max_components=2)
        marked = sum(1 for v in filtered.data if v != 0)
        # Exactly 2 components should remain (16 + 16 = 32)
        assert marked == 32

    def test_does_not_modify_original(self) -> None:
        mask = _make_mask(10, 10)
        mask.data[0] = 255
        filtered = filter_components(mask, min_size=10)
        assert mask.data[0] == 255  # original unchanged
        assert filtered.data[0] == 0  # filtered is empty

    def test_empty_mask(self) -> None:
        mask = _make_mask(10, 10)
        filtered = filter_components(mask, min_size=10)
        assert filtered.marked == 0


# ---------------------------------------------------------------------------
# closing
# ---------------------------------------------------------------------------


class TestClosing:
    def test_fills_narrow_gap(self) -> None:
        mask = _make_mask(10, 10)
        # Left block
        for y in range(5):
            for x in range(3):
                mask.data[y * 10 + x] = 255
        # Right block
        for y in range(5):
            for x in range(7, 10):
                mask.data[y * 10 + x] = 255
        # Gap in middle (columns 3-6)
        result = closing(mask, radius=2)
        # The gap should be bridged
        gap_filled = sum(1 for y in range(5) for x in range(3, 7) if result.data[y * 10 + x] != 0)
        assert gap_filled > 0

    def test_no_change_no_gap(self) -> None:
        mask = _make_mask(10, 10)
        # Create a solid 8x8 block in the center
        for y in range(1, 9):
            for x in range(1, 9):
                mask.data[y * 10 + x] = 255
        result = closing(mask, radius=1)
        # Closing with radius=1 can only expand outward by 1 pixel, not shrink
        # Check that no pixels within the original region were removed
        for y in range(1, 9):
            for x in range(1, 9):
                assert result.data[y * 10 + x] == 255, f"pixel ({x},{y}) was removed"

    def test_rectangular_mask_uses_real_height(self) -> None:
        mask = _make_mask(12, 5)
        for y in range(1, 4):
            for x in range(3, 9):
                mask.data[y * mask.width + x] = 255

        result = closing(mask, radius=1)

        assert (result.width, result.height) == (12, 5)
        assert len(result.data) == 60

    def test_preserves_corner_foreground(self) -> None:
        mask = _make_mask(5, 5)
        mask.data[0] = 255

        result = closing(mask, radius=1)

        assert result.data[0] == 255

    def test_preserves_edge_foreground(self) -> None:
        mask = _make_mask(7, 5)
        for y in range(1, 4):
            mask.data[y * mask.width] = 255

        result = closing(mask, radius=1)

        for y in range(1, 4):
            assert result.data[y * mask.width] == 255


# ---------------------------------------------------------------------------
# feather_blend
# ---------------------------------------------------------------------------


class TestFeatherBlend:
    def test_no_feather_hard_composites_masked_pixels(self) -> None:
        src = _make_raster(w=10, h=10, ch=3, fill=100)
        inpainted = _make_raster(w=10, h=10, ch=3, fill=200)
        mask = _make_mask(w=10, h=10)
        mask.data[5 * 10 + 5] = 255

        result = feather_blend(src, inpainted, mask, feather_radius=0)

        assert bytes(result.data[:3]) == bytes(src.data[:3])
        center = (5 * 10 + 5) * 3
        assert bytes(result.data[center : center + 3]) == b"\xc8\xc8\xc8"

    def test_feather_blends(self) -> None:
        src = _make_raster(w=10, h=10, ch=3, fill=100)
        inpainted = _make_raster(w=10, h=10, ch=3, fill=200)
        mask = _make_mask(w=10, h=10, fill=255)
        result = feather_blend(src, inpainted, mask, feather_radius=3)
        # Result should be between 100 and 200 due to blending
        # Near the boundary, values should differ from both originals
        # After smoothstep, edge pixels get blended
        assert result.width == 10
        assert result.height == 10
        assert result.channels == 3

    def test_dimension_mismatch_raises(self) -> None:
        src = _make_raster(w=10, h=10, ch=3, fill=100)
        inpainted = _make_raster(w=10, h=10, ch=3, fill=200)
        mask = _make_mask(w=5, h=5, fill=255)
        with pytest.raises(ValueError):
            feather_blend(src, inpainted, mask, feather_radius=3)

    def test_channel_mismatch_raises(self) -> None:
        src = _make_raster(w=10, h=10, ch=3, fill=100)
        inpainted = _make_raster(w=10, h=10, ch=4, fill=200)
        mask = _make_mask(w=10, h=10, fill=255)
        with pytest.raises(ValueError, match="channel count"):
            feather_blend(src, inpainted, mask, feather_radius=0)
