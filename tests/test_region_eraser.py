"""Tests for the universal region eraser.

Vectors adapted from upstream remove-ai-watermarks ``tests/test_region_eraser.py``
(boxes→mask, padded crop bounds, per-backend fills, depth/channel carry); the
learned-tier fake sessions and the end-to-end fixture runs through the
ticket-02 ``image_io`` layer are ours.  Seam: the region-eraser public API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from image_io import imwrite, load_image_bgr
from region_eraser import FILL_BACKENDS, Backend, boxes_to_mask, erase


class TestPaddedCropBox:
    """The padded bounding box that bounds the learned backends' ONNX working set."""

    def test_empty_mask_returns_none(self):
        from region_eraser import _padded_crop_box

        assert (
            _padded_crop_box(np.zeros((100, 100), np.uint8), 100, 100, pad_frac=0.1, pad_min=8)
            is None
        )

    def test_pad_min_dominates_and_clamps_at_border(self):
        from region_eraser import _padded_crop_box

        mask = np.zeros((100, 100), np.uint8)
        mask[0:5, 0:5] = 255  # 5-px mark in the top-left corner
        # pad = max(8, int(0.1*5)) = 8; x0 clamps to 0 (not -8), x1 = min(100, 4+1+8) = 13.
        assert _padded_crop_box(mask, 100, 100, pad_frac=0.1, pad_min=8) == (0, 0, 13, 13)

    def test_pad_frac_dominates_for_large_mark(self):
        from region_eraser import _padded_crop_box

        mask = np.zeros((400, 400), np.uint8)
        mask[100:300, 100:300] = 255  # 200-px span
        # pad = max(8, int(0.2*200)) = 40; box = (100-40, .., 299+1+40, ..).
        assert _padded_crop_box(mask, 400, 400, pad_frac=0.2, pad_min=8) == (60, 60, 340, 340)

    def test_clamps_at_far_border(self):
        from region_eraser import _padded_crop_box

        mask = np.zeros((50, 60), np.uint8)
        mask[45:50, 55:60] = 255  # bottom-right corner
        _x0, _y0, x1, y1 = _padded_crop_box(mask, 50, 60, pad_frac=0.1, pad_min=8)
        assert x1 == 60  # clamped to w, no overflow
        assert y1 == 50  # clamped to h, no overflow


class TestBoxesToMask:
    def test_mask_set_inside_box(self):
        mask = boxes_to_mask((100, 100), [(10, 20, 30, 40)], dilate=0)
        assert mask[25, 15] == 255  # inside
        assert mask[0, 0] == 0  # outside
        assert mask.shape == (100, 100)

    def test_multiple_boxes(self):
        mask = boxes_to_mask((100, 100), [(0, 0, 10, 10), (90, 90, 10, 10)], dilate=0)
        assert mask[5, 5] == 255
        assert mask[95, 95] == 255
        assert mask[50, 50] == 0

    def test_dilate_grows_mask(self):
        m0 = boxes_to_mask((100, 100), [(40, 40, 10, 10)], dilate=0)
        m5 = boxes_to_mask((100, 100), [(40, 40, 10, 10)], dilate=5)
        assert m5.sum() > m0.sum()

    def test_box_clipped_to_bounds(self):
        # box partly outside the image must not raise and stays in-bounds
        mask = boxes_to_mask((50, 50), [(40, 40, 100, 100)], dilate=0)
        assert mask[45, 45] == 255


class TestEraseCv2:
    def _image_with_logo(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        img = np.full((200, 200, 3), 120, np.uint8)  # flat gray background
        box = (140, 160, 50, 30)
        x, y, w, h = box
        img[y : y + h, x : x + w] = (255, 255, 255)  # bright "logo"
        return img, box

    def test_erase_changes_region(self):
        img, box = self._image_with_logo()
        out = erase(img, boxes=[box], backend="cv2")
        x, y, w, h = box
        # on a flat background the logo region should be repainted near gray
        region = out[y : y + h, x : x + w]
        assert abs(float(region.mean()) - 120) < 20
        assert not np.array_equal(out, img)

    def test_pixels_outside_box_untouched(self):
        img, box = self._image_with_logo()
        out = erase(img, boxes=[box], backend="cv2", dilate=0)
        # a far corner must be identical
        assert np.array_equal(img[:50, :50], out[:50, :50])

    def test_no_boxes_returns_copy(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        out = erase(img, boxes=[], backend="cv2")
        assert np.array_equal(img, out)

    def test_empty_mask_returns_copy(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        out = erase(img, mask=np.zeros((100, 100), np.uint8), backend="cv2")
        assert np.array_equal(img, out)


class TestNonBgrInputs:
    """cv2.inpaint rejects 4-channel BGRA and 2D-only entry points must work."""

    def test_grayscale_2d_does_not_raise(self):
        gray = np.full((100, 100), 120, np.uint8)
        out = erase(gray, boxes=[(40, 40, 20, 20)], backend="cv2")
        assert out.shape == gray.shape

    def test_bgra_preserves_alpha_and_does_not_raise(self):
        bgra = np.full((100, 100, 4), 120, np.uint8)
        bgra[..., 3] = 200  # opaque-ish alpha plane
        out = erase(bgra, boxes=[(40, 40, 20, 20)], backend="cv2", dilate=0)
        assert out.shape == bgra.shape
        # alpha plane is carried through unchanged
        assert np.array_equal(out[..., 3], bgra[..., 3])


class TestBackendTable:
    """The fill-backend names are stated in several places; they must agree.

    Deliberately separate LITERALS (not derived through an import) so the
    registry and the CLI can state their choices without importing this
    cv2-loading module at their import time.
    """

    def test_executable_backends_are_the_table_minus_auto(self):
        import typing

        assert set(typing.get_args(Backend)) == set(FILL_BACKENDS) - {"auto"}

    def test_auto_is_a_caller_facing_choice(self):
        assert "auto" in FILL_BACKENDS

    def test_unknown_backend_degrades_to_cv2_instead_of_raising(self):
        """``erase`` is public: a library caller passing ``"auto"`` (or a typo) must get
        the classical fill, not a KeyError."""
        img = np.full((64, 64, 3), 100, np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[20:40, 20:40] = 255
        assert erase(img, mask=mask, backend="auto").shape == img.shape  # type: ignore[arg-type]


class TestSixteenBitInputs:
    """A 16-bit source must survive the fill, whatever the backend can hold.

    cv2.inpaint takes 16-bit only as a single channel (its colour path is 8-bit),
    so a uint16 colour image fills through an 8-bit copy and every pixel OUTSIDE
    the mask stays bit-exact at the source's depth -- feeding a 16-bit colour
    image to cv2 used to raise a bare OpenCV "Unsupported format" error.
    """

    @staticmethod
    def _gradient16() -> np.ndarray:
        column = np.linspace(0, 65535, 96, dtype=np.uint16)
        return np.repeat(np.tile(column, (96, 1))[:, :, None], 3, axis=2).copy()

    def test_cv2_fills_a_16_bit_image_without_raising(self):
        img = self._gradient16()
        out = erase(img, boxes=[(20, 20, 30, 30)], backend="cv2", dilate=3)
        assert out.dtype == np.uint16
        assert out.shape == img.shape

    def test_cv2_leaves_everything_outside_the_mask_bit_exact(self):
        img = self._gradient16()
        mask = boxes_to_mask(img.shape[:2], [(20, 20, 30, 30)], dilate=3)
        out = erase(img, mask=mask, backend="cv2")
        keep = mask <= 127
        assert np.array_equal(out[keep], img[keep])
        assert not np.array_equal(out[mask > 127], img[mask > 127])

    def test_the_fill_lands_in_the_source_s_range_not_the_8_bit_one(self):
        # The narrow-then-widen round trip must scale back up: leaving the 8-bit
        # levels in a uint16 array would paint the mask near-black.
        img = np.full((96, 96, 3), 40000, np.uint16)
        mask = boxes_to_mask(img.shape[:2], [(30, 30, 20, 20)], dilate=3)
        filled = erase(img, mask=mask, backend="cv2")[mask > 127]
        assert filled.min() > 30000

    def test_cv2_preserves_a_16_bit_alpha_plane(self):
        bgra = np.full((96, 96, 4), 40000, np.uint16)
        bgra[..., 3] = np.arange(96 * 96, dtype=np.uint16).reshape(96, 96) * 7

        out = erase(bgra, boxes=[(30, 30, 20, 20)], backend="cv2", dilate=3)

        assert out.dtype == np.uint16
        assert np.array_equal(out[..., 3], bgra[..., 3])

    def test_a_float_image_says_what_is_wrong(self):
        img = np.full((64, 64, 3), 0.5, np.float32)
        with pytest.raises(RuntimeError, match="integer images"):
            erase(img, boxes=[(10, 10, 20, 20)], backend="cv2")


class TestRegionFillEndToEnd:
    """cv2-telea and cv2-ns region fills work end-to-end on a fixture file.

    The full IO path runs through the ticket-02 image_io layer: fixture written
    with imwrite, loaded with load_image_bgr, erased, written back, reloaded.
    """

    @pytest.mark.parametrize("method", ["telea", "ns"])
    def test_fixture_file_region_fill(self, tmp_path: Path, method: str) -> None:
        src = tmp_path / "fixture.png"
        background = np.full((120, 160, 3), 90, np.uint8)
        background[100:, :] = (60, 130, 60)  # ground band for local structure
        background[10:60, 100:150] = (200, 200, 200)  # bright "watermark" block
        assert imwrite(src, background)

        img = load_image_bgr(src)
        assert img is not None
        out = erase(img, boxes=[(100, 10, 50, 50)], backend="cv2", cv2_method=method, dilate=2)

        dest = tmp_path / f"cleaned-{method}.png"
        assert imwrite(dest, out)
        reloaded = load_image_bgr(dest)
        assert reloaded is not None
        # far corner (>= dilate away from the box) is preserved exactly
        assert np.array_equal(reloaded[:5, :5], background[:5, :5])
        # the watermark block was repainted towards its surroundings, not left bright
        region = reloaded[12:58, 102:148]
        assert float(region.mean()) < 150
