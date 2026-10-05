"""Vendor engine wave 1 tests — the calibrated CJK set (ticket 08).

The upstream vectors (test_jimeng_engine.py, test_baidu_engine.py, test_yuanbao_engine.py,
test_doubao_engine.py, test_kling_engine.py, test_qwen_engine.py,
test_generic_ai_label_engine.py, test_pill_engine.py) are adapted to this tree's
constraints: fixtures compose each engine's own renderer-synthetic silhouette at the
engine's measured geometry (the upstream capture-solved assets are forbidden here), and
removal exercises the shared fill directly through ``region_eraser`` — the registry that
wraps these engines is a later ticket's seam.

What is pinned per engine: detect on a synthetic fixture, the correct corner, NCC within
its calibrated gate, quiet verdicts on non-matching images, the bounded/degenerate-input
contract, and the provenance posture (strict-only engines never relax; the generic label
never relaxes at all).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from baidu_engine import BaiduEngine
from jimeng_engine import _ALPHA_WIDTH_FRAC as _JIMENG_ALPHA_WFRAC
from jimeng_engine import DETECT_NCC_THRESHOLD as _JIMENG_GATE
from jimeng_engine import JimengEngine
from jimeng_engine import _alpha_template as _jimeng_alpha
from jimeng_engine import _glyph_silhouette as _jimeng_silhouette
from jimeng_engine import _template_match_score as _jimeng_score
from numpy.typing import NDArray
from region_eraser import erase as erase_mask
from yuanbao_engine import YuanbaoEngine

ASSETS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts" / "assets"


def _asset(name: str) -> NDArray[np.float32]:
    """A shipped silhouette as float [0,1], or assert (every engine's asset ships)."""
    img = cv2.imread(str(ASSETS / name), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"asset missing: {name}"
    return img.astype(np.float32) / 255.0


def _compose_mark(
    asset_name: str,
    w: int,
    h: int,
    *,
    width_frac: float,
    aspect: float,
    bg: float | tuple[float, float, float] = 100.0,
    foreground: float = 255.0,
    alpha: float = 1.0,
    basis: str = "short",
    margin_right: float = 0.015,
    margin_bottom: float = 0.015,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """Composite a vendor silhouette at its measured geometry onto a flat background.

    Mirrors every upstream engine test's forward model: ``out = a*fg + (1-a)*bg``.
    ``basis`` sizes the mark off the short side (the engine's scale basis for all
    bottom-right marks here) or the width; the mark anchors into the bottom-right
    corner with the given margins. Returns ``(image, (ax, ay, gw, gh))``.
    """
    img = np.empty((h, w, 3), np.float32)
    img[:] = bg
    at = _asset(asset_name)
    base = min(w, h) if basis == "short" else w
    gw = max(4, int(width_frac * base))
    gh = max(4, int(gw * aspect))
    ax = w - int(margin_right * base) - gw
    ay = h - int(margin_bottom * base) - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh)) * alpha
    a3 = amap[:, :, None]
    wm = (a3 * foreground + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, (ax, ay, gw, gh)


def _fill(engine: Any, img: NDArray[np.uint8], **kwargs: Any) -> NDArray[np.uint8]:
    """The shared fill: engine footprint mask -> region_eraser cv2 backend."""
    mask = engine.footprint_mask(img, **kwargs)
    assert mask is not None
    out = erase_mask(img, mask=mask, backend="cv2")
    assert out.shape == img.shape
    return out


class TestJimengEngine:
    """Vectors adapted from upstream test_jimeng_engine.py.

    Deviation from upstream: the capture-solved Jimeng wordmark silhouette is
    forbidden in this tree, so the engine carries the shipped synthetic
    ``jimeng_label_alpha.png`` as its detection silhouette (asset name, alpha
    geometry, and this compose helper all follow the shipped asset); the calibrated
    corner, gates, and provenance factor are upstream's.
    """

    def test_locate_anchored_bottom_right(self):
        eng = JimengEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) < int(2048 * 0.03)
        assert 2048 - (loc.y + loc.h) < int(2048 * 0.03)

    def test_locate_scales_with_width(self):
        eng = JimengEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)

    def test_clean_gradient_not_detected(self):
        eng = JimengEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = JimengEngine()
        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads_binary(self):
        sil = _jimeng_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_match_score_shape_sensitive(self):
        """The glyph silhouette correlates with itself, not with a filled block."""
        sil = _jimeng_silhouette()
        h, w = sil.shape
        box = np.zeros((h + 8, w * 3), np.uint8)
        box[4 : 4 + h, 4 : 4 + w] = sil
        # scale_base that puts the detection template at the silhouette's own size
        # (the substituted asset's nominal width), so "self" really is self.
        nominal_base = int(w / _JIMENG_ALPHA_WFRAC)
        assert _jimeng_score(box, nominal_base) >= _JIMENG_GATE
        solid = np.full_like(box, 255)
        assert _jimeng_score(solid, nominal_base) < _JIMENG_GATE

    def test_alpha_template_loads(self):
        at = _jimeng_alpha()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0 and float(at.max()) <= 1.0

    def test_synthetic_mark_detected_at_calibrated_gate(self):
        eng = JimengEngine()
        wm, _ = _compose_mark(
            "jimeng_label_alpha.png", 2048, 2048, width_frac=0.2021, aspect=128 / 353
        )
        det = eng.detect(wm)
        assert det.detected
        assert det.confidence >= _JIMENG_GATE

    def test_small_image_guarded_from_false_positive(self):
        """Below the minimum short side a tiny shape can spuriously NCC-match a CJK
        silhouette; the size guard suppresses detection. Bracket it: detected native,
        suppressed downscaled."""
        eng = JimengEngine()
        wm, _ = _compose_mark(
            "jimeng_label_alpha.png", 2048, 2048, width_frac=0.2021, aspect=128 / 353
        )
        assert eng.detect(wm).detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected

    def test_footprint_mask_in_bottom_right(self):
        eng = JimengEngine()
        wm, _ = _compose_mark(
            "jimeng_label_alpha.png", 2048, 2048, width_frac=0.2021, aspect=128 / 353
        )
        mask = eng.footprint_mask(wm)
        assert mask is not None
        assert mask.shape == wm.shape[:2]
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2
        assert xs.mean() > wm.shape[1] / 2

    def test_fill_clears_and_leaves_far_region_exact(self):
        """localize -> shared cv2 fill: re-detect goes quiet, the far quadrant is
        byte-identical (the fill never leaks outside the mask)."""
        eng = JimengEngine()
        wm, _ = _compose_mark(
            "jimeng_label_alpha.png", 1440, 2560, width_frac=0.2021, aspect=128 / 353
        )
        before = eng.detect(wm)
        assert before.detected
        out = _fill(eng, wm, detection=before)
        assert eng.detect(out).confidence < before.confidence
        h, w = wm.shape[:2]
        assert np.array_equal(out[: h // 2, : w // 2], wm[: h // 2, : w // 2])

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w: int, h: int):
        eng = JimengEngine()
        img = np.zeros((h, w, 3), np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is None or mask.shape == (h, w)

    def test_grayscale_2d_does_not_raise(self):
        eng = JimengEngine()
        gray = np.zeros((2048, 2048), np.uint8)
        mask = eng.footprint_mask(gray, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    def test_bgra_4channel_does_not_raise(self):
        eng = JimengEngine()
        bgra = np.zeros((2048, 2048, 4), np.uint8)
        mask = eng.footprint_mask(bgra, force=True)
        assert mask is None or mask.shape == (2048, 2048)

    def test_rival_margin_carries_the_doubao_template(self):
        """The two ByteDance marks share the corner and a script; Jimeng's detector is
        competitive against the shipped Doubao outline silhouette (upstream rival,
        adapted to this tree's synthetic asset name)."""
        assert "doubao_outline_alpha.png" in JimengEngine().config.rivals


class TestBaiduEngine:
    """Vectors adapted from upstream test_baidu_engine.py."""

    def _compose(self, w: int = 1024, h: int = 1024, bg: float = 100.0):
        """The 百度 text run at its measured layout plus the solid white tag between
        the text and the corner (upstream's compose, at this tree's asset)."""
        short = min(w, h)
        wm, (ax, ay, gw, gh) = _compose_mark(
            "baidu_alpha.png",
            w,
            h,
            width_frac=0.090,
            aspect=187 / 366,
            bg=bg,
            margin_right=0.099,
            margin_bottom=0.006,
        )
        # the white rounded tag between the text run and the corner
        tx0 = w - int(0.015 * short) - int(0.075 * short)
        wm[ay - gh // 8 : ay + gh + gh // 8, tx0 : w - int(0.015 * short)] = 255
        return wm, (ax, ay, gw, gh, tx0)

    def test_locate_anchored_bottom_right(self):
        eng = BaiduEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        assert 2048 - (loc.x + loc.w) < 40
        assert 2048 - (loc.y + loc.h) < 40

    def test_locate_scales_with_short_side_not_width(self):
        eng = BaiduEngine()
        landscape = eng.locate(np.zeros((640, 1280, 3), np.uint8))
        wider = eng.locate(np.zeros((640, 2560, 3), np.uint8))
        assert wider.w == landscape.w

    def test_config_pins(self):
        eng = BaiduEngine()
        assert eng.config.detect_frontend == "tophat"
        assert eng.config.scale_basis == "short"
        assert eng.config.provenance_ncc_factor == 1.0  # strict only, unmeasured band
        assert eng.config.detect_ncc_threshold >= 0.48
        assert "doubao_outline_alpha.png" in eng.config.rivals
        assert "qwen_alpha.png" in eng.config.rivals

    def test_detects_composed_mark(self):
        eng = BaiduEngine()
        wm, _ = self._compose()
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_clean_frame_stays_quiet(self):
        eng = BaiduEngine()
        assert not eng.detect(np.full((1024, 1024, 3), 100, np.uint8)).detected

    def test_mask_extends_to_the_corner_tag(self):
        """Regression (upstream 2026-07-22): the tag's flat white interior gives no
        top-hat response, so a blob-bbox mask ends at the text run and the fill
        leaves the tag as a ghost. The footprint must extend right to the corner."""
        eng = BaiduEngine()
        wm, (ax, _ay, gw, _gh, tx0) = self._compose()
        mask = eng.footprint_mask(wm)
        assert mask is not None
        _ys, xs = np.where(mask > 0)
        assert xs.min() <= ax + int(0.1 * gw)
        assert xs.max() >= tx0 + 10

    def test_no_mask_on_clean_frame(self):
        eng = BaiduEngine()
        assert eng.footprint_mask(np.full((1024, 1024, 3), 100, np.uint8)) is None

    def test_force_masks_the_whole_locate_box_on_a_clean_frame(self):
        """``force`` takes priority over detection for this mark, unlike the base
        policy: a --no-detect caller named the mark, so the honest footprint is the
        whole geometry box."""
        eng = BaiduEngine()
        img = np.full((1024, 1024, 3), 100, np.uint8)
        mask = eng.footprint_mask(img, force=True)
        assert mask is not None
        bx, by, bw, bh = eng.locate(img).bbox
        ys, xs = np.where(mask > 0)
        assert xs.min() <= bx
        assert xs.max() >= bx + bw - 1
        assert ys.min() <= by
        assert ys.max() >= by + bh - 1

    def test_fill_clears_and_leaves_far_region_exact(self):
        eng = BaiduEngine()
        wm, _ = self._compose(1024, 1024)
        before = eng.detect(wm)
        assert before.detected
        out = _fill(eng, wm, detection=before)
        assert not eng.detect(out).detected
        h, w = wm.shape[:2]
        assert np.array_equal(out[: h // 2, : w // 2], wm[: h // 2, : w // 2])


class TestYuanbaoEngine:
    """Vectors adapted from upstream test_yuanbao_engine.py."""

    def _compose(self, w: int, h: int, *, bg: float, foreground: float):
        return _compose_mark(
            "yuanbao_alpha.png",
            w,
            h,
            width_frac=0.08,
            aspect=0.0446 / 0.08,
            bg=bg,
            foreground=foreground,
            margin_right=0.028,
            margin_bottom=0.031,
        )

    def test_config_pins(self):
        eng = YuanbaoEngine()
        assert eng.config.detect_frontend == "contrast"  # two-polarity mark
        assert eng.config.provenance_ncc_factor == 1.0  # strict only

    def test_detects_light_mark_on_dark_background(self):
        watermark, _ = self._compose(1024, 1024, bg=60, foreground=230)
        detection = YuanbaoEngine().detect(watermark)
        assert detection.detected
        assert detection.confidence >= 0.80

    def test_detects_dark_mark_on_light_background(self):
        """Yuanbao switches mark polarity with the background; a white top-hat alone
        misses the dark-gray stamp used on pale scenes."""
        watermark, _ = self._compose(1024, 1024, bg=230, foreground=110)
        detection = YuanbaoEngine().detect(watermark)
        assert detection.detected
        assert detection.confidence >= 0.80

    def test_clean_gradient_stays_quiet(self):
        ramp = np.tile(np.linspace(40, 220, 1024, dtype=np.uint8), (1024, 1))
        image = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not YuanbaoEngine().detect(image).detected

    def test_match_must_hug_bottom_right_anchor(self):
        """Shift the mark 60 px in from the corner: still inside the locate window, so
        the sweep finds it, but beyond the 0.04 anchor margin, so only the anchor gate
        can demote it. A far shift would fall outside the window and pass regardless."""

        class Ungated(YuanbaoEngine):
            def _post_gate(self, det, scan):  # type: ignore[no-untyped-def]
                return det

        watermark, (ax, ay, gw, gh) = self._compose(1024, 1024, bg=60, foreground=230)
        assert YuanbaoEngine().detect(watermark).detected
        shifted = np.full_like(watermark, 60)
        shifted[ay - 60 : ay - 60 + gh, ax - 60 : ax - 60 + gw] = watermark[
            ay : ay + gh, ax : ax + gw
        ]
        assert Ungated().detect(shifted).detected
        assert not YuanbaoEngine().detect(shifted).detected

    def test_mask_uses_detector_box_for_dark_mark(self):
        watermark, (ax, ay, gw, gh) = self._compose(1024, 1024, bg=230, foreground=110)
        mask = YuanbaoEngine().footprint_mask(watermark)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert xs.min() <= ax + int(0.05 * gw)
        assert xs.max() >= ax + gw - int(0.05 * gw)
        assert ys.min() <= ay + gh // 2 <= ys.max()

    def test_fill_clears_detector(self):
        watermark, _ = self._compose(1024, 1024, bg=60, foreground=230)
        eng = YuanbaoEngine()
        out = _fill(eng, watermark)
        assert not eng.detect(out).detected
