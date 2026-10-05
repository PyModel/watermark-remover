"""Shared text-mark engine tests (ticket 07).

The upstream vectors (test_text_mark_engine.py, test_text_mark_memory.py,
test_text_mark_faint_mask.py) are policy tests pinned to per-vendor engine
modules; no vendor engine ships in this ticket, so every vector is adapted to
synthetic :class:`TextMarkConfig` instances over the in-tree renderer-synthetic
silhouette assets. Vendor-specific tuning stays in (future) vendor configs;
these tests pin the shared engine's behavior: locate -> NCC score -> footprint
mask -> shared region-eraser fill -> validate.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

import _text_mark_engine as tme
from _text_mark_engine import (
    TextMarkConfig,
    TextMarkEngine,
    glyph_silhouette,
    load_alpha_template,
)

ASSET = "kling_alpha.png"  # in-tree renderer-synthetic CJK strip (347x83)


def _config(**overrides: Any) -> TextMarkConfig:
    """A synthetic bottom-right strip mark with the calibrated br-strip defaults."""
    fields: dict[str, Any] = {
        "name": "SyntheticMark",
        "asset_name": ASSET,
        "corner": "br",
        "margin_floor": 4,
        "width_frac": 0.22,
        "height_frac": 0.075,
        "margin_x_frac": 0.004,
        "margin_bottom_frac": 0.004,
        "max_saturation": 55,
        "logo_min_luma": 150,
        "tophat_delta": 12,
        "morph_open_size": 5,
        "detect_min_coverage": 0.04,
        "detect_ncc_threshold": 0.40,
        "alpha_width_frac": 0.1636,
        "alpha_height_frac": 0.0405,
        "min_gw": 8,
    }
    fields.update(overrides)
    return TextMarkConfig(**fields)


def _engine(**overrides: Any) -> TextMarkEngine:
    return TextMarkEngine(_config(**overrides))


def _flat(w: int = 2048, h: int = 1536, level: int = 100) -> NDArray[np.uint8]:
    return np.full((h, w, 3), level, np.uint8)


def _stamp(
    engine: TextMarkEngine,
    image: NDArray[np.uint8],
    *,
    alpha: float = 0.6,
    textured: bool = False,
    scale: float = 1.0,
) -> NDArray[np.uint8]:
    """Composite the engine's own glyph silhouette onto ``image`` (white overlay).

    Same forward model the marks use: ``stamped = (1-a)*bg + a*white``. ``scale``
    resizes the stamped glyph off the nominal size; ``textured`` adds fine
    luminance noise so the top-hat responds outside the glyph too.
    """
    tmpl = load_alpha_template(engine.config.asset_name)
    assert tmpl is not None
    img = image.copy()
    if textured:
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 14, img.shape[:2] + (1,)).repeat(3, axis=2)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    loc = engine.locate(img)
    base = engine.scale_base(img)
    gw = max(engine.config.min_gw, int(engine.config.alpha_width_frac * base * scale))
    gh = max(4, int(engine.config.alpha_height_frac * base * scale))
    if gw >= loc.w or gh >= loc.h:
        pytest.fail("fixture glyph does not fit the locate box")
    a = cv2.resize(tmpl, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) * alpha
    x = loc.x + (loc.w - gw) // 2
    y = loc.y + (loc.h - gh) // 2
    roi = img[y : y + gh, x : x + gw].astype(np.float32)
    a3 = a[..., None]
    img[y : y + gh, x : x + gw] = np.clip(roi * (1 - a3) + 255.0 * a3, 0, 255).astype(np.uint8)
    return img


class TestLocateAndScale:
    """locate() anchors the box from the config, scaled by the configured basis."""

    def test_locate_anchors_the_box_bottom_right(self):
        eng = _engine()
        img = _flat()
        loc = eng.locate(img)
        base = eng.scale_base(img)
        assert loc.w == max(40, int(base * 0.22))
        assert loc.h == max(16, int(base * 0.075))
        # bottom-right: the box hugs the corner with only the configured margin
        assert loc.x + loc.w <= 2048
        assert loc.y + loc.h <= 1536
        assert 2048 - (loc.x + loc.w) <= max(4, int(base * 0.004)) + 1
        assert 1536 - (loc.y + loc.h) <= max(4, int(base * 0.004)) + 1

    def test_scale_basis_is_config_driven(self):
        portrait = np.zeros((1600, 900, 3), np.uint8)
        landscape = np.zeros((900, 1600, 3), np.uint8)
        short = _engine(scale_basis="short")
        wide = _engine(scale_basis="width")
        tall = _engine(scale_basis="long")
        assert short.scale_base(portrait) == short.scale_base(landscape) == 900
        assert wide.scale_base(landscape) == 1600
        assert tall.scale_base(portrait) == 1600

    def test_landscape_box_stays_inside_the_frame(self):
        """The upstream landscape overshoot failure: a box scaled on the wrong
        basis must clamp to the frame instead of running off the image."""
        eng = _engine(scale_basis="short")
        landscape = np.zeros((900, 2400, 3), np.uint8)
        loc = eng.locate(landscape)
        assert loc.w < int(2400 * eng.config.width_frac)
        assert loc.x >= 0 and loc.y >= 0
        assert loc.x + loc.w <= 2400
        assert loc.y + loc.h <= 900

    def test_degenerate_frame_locates_without_raising(self):
        eng = _engine()
        loc = eng.locate(np.zeros((10, 10, 3), np.uint8))
        assert loc.x == 0 and loc.y == 0
        assert loc.w == 10 and loc.h == 10


class TestTemplates:
    """Asset templates load from the in-tree synthetic silhouettes, cached."""

    def test_load_alpha_template_reads_an_in_tree_asset(self):
        at = load_alpha_template(ASSET)
        assert at is not None
        assert at.dtype == np.float32
        assert 0.0 <= float(at.min()) and float(at.max()) <= 1.0
        assert float(at.max()) > 0.9  # real glyph pixels exist

    def test_glyph_silhouette_is_binary(self):
        sil = glyph_silhouette(ASSET)
        assert sil is not None
        assert sil.dtype == np.uint8
        assert set(np.unique(sil)).issubset({0, 255})
        assert (sil == 255).mean() > 0.05

    def test_unknown_asset_is_rejected_stably(self):
        assert load_alpha_template("no_such_asset.png") is None
        assert glyph_silhouette("no_such_asset.png") is None
        box = np.zeros((50, 50), np.uint8)
        missing = _config(asset_name="no_such_asset.png")
        assert tme.template_match_score(box, 200, missing) == 0.0
        # A missing silhouette scores 0.0 and reports no match box -- a stable
        # rejection, never a raise.
        assert tme._template_match_best(box, 200, missing) == (0.0, None)

    def test_flat_box_matches_score_zero_inside_the_box(self):
        """A constant box yields the 0.0 floor with the template's ROI-local box,
        clamped inside the searched box (bounded matchTemplate work)."""
        box = np.zeros((50, 50), np.uint8)
        score, match = tme._template_match_best(box, 200, _config())
        assert score == 0.0
        assert match is not None
        x0, y0, x1, y1 = match
        assert 0 <= x0 <= x1 < 50 and 0 <= y0 <= y1 < 50

    def test_alpha_cache_caches_successful_loads(self):
        first = load_alpha_template(ASSET)
        again = load_alpha_template(ASSET)
        assert first is again  # identical object: served from the cache


class TestExtractMask:
    """extract_mask() returns the box-sized glyph mask (O(footprint) memory)."""

    def test_returns_box_sized_mask(self):
        eng = _engine()
        img = _stamp(eng, _flat())
        loc = eng.locate(img)
        box = eng.extract_mask(img, loc)
        assert box.dtype == np.uint8
        # Shape == loc.bbox, i.e. a full-frame mask cropped to the box would give
        # the same bytes -- but no (h, w) allocation ever happens (memory guard).
        assert box.shape == (loc.h, loc.w)
        assert box.size * 4 < img.shape[0] * img.shape[1]

    def test_finds_the_stamped_glyph(self):
        eng = _engine()
        img = _stamp(eng, _flat())
        loc = eng.locate(img)
        box = eng.extract_mask(img, loc)
        # 20 = the engine's minimum-glyph floor (a real strip covers hundreds).
        assert (box > 0).sum() >= 20
        ys, xs = np.where(box > 0)
        # the glyph sits mid-box (the fixture centers it), not at the edges
        assert xs.min() > 0 and xs.max() < loc.w - 1

    def test_clean_box_yields_no_glyph(self):
        eng = _engine()
        img = _flat()
        loc = eng.locate(img)
        assert (eng.extract_mask(img, loc) > 0).sum() == 0

    def test_degenerate_roi_returns_a_zero_mask(self):
        eng = _engine()
        img = _flat()
        loc = tme.TextMarkLocation(x=0, y=0, w=8, h=8)
        box = eng.extract_mask(img, loc)
        assert box.shape == (8, 8)
        assert int(box.sum()) == 0


class TestTemplateMatch:
    """template_match_score() = zero-mean NCC of the silhouette vs the blob."""

    def test_scores_the_stamped_glyph_high(self):
        eng = _engine()
        img = _stamp(eng, _flat())
        loc = eng.locate(img)
        box = eng.extract_mask(img, loc)
        score = tme.template_match_score(box, eng.scale_base(img), eng.config)
        assert score > 0.5

    def test_flat_box_scores_at_or_below_zero(self):
        eng = _engine()
        img = _flat()
        loc = eng.locate(img)
        box = eng.extract_mask(img, loc)
        assert tme.template_match_score(box, eng.scale_base(img), eng.config) <= 0.0

    def test_template_is_clamped_to_the_box(self):
        """A scaled template never exceeds the box it matches into (bounded
        matchTemplate allocation on untrusted image geometry): the match box stays
        inside the box and a flat box still scores the 0.0 floor."""
        eng = _engine(alpha_width_frac=9.9, alpha_height_frac=9.9)
        box = np.zeros((40, 200), np.uint8)
        score, match = tme._template_match_best(box, 1000, eng.config)
        assert score == 0.0
        assert match is not None
        assert match[2] < 200 and match[3] < 40
