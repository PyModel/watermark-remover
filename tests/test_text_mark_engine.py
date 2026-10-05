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


class TestDetect:
    """detect() gates on NCC + (for the binary front-end) the coverage gate."""

    def test_detects_the_stamped_mark(self):
        eng = _engine()
        img = _stamp(eng, _flat())
        det = eng.detect(img)
        assert det.detected
        assert det.confidence > 0.5
        assert det.region == eng.locate(img).bbox
        assert det.coverage > eng.config.detect_min_coverage
        assert det.match_box is not None
        assert det.template_asset == ASSET
        assert det.provenance is False

    def test_rejects_the_clean_frame(self):
        eng = _engine()
        det = eng.detect(_flat())
        assert not det.detected
        assert det.confidence == 0.0  # below the coverage gate: no score consulted

    def test_provenance_relaxes_only_the_threshold(self):
        """The relaxed gate is threshold * provenance_ncc_factor; the scan is shared.

        A partially-erased stamp scores mid-range: pick the gate pair around the
        measured score so strict rejects and provenance accepts, with the SAME
        confidence either way (the trust level scales the threshold, nothing else).
        """
        eng = _engine()
        img = _stamp(eng, _flat())
        strict = eng.detect(img, provenance=False)
        relaxed = eng.detect(img, provenance=True)
        assert strict.confidence == relaxed.confidence > 0.5
        assert strict.detected and relaxed.detected
        # Now a config gated strictly above the fixture's score, with a relaxation
        # factor that brings the gate back under it.
        score = strict.confidence
        tight = _engine(detect_ncc_threshold=score + 0.05, provenance_ncc_factor=0.5)
        assert not tight.detect(img, provenance=False).detected
        assert tight.detect(img, provenance=True).detected

    def test_small_images_are_skipped_stably(self):
        """Below the 200px short-side floor, detection skips instead of correlating
        an 8px template (the upstream icon/thumbnail false-positive band)."""
        eng = _engine()
        img = _stamp(eng, _flat(w=160, h=120))
        det = eng.detect(img)
        assert not det.detected
        assert det.region == (0, 0, 0, 0)
        assert det.confidence == 0.0
        assert det.match_box is None

    def test_empty_and_none_inputs_reject_stably(self):
        eng = _engine()
        for bad in (None, np.zeros((0, 0, 3), np.uint8)):
            det = eng.detect(bad)
            assert not det.detected
            assert det.region == (0, 0, 0, 0)
        assert eng.footprint_mask(None) is None
        assert eng.footprint_mask(np.zeros((0, 0, 3), np.uint8)) is None

    def test_sliver_frames_reject_stably(self):
        """h < 32 or w < 64 cannot carry the mark; footprint must say None, not
        feed cv2 a degenerate Mat."""
        eng = _engine()
        assert eng.footprint_mask(_flat(w=200, h=16)) is None
        assert eng.footprint_mask(_flat(w=48, h=400)) is None


class TestRivalMargin:
    """Same-corner detection is competitive: the rival's template scores the same blob."""

    def test_a_rival_shaped_blob_loses_the_margin(self):
        """The decisive case: a blob matching the RIVAL at least as well as this mark
        must not be booked as this mark, however high the absolute score. The compact
        yuanbao pair is byte-identical (per assets/PROVENANCE.md), so both templates
        score ~1.0 and the margin always rejects."""
        engine_asset, rival_asset = "yuanbao_compact_alpha.png", "yuanbao_compact_gray_alpha.png"
        rival_cfg = _config(
            name="Rival",
            asset_name=rival_asset,
            alpha_width_frac=0.1636,
            alpha_height_frac=0.1636 * 222 / 329,  # the compact stamp's own aspect
        )
        tme.register_rival_config(rival_asset, rival_cfg)
        try:
            eng = _engine(
                asset_name=engine_asset,
                rivals=(rival_asset,),
                rival_margin=0.10,
                alpha_height_frac=0.1636 * 222 / 329,
            )
            blob = glyph_silhouette(engine_asset)
            assert blob is not None
            canvas = np.zeros((blob.shape[0] + 20, blob.shape[1] + 20), np.uint8)
            canvas[10 : 10 + blob.shape[0], 10 : 10 + blob.shape[1]] = blob
            width = int(blob.shape[1] / eng.config.alpha_width_frac)
            score = eng._template_match_score(canvas, width)
            assert score > 0.5  # a strong ABSOLUTE match...
            assert not eng._rival_margin_ok(score, canvas, width)  # ...that loses the margin
        finally:
            tme._RIVAL_CONFIGS.pop(rival_asset, None)

    def test_no_rivals_declared_means_unaffected(self):
        eng = _engine(rivals=())
        assert eng._rival_margin_ok(0.42, np.zeros((10, 10), np.uint8), 100)

    def test_unresolvable_rival_degrades_to_accept(self):
        """A rival asset with no registered config and no module is skipped, not a
        raise (a missing/renamed engine must not break detection)."""
        eng = _engine(rivals=("no_such_asset.png",))
        assert eng._rival_margin_ok(0.42, np.zeros((10, 10), np.uint8), 100)

    def test_registered_rival_beats_the_module_table(self):
        """Registration wins over the lazy module lookup, so a vendor engine can pin
        its exact config without this module knowing vendor layout."""
        rival_asset = "kling_latin_alpha.png"
        rival_cfg = _config(name="Pinned", asset_name=rival_asset)
        tme.register_rival_config(rival_asset, rival_cfg)
        try:
            assert tme._rival_config(rival_asset) is rival_cfg
        finally:
            tme._RIVAL_CONFIGS.pop(rival_asset, None)


class TestContinuousFrontends:
    """The tophat/gray/contrast front-ends sweep the ladder on the continuous response."""

    def test_tophat_response_is_contrast_invariant(self):
        """A faint mark and a bold one produce the same max-normalized response."""
        eng = _engine(detect_frontend="tophat")
        img = _flat(w=2048, h=1536)
        loc = eng.locate(img)
        out = []
        for amplitude in (12, 90):  # a barely-there overlay and a bold one
            probe = img.copy()
            x, y, bw, bh = loc.bbox
            probe[y + bh // 3 : y + 2 * bh // 3, x + bw // 4 : x + 3 * bw // 4] = 100 + amplitude
            resp = eng.tophat_response(probe, loc)
            assert resp is not None
            out.append(resp)
        assert abs(int(out[0].max()) - int(out[1].max())) <= 1

    def test_flat_input_yields_no_response(self):
        eng = _engine(detect_frontend="tophat")
        img = _flat()
        assert eng.tophat_response(img, eng.locate(img)) is None

    def test_binary_frontend_never_builds_a_ladder_response(self):
        eng = _engine(detect_frontend="binary")
        with pytest.raises(ValueError, match="no ladder response"):
            eng._detect_response(_flat(), eng.locate(_flat()))

    def test_faint_mark_found_only_by_the_continuous_frontend(self):
        """The upstream RunningHub story, config-driven: a faint stamp shatters under
        the binary coverage gate but scores through the continuous tophat response."""
        faint_alpha = 0.08
        binary = _engine()
        tophat = _engine(detect_frontend="tophat", detect_ncc_threshold=0.50)
        img = _stamp(binary, _flat(), alpha=faint_alpha)
        loc = binary.locate(img)
        # Premise: the binarizing path cannot segment it.
        assert (binary.extract_mask(img, loc) > 0).sum() < 20
        assert not binary.detect(img).detected
        # The continuous front-end still finds it.
        det = tophat.detect(img)
        assert det.detected
        assert det.match_box is not None

    def test_contrast_frontend_sees_the_dark_on_light_polarity(self):
        """A dark glyph on a bright corner has no WHITE top-hat; the absolute
        residual front-end must still find it (config-driven renderer polarity)."""
        bright_bg = _flat(level=220)
        tophat = _engine(detect_frontend="tophat", logo_min_luma=0)
        contrast = _engine(detect_frontend="contrast", detect_ncc_threshold=0.50)
        dark = _stamp_dark(tophat, bright_bg, level=60)
        assert not tophat.detect(dark).detected
        assert contrast.detect(dark).detected

    def test_gray_frontend_is_polarity_dependent_where_contrast_is_not(self):
        """The gray front-end correlates the RAW grayscale, so its sign tracks the
        mark's polarity: it accepts the light-on-dark stamp and rejects the dark-on-
        light inversion of the same silhouette. The contrast front-end (absolute
        residual) accepts both. Upstream's per-mark gray-arm precision numbers were
        capture-corpus measurements; the port pins the mechanism, and per-mark
        calibration belongs to the vendor tickets."""
        gray = _engine(detect_frontend="gray", detect_ncc_threshold=0.40)
        contrast = _engine(detect_frontend="contrast", detect_ncc_threshold=0.50)
        light_mark = _stamp(gray, _flat(), alpha=0.6)  # bright glyph on dark bg
        dark_mark = _stamp_dark(gray, _flat(level=220), level=60)  # inverted polarity
        assert gray.detect(light_mark).detected
        assert not gray.detect(dark_mark).detected
        assert contrast.detect(light_mark).detected
        assert contrast.detect(dark_mark).detected

    def test_ladder_sweep_finds_the_off_nominal_scale(self):
        """The ladder is config-driven: a stamp rendered 25% off the nominal size is
        found by the default 3-rung comb and missed by a single 1.0 rung."""
        img = _stamp(_engine(detect_frontend="tophat"), _flat(), alpha=0.6, scale=1.25)
        swept = _engine(detect_frontend="tophat", detect_ncc_threshold=0.50)
        pinned = _engine(detect_frontend="tophat", detect_ncc_threshold=0.50, ladder=(1.0,))
        assert swept.detect(img).detected
        assert not pinned.detect(img).detected


def _stamp_dark(engine: TextMarkEngine, image: NDArray[np.uint8], *, level: int) -> NDArray[np.uint8]:
    """Composite the engine's glyph as a DARK overlay (light-on-dark inverted)."""
    tmpl = load_alpha_template(engine.config.asset_name)
    assert tmpl is not None
    img = image.copy()
    loc = engine.locate(img)
    base = engine.scale_base(img)
    gw = max(engine.config.min_gw, int(engine.config.alpha_width_frac * base))
    gh = max(4, int(engine.config.alpha_height_frac * base))
    a = cv2.resize(tmpl, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = loc.x + (loc.w - gw) // 2
    y = loc.y + (loc.h - gh) // 2
    roi = img[y : y + gh, x : x + gw].astype(np.float32)
    a3 = a[..., None]
    img[y : y + gh, x : x + gw] = np.clip(roi * (1 - a3) + level * a3, 0, 255).astype(np.uint8)
    return img
