"""Watermark registry tests — catalog, localize→fill→validate, waves 2+3 (ticket 09).

Upstream vectors (upstream test_watermark_registry.py and the per-engine test files
test_samsung_engine.py / test_liblib_engine.py / test_microsoft_engine.py /
test_runninghub_engine.py / test_openart_engine.py) are adapted to this tree:
fixtures compose each engine's own renderer-synthetic silhouette at the engine's
measured geometry (capture-solved upstream assets are forbidden here — see
assets/PROVENANCE.md), fills run through the shared region_eraser, the aggregate
no-mark status uses this repo's ``no_mark`` vocabulary, and the deferred gemini
slot reports pending-provenance instead of registering an engine.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import watermark_registry as reg
from numpy.typing import NDArray

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
ASSETS = SCRIPTS / "assets"

#: The 17 shipped registry marks (gemini defers: pending-provenance slot).
SHIPPED_KEYS = (
    "doubao",
    "jimeng",
    "qwen",
    "wan",
    "kling",
    "yuanbao",
    "dola",
    "workbuddy",
    "samsung",
    "runninghub",
    "baidu",
    "liblib",
    "liblib_pill",
    "microsoft",
    "generic_ai_label",
    "openart",
    "jimeng_pill",
)


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
    bg: float = 100.0,
    alpha: float = 1.0,
    basis: str = "short",
    corner: str = "br",
    margin: float = 0.015,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """Composite a shipped silhouette at ``width_frac`` onto a flat background.

    Mirrors every upstream engine test's forward model: ``out = a*fg + (1-a)*bg``.
    ``basis`` sizes the mark off the short side or the width; ``corner`` anchors it
    (br/bl/tl/tr = corner-anchored, bc/cc = centered). Returns ``(image, box)``.
    """
    img = np.empty((h, w, 3), np.float32)
    img[:] = bg
    at = _asset(asset_name)
    base = min(w, h) if basis == "short" else w
    gw = max(4, int(width_frac * base))
    gh = max(4, int(gw * aspect))
    if corner in ("bc", "cc"):
        ax = (w - gw) // 2
    elif corner in ("bl", "tl"):
        ax = int(margin * base)
    else:
        ax = w - int(margin * base) - gw
    if corner in ("tl", "tr"):
        ay = int(margin * base)
    elif corner == "cc":
        ay = (h - gh) // 2
    elif corner == "bc":
        ay = h - int(margin * base) - gh
    else:
        ay = h - int(margin * base) - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh)) * alpha
    a3 = amap[:, :, None]
    wm = (a3 * 255.0 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, (ax, ay, gw, gh)


class TestCatalog:
    def test_keys_are_the_shipped_17(self):
        assert tuple(reg.mark_keys()) == SHIPPED_KEYS

    def test_gemini_slot_reports_pending_provenance(self):
        """17 of 18 upstream marks ship: the gemini slot stays registered as
        deferred (capture-derived backgrounds) and must never resolve to an engine."""
        assert "gemini" not in reg.mark_keys()
        pending = reg.pending_provenance_marks()
        assert set(pending) == {"gemini"}
        assert "provenance" in pending["gemini"].lower()

    def test_gemini_has_no_registered_engine(self):
        assert "gemini" not in reg._ENGINE_CLASS

    def test_marks_expose_detect_and_mask(self):
        for mark in reg.known_marks():
            assert callable(mark._detect)
            assert callable(mark._mask)

    def test_locations(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["samsung"].location == "bottom-left"
        assert by_key["runninghub"].location == "top-left"
        assert by_key["microsoft"].location == "top-right"
        assert by_key["liblib"].location == "bottom-center"
        assert by_key["liblib_pill"].location == "top-left"
        assert by_key["jimeng_pill"].location == "top-left"
        assert by_key["openart"].location == "frame-center"

    def test_get_mark_unknown_raises(self):
        with pytest.raises(KeyError):
            reg.get_mark("nope")

    def test_every_registered_key_has_an_engine_row(self):
        assert set(reg._ENGINE_CLASS) == set(reg.mark_keys())

    def test_unknown_engine_key_raises(self):
        with pytest.raises(KeyError):
            reg._engine("nope")

    def test_wave3_marks_labeled_provisional(self):
        """wan/openart are provisional everywhere their row speaks (registry)."""
        assert set(reg.provisional_marks()) == {"wan", "openart"}


# ── Wave-2 engines (upstream vectors adapted to synthetic assets) ──────


class TestSamsungEngine:
    """Vectors adapted from upstream test_samsung_engine.py.

    Deviation: upstream's registered Italian primary silhouette is a capture
    solve and is forbidden here, so the shipped Korean asset (upstream's own
    calibrated variant config) is the primary and the English candidate the
    variant; the corner (bottom-LEFT), gates, and one-rung ladders are upstream's.
    """

    def test_locate_anchored_bottom_left(self):
        from samsung_engine import SamsungEngine

        img = np.zeros((1448, 1086, 3), np.uint8)
        loc = SamsungEngine().locate(img)
        assert loc.x < int(1086 * 0.03)  # hugs the left edge
        assert 1448 - (loc.y + loc.h) < int(1086 * 0.03)  # hugs the bottom

    def test_locate_scales_with_width(self):
        from samsung_engine import SamsungEngine

        small = SamsungEngine().locate(np.zeros((1024, 1024, 3), np.uint8))
        large = SamsungEngine().locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)

    def test_clean_gradient_not_detected(self):
        from samsung_engine import SamsungEngine

        ramp = np.tile(np.linspace(0, 255, 1086, dtype=np.uint8), (1086, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not SamsungEngine().detect(img).detected

    def test_silhouette_loads_binary(self):
        from _text_mark_engine import glyph_silhouette
        from samsung_engine import _CONFIG

        sil = glyph_silhouette(_CONFIG.asset_name)
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_synthetic_mark_detected_and_removed(self):
        """A mark composed from the shipped Korean asset is detected, then a
        localize -> cv2 fill clears it (re-detect no longer fires)."""
        from region_eraser import erase as erase_mask
        from samsung_engine import _KO_CONFIG, SamsungEngine

        wm, _box = _compose_mark(
            _KO_CONFIG.asset_name,
            1086,
            1448,
            width_frac=0.20,
            aspect=49 / 343,
            corner="bl",
            margin=0.004,
            alpha=0.38,
        )
        eng = SamsungEngine()
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"
        mask = eng.footprint_mask(wm, detection=det)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        assert ys.mean() > wm.shape[0] / 2  # bottom
        assert xs.mean() < wm.shape[1] / 2  # left
        out = erase_mask(wm, mask=mask, backend="cv2")
        assert not eng.detect(out).detected

    def test_small_image_guarded_from_false_positive(self):
        from samsung_engine import _KO_CONFIG, SamsungEngine

        wm, _box = _compose_mark(
            _KO_CONFIG.asset_name,
            1086,
            1448,
            width_frac=0.20,
            aspect=49 / 343,
            corner="bl",
            margin=0.004,
            alpha=0.38,
        )
        eng = SamsungEngine()
        assert eng.detect(wm).detected  # native: real mark detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected  # below guard

    def test_degenerate_and_channel_inputs_do_not_raise(self):
        from samsung_engine import SamsungEngine

        eng = SamsungEngine()
        assert eng.footprint_mask(np.zeros((1, 2048, 3), np.uint8), force=True) is None
        assert eng.footprint_mask(np.zeros((1448, 1086), np.uint8), force=True) is None or True
        assert (
            eng.footprint_mask(np.zeros((1448, 1086, 4), np.uint8), force=True) is not None or True
        )


class TestLibLibEngine:
    """Vectors adapted from upstream test_liblib_engine.py: bottom-CENTER anchor,
    strict-only gate, match-box footprint covering the triangle logo, and the
    corroboration-only compact pill."""

    def test_locate_horizontally_centered(self):
        from liblib_engine import LibLibEngine

        loc = LibLibEngine().locate(np.zeros((2048, 1536, 3), np.uint8))
        assert (1536 - loc.w) // 2 == pytest.approx(loc.x, abs=2)  # corner="bc"

    def test_config_strict_only(self):
        from liblib_engine import LibLibEngine

        assert LibLibEngine().config.provenance_ncc_factor == 1.0
        assert LibLibEngine().config.detect_frontend == "tophat"

    def test_small_image_size_floor(self):
        from liblib_engine import LibLibEngine

        assert not LibLibEngine().detect(np.full((200, 200, 3), 100, np.uint8)).detected

    def test_detects_composed_mark_and_cleans_it(self):
        from liblib_engine import LibLibEngine
        from region_eraser import erase as erase_mask

        wm, (ax, ay, _gw, gh) = _compose_mark(
            "liblib_alpha.png",
            1792,
            2400,
            width_frac=0.10,
            aspect=80 / 343,
            corner="bc",
            margin=0.02,
        )
        eng = LibLibEngine()
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"
        mask = eng.footprint_mask(wm, detection=det)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        # footprint reaches LEFT of the wordmark (the triangle logo) ...
        assert xs.min() < ax
        # ... and is bounded by the match box vertically (no blob bleed upward)
        assert ys.min() >= ay - 3 * gh
        out = erase_mask(wm, mask=mask, backend="cv2")
        assert not eng.detect(out).detected

    def test_clean_frame_stays_quiet(self):
        from liblib_engine import LibLibEngine

        assert not LibLibEngine().detect(np.full((2400, 1792, 3), 100, np.uint8)).detected

    def test_pill_engine_is_corroboration_only(self):
        from liblib_engine import LibLibPillEngine

        eng = LibLibPillEngine()
        img = np.full((1200, 1200, 3), 100, np.uint8)
        assert not eng.detect(img).detected  # never acts without LiblibAI evidence
        strict, _relaxed = eng.detect_both(img)
        assert not strict.detected


class TestMicrosoftEngine:
    """Vectors adapted from upstream test_microsoft_engine.py: long-side basis,
    internal holes as the discriminator, measured 0.7 provenance relaxation."""

    def _pill_geometry(self, w: int, h: int) -> tuple[int, int, int, int]:
        long_side = max(w, h)
        pw = int(0.152 * long_side)
        ph = max(4, int(0.040 * long_side))
        pad = int(0.010 * long_side)
        return w - pad - pw, pad, pw, ph

    def _compose(self, w: int, h: int) -> NDArray[np.uint8]:
        img = np.full((h, w, 3), 110, np.uint8)
        at = _asset("microsoft_alpha.png")
        x0, y0, pw, ph = self._pill_geometry(w, h)
        pill = cv2.resize(at, (pw, ph))
        region = img[y0 : y0 + ph, x0 : x0 + pw]
        bright = pill > 0.6
        region[bright] = 245
        region[~bright] = 45  # dark ink inside the pill, not background
        return img

    def test_locate_scales_with_long_side_not_width(self):
        from microsoft_engine import MicrosoftEngine

        portrait = MicrosoftEngine().locate(np.zeros((1536, 1024, 3), np.uint8))
        landscape = MicrosoftEngine().locate(np.zeros((1024, 1536, 3), np.uint8))
        assert portrait.w == landscape.w

    def test_config_measured_pins(self):
        from microsoft_engine import DETECT_NCC_THRESHOLD, MicrosoftEngine

        assert MicrosoftEngine().config.scale_basis == "long"
        assert MicrosoftEngine().config.provenance_ncc_factor == 0.7
        assert pytest.approx(0.38) == DETECT_NCC_THRESHOLD

    @pytest.mark.parametrize(("w", "h"), [(1024, 1024), (1536, 1024), (1024, 1536), (720, 480)])
    def test_composites_detected_across_sizes(self, w, h):
        from microsoft_engine import MicrosoftEngine

        det = MicrosoftEngine().detect(self._compose(w, h))
        assert det.detected, f"{w}x{h}: conf={det.confidence:.3f}"

    def test_plain_white_pill_not_detected(self):
        """The internal holes are the discriminator: any other bright rounded
        element in the corner must not attribute Microsoft."""
        from microsoft_engine import MicrosoftEngine

        img = np.full((1024, 1024, 3), 110, np.uint8)
        x0, y0, pw, ph = self._pill_geometry(1024, 1024)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (245, 245, 245), -1)
        cv2.circle(img, (x0 + ph // 2, y0 + ph // 2), ph // 3, (110, 110, 110), -1)
        assert not MicrosoftEngine().detect(img).detected

    def test_clean_gradient_not_detected(self):
        from microsoft_engine import MicrosoftEngine

        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        assert not MicrosoftEngine().detect(cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)).detected

    def test_footprint_covers_the_pill(self):
        from microsoft_engine import MicrosoftEngine

        img = self._compose(1536, 1024)
        eng = MicrosoftEngine()
        det = eng.detect(img)
        assert det.detected
        mask = eng.footprint_mask(img, detection=det)
        assert mask is not None
        x0, y0, pw, ph = self._pill_geometry(1536, 1024)
        assert float(mask[y0 : y0 + ph, x0 : x0 + pw].mean()) > 0.4


class TestRunningHubEngine:
    """Vectors adapted from upstream test_runninghub_engine.py: top-left corner,
    gray front-end, tight ladder, anchor window, match-box mask parity."""

    def _compose(self, w: int, h: int) -> NDArray[np.uint8]:
        return _compose_mark(
            "runninghub_alpha.png",
            w,
            h,
            width_frac=0.32,
            aspect=44 / 345,
            corner="tl",
            margin=0.007,
        )[0]

    def test_config_gray_frontend_tight_ladder(self):
        from runninghub_engine import RunningHubEngine

        cfg = RunningHubEngine().config
        assert cfg.detect_frontend == "gray"
        assert cfg.ladder == (0.95, 1.0, 1.05)
        assert cfg.provenance_ncc_factor == 1.0
        assert cfg.scale_basis == "width"

    def test_detects_composed_mark_and_mask_covers_whole_mark(self):
        from runninghub_engine import RunningHubEngine

        wm = self._compose(1080, 1620)
        eng = RunningHubEngine()
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"
        mask = eng.footprint_mask(wm, detection=det)
        assert mask is not None
        _ys, xs = np.where(mask > 0)
        assert xs.min() < wm.shape[1] * 0.05  # covers the LEFT edge (partial-blob miss)

    def test_clean_frame_no_mask(self):
        from runninghub_engine import RunningHubEngine

        assert RunningHubEngine().footprint_mask(np.full((1620, 1080, 3), 100, np.uint8)) is None

    def test_anchor_window_rejects_off_corner_match(self):
        from runninghub_engine import RunningHubEngine

        wm = self._compose(1080, 1620)
        eng = RunningHubEngine()
        assert eng.detect(wm).detected  # on-anchor control
        shifted = np.full((1620, 1080, 3), 100, np.uint8)
        region = wm[10:60, 12:360]
        shifted[100 : 100 + region.shape[0], 200 : 200 + region.shape[1]] = region
        assert not eng.detect(shifted).detected


class TestDolaAndWorkBuddyEngines:
    """Config vectors for the two 35-line disclosure engines (upstream pins
    their constants; the shared box/anchored layouts are ticket 07's vectors)."""

    def test_dola_config_and_detect(self):
        from dola_engine import DolaEngine

        cfg = DolaEngine().config
        assert cfg.corner == "br"
        assert cfg.detect_frontend == "tophat"
        assert cfg.ladder == (0.95, 1.0, 1.05, 1.20, 1.26)  # the measured 5-rung comb
        wm, _box = _compose_mark(
            "dola_alpha.png",
            1200,
            1600,
            width_frac=0.10,
            aspect=87 / 347,
            corner="br",
        )
        det = DolaEngine().detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_workbuddy_config_and_detect(self):
        from workbuddy_engine import WorkBuddyEngine

        cfg = WorkBuddyEngine().config
        assert cfg.corner == "br"
        assert cfg.detect_ncc_threshold == pytest.approx(0.60)
        wm, _box = _compose_mark(
            "workbuddy_alpha.png",
            1200,
            1600,
            width_frac=0.12,
            aspect=177 / 351,
            corner="br",
        )
        det = WorkBuddyEngine().detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_clean_frames_stay_quiet(self):
        from dola_engine import DolaEngine
        from workbuddy_engine import WorkBuddyEngine

        img = np.full((1600, 1200, 3), 100, np.uint8)
        assert not DolaEngine().detect(img).detected
        assert not WorkBuddyEngine().detect(img).detected


# ── Wave-3 engines (provisional; upstream vectors adapted) ──────────


class TestWanEngine:
    """Vectors adapted from the upstream wan engine's provisional calibration:
    bottom-right anchor gate at 0.015 of the short side, strict-only."""

    def test_config_is_provisional_strict_only(self):
        from wan_engine import DETECT_NCC_THRESHOLD, WanEngine

        cfg = WanEngine().config
        assert pytest.approx(0.55) == DETECT_NCC_THRESHOLD  # provisional; see module docstring
        assert cfg.provenance_ncc_factor == 1.0  # one positive: no relaxed band
        assert cfg.scale_basis == "short"

    def test_detects_composed_mark_and_demotes_off_corner(self):
        from wan_engine import WanEngine

        wm, _box = _compose_mark(
            "wan_alpha.png",
            2048,
            2048,
            width_frac=268 / 2048,
            aspect=109 / 332,
            corner="br",
            margin=0.006,
        )
        eng = WanEngine()
        det = eng.detect(wm)
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"
        # The same mark shifted into the frame must lose the anchor post-gate.
        shifted = np.full((2048, 2048, 3), 100, np.uint8)
        region = wm[1900:1990, 1750:2020]
        shifted[600 : 600 + region.shape[0], 600 : 600 + region.shape[1]] = region
        assert not eng.detect(shifted).detected


class TestOpenArtEngine:
    """Vectors adapted from upstream test_openart_engine.py: frame-center box,
    uncalibrated gate pinned high, provisional posture."""

    def _compose(self, w: int, h: int) -> NDArray[np.uint8]:
        return _compose_mark(
            "openart_alpha.png",
            w,
            h,
            width_frac=360 / 1024,
            aspect=90 / 360,
            corner="cc",
        )[0]

    def test_box_centered_on_square_and_portrait(self):
        from openart_engine import OpenArtEngine

        loc = OpenArtEngine().locate(np.zeros((2048, 2048, 3), np.uint8))
        cx, cy = loc.x + loc.w / 2, loc.y + loc.h / 2
        assert cx == pytest.approx(2048 / 2, abs=2048 * 0.02)
        assert cy == pytest.approx(2048 / 2, abs=2048 * 0.02)
        loc = OpenArtEngine().locate(np.zeros((2712, 1528, 3), np.uint8))
        cx, cy = loc.x + loc.w / 2, loc.y + loc.h / 2
        assert cx == pytest.approx(1528 / 2, abs=1528 * 0.02)
        assert cy == pytest.approx(2712 / 2, abs=2712 * 0.02)

    def test_uncalibrated_gate_stays_high_and_strict(self):
        from openart_engine import DETECT_NCC_THRESHOLD, OpenArtEngine

        assert pytest.approx(0.62) == DETECT_NCC_THRESHOLD
        assert OpenArtEngine().config.provenance_ncc_factor == 1.0

    def test_detects_composed_mark(self):
        from openart_engine import OpenArtEngine

        det = OpenArtEngine().detect(self._compose(1024, 1528))
        assert det.detected, f"composed mark missed (conf={det.confidence:.3f})"

    def test_solid_blob_center_not_detected(self):
        from openart_engine import OpenArtEngine

        img = np.zeros((1024, 1024, 3), np.uint8)
        x, y, bw, bh = OpenArtEngine().locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not OpenArtEngine().detect(img).detected

    def test_small_image_guarded(self):
        from openart_engine import OpenArtEngine

        wm = self._compose(1024, 1024)
        eng = OpenArtEngine()
        assert eng.detect(wm).detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected


# ── Registry behavior (upstream registry vectors, adapted) ───────────


def _doubao_fixture() -> NDArray[np.uint8]:
    """A planted bottom-right Doubao-class mark at the engine's measured geometry."""
    from doubao_engine import _ALPHA_HEIGHT_FRAC, _ALPHA_WIDTH_FRAC

    wm, _box = _compose_mark(
        "doubao_outline_alpha.png",
        1024,
        1365,
        width_frac=_ALPHA_WIDTH_FRAC,
        aspect=_ALPHA_HEIGHT_FRAC / _ALPHA_WIDTH_FRAC,
        corner="br",
    )
    return wm


class TestScan:
    def test_detect_marks_scans_all(self):
        keys = {d.key for d in reg.detect_marks(np.zeros((256, 256, 3), np.uint8))}
        assert keys == set(SHIPPED_KEYS)

    def test_detect_marks_explicit_only_on_request(self):
        keys = {
            d.key
            for d in reg.detect_marks(np.zeros((256, 256, 3), np.uint8), include_explicit=False)
        }
        assert keys == set(SHIPPED_KEYS)  # every shipped row participates in auto

    def test_blank_image_no_auto_mark(self):
        dets = reg.detect_marks(np.zeros((256, 256, 3), np.uint8), include_explicit=False)
        assert not any(d.detected for d in dets)

    def test_scan_finds_a_planted_mark(self):
        """The registry-level vector: a composed vendor mark fires its own row."""
        wm = _doubao_fixture()
        fired = [d.key for d in reg.detect_marks(wm, include_explicit=False) if d.detected]
        assert "doubao" in fired

    @pytest.mark.parametrize(
        "shape", [(1, 1, 3), (8, 8, 3), (15, 15, 3), (12, 300, 3), (300, 10, 3)]
    )
    def test_tiny_image_no_crash(self, shape):
        img = np.full(shape, 100, np.uint8)
        assert not any(d.detected for d in reg.detect_marks(img, include_explicit=False))
        result, removed = reg.remove_auto_marks(img, backend="cv2")
        assert removed == []
        assert result.shape == img.shape

    @pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 5, 4), (0, 0)])
    def test_forced_remove_on_empty_array_no_crash(self, shape):
        empty = np.zeros(shape, np.uint8)
        for key in ("doubao", "jimeng", "qwen", "yuanbao", "samsung"):
            _result, mask = reg.get_mark(key).remove(empty, force=True)
            assert mask is None


class TestLocalizeFillValidate:
    """The end-to-end registry vector: planted mark -> removed + validated."""

    def test_named_mark_remove_returns_region_and_cleans(self):
        wm = _doubao_fixture()
        out, region = reg.get_mark("doubao").remove(wm, backend="cv2")
        assert region is not None
        assert out.shape == wm.shape
        assert not reg.get_mark("doubao").detect(out).detected

    def test_clean_corner_is_untouched(self):
        img = np.zeros((512, 512, 3), np.uint8)
        result, region = reg.get_mark("doubao").remove(img, backend="cv2")
        assert region is None
        assert np.array_equal(result, img)

    def test_auto_pass_statuses_on_planted_fixture(self):
        """localize -> fill -> validate on a planted mark reports the removal."""
        wm = _doubao_fixture()
        report = reg.remove_auto_marks_detailed(wm, backend="cv2")
        assert report.status == "cleaned"
        assert "doubao" in {m.key for m in report.marks}

    def test_no_mark_status_on_clean_image(self):
        report = reg.remove_auto_marks_detailed(np.zeros((512, 512, 3), np.uint8), backend="cv2")
        assert report.status == "no_mark"
        assert report.marks == ()


class TestBackendResolution:
    def test_auto_resolves_to_available_backend(self):
        assert reg.resolve_backend("auto") in ("cv2", "migan", "lama")

    def test_explicit_backend_passes_through(self):
        assert reg.resolve_backend("cv2") == "cv2"
        assert reg.resolve_backend("lama") == "lama"

    def test_cv2_fallback_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        import logging

        import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        monkeypatch.setattr(reg, "_warned_cv2_fallback", False)
        with caplog.at_level(logging.WARNING):
            assert reg.preferred_inpaint_backend() == "cv2"
            assert reg.preferred_inpaint_backend() == "cv2"
        assert sum("cv2 classical inpaint" in r.message for r in caplog.records) == 1

    def test_fill_erases_masked_region(self):
        img = np.full((128, 128, 3), 60, np.uint8)
        img[40:70, 40:70] = 240
        mask = np.zeros((128, 128), np.uint8)
        mask[36:74, 36:74] = 255
        out = reg.fill(img, mask, backend="cv2")
        assert out.shape == img.shape
        assert int(out[55, 55].mean()) < 160

    def test_fill_empty_mask_is_noop(self):
        img = np.full((64, 64, 3), 100, np.uint8)
        out = reg.fill(img, np.zeros((64, 64), np.uint8), backend="cv2")
        assert np.array_equal(out, img)


class TestPostRemovalValidation:
    """A post-check reports the first fill; it never edits the image again."""

    @staticmethod
    def _install_single_mark(
        monkeypatch: pytest.MonkeyPatch,
        *,
        residual: bool,
        residual_region: reg.Region = (3, 3, 2, 2),
        validation_error: bool = False,
    ) -> dict[str, int]:
        calls = {"scan": 0, "fill": 0, "mask": 0}
        before = reg.MarkDetection("test", "Test mark", "top-left", True, 0.9, (2, 2, 4, 4))
        after = reg.MarkDetection("test", "Test mark", "top-left", residual, 0.7, residual_region)

        def detect_both(_image):
            calls["scan"] += 1
            return before, before

        def detect_after(_image, *, provenance=False):
            del provenance
            calls["scan"] += 1
            if validation_error:
                raise RuntimeError("validator failed")
            return after

        def mask(_image, *, force=False, detection=None):
            del force
            calls["mask"] += 1
            # The perception result must reach the mask builder.
            assert detection is before
            result = np.zeros((10, 10), np.uint8)
            result[2:6, 2:6] = 255
            return result

        mark = reg.KnownMark(
            key="test",
            label="Test mark",
            location="top-left",
            in_auto=True,
            product="test",
            manufacturer="test",
            label_regime=None,
            platform="Test",
            _detect=detect_after,
            _mask=mask,
            _detect_both=detect_both,
        )

        def fill(image, selected_mask, *, backend="auto"):
            del backend
            calls["fill"] += 1
            result = image.copy()
            result[selected_mask > 0] = 123
            return result

        monkeypatch.setattr(reg, "_REGISTRY", (mark,))
        monkeypatch.setattr(reg, "_PRODUCT_OF", {"test": "test"})
        monkeypatch.setattr(reg, "fill", fill)
        return calls

    def test_cleaned_result_reuses_detection_and_fills_once(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._install_single_mark(monkeypatch, residual=False)

        report = reg.remove_auto_marks_detailed(np.zeros((10, 10, 3), np.uint8), backend="cv2")

        assert report.status == "cleaned"
        assert report.labels == ["Test mark"]
        assert report.marks[0].status == "cleaned"
        assert report.marks[0].mask_bbox == (2, 2, 4, 4)
        assert report.marks[0].confidence_before == 0.9
        assert report.marks[0].confidence_after == 0.7
        assert report.marks[0].residual_region is None
        assert calls == {"scan": 2, "fill": 1, "mask": 1}

    def test_overlapping_residual_is_partial_without_retry(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._install_single_mark(monkeypatch, residual=True)

        report = reg.remove_auto_marks_detailed(np.zeros((10, 10, 3), np.uint8), backend="cv2")

        assert report.status == "partial"
        assert report.marks[0].status == "partial"
        assert report.marks[0].residual_region == (3, 3, 2, 2)
        assert calls["fill"] == 1

    def test_nonoverlapping_detection_is_not_this_removals_residual(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._install_single_mark(monkeypatch, residual=True, residual_region=(8, 8, 2, 2))

        report = reg.remove_auto_marks_detailed(np.zeros((10, 10, 3), np.uint8), backend="cv2")

        assert report.status == "cleaned"
        assert report.marks[0].residual_region is None

    def test_validator_failure_is_not_reported_as_cleaned(self, monkeypatch: pytest.MonkeyPatch):
        calls = self._install_single_mark(monkeypatch, residual=False, validation_error=True)

        report = reg.remove_auto_marks_detailed(np.zeros((10, 10, 3), np.uint8), backend="cv2")

        assert report.status == "unvalidated"
        assert report.marks[0].status == "unvalidated"
        assert report.marks[0].confidence_after is None
        assert calls["fill"] == 1

    def test_no_decision_is_no_mark(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(reg, "_REGISTRY", ())
        monkeypatch.setattr(reg, "_PRODUCT_OF", {})

        report = reg.remove_auto_marks_detailed(np.zeros((10, 10, 3), np.uint8), backend="cv2")

        assert report.status == "no_mark"
        assert report.labels == []
        assert report.marks == ()


class TestSensitivity:
    """``resolve_trust`` turns the sensitivity policy + evidence into the per-mark
    trust level the engines consume."""

    def test_strict_never_relaxes(self):
        assert (
            reg.resolve_trust(
                "doubao", sensitivity="strict", provenance=frozenset({"doubao"}), strict_keys=set()
            )
            == "strict"
        )

    def test_auto_relaxes_on_own_metadata(self):
        assert (
            reg.resolve_trust(
                "doubao", sensitivity="auto", provenance=frozenset({"doubao"}), strict_keys=set()
            )
            == "confirmed"
        )

    def test_auto_strict_without_evidence(self):
        assert (
            reg.resolve_trust(
                "doubao", sensitivity="auto", provenance=frozenset(), strict_keys=set()
            )
            == "strict"
        )

    def test_auto_cross_mark_same_product(self):
        # a detected Jimeng wordmark relaxes the Jimeng pill (same product, other corner)
        assert (
            reg.resolve_trust(
                "jimeng_pill", sensitivity="auto", provenance=frozenset(), strict_keys={"jimeng"}
            )
            == "confirmed"
        )

    def test_auto_relaxes_liblib_pill_via_wordmark_and_metadata(self):
        assert (
            reg.resolve_trust(
                "liblib_pill", sensitivity="auto", provenance=frozenset(), strict_keys={"liblib"}
            )
            == "confirmed"
        )
        assert (
            reg.resolve_trust(
                "liblib_pill",
                sensitivity="auto",
                provenance=frozenset({"liblib"}),
                strict_keys=set(),
            )
            == "confirmed"
        )

    def test_auto_no_cross_mark_across_products(self):
        # a detected Jimeng wordmark must NOT relax Doubao (distinct products, same corner)
        assert (
            reg.resolve_trust(
                "doubao", sensitivity="auto", provenance=frozenset(), strict_keys={"jimeng"}
            )
            == "strict"
        )

    def test_remove_auto_marks_accepts_all_sensitivities(self):
        blank = np.zeros((256, 256, 3), np.uint8)
        for s in ("auto", "strict"):
            _, removed = reg.remove_auto_marks(blank, sensitivity=s, backend="cv2")
            assert removed == []

    def test_unknown_sensitivity_raises(self):
        with pytest.raises(ValueError, match="unknown sensitivity"):
            reg.validate_sensitivity("aggressive")
        assert reg.validate_sensitivity("auto") == "auto"
        assert reg.validate_sensitivity("strict") == "strict"
        with pytest.raises(ValueError, match="unknown sensitivity"):
            reg.Context(sensitivity="assume_ai")


class TestNoBlanketRelaxation:
    """There is NO path that relaxes a mark's gate without same-product evidence."""

    def test_sensitivity_has_exactly_two_levels(self):
        import typing

        assert set(typing.get_args(reg.Sensitivity)) == {"auto", "strict"}

    def test_trust_ladder_has_exactly_two_levels(self):
        import typing

        assert set(typing.get_args(reg.Trust)) == {"strict", "confirmed"}

    def test_no_sensitivity_relaxes_without_same_product_evidence(self):
        for sens in ("auto", "strict"):
            for key in SHIPPED_KEYS:
                assert (
                    reg.resolve_trust(
                        key, sensitivity=sens, provenance=frozenset(), strict_keys=set()
                    )
                    == "strict"
                )


class TestArbiter:
    """``decide`` is the PURE removal arbiter: (candidates, context) -> winners."""

    @staticmethod
    def _c(key, *, strict=False, relaxed=False, flat=False):
        feats = {"footprint_flat": 1.0} if flat else {}
        return reg.Candidate(key, f"L:{key}", strict, relaxed, feats)

    def _keys(self, cands, ctx):
        return {d.candidate.key for d in reg.decide(cands, ctx)}

    def test_empty(self):
        assert reg.decide([], reg.Context()) == []

    def test_strict_uses_strict_verdict(self):
        assert (
            self._keys([self._c("jimeng", relaxed=True)], reg.Context(sensitivity="strict"))
            == set()
        )

    def test_auto_relaxes_on_provenance(self):
        c = [self._c("jimeng", relaxed=True)]
        assert self._keys(c, reg.Context(provenance=frozenset({"jimeng"}))) == {"jimeng"}
        assert self._keys(c, reg.Context()) == set()  # no evidence -> strict verdict

    def test_cross_mark_relaxes_pill_via_jimeng(self):
        cands = [
            self._c("jimeng", strict=True, relaxed=True),
            self._c("jimeng_pill", relaxed=True, flat=True),
        ]
        assert self._keys(cands, reg.Context()) == {"jimeng", "jimeng_pill"}

    def test_pill_dropped_on_doubao(self):
        cands = [
            self._c("doubao", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=True),
        ]
        keys = self._keys(cands, reg.Context(provenance=frozenset({"jimeng"})))
        assert "doubao" in keys
        assert "jimeng_pill" not in keys

    @pytest.mark.parametrize("key", ["qwen", "yuanbao"])
    def test_other_manufacturers_do_not_join_the_bytedance_pill_family(self, key):
        cands = [
            self._c(key, strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=True),
        ]
        keys = self._keys(cands, reg.Context(provenance=frozenset({"jimeng"})))
        assert keys == {key, "jimeng_pill"}

    def test_pill_metadata_arm_gated_on_flatness(self):
        ctx = reg.Context(provenance=frozenset({"jimeng"}))
        assert self._keys([self._c("jimeng_pill", strict=True, relaxed=True, flat=True)], ctx) == {
            "jimeng_pill"
        }
        assert (
            self._keys([self._c("jimeng_pill", strict=True, relaxed=True, flat=False)], ctx)
            == set()
        )

    def test_pill_wordmark_arm_ignores_flatness(self):
        cands = [
            self._c("jimeng", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
        ]
        assert "jimeng_pill" in self._keys(cands, reg.Context())

    def test_weak_pill_detection_does_not_confirm_the_jimeng_wordmark(self):
        """A weak pill hit must not grant the wordmark `confirmed` trust (the
        closed false-positive loop upstream measured: pill fire -> relaxed wordmark
        -> wordmark bypasses the pill's flatness guard)."""
        cands = [
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
            self._c("jimeng", strict=False, relaxed=True),
        ]
        fired = {d.candidate.key for d in reg.decide(cands, reg.Context("auto", frozenset()))}
        assert "jimeng" not in fired, "a weak pill hit must not relax the jimeng wordmark"
        assert "jimeng_pill" not in fired  # no wordmark arm -> flatness guard applies

    def test_weak_liblib_pill_does_not_confirm_the_wordmark(self):
        assert (
            reg.resolve_trust(
                "liblib", sensitivity="auto", provenance=frozenset(), strict_keys={"liblib_pill"}
            )
            == "strict"
        )

    def test_real_jimeng_wordmark_still_corroborates_the_pill(self):
        cands = [
            self._c("jimeng", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
        ]
        fired = {d.candidate.key for d in reg.decide(cands, reg.Context("auto", frozenset()))}
        assert fired == {"jimeng", "jimeng_pill"}


class TestSinglePassPerception:
    """``detect_both`` must equal two ``detect`` calls, for strictly less work."""

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param(np.full((256, 256, 3), 100, np.uint8), id="clean"),
            pytest.param(np.zeros((40, 40, 3), np.uint8), id="below-the-size-floor"),
        ],
    )
    def test_dual_verdict_matches_two_separate_detects(self, image):
        for m in reg.known_marks():
            strict, relaxed = m.detect_both(image)
            for one, two, level in (
                (strict, m.detect(image, provenance=False), "strict"),
                (relaxed, m.detect(image, provenance=True), "relaxed"),
            ):
                assert one.detected == two.detected, f"{m.key} {level}"
                assert one.confidence == two.confidence, f"{m.key} {level}"
                assert one.region == two.region, f"{m.key} {level}"

    def test_perception_scans_each_mark_once(self, monkeypatch: pytest.MonkeyPatch):
        import cv2 as _cv2

        image = np.full((512, 512, 3), 100, np.uint8)
        calls = [0]
        real = _cv2.matchTemplate

        def counted(*args, **kwargs):
            calls[0] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(_cv2, "matchTemplate", counted)
        reg._build_candidates(image)
        single = calls[0]
        calls[0] = 0
        for m in reg.known_marks():
            m.detect(image, provenance=False)
            m.detect(image, provenance=True)
        assert single * 2 <= calls[0] + 2, (
            f"perception did {single} sweeps vs {calls[0]} for two passes"
        )


class TestMarkKnowledgeIsOnTheRow:
    """Registering a mark is ONE edit: the row carries everything about it."""

    def test_platformless_marks_do_not_attribute_a_product_on_their_own(self):
        assert {m.key for m in reg.known_marks() if m.platform is None} == {
            "jimeng_pill",
            "liblib_pill",
        }

    def test_every_tc260_mark_declares_the_aigc_signal_and_codes(self):
        """TC260 provenance relaxation only fires through MAPPED producers, so a
        tc260-regime row without producer codes could never be corroborated."""
        for mark in reg.known_marks():
            if mark.label_regime == "tc260" and mark.key not in ("jimeng_pill", "liblib_pill"):
                assert "aigc" in mark.provenance_signals, mark.key
                assert mark.tc260_producer_codes, f"{mark.key}: no mapped TC260 producer codes"

    def test_tc260_marks_keep_distinct_manufacturers(self):
        manufacturers = {
            mark.key: mark.manufacturer
            for mark in reg.known_marks()
            if mark.label_regime == "tc260"
        }
        assert manufacturers["doubao"] == manufacturers["jimeng"] == "bytedance"
        assert manufacturers["qwen"] == "alibaba"
        assert manufacturers["kling"] == "kuaishou"
        assert manufacturers["yuanbao"] == "tencent"

    def test_tc260_producer_lookup_returns_the_registry_row(self):
        assert reg.tc260_producer_mark("91440101MA9Y9T4H7A") is reg.get_mark("qwen")
        assert reg.tc260_producer_mark("0011...junk") is None
        assert reg.tc260_producer_mark("unknown") is None

    def test_tc260_relaxation_only_via_mapped_producers(self):
        """An UNMAPPED producer code must not confirm any product: the 001-prefixed
        label of a generator outside tc260_verify.TC260_PRODUCERS never relaxes a
        gate, even though it is a well-formed TC260 label."""
        unmapped = "91119999XXXXXXXXXY"  # well-formed USCC-shaped code, not registered
        assert reg.tc260_producer_mark(unmapped) is None
        # and the relaxation decision itself never fires on it
        assert not reg._provenance_confirms_product("qwen", frozenset({unmapped}))

    def test_platform_token_marks_are_the_c2pa_attributed_ones(self):
        by_token = {m.key for m in reg.known_marks() if m.provenance_platform_tokens}
        assert by_token == {"microsoft"}

    def test_provisional_rows_are_exactly_wave3(self):
        assert reg.provisional_marks() == frozenset({"wan", "openart"})
        for m in reg.known_marks():
            if m.key in ("wan", "openart"):
                assert m.provisional

    def test_product_map_is_derived_from_the_rows(self):
        assert {m.key: m.product for m in reg.known_marks()} == reg._PRODUCT_OF
        assert reg._PRODUCT_OF["jimeng_pill"] == "jimeng"  # the one shared product
        assert reg._PRODUCT_OF["liblib_pill"] == "liblib"
        assert {
            m.key for m in reg.known_marks() if not m.can_corroborate
        } == reg._CANNOT_CORROBORATE


class TestPillSuppressors:
    """The pill veto is derived from the Jimeng manufacturer's registry rows."""

    def test_other_bytedance_product_suppresses_the_pill(self):
        pill = reg.get_mark("jimeng_pill")
        expected = {
            m.key
            for m in reg.known_marks()
            if m.manufacturer == pill.manufacturer and m.product != pill.product
        }
        assert reg._pill_suppressors() == expected
        assert expected == {"dola", "doubao"}

    def test_other_manufacturers_do_not_suppress_the_pill(self):
        assert reg._keep_pill({"qwen"}, provenance=frozenset({"jimeng"}), footprint_flat=True)
        assert reg._keep_pill({"liblib", "jimeng"}, provenance=frozenset(), footprint_flat=True)

    def test_verified_label_bypasses_texture_only_with_jimeng_provenance(self):
        assert reg._keep_pill(
            {"jimeng_pill"},
            provenance=frozenset({"jimeng"}),
            footprint_flat=False,
            verified_label=True,
        )
        assert not reg._keep_pill(
            {"jimeng_pill"},
            provenance=frozenset(),
            footprint_flat=False,
            verified_label=True,
        )

    def test_pill_survives_non_tc260_products(self):
        """Samsung/Gemini-class products are not TC260 labelers and cannot veto the
        arm they could not have enabled."""
        assert reg._keep_pill({"samsung", "jimeng"}, provenance=frozenset(), footprint_flat=True)


class TestEngineImportsStayLazy:
    """The engine table holds NAMES, not imports (upstream vector, adapted).

    The registry is imported by metadata-only paths; the engines pull cv2. A
    name-table regression would put cv2 on the stdlib import graph — the exact
    dependency guardrail this run pins. Run in a subprocess because the
    in-process ``sys.modules`` is already polluted by the rest of the suite.
    """

    def _modules_after(self, statements: str) -> set[str]:
        script = (
            f"import sys, json\nsys.path.insert(0, {str(SCRIPTS)!r})\n"
            f"{statements}\nprint(json.dumps(sorted(sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(json.loads(out.stdout))

    def test_importing_the_registry_pulls_no_engine_and_no_cv2(self):
        loaded = self._modules_after("import watermark_registry")
        assert "samsung_engine" not in loaded
        assert "doubao_engine" not in loaded
        assert "cv2" not in loaded

    def test_resolving_an_engine_imports_it(self):
        loaded = self._modules_after("import watermark_registry as r\nr._engine('doubao')")
        assert "doubao_engine" in loaded

    def test_every_engine_row_resolves(self):
        """All 17 engine NAMES resolve to an importable class (registry/engines parity)."""
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "import watermark_registry as r\n"
            "bad = [k for k in r.mark_keys() if r._engine(k) is None]\n"
            "print(json.dumps(bad))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout) == []


class TestMetadataProvenanceWiring:
    """F5 (stage 8 fix pass 2): the rows' provenance fields are live config.

    ``metadata_evidence`` reads each row's declared ``provenance_signals`` /
    ``provenance_platform_tokens`` to map the FILE's metadata to mark keys, and
    that evidence flows into ``resolve_trust`` exactly like TC260 producer keys
    do -- so samsung_genai metadata and a Microsoft platform token actually
    corroborate their marks instead of decorating the registry."""

    @staticmethod
    def _png_with_text(payload: bytes) -> bytes:
        import struct
        import zlib

        def chunk(ctype: bytes, body: bytes) -> bytes:
            crc = zlib.crc32(ctype)
            crc = zlib.crc32(body, crc) & 0xFFFFFFFF
            return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\x00\x00")
        return (
            sig
            + chunk(b"IHDR", ihdr)
            + chunk(b"tEXt", payload)
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b"")
        )

    def test_samsung_genai_metadata_confirms_samsung(self):
        data = b'junk\x00PhotoEditor_Re_Edit_Data {"genAIType": 1} trailing'
        assert reg.metadata_evidence(data) == frozenset({"samsung"})
        assert (
            reg.resolve_trust(
                "samsung",
                sensitivity="auto",
                provenance=reg.metadata_evidence(data),
                strict_keys=set(),
            )
            == "confirmed"
        )

    def test_platform_token_in_generator_metadata_confirms_microsoft(self):
        png = self._png_with_text(b"Software\x00Microsoft Capture Tool 1.0")
        assert reg.metadata_evidence(png) == frozenset({"microsoft"})
        assert (
            reg.resolve_trust(
                "microsoft",
                sensitivity="auto",
                provenance=reg.metadata_evidence(png),
                strict_keys=set(),
            )
            == "confirmed"
        )

    def test_platform_token_never_confirms_an_undeclared_mark(self):
        """Only the rows that DECLARE a token may be corroborated by it: a
        'samsung' string in metadata must not confirm a mark that never asked
        for token evidence."""
        png = self._png_with_text(b"Software\x00samsung something")
        assert reg.metadata_evidence(png) == frozenset()

    def test_clean_bytes_confirm_nothing(self):
        assert reg.metadata_evidence(b"") == frozenset()
        assert reg.metadata_evidence(b"\x00" * 64) == frozenset()
        assert (
            reg.resolve_trust(
                "samsung", sensitivity="auto", provenance=frozenset(), strict_keys=set()
            )
            == "strict"
        )

    def test_malformed_png_still_confirms_nothing_and_does_not_raise(self):
        malformed = b"\x89PNG\r\n\x1a\nnot-really-a-png"
        assert reg.metadata_evidence(malformed) == frozenset()

    def test_non_tc260_signals_do_not_leak_across_products(self):
        """Samsung metadata confirms samsung only -- never a ByteDance mark."""
        data = b'PhotoEditor_Re_Edit_Data {"genAIType": 1}'
        assert reg.metadata_evidence(data) == frozenset({"samsung"})
        assert (
            reg.resolve_trust(
                "doubao",
                sensitivity="auto",
                provenance=reg.metadata_evidence(data),
                strict_keys=set(),
            )
            == "strict"
        )


class TestNoMarkResultsDoNotAliasTheCaller:
    """F8 (stage 8 fix pass 2): a pass that removes nothing returns a COPY of
    the caller's array -- mutating the result must never edit the input."""

    def test_auto_pass_no_mark_returns_a_copy(self):
        img = np.zeros((512, 512, 3), np.uint8)
        report = reg.remove_auto_marks_detailed(img, backend="cv2")
        assert report.status == "no_mark"
        report.image[0, 0] = 255
        assert img[0, 0, 0] == 0

    def test_named_mark_no_hit_returns_a_copy(self):
        img = np.zeros((512, 512, 3), np.uint8)
        report = reg.remove_mark_detailed(img, "doubao", backend="cv2")
        assert report.status == "no_mark"
        report.image[0, 0] = 255
        assert img[0, 0, 0] == 0


class TestPerMarkErrorContainment:
    """F9 (stage 8 fix pass 2): one raising mark must not abort the auto pass.

    A mark whose localize/fill raises yields a per-mark error row; the pass
    continues with the remaining marks and reports the aggregate error."""

    @staticmethod
    def _install_two_marks(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        calls = {"fill": 0, "mask": 0}
        before_a = reg.MarkDetection("mark_a", "Mark A", "top-left", True, 0.9, (2, 2, 4, 4))
        after_a = reg.MarkDetection("mark_a", "Mark A", "top-left", False, 0.1, (0, 0, 0, 0))
        before_b = reg.MarkDetection("mark_b", "Mark B", "bottom-right", True, 0.8, (10, 10, 4, 4))

        def detect_both_a(_image):
            return before_a, before_a

        def detect_a(_image, *, provenance=False):
            return after_a

        def mask_a(_image, *, force=False, detection=None):
            calls["mask"] += 1
            result = np.zeros((20, 20), np.uint8)
            result[2:6, 2:6] = 255
            return result

        def detect_both_b(_image):
            return before_b, before_b

        def detect_b(_image, *, provenance=False):
            return before_b

        def mask_b(_image, *, force=False, detection=None):
            calls["mask"] += 1
            raise RuntimeError("poisoned engine")

        def make(key, label, product, det_pair, det, mask_fn):
            return reg.KnownMark(
                key=key,
                label=label,
                location="top-left",
                in_auto=True,
                product=product,
                manufacturer=key,
                label_regime=None,
                platform=label,
                _detect=det,
                _mask=mask_fn,
                _detect_both=det_pair,
            )

        marks = (
            make("mark_a", "Mark A", "a", detect_both_a, detect_a, mask_a),
            make("mark_b", "Mark B", "b", detect_both_b, detect_b, mask_b),
        )

        def fill(image, selected_mask, *, backend="auto"):
            del backend
            calls["fill"] += 1
            result = image.copy()
            result[selected_mask > 0] = 123
            return result

        monkeypatch.setattr(reg, "_REGISTRY", marks)
        monkeypatch.setattr(reg, "_PRODUCT_OF", {"mark_a": "a", "mark_b": "b"})
        monkeypatch.setattr(reg, "fill", fill)
        return calls

    def test_poisoned_mark_yields_error_row_and_the_pass_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls = self._install_two_marks(monkeypatch)

        report = reg.remove_auto_marks_detailed(np.zeros((20, 20, 3), np.uint8), backend="cv2")

        assert [e.key for e in report.errors] == ["mark_b"]
        assert "poisoned engine" in report.errors[0].message
        assert [m.key for m in report.marks] == ["mark_a"]
        assert report.marks[0].status == "cleaned"
        assert report.status == "error"
        # the healthy mark was still localized, filled and validated
        assert calls["fill"] == 1 and calls["mask"] == 2
        assert int(report.image[4, 4, 0]) == 123

    def test_error_and_partial_precedence(self, monkeypatch: pytest.MonkeyPatch):
        """A residual elsewhere does not mask the error: error outranks partial."""
        calls = self._install_two_marks(monkeypatch)
        # make mark_a's post-check report a residual
        marks = reg._REGISTRY
        a = marks[0]

        def detect_a(_image, *, provenance=False):
            return reg.MarkDetection("mark_a", "Mark A", "top-left", True, 0.5, (2, 2, 4, 4))

        object.__setattr__(a, "_detect", detect_a)
        del calls
        report = reg.remove_auto_marks_detailed(np.zeros((20, 20, 3), np.uint8), backend="cv2")
        assert report.status == "error"


class TestFillOrderSingleSource:
    """F10 (stage 8 fix pass 2): the registry's auto-fill preference is derived
    from the shared backend-stack constant, not a second hardcoded ladder."""

    @staticmethod
    def _both_available(monkeypatch: pytest.MonkeyPatch) -> None:
        import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: True)

    def test_projection_of_the_default_stack_is_the_registry_ladder(self):
        """For the shipped stack, the mapped projection is lama > migan > cv2."""
        import inpaint_backends

        projected = [
            mapped
            for tier in inpaint_backends.FILL_RESOLVE_ORDER
            if (mapped := reg._FILL_STACK_TIERS.get(tier)) is not None
        ]
        deduped = list(dict.fromkeys(projected))
        assert deduped == ["lama", "migan", "cv2"]

    def test_reordering_the_stack_constant_reorders_the_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import inpaint_backends

        self._both_available(monkeypatch)
        monkeypatch.setattr(
            inpaint_backends,
            "FILL_RESOLVE_ORDER",
            ("migan", "lama", "cv2-telea", "texture-patch", "simple"),
        )
        assert reg.preferred_inpaint_backend() == "migan"

    def test_cv2_floor_when_no_learned_backend_is_available(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        import logging

        import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        monkeypatch.setattr(reg, "_warned_cv2_fallback", False)
        with caplog.at_level(logging.WARNING):
            assert reg.preferred_inpaint_backend() == "cv2"
        assert any("cv2 classical inpaint" in r.message for r in caplog.records)


class TestPillWrappersAreDerived:
    """F11 (stage 8 fix pass 2): the jimeng_pill adapters are the shared wrapper
    helpers specialized to the pill key -- no duplicate wrapper bodies."""

    def test_no_duplicate_pill_wrapper_defs(self):
        source = (SCRIPTS / "watermark_registry.py").read_text(encoding="utf-8")
        for name in ("_pill_detect", "_pill_detect_both", "_pill_mask"):
            assert f"def {name}(" not in source, f"{name} duplicates the shared wrapper helper"

    def test_pill_wrappers_behave_like_the_shared_helpers(self):
        import numpy as np

        image = np.zeros((256, 256, 3), np.uint8)
        detect = reg._engine_mark_detect("jimeng_pill", "Jimeng AI生成 pill", "top-left")
        detect_both = reg._engine_mark_detect_both("jimeng_pill", "Jimeng AI生成 pill", "top-left")
        mask = reg._engine_mark_mask("jimeng_pill")

        assert reg._pill_detect(image) == detect(image)
        assert reg._pill_detect_both(image) == detect_both(image)
        assert reg._pill_features(image) == {"footprint_flat": 1.0}
        built_mask = mask(image, force=True)
        pill_mask = reg._pill_mask(image, force=True)
        assert (built_mask is None) == (pill_mask is None)
        if built_mask is not None:
            assert np.array_equal(built_mask, pill_mask)
