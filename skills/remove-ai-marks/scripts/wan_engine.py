# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/wan_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package and the detection-silhouette aspect follows this tree's rendered ``wan_alpha.png`` (332x109; upstream's hexagon approximation regenerated with this tree's fonts).
"""Alibaba Wan (通义万相) visible watermark detector/localizer.

Wan image exports carry a white, semi-transparent logo plus "Wan" wordmark in the
bottom-right corner, alongside the TC260 label in XMP whose ContentProducer names
Tongyi Yunqi (Hangzhou) Information Technology (USCC 91330106MA2CFLDG4R, owned by
Alibaba Cloud and Tongyi Lab). The service API's ``noWaterMark: true`` flag did not
remove it on the measured export, so the flag is not evidence of a clean file.

Detection matches the bundled synthetic silhouette (``assets/wan_alpha.png``, drawn
by ``render_vendor_silhouettes.py::draw_wan``, no export pixels) against the
corner; removal is the shared localize -> fill.

**Calibration is PROVISIONAL** (upstream, 2026-09-23/24): one real positive exists
(a 2048 px square Wan 2.7 Pro export); the mark measures 0.131 x 0.050 of the short
side with ~0.005 margins and scores 0.61-0.67 at 512-2048 px, while over 3092 local
images the maximum was 0.519. The mark hugs the corner, and requiring the match to
end within 0.015 of the short side from the right and bottom edges leaves no
negative at 0.45 or above. Both gates apply. Re-measure on a real Wan cohort before
any precision claim. The registry labels this mark provisional.
"""

from __future__ import annotations

from _text_mark_engine import BottomRightAnchoredEngine, TextMarkConfig

# Locate box as a fraction of the SHORT side: the mark plus slack for the ladder.
WM_WIDTH_FRAC = 0.17
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.002
MARGIN_BOTTOM_FRAC = 0.002

MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04  # unused by the tophat front-end (kept for config parity)
DETECT_NCC_THRESHOLD = 0.55  # provisional; see the module docstring

_ALPHA_WIDTH_FRAC = 268 / 2048
_ALPHA_HEIGHT_FRAC = (268 / 2048) * 109 / 332

_CONFIG = TextMarkConfig(
    name="Wan",
    asset_name="wan_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="tophat",
    scale_basis="short",
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    # STRICT ONLY: one positive cannot calibrate a relaxed provenance band.
    provenance_ncc_factor=1.0,
)


class WanEngine(BottomRightAnchoredEngine):
    """Detect/localize the bottom-right Wan logo plus wordmark (locate -> mask -> fill)."""

    # Match must end this close to the right and bottom edges (fraction of the short
    # side). Measured positive: 0.005-0.008 at 512-2048 px.
    _ANCHOR_MAX_RIGHT = 0.015
    _ANCHOR_MAX_BOTTOM = 0.015

    def __init__(self) -> None:
        super().__init__(_CONFIG)
