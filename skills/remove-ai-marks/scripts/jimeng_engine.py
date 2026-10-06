# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/jimeng_engine.py, Apache-2.0; modified: detection silhouette swapped from the upstream capture-solved wordmark asset to this tree's renderer-synthetic ``jimeng_label_alpha.png`` (capture-solve outputs are forbidden here — see assets/PROVENANCE.md), so the alpha-map geometry fractions are re-derived from the shipped asset's aspect while keeping the calibrated corner, gates, provenance factor, and rival wiring; scripts package imports are flattened to this repo's stdlib script package (top-level module names).
"""Jimeng visible watermark detector/localizer.

Jimeng (即梦AI, ByteDance) stamps generated images with a visible wordmark strip
in the bottom-right corner -- a near-white semi-transparent overlay, the same
overlay class as the Doubao text strip.

Detection matches the bundled glyph silhouette against the corner; removal is the
shared **localize -> fill** (the glyph footprint feeds ``region_eraser``), NOT
reverse-alpha. This module shares
:class:`_text_mark_engine.TextMarkEngine` and supplies only Jimeng's tuned
:class:`TextMarkConfig` (bottom-right corner, ``assets/jimeng_label_alpha.png``
-- the renderer-synthetic detection silhouette shipped by this tree; the
upstream engine's silhouette was recovered from a real carrier capture and is
deliberately not shipped here, see ``assets/PROVENANCE.md``). Jimeng images are
also caught by the China TC260 AIGC metadata label. The visual detector also
feeds ``identify`` when metadata is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import _text_mark_engine
from _text_mark_engine import TextMarkConfig, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (mark scales with width, bottom-right).
WM_WIDTH_FRAC = 0.27
WM_HEIGHT_FRAC = 0.092
MARGIN_RIGHT_FRAC = 0.008
MARGIN_BOTTOM_FRAC = 0.010

# Glyph appearance: a light, low-saturation gray brighter than the local background.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

# Shape-consistent detection. Threshold 0.45 cleanly separates real Jimeng marks
# (>=0.81) from the Doubao strip (0.21), so the two ByteDance marks do not cross-fire.
#
# That separation holds at the STRICT threshold ONLY. Relaxed under provenance it
# collapses: at the old shared 0.7 factor, many false additions were Doubao marks.
# That was patched with a tighter 0.85
# factor at the time; the competitive rival margin replaced it -- see below.
DETECT_MIN_COVERAGE = 0.02
DETECT_NCC_THRESHOLD = 0.45

# Back to the 0.70 default. The 0.85 patch (2026-07-18) existed only to blunt the
# Doubao cross-fire by sacrificing recall; the competitive RIVAL MARGIN below
# discriminates directly, so the recall the patch gave up is no longer the price of
# precision. See TextMarkEngine._rival_margin_ok.
PROVENANCE_NCC_FACTOR = 0.7

# Detection-silhouette geometry, sized from the shipped renderer-synthetic asset
# (353x128 px): the width fraction sizes the detection silhouette for the match;
# removal is the template-free glyph footprint mask.
_ALPHA_WIDTH_FRAC = 0.2021
_ALPHA_HEIGHT_FRAC = 0.2021 * 128 / 353

_CONFIG = TextMarkConfig(
    name="Jimeng",
    asset_name="jimeng_label_alpha.png",
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
    provenance_ncc_factor=PROVENANCE_NCC_FACTOR,
    rivals=("doubao_outline_alpha.png",),
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Jimeng alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary glyph silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Jimeng glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class JimengEngine(TextMarkEngine):
    """Detect/localize the visible Jimeng bottom-right wordmark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
