# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/yuanbao_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package; the compact-variant assets and provenance comments are unchanged.
"""Tencent Yuanbao visible watermark detector and localizer.

Yuanbao stamps a compact italic two-line mark, ``元宝`` over ``AI生成``, in the
bottom-right corner. The same silhouette is rendered light on dark scenes and
dark on pale scenes, so a one-polarity white top-hat cannot detect it reliably.
This engine uses the shared text-mark pipeline with the ``contrast`` front-end:
normalized absolute local-luma residual followed by silhouette NCC.

The bundled silhouette is synthetic and font-rendered by
``render_vendor_silhouettes.py``. Removal follows the shared localize-then-fill
path and uses the detector's own match box.

Calibration (2026-07-25) used the metadata-harvested Tencent cohort after byte
deduplication and visual adjudication. The standard two-line variant was detected
on 26 of 28 unique marked carriers (92.9%) at gate 0.38, with 0 fires on 286
hand-labeled clean frames. The separate photographer-overlay variant is not
covered by this silhouette.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import _text_mark_engine
from _text_mark_engine import BottomRightAnchoredEngine, TextMarkConfig
from _text_mark_variants import BottomRightBoxTextMarkEngine, TextMarkVariantsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

WM_WIDTH_FRAC = 0.20
WM_HEIGHT_FRAC = 0.15
MARGIN_RIGHT_FRAC = 0.002
MARGIN_BOTTOM_FRAC = 0.002

MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.38

_ALPHA_WIDTH_FRAC = 0.08
_ALPHA_HEIGHT_FRAC = 0.0446
_LADDER = (0.95, 1.0, 1.05)

_CONFIG = TextMarkConfig(
    name="Tencent Yuanbao",
    asset_name="yuanbao_alpha.png",
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
    detect_frontend="contrast",
    scale_basis="short",
    ladder=_LADDER,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=32,
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any]:
    """The bundled Yuanbao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


class YuanbaoEngine(TextMarkVariantsMixin, BottomRightAnchoredEngine):
    """Detect and localize the bottom-right Yuanbao mark."""

    _ANCHOR_MAX_RIGHT = 0.04
    _ANCHOR_MAX_BOTTOM = 0.04

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._variants = (
            BottomRightBoxTextMarkEngine(
                replace(
                    _CONFIG,
                    asset_name="yuanbao_compact_alpha.png",
                    width_frac=0.28,
                    height_frac=0.13,
                    scale_basis="width",
                    alpha_width_frac=0.09,
                    alpha_height_frac=0.09 * 238 / 387 * 0.75,
                    ladder=(0.95, 1.0, 1.05),
                    detect_ncc_threshold=0.50,
                    provenance_ncc_factor=0.62,
                    max_saturation=255,
                )
            ),
            BottomRightBoxTextMarkEngine(
                replace(
                    _CONFIG,
                    asset_name="yuanbao_compact_gray_alpha.png",
                    width_frac=0.28,
                    height_frac=0.13,
                    scale_basis="width",
                    alpha_width_frac=0.06,
                    alpha_height_frac=0.06 * 238 / 387 * 0.90,
                    ladder=(0.95, 1.0, 1.05),
                    detect_frontend="gray",
                    detect_ncc_threshold=0.50,
                    provenance_ncc_factor=0.50,
                    max_saturation=255,
                )
            ),
        )
