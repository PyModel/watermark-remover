# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/dola_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package and the detection-silhouette aspect follows this tree's rendered ``dola_alpha.png`` (347x87). Upstream's video-form Hershey templates are out of scope here (no video engine ships).
"""Detect the bottom-right Dola AI image disclosure, independently of video marks."""

from __future__ import annotations

from _text_mark_engine import TextMarkConfig
from _text_mark_variants import BoxTextMarkEngine

_CONFIG = TextMarkConfig(
    name="Dola AI image disclosure",
    asset_name="dola_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=0.28,
    height_frac=0.09,
    margin_x_frac=0.002,
    margin_bottom_frac=0.002,
    max_saturation=255,
    logo_min_luma=110,
    tophat_delta=8,
    morph_open_size=3,
    detect_min_coverage=0.01,
    detect_ncc_threshold=0.50,
    alpha_width_frac=0.10,
    alpha_height_frac=0.10 * 87 / 347,
    detect_frontend="tophat",
    scale_basis="width",
    ladder=(0.95, 1.0, 1.05, 1.20, 1.26),
    provenance_ncc_factor=1.0,
    min_gw=24,
)


class DolaEngine(BoxTextMarkEngine):
    """Detect and localize the Dola AI image wordmark."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
