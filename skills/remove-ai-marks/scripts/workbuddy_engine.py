# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/workbuddy_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package and the detection-silhouette aspect follows this tree's rendered ``workbuddy_alpha.png`` (351x177, byte-identical to upstream's commit).
"""Detect the corner AI生成 / WORKBUDDY disclosure without inferring a base model."""

from __future__ import annotations

from _text_mark_engine import TextMarkConfig
from _text_mark_variants import BottomRightBoxTextMarkEngine

_CONFIG = TextMarkConfig(
    name="WorkBuddy AI disclosure",
    asset_name="workbuddy_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=0.22,
    height_frac=0.13,
    margin_x_frac=0.002,
    margin_bottom_frac=0.002,
    max_saturation=255,
    logo_min_luma=110,
    tophat_delta=8,
    morph_open_size=3,
    detect_min_coverage=0.01,
    detect_ncc_threshold=0.60,
    alpha_width_frac=0.12,
    alpha_height_frac=0.12 * 177 / 351,
    detect_frontend="tophat",
    scale_basis="width",
    ladder=(0.95, 1.0, 1.05),
    provenance_ncc_factor=1.0,
    min_gw=24,
)


class WorkBuddyEngine(BottomRightBoxTextMarkEngine):
    """Detect and localize the two-line WorkBuddy disclosure."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
