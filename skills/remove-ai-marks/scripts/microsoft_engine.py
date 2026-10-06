# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/microsoft_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package (region_eraser imported top-level), the upstream pyright blanket comment is dropped (this repo lints with ruff), and the text-variant detection-silhouette aspect follows this tree's rendered ``microsoft_text_alpha.png`` (342x53).
"""Microsoft top-right AI-badge detector/localizer.

This engine covers one measured Microsoft output variant: a white pill with dark
internal shapes in the top-right corner. The evaluated files used both "Made with
AI" and "AI-Generated" wording. This is narrower than Microsoft's documented
watermark feature, which can use a Copilot icon or text and can place the mark in
other positions. A Microsoft provenance signal therefore does not establish that
this exact visible variant is present.

Detection matches a synthetic pill silhouette (white pill with the sparkle and
text KNOCKED OUT) against the top-hat blob of the located box: the holes are what
discriminate this pill from any other bright rounded element in the corner.
Removal is the shared **localize -> fill**; the glyph-bbox :meth:`footprint_mask`
covers the whole pill including its text.

The tuned numbers below were remeasured upstream on 2026-08-27 with the registered
engine (17 visually confirmed carriers, 343 Microsoft-provenance files, 1200
non-overlapping no-signal controls); they are ported, not re-measured here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import _text_mark_engine
import region_eraser
from _text_mark_engine import TextMarkConfig, TextMarkDetection
from _text_mark_variants import GlyphTextMarkEngine, TextMarkVariantsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image LONG side (measured; scale_basis="long":
# on 1024x1536 portraits the pill tracks the 1536, and a width basis undersized
# the template until the portrait carriers fell to 0.15-0.32 NCC).
WM_WIDTH_FRAC = 0.170
WM_HEIGHT_FRAC = 0.055
MARGIN_RIGHT_FRAC = 0.004
MARGIN_TOP_FRAC = 0.003

# Glyph appearance: a bright near-white pill (luma ~245), gray-scale (sat < 60).
MAX_SATURATION = 60
LOGO_MIN_LUMA = 170
TOPHAT_DELTA = 10

# Calibrated 2026-08-27 upstream: no-signal controls (n=1200) max 0.293; visually
# confirmed carriers (n=17) p50 0.519 / max 0.579, with 15/17 above the gate.
DETECT_MIN_COVERAGE = 0.30  # the pill fills most of its box; content corners do not
DETECT_NCC_THRESHOLD = 0.38

# Pill silhouette geometry (fraction of the long side): 0.152 x 0.040, aspect ~3.78.
_ALPHA_WIDTH_FRAC = 0.152
_ALPHA_HEIGHT_FRAC = 0.040

_CONFIG = TextMarkConfig(
    name="Microsoft top-right AI badge",
    asset_name="microsoft_alpha.png",
    corner="tr",
    margin_floor=2,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_TOP_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=24,
    detect_frontend="binary",
    scale_basis="long",
    provenance_ncc_factor=0.7,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Microsoft pill template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


class MicrosoftEngine(TextMarkVariantsMixin):
    """Detect/localize the measured Microsoft top-right AI badge."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._variants = (
            GlyphTextMarkEngine(
                replace(
                    _CONFIG,
                    asset_name="microsoft_text_alpha.png",
                    width_frac=0.20,
                    height_frac=0.065,
                    margin_x_frac=0.002,
                    margin_bottom_frac=0.002,
                    scale_basis="width",
                    alpha_width_frac=0.14,
                    alpha_height_frac=0.14 * 53 / 342,
                    detect_frontend="contrast",
                    ladder=(0.95, 1.0, 1.05),
                    detect_ncc_threshold=0.60,
                    provenance_ncc_factor=1.0,
                    max_saturation=255,
                )
            ),
        )

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Cover the enclosing beige pill for the text-only layout."""
        if force or image is None or image.size == 0:
            return super().footprint_mask(image, force=force, dilate=dilate, detection=detection)
        det = detection if detection is not None else self.detect(image)
        engine = self.variant_engine(det)
        if engine is self or not det.detected or det.match_box is None:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)

        loc = engine.locate(image)
        x0, y0, x1, y1 = det.match_box
        pad_x = int(image.shape[1] * 0.012)
        pad_y = int(image.shape[1] * 0.014)
        box = (
            max(0, loc.x + x0 - pad_x),
            max(0, loc.y + y0 - pad_y),
            x1 - x0 + 1 + 2 * pad_x,
            y1 - y0 + 1 + 2 * pad_y,
        )
        return region_eraser.boxes_to_mask(
            image.shape[:2],
            [box],
            dilate=0 if dilate is None else max(0, dilate),
        )
