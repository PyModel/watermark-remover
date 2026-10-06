# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/samsung_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package, the upstream pyright blanket comment is dropped (this repo lints with ruff), and the PRIMARY silhouette is this tree's renderer-synthetic ``samsung_ko_alpha.png`` with ``samsung_en_alpha.png`` as the variant — upstream's registered Italian ``samsung_alpha.png`` is a capture solve and is forbidden here (see assets/PROVENANCE.md), so the Korean arm carries upstream's own calibrated variant config and the English candidate inherits the registered Italian geometry with the shipped asset's aspect.
"""Samsung Galaxy AI visible watermark detector/localizer.

Samsung's on-device Generative AI photo edits burn a visible AI-content label into
the bottom-LEFT corner. The string is locale-specific: DETECTION only matches the
silhouettes shipped here (Korean "AI로 생성한 콘텐츠" primary, English
"AI-generated content" + sparkle variant), so other locales are not yet detected;
the localize -> fill removal itself is locale-agnostic.

Detection matches the bundled glyph silhouette against the corner; removal is the
shared **localize -> fill** (a detector-aligned alpha :meth:`footprint_mask` feeds
``region_eraser``), NOT reverse-alpha. This module shares
:class:`_text_mark_engine.TextMarkEngine` and supplies only Samsung's tuned
:class:`TextMarkConfig` values. Samsung Galaxy AI edits are also caught by C2PA +
the ``genAIType`` marker, so this is the visible-mark *removal* path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import _text_mark_engine
import image_io
from _text_mark_engine import TextMarkConfig, TextMarkDetection
from _text_mark_variants import GlyphTextMarkEngine, TextMarkVariantsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (mark scales with width, bottom-LEFT).
WM_WIDTH_FRAC = 0.40
WM_HEIGHT_FRAC = 0.060
MARGIN_LEFT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.002

# Glyph appearance: a light, low-saturation gray. LOGO_MIN_LUMA is lower than Jimeng's
# because the mark is faint (peak alpha ~0.38), so on a mid/dark background its glyph
# luma is lower; a white-paper document is still left untouched.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 110
TOPHAT_DELTA = 8

# Shape-consistent detection, inherited from the registered Italian arm's calibrated
# gate (continuous top-hat separated retained real positives at 0.82-0.90 from 421
# clean controls no higher than 0.38) and the Korean contrast variant's measured 0.46.
DETECT_MIN_COVERAGE = 0.01
DETECT_NCC_THRESHOLD = 0.40

# Detection-silhouette geometry (fraction of width). The Italian capture solved a
# 0.3195-wide mark at native width 1086; the shipped synthetic assets carry that
# width with their own rendered aspect.
_ALPHA_WIDTH_FRAC = 0.3195
_ALPHA_NATIVE_WIDTH = 1086

# The solved-alpha footprint floor/dilation upstream measured: a 0.05 floor plus
# one-pixel dilation covered the visible strokes while keeping the fill sparse.
_FOOTPRINT_ALPHA_FLOOR = 0.05
_FOOTPRINT_DILATE = 1

_CONFIG = TextMarkConfig(
    # PRIMARY: Korean label (upstream's calibrated variant config; contrast frontend).
    name="Samsung Galaxy AI",
    asset_name="samsung_ko_alpha.png",
    corner="bl",
    margin_floor=2,
    width_frac=0.28,
    height_frac=0.065,
    margin_x_frac=MARGIN_LEFT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=255,
    logo_min_luma=110,
    tophat_delta=8,
    morph_open_size=3,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=0.46,
    detect_frontend="contrast",
    ladder=(0.95, 1.0, 1.05),
    provenance_ncc_factor=1.0,
    scale_basis="width",
    alpha_width_frac=0.20,
    alpha_height_frac=0.20 * 49 / 343,
    min_gw=16,
)

#: The Korean primary config, exported for the registry test vectors.
_KO_CONFIG = _CONFIG

# VARIANT: English label + four-point sparkle; upstream registered only a candidate
# silhouette for this locale, so it inherits the registered Italian geometry (locate
# box, one-rung ladder, top-hat gate) with the shipped asset's own aspect.
_EN_CONFIG = replace(
    _CONFIG,
    asset_name="samsung_en_alpha.png",
    name="Samsung Galaxy AI (en)",
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    detect_frontend="tophat",
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    ladder=(1.0,),
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_WIDTH_FRAC * 31 / 314,
    min_gw=16,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Samsung alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary Galaxy AI label silhouette (255 = glyph), or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Samsung glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class SamsungEngine(TextMarkVariantsMixin):
    """Detect/localize the visible Samsung Galaxy AI text mark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
        self._variants = (GlyphTextMarkEngine(_EN_CONFIG),)

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Return the detector-aligned Samsung glyph footprint.

        The mark's peak opacity is only about 0.38, so filling the enclosing wordmark
        rectangle destroys substantially more real content than the overlay damaged.
        The continuous detector already found the silhouette at one exact box; align
        the solved alpha to that box and mask its strokes only. Explicit ``force`` has
        no trustworthy alignment and retains the shared geometry box.
        """
        if force or image is None or image.size == 0:
            return super().footprint_mask(image, force=force, dilate=dilate, detection=detection)

        image = image_io.to_bgr(image)
        det = detection if detection is not None else self.detect(image)
        engine = self.variant_engine(det)
        radius = _FOOTPRINT_DILATE if dilate is None else max(0, dilate)
        if engine is not self and det.detected and det.match_box is not None:
            import region_eraser

            x0, y0, x1, y1 = det.match_box
            loc = engine.locate(image)
            glyph_height = y1 - y0 + 1
            return region_eraser.boxes_to_mask(
                image.shape[:2],
                [
                    (
                        max(0, loc.x + x0 - round(1.3 * glyph_height)),
                        loc.y + y0,
                        x1 - x0 + 1 + round(1.3 * glyph_height),
                        glyph_height,
                    )
                ],
                dilate=radius,
            )
        alpha = _text_mark_engine.load_alpha_template(engine.config.asset_name)
        if alpha is not None:
            aligned = engine._aligned_alpha_mask(
                image,
                det,
                alpha,
                alpha_floor=_FOOTPRINT_ALPHA_FLOOR,
                dilate=radius,
            )
            if aligned is not None:
                return aligned
        return super().footprint_mask(image, force=False, dilate=dilate, detection=det)
