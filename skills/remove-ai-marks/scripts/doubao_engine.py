# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/doubao_engine.py, Apache-2.0; modified: the primary detection silhouette is this tree's renderer-synthetic ``doubao_outline_alpha.png`` (the upstream capture-solved strip asset is forbidden here — see assets/PROVENANCE.md); the upstream outlined-variant arm is NOT ported — its variant routing keys on a distinct second silhouette asset, and this tree ships exactly one Doubao silhouette, so a shared asset name would mis-route the base detection into the variant's footprint policy (the variant's measured gates were calibrated on real captured outlined marks and are not synthetically reproducible); scripts package imports are flattened to this repo's stdlib script package.
"""Doubao visible watermark detector/localizer.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner -- the explicit AIGC
label mandated by China's TC260 standard, a near-white semi-transparent overlay.

Detection matches the bundled glyph silhouette against the corner candidate; removal
is the shared **localize -> fill**. Doubao reuses that aligned silhouette as a sparse
glyph footprint instead of bounding the thresholded corner response: bright scene
texture behind this bottom-edge mark can otherwise expand a solid mask to the frame
edges, where classical inpainting has no outside context. This module shares
:class:`_text_mark_engine.TextMarkEngine` and supplies only Doubao's tuned
:class:`TextMarkConfig` (bottom-right corner, ``assets/doubao_outline_alpha.png``
-- the renderer-synthetic detection silhouette shipped by this tree; the upstream
primary silhouette was recovered from a real carrier capture and is deliberately
not shipped here, see ``assets/PROVENANCE.md``). Arbitrary-region inpainting still
lives in ``region_eraser`` / the ``erase`` command.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import _text_mark_engine
import image_io
import numpy as np
from _text_mark_engine import (
    TextMarkConfig,
    TextMarkDetection,
    TextMarkEngine,
    TextMarkScan,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the mark's scale basis, anchored bottom-right. The
# box is GENEROUSLY wider than the mark and reaches close to the corner so a
# per-image re-rasterization shift stays inside the NCC alignment search.
WM_WIDTH_FRAC = 0.22
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.004

# Glyph appearance: a light glyph rendered brighter than the local background
# (white top-hat). Preserve the original low-saturation front end, which can
# score better on pale backgrounds than a wider color filter.
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
_TINTED_MAX_SATURATION = 100
_TINTED_CORNER_FRACTION = 0.10
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Shape-consistent detection: match the bundled alpha glyph silhouette against the
# corner candidate via TM_CCOEFF_NORMED (keys on glyph SHAPE, not coverage).
# NOTE: this gate is FRONT-END SPECIFIC. The continuous top-hat front-end scores
# higher overall than the binary one (mean 0.809 vs 0.723 on the same 90 upstream
# positives), so the binary-era 0.40 left the provenance-relaxed gate (x0.7) far too
# low and admitted false fires. Calibrated upstream on the 240-image unbiased recall
# sample, full auto path:
#
#   gate   relaxed   recall   precision   true   false
#   0.40     0.280      96%         91%     86       8
#   0.45     0.315      94%         93%     85       6
#   0.50     0.350      92%         99%     83       1   <- chosen
#   0.60     0.420      87%         99%     78       1
#
# 0.50 beats the binary front-end on recall (92% vs 89%) at identical precision (99%),
# which is the only reason the front-end switch is worth it. Do not port this number to
# a binary-front-end mark; re-calibrate per front-end.
DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.50

# Detection-silhouette geometry, sized from the shipped renderer-synthetic outline
# asset (346x77 px): sizes the glyph silhouette for the TM_CCOEFF_NORMED detection
# match; Doubao also reuses the aligned alpha as its sparse removal footprint.
_ALPHA_WIDTH_FRAC = 0.1636
_ALPHA_HEIGHT_FRAC = 0.1636 * 77 / 346

# The captured alpha is also the removal footprint after the detector aligns it.
# Keep faint anti-aliased edges, then grow by one pixel so cv2 does not leave a halo.
# A solid enclosing rectangle is unsafe here: the mark sits close to two frame edges,
# and a textured branch in the canonical sample made the thresholded response reach
# both edges before OpenCV inpainted the whole block into triangular wedges.
_FOOTPRINT_ALPHA_FLOOR = 0.05
_FOOTPRINT_DILATE = 1

_CONFIG = TextMarkConfig(
    name="Doubao",
    asset_name="doubao_outline_alpha.png",
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
    scale_basis="short",  # measured: recovers 56% of landscape misses (see scale_base)
    # No rival margin: measured 2026-07-18, the symmetric gate cost Doubao 7 genuine
    # detections to prevent 5 false ones (1.4:1 against). Doubao's absolute detector
    # is already 86% precise, so it has nothing to buy; Jimeng's is 38% and gains 25pp
    # for free. The confusion is asymmetric, so the remedy is too.
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
)

# A translucent white mark over yellow/orange imagery can retain enough of the
# background color to miss the original 55-level filter. Search with a wider
# filter only when the corner is substantially colorful. Keep already accepted
# detections and otherwise retain the stronger of the two scans.
_TINTED_ENGINE = TextMarkEngine(dataclasses.replace(_CONFIG, max_saturation=_TINTED_MAX_SATURATION))


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Doubao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "豆包AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Doubao glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class DoubaoEngine(TextMarkEngine):
    """Detect/localize the visible Doubao "豆包AI生成" watermark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def _scan(self, image: NDArray[Any] | None) -> TextMarkScan:
        base = super()._scan(image)
        if base.loc is None or image is None or (base.score or 0) >= DETECT_NCC_THRESHOLD:
            return base

        x, y, w, h = base.loc.bbox
        corner = image_io.to_bgr(image[y : y + h, x : x + w])
        channel_spread = corner.max(axis=2) - corner.min(axis=2)
        colored = np.count_nonzero(channel_spread >= _TINTED_MAX_SATURATION)
        if colored < _TINTED_CORNER_FRACTION * channel_spread.size:
            return base

        tinted = _TINTED_ENGINE._scan(image)
        return tinted if (tinted.score or 0) > (base.score or 0) else base

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Return the detector-aligned Doubao glyph footprint.

        The continuous top-hat response is suitable for locating the wordmark but is
        not a safe removal mask: unrelated bright texture can join the response and
        enlarge its bounding rectangle. The detector already carries the winning
        template box, so resize the alpha to that exact box and mask only its
        glyphs. Explicit ``force`` has no trustworthy alignment and retains the shared
        geometry-box behavior.
        """
        if force:
            return super().footprint_mask(image, force=True, dilate=dilate, detection=detection)
        if image is None or image.size == 0:
            return None

        image = image_io.to_bgr(image)
        det = detection if detection is not None else self.detect(image)
        if not det.detected or det.match_box is None:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)
        alpha = _text_mark_engine.load_alpha_template(self.config.asset_name)
        if alpha is None:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)

        radius = _FOOTPRINT_DILATE if dilate is None else max(0, dilate)
        return self._aligned_alpha_mask(
            image,
            det,
            alpha,
            alpha_floor=_FOOTPRINT_ALPHA_FLOOR,
            dilate=radius,
        )
