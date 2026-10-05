"""Combine independently gated text layouts and retain their mask geometry.

A vendor mark can re-render its label as several layouts (the calibrated wordmark, an
outlined variant, a compact restack, a second language) that share one corner but
detect and mask differently. A variants engine composes a base
:class:`_text_mark_engine.TextMarkEngine` with a tuple of sibling engines: detection
returns the best layout verdict at each trust level, and the removal mask is built by
whichever layout won, so each footprint keeps its own geometry.

All layout tuning lives in the sibling engines' :class:`TextMarkConfig` objects;
nothing here is vendor-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
from _text_mark_engine import (
    BottomRightAnchoredEngine,
    TextMarkConfig,
    TextMarkDetection,
    TextMarkEngine,
    TextMarkLocation,
    best_detection,
    load_alpha_template,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TextMarkVariantsMixin(TextMarkEngine):
    """Add independently configured layouts to an existing text-mark engine."""

    _variants: tuple[TextMarkEngine, ...] = ()
    config: TextMarkConfig

    def detect_both(
        self, image: NDArray[Any] | None
    ) -> tuple[TextMarkDetection, TextMarkDetection]:
        pairs = [
            super().detect_both(image),
            *(engine.detect_both(image) for engine in self._variants),
        ]
        return best_detection(*(pair[0] for pair in pairs)), best_detection(
            *(pair[1] for pair in pairs)
        )

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> TextMarkDetection:
        return self.detect_both(image)[int(provenance)]

    def variant_engine(self, detection: TextMarkDetection) -> TextMarkEngine:
        for engine in self._variants:
            if engine.config.asset_name == detection.template_asset:
                return engine
        return self

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        if image is None or image.size == 0 or force:
            return super().footprint_mask(image, force=force, dilate=dilate, detection=detection)
        det = detection if detection is not None else self.detect(image)
        engine = self.variant_engine(det)
        if engine is self:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)
        return engine.footprint_mask(image, force=False, dilate=dilate, detection=det)


class GlyphTextMarkEngine(TextMarkEngine):
    """Use the accepted template's aligned strokes instead of a textured ROI.

    Bright scene texture behind a bottom-edge mark can expand a solid mask to the
    frame edges, where classical inpainting has no outside context. When the mark's
    alpha asset is trustworthy, the detector's winning match already fixes the
    template's position and scale, so the aligned strokes are the tightest honest
    footprint.
    """

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        if force:
            return super().footprint_mask(image, force=True, dilate=dilate, detection=detection)
        if image is None or image.size == 0:
            return None
        det = detection if detection is not None else self.detect(image)
        alpha = load_alpha_template(self.config.asset_name)
        if alpha is None or not det.detected:
            return None
        return self._aligned_alpha_mask(
            image,
            det,
            alpha,
            alpha_floor=0.05,
            dilate=2 if dilate is None else max(0, dilate),
        )


class BottomRightGlyphTextMarkEngine(GlyphTextMarkEngine, BottomRightAnchoredEngine):
    """Glyph-mask variant whose accepted match must hug the bottom-right corner."""

    _ANCHOR_MAX_RIGHT = 0.04
    _ANCHOR_MAX_BOTTOM = 0.04

    def _ladder_best(
        self, image: NDArray[Any], loc: TextMarkLocation
    ) -> tuple[float, tuple[int, int, int, int] | None]:
        """Find the strongest match among placements that satisfy the anchor.

        The unanchored sweep can win a HIGHER score on an off-corner placement of the
        same silhouette, which the post-gate would only demote after the fact -- and
        demotion loses the match box a faint mark needs for its footprint. Constraining
        the sweep to anchored placements keeps the win inside the corner instead.
        """
        response = self._detect_response(image, loc)
        silhouette = self._glyph_silhouette()
        if response is None or silhouette is None:
            return 0.0, None
        frame_h, frame_w = image.shape[:2]
        anchor_base = min(frame_h, frame_w)
        base = self.scale_base(image)
        best_score = 0.0
        best_box: tuple[int, int, int, int] | None = None
        for scale in self.config.ladder:
            glyph_w = max(self.config.min_gw, int(self.config.alpha_width_frac * base * scale))
            glyph_h = max(4, int(self.config.alpha_height_frac * base * scale))
            if glyph_w >= response.shape[1] or glyph_h >= response.shape[0]:
                continue
            template = cv2.resize(silhouette, (glyph_w, glyph_h), interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(response, template, cv2.TM_CCOEFF_NORMED)
            min_x = max(0, round(frame_w - loc.x - glyph_w - self._ANCHOR_MAX_RIGHT * anchor_base))
            max_x = min(scores.shape[1] - 1, frame_w - loc.x - glyph_w)
            min_y = max(0, round(frame_h - loc.y - glyph_h - self._ANCHOR_MAX_BOTTOM * anchor_base))
            max_y = min(scores.shape[0] - 1, frame_h - loc.y - glyph_h)
            if min_x > max_x or min_y > max_y:
                continue
            anchored = scores[min_y : max_y + 1, min_x : max_x + 1]
            _, score, _, point = cv2.minMaxLoc(anchored)
            if score > best_score:
                x, y = min_x + int(point[0]), min_y + int(point[1])
                best_score = float(score)
                best_box = (x, y, x + glyph_w - 1, y + glyph_h - 1)
        return best_score, best_box


def _box_footprint(
    engine: GlyphTextMarkEngine,
    image: NDArray[Any] | None,
    *,
    force: bool,
    dilate: int | None,
    detection: TextMarkDetection | None,
    pad_left_height: float,
    pad_right_height: float,
) -> NDArray[Any] | None:
    """Mask the full winning text rectangle for outlined or stacked layouts.

    An outlined glyph's sparse strokes leave the outline's INTERIOR unfilled, and an
    inpaint that keeps interior texture reads as a ghost; the enclosing rectangle (plus
    config-driven side pads, a fraction of the match height) is the honest footprint.
    """
    if image is None or image.size == 0 or force:
        return GlyphTextMarkEngine.footprint_mask(
            engine,
            image,
            force=force,
            dilate=dilate,
            detection=detection,
        )
    det = detection if detection is not None else engine.detect(image)
    if not det.detected or det.match_box is None:
        return None
    import region_eraser

    loc = engine.locate(image)
    x0, y0, x1, y1 = det.match_box
    extra_left = round((y1 - y0 + 1) * pad_left_height)
    extra_right = round((y1 - y0 + 1) * pad_right_height)
    pad = 2 if dilate is None else max(0, dilate)
    return region_eraser.boxes_to_mask(
        image.shape[:2],
        [
            (
                loc.x + x0 - extra_left,
                loc.y + y0,
                x1 - x0 + 1 + extra_left + extra_right,
                y1 - y0 + 1,
            )
        ],
        dilate=pad,
    )


class BoxTextMarkEngine(GlyphTextMarkEngine):
    """Text detector with a solid detector-aligned removal footprint."""

    _BOX_PAD_LEFT_HEIGHT = 0.0
    _BOX_PAD_RIGHT_HEIGHT = 0.0

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        return _box_footprint(
            self,
            image,
            force=force,
            dilate=dilate,
            detection=detection,
            pad_left_height=self._BOX_PAD_LEFT_HEIGHT,
            pad_right_height=self._BOX_PAD_RIGHT_HEIGHT,
        )


class BottomRightBoxTextMarkEngine(BottomRightGlyphTextMarkEngine):
    """Anchored text detector with a solid detector-aligned footprint."""

    _BOX_PAD_LEFT_HEIGHT = 0.0
    _BOX_PAD_RIGHT_HEIGHT = 0.0

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        return _box_footprint(
            self,
            image,
            force=force,
            dilate=dilate,
            detection=detection,
            pad_left_height=self._BOX_PAD_LEFT_HEIGHT,
            pad_right_height=self._BOX_PAD_RIGHT_HEIGHT,
        )
