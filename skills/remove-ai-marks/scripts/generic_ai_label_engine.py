# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/generic_ai_label_engine.py, Apache-2.0; modified: the Doubao rival asset name follows this tree's renderer-synthetic ``doubao_outline_alpha.png`` (the upstream capture-solved asset is forbidden here — see assets/PROVENANCE.md), scripts package imports are flattened to this repo's stdlib script package, and the registry-facing mentions are dropped (the registry is a later ticket here); upstream research citations and calibration tables are trimmed to the load-bearing numbers.
"""Generic bare "AI生成" text-mark detector/localizer (brand-less TC260 fallback).

Every other text-mark engine in this package is tuned to one vendor's exact wordmark
silhouette ("豆包AI生成" for Doubao, and so on). This engine instead catches a BARE
"AI生成" (or "AI generated") corner strip with **no brand wordmark in front of it** --
the China GB 45438-2025 / TC260 explicit-AIGC label stamped by OS- and gallery-level
AI-edit tools that do not carry a distinct generator brand of their own, as opposed to
a named generator product like Doubao or Kling.

The label is compliance-driven, not brand-driven, so several manufacturers' gallery
apps plausibly render the exact same "AI生成" string with only styling differences
(font, position, background pill) -- which is why this is one generic fallback keyed
on the bare text, not a family of near-duplicate per-OEM configs.

Detection reuses the shared physical assumption every other text mark in this family
already encodes (``MAX_SATURATION`` / ``LOGO_MIN_LUMA`` / ``TOPHAT_DELTA``): a light,
low-saturation glyph rendered brighter than its local background, mandated by
GB 45438-2025's own "light-colored watermark" house style, not a vendor-specific
measurement. Everything else here is UNMEASURED and synthetic: the mark has no logo
element to capture -- only literal text -- so the shared capture-less font-rendering
pipeline (``render_vendor_silhouettes.py``) renders "AI生成" directly as the
alpha-template asset (``assets/generic_ai_label_alpha.png``), which plugs into the
same :class:`TextMarkConfig` machinery every other mark uses.

CALIBRATION CAVEAT: ``DETECT_NCC_THRESHOLD`` below is fit on a SYNTHETIC corpus only
(the rendered glyph composited onto generated backgrounds, scored against generated
clean negatives and the existing brand marks' own composited examples as a
false-positive/rival check). Treat this detector as WEAKER evidence than the
brand-tuned engines until it is recalibrated on independently labeled real captures;
the ``rivals`` margin below exists specifically so a genuine
Doubao/Qwen/Baidu/Kling/Yuanbao mark -- every one of which literally contains the
substring "AI生成" in its own glyphs -- is attributed to its own tuned engine
instead of double-firing here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import _text_mark_engine
import image_io
from _text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (unmeasured basis -- the default every
# unconfirmed vendor starts from; see scale_basis below). The box must comfortably
# exceed the glyph at the TOP of the search ladder (alpha_*_frac * max(ladder)) or the
# search crop clips the glyph before NCC ever sees it -- at this asset's shape that
# floor is 0.1724*1.25=0.216 / 0.0625*1.25=0.078; these fractions keep ~1.3x headroom
# above it, matching the other bottom-right marks' convention.
WM_WIDTH_FRAC = 0.28
WM_HEIGHT_FRAC = 0.10
MARGIN_RIGHT_FRAC = 0.010
MARGIN_BOTTOM_FRAC = 0.010

# Glyph appearance: shared TC260 "light watermark" house style (see module docstring),
# not a vendor-specific measurement -- identical to Doubao/Jimeng/Kling.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

# Shape-consistent detection via the continuous top-hat front-end (TM_CCOEFF_NORMED
# against the rendered "AI生成" silhouette). Calibrated SYNTHETICALLY (see the module
# docstring's caveat); upstream's sweep on that corpus:
#
#   gate   margin   recall   precision   TP    clean_fp   rival_fp
#   0.45    0.10      67%       100%     24        0        0
#   0.50    0.10      67%       100%     24        0        0   <- chosen
#   0.55    0.10      42%       100%     15        0        0
#
# 0.50 over the tied 0.45 (identical recall/precision): a synthetic clean-background
# removal leaves a faint TELEA-inpainting residual that still correlates ~0.45-0.46
# against this template, so 0.45 leaves detection re-firing on the library's OWN
# removal output with no headroom. 0.50 costs nothing on this corpus and clears that
# residual.
#
# The plain margin=0.10 shared by every other mark already separates this template
# from the rivals' binarized glyph blobs once the absolute gate clears 0.45, so no
# widened rival margin is needed.
DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.50

# Never relaxed under provenance: this mark cannot be attributed to one vendor, so
# there is no producer identity that could confirm it the way a TC260 USCC confirms
# the brand marks, and the synthetic-only calibration has no measured sub-gate band
# to relax into safely.
PROVENANCE_NCC_FACTOR = 1.0

# Detection-silhouette geometry, emitted by render_vendor_silhouettes.py
# (font-rendered, not captured) at this reference width.
_ALPHA_WIDTH_FRAC = 0.1724
_ALPHA_HEIGHT_FRAC = 0.0625

# The rendered alpha is also the removal footprint after the detector aligns it,
# mirroring Doubao/Kling: a solid enclosing rectangle risks reaching frame edges on
# textured backgrounds, so the sparse glyph footprint is safer for an
# as-yet-uncalibrated mark.
_FOOTPRINT_ALPHA_FLOOR = 0.05
_FOOTPRINT_DILATE = 1

# Rival marks sharing the bottom-right corner whose OWN glyphs contain "AI生成" as a
# substring (Doubao "豆包AI生成", Baidu/Kling/Qwen/Yuanbao's Chinese text runs).
# Jimeng is excluded on purpose: its product wordmark is a distinct mark without that
# substring and scores low against this template. Without this margin, this generic
# detector would fire a SECOND time on every rival's mark, which are already
# correctly attributed by their own tuned engines.
_RIVALS = (
    "doubao_outline_alpha.png",
    "qwen_alpha.png",
    "baidu_alpha.png",
    "kling_alpha.png",
    "yuanbao_alpha.png",
)

_CONFIG = TextMarkConfig(
    name="Generic AI-generated label",
    asset_name="generic_ai_label_alpha.png",
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
    scale_basis="width",
    rivals=_RIVALS,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    provenance_ncc_factor=PROVENANCE_NCC_FACTOR,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled generic-label alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the generic-label glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class GenericAiLabelEngine(TextMarkEngine):
    """Detect/localize a brand-less bare "AI生成" text mark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)

    def footprint_mask(
        self,
        image: NDArray[Any] | None,
        *,
        force: bool = False,
        dilate: int | None = None,
        detection: TextMarkDetection | None = None,
    ) -> NDArray[Any] | None:
        """Return the detector-aligned generic-label glyph footprint.

        Mirrors :meth:`DoubaoEngine.footprint_mask`: the continuous top-hat response
        locates the mark, but the rendered alpha resized onto the detector's winning
        template box is the safer removal mask on textured backgrounds. ``force`` falls
        back to the shared geometry-box behavior (no trustworthy alignment).
        """
        if force:
            return super().footprint_mask(image, force=True, dilate=dilate, detection=detection)
        if image is None or image.size == 0:
            return None

        image = image_io.to_bgr(image)
        det = detection if detection is not None else self.detect(image)
        if not det.detected or det.match_box is None:
            return super().footprint_mask(image, force=False, dilate=dilate, detection=det)
        alpha = _alpha_template()
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
