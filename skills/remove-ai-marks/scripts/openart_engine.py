# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/openart_engine.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package and the upstream pyright blanket comment is dropped (this repo lints with ruff). The shipped ``openart_alpha.png`` is the procedural reconstruction regenerated with the pixel extra's OpenCV 5 (see assets/PROVENANCE.md).
"""OpenArt visible watermark detector/localizer.

OpenArt (openart.ai) stamps free-plan generations with a semi-transparent white
"OpenArt" wordmark (a bowtie/infinity icon followed by the brand name) placed
**near the center of the frame**, not in a corner like every other text mark
this repository detects. Neither C2PA, EXIF AI tags, nor the China AIGC/TC260
label were present on the observed carrier -- OpenArt apparently ships no
machine-readable provenance signal alongside this visible mark.

**Calibration is PROVISIONAL, not measured.** No such corpus exists for OpenArt:
there is no committed real capture (see ``render_vendor_silhouettes.py`` /
assets/PROVENANCE.md for the procedural reconstruction), so
``DETECT_NCC_THRESHOLD`` below is set conservatively high rather than measured,
specifically because a center-frame locate box (``corner == "cc"``) sits over far
more varied image content than a corner box, and an uncalibrated false-fire there
destroys pixels in the middle of the photo. Do not lower this threshold or add
OpenArt to any auto-scan precision claim without a real measured corpus. The
registry labels this mark provisional.

Detection matches the bundled (procedurally reconstructed, not captured) glyph
silhouette against the center-frame candidate; removal is the shared **localize
-> fill**, using the default template-free rectangular footprint (no override --
unlike Doubao, this mark does not sit near a frame edge where a solid rectangular
fill risks bleeding into un-inpaintable border texture).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import _text_mark_engine
from _text_mark_engine import TextMarkConfig, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Locate geometry as a fraction of the image SHORT side (portrait is the only
# observed carrier so far; "short" keeps the box a sane size on landscape
# inputs too). Generously oversized relative to the alpha template below so the
# scale ladder and NCC alignment search both have slack; unlike a corner mark
# this box floats free in the frame, so accuracy depends entirely on the NCC
# score, not the corner anchor.
WM_WIDTH_FRAC = 0.55
WM_HEIGHT_FRAC = 0.18
MARGIN_X_FRAC = 0.0  # unused: "cc" centers horizontally regardless of margin_x_frac
MARGIN_BOTTOM_FRAC = 0.0  # unused: "cc" centers vertically regardless of margin_bottom_frac

# Glyph appearance: a light, low-saturation near-white overlay (white top-hat),
# the same rendering class as every other bundled text mark.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 150
TOPHAT_DELTA = 12

# Inert for the "tophat" front-end (the coverage gate only guards the "binary"
# front-end), kept for the dataclass contract.
DETECT_MIN_COVERAGE = 0.02

# UNCALIBRATED -- see the module docstring. Set well above Doubao/Jimeng's
# measured 0.45-0.50 specifically because this mark floats in the frame center
# rather than a corner. Re-measure against a real captured corpus before relying
# on this number for a precision claim.
DETECT_NCC_THRESHOLD = 0.62

# No independent provenance signal is known for OpenArt, so the provenance-relaxed
# gate can never be legitimately reached today; keep it at parity with the strict
# gate rather than inventing a relaxation factor with nothing to calibrate it.
PROVENANCE_NCC_FACTOR = 1.0

# Detection-silhouette geometry for the bundled asset (canvas 360x90, aspect 4:1).
# The asset's own fractions are self-consistent against 360/1024 and 90/1024: at
# scale_base == 1024 the resized template comes back at its native pixel size.
_ALPHA_NATIVE_WIDTH = 1024
_ALPHA_WIDTH_FRAC = 360 / 1024
_ALPHA_HEIGHT_FRAC = 90 / 1024

_CONFIG = TextMarkConfig(
    name="OpenArt",
    asset_name="openart_alpha.png",
    corner="cc",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_X_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    provenance_ncc_factor=PROVENANCE_NCC_FACTOR,
    detect_frontend="tophat",
    scale_basis="short",
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=12,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled (procedurally reconstructed) OpenArt alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "OpenArt" wordmark silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the OpenArt glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class OpenArtEngine(TextMarkEngine):
    """Detect/localize the visible, center-frame OpenArt wordmark (locate -> mask; mask feeds the fill)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
