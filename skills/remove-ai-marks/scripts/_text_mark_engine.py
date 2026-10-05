"""Shared base for visible text-mark detectors and localizers.

Each mark supplies a :class:`TextMarkConfig` with its silhouette, expected area,
scale ladder, detector frontend, and calibrated gates. The shared engine locates
the candidate, scores the silhouette with normalized correlation, and builds a
removal footprint handed to the shared region-eraser fill. Individual engines may
override detection or footprint behavior when their measured variant requires it;
all per-mark tuning lives in the config, never in this module.

The removal path never performs reverse-alpha pixel recovery. This module is on
the pixel path: it imports numpy and OpenCV at import time (like the region
eraser), so only pixel-extra code should import it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum image short side (px) for text-mark DETECTION. Below this the glyph
# template degrades to the ``min_gw`` floor (~8 px) and TM_CCOEFF_NORMED on a few
# pixels is noise, so an unrelated small geometric shape can spuriously correlate
# with the CJK silhouette (upstream 2026-06-26 FP: a 48x48 app icon scored real
# marks 0.41-0.47, above their thresholds; upscaled it collapses to ~0.06-0.10).
# Detection is skipped (verdict stays "not detected", the safe default) rather
# than risk a false positive; removal is gated on detection, so it is suppressed
# too.
_MIN_DETECT_SHORT_SIDE = 200

# Provenance-confirmed NCC relaxation. When external metadata already confirms the
# vendor (so the mark is present with high prior), a faint or slightly re-rendered
# glyph that scores just below the standard NCC gate is still trusted. The relaxed
# gate is ``detect_ncc_threshold * provenance_ncc_factor``; the coverage gate still
# applies on top.
#
# This is a DEFAULT, not a shared constant: upstream measured that one shared
# relaxation factor meant two different things per mark (doubao 76% precision at
# 0.70, jimeng 17% at the same number), so the factor is per-mark config. Vendor
# tickets must calibrate their own factor before shipping a relaxation.
_DEFAULT_PROVENANCE_NCC_FACTOR = 0.7


@dataclass(frozen=True)
class TextMarkConfig:
    """All per-mark tuning for a text-mark detector/localizer."""

    name: str  # short label for log lines (e.g. "Doubao")
    asset_name: str  # bundled alpha PNG under assets/ (e.g. "kling_alpha.png")
    corner: Literal[
        "br", "bl", "tl", "tr", "bc", "cc"
    ]  # br (Doubao/Jimeng), bl (Samsung), tl (RunningHub), tr (Microsoft), bc (LiblibAI), cc (OpenArt)
    margin_floor: int  # min margin in px for locate (4 for br marks, 2 for Samsung)
    # locate geometry (fraction of scale_base -- see TextMarkEngine.scale_base)
    width_frac: float
    height_frac: float
    margin_x_frac: float  # right margin (br/tr) or left margin (bl/tl)
    margin_bottom_frac: float
    # glyph appearance
    max_saturation: float
    logo_min_luma: float
    tophat_delta: float
    morph_open_size: int  # MORPH_OPEN kernel side (5 for br marks, 3 for Samsung)
    # detection
    detect_min_coverage: float
    detect_ncc_threshold: float
    # alpha-map glyph geometry (fraction of scale_base) sizing the detection
    # silhouette for template_match_score
    alpha_width_frac: float
    alpha_height_frac: float
    min_gw: int  # minimum glyph width for the template match (8 br, 16 Samsung)
    # Asset names of RIVAL marks that occupy the same corner and can therefore be
    # scored against the same glyph blob. Detection becomes COMPETITIVE: this mark's
    # template must beat every rival's by ``rival_margin``. See
    # TextMarkEngine._rival_margin_ok.
    # Detection front-end. "binary" thresholds the top-hat into a glyph blob and
    # correlates a binary silhouette against it; "tophat" correlates the CONTINUOUS
    # top-hat response against a soft template and never binarizes. See
    # TextMarkEngine.tophat_response. "gray" correlates the silhouette against the
    # raw GRAYSCALE of the locate box: for a faint mid-gray mark the top-hat's
    # background subtraction and max-normalization suppress the response, while raw
    # gray NCC separates. Contrast-DEPENDENT, unlike tophat. "contrast" correlates
    # against the ABSOLUTE local-luma residual, for a mark whose renderer switches
    # between light-on-dark and dark-on-light while preserving one silhouette.
    detect_frontend: Literal["binary", "tophat", "gray", "contrast"] = "binary"
    # Gaussian sigma applied to the template in the "tophat" front-end (0 = none).
    template_blur: float = 0.0
    # Which image dimension the mark's size and margins scale with. VENDOR-SPECIFIC,
    # measured, not assumed -- see TextMarkEngine.scale_base. "short" = min(h, w),
    # "width" = w, "long" = max(h, w).
    scale_basis: Literal["short", "width", "long"] = "width"
    # Scale rungs the ladder sweep covers (the detection comb). PER-MARK: a vendor
    # whose stamp sizes do not land on the shared 3-rung comb carries its own
    # ladder. Densifying the SHARED ladder for everyone was measured and rejected
    # upstream (false fire 2.52% -> 3.05%), so the default stays 3 rungs and a
    # deviation must be calibrated per mark, never ported.
    ladder: tuple[float, ...] = (0.8, 1.0, 1.25)
    rivals: tuple[str, ...] = ()
    rival_margin: float = 0.10
    # Multiplier applied to detect_ncc_threshold when provenance confirms the vendor.
    # Per-mark, NOT shared: see _DEFAULT_PROVENANCE_NCC_FACTOR. Last field so it can
    # carry a default.
    provenance_ncc_factor: float = _DEFAULT_PROVENANCE_NCC_FACTOR


@dataclass
class TextMarkLocation:
    """Located watermark box, in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class TextMarkDetection:
    """Result of visible text-mark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels
    # ROI-local (x0, y0, x1, y1) of the template's best match, in the LOCATED BOX's
    # coordinates. Continuous front ends get it from the ladder sweep; ``binary``
    # gets it from its single nominal-size match. None whenever no template matched.
    # ``footprint_mask`` bounds the fill with it, so carrying it here is what stops
    # the mask path re-running a search the detector already ran.
    match_box: tuple[int, int, int, int] | None = None
    # The trust level this detection was taken at, mirroring detect()'s
    # ``provenance``. ``footprint_mask`` reuses a threaded detection only when it
    # matches the STRICT level its own re-detect would have used -- see
    # TextMarkEngine._strict_detection.
    provenance: bool = False
    template_asset: str | None = None


def best_detection(*detections: TextMarkDetection) -> TextMarkDetection:
    """Prefer an accepted variant, then retain the strongest rejected score."""
    return max(detections, key=lambda detection: (detection.detected, detection.confidence))


@dataclass
class TextMarkScan:
    """The trust-level-BLIND half of text-mark detection, reusable across both levels.

    ``loc is None`` means detection stopped before any scan (empty or too-small image).
    ``score is None`` means the binary front-end fell below its coverage gate, which is
    a verdict of "not detected, confidence 0.0" without consulting the rival margin.
    """

    loc: TextMarkLocation | None
    box: NDArray[Any] | None  # box-sized binary glyph mask
    base: int  # scale_base(image)
    frame: tuple[int, int] = (0, 0)  # (h, w) of the scanned image
    coverage: float = 0.0
    score: float | None = None
    match_box: tuple[int, int, int, int] | None = None


# Alpha / silhouette templates, cached per asset name. This shared cache lets every
# text-mark engine reuse the loader without re-reading an asset. Only SUCCESSFUL loads
# are cached, so a missing asset is retried.
_alpha_cache: dict[str, NDArray[Any]] = {}
_silhouette_cache: dict[str, NDArray[Any]] = {}


def load_alpha_template(asset_name: str) -> NDArray[Any] | None:
    """Lazily load the bundled alpha template (float [0,1]) for ``asset_name``, or None.

    A missing (or unreadable) asset is a stable ``None``, never a raise: engines are
    constructed from configs and a typo'd asset name must degrade to "no detection",
    not crash the identify path.
    """
    cached = _alpha_cache.get(asset_name)
    if cached is not None:
        return cached
    path = Path(__file__).parent / "assets" / asset_name
    img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.debug("alpha template %s unavailable", asset_name)
        return None
    _alpha_cache[asset_name] = img.astype(np.float32) / 255.0
    return _alpha_cache[asset_name]


def glyph_silhouette(asset_name: str) -> NDArray[Any] | None:
    """Binary glyph silhouette (255 = glyph) from the bundled alpha map, or None."""
    cached = _silhouette_cache.get(asset_name)
    if cached is not None:
        return cached
    at = load_alpha_template(asset_name)
    if at is None:
        return None
    _silhouette_cache[asset_name] = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache[asset_name]


# Rival-resolution wiring: maps a rival ASSET NAME to the module that carries its
# engine (and therefore its config). This is wiring, not detection logic -- the
# mark's own tuning always lives in its config. Looked up LAZILY so engine modules
# (which import this one) never form an import cycle; a module that is missing or
# renamed degrades to "no rival margin" instead of breaking detection. An engine
# module whose name does not follow this table calls :func:`register_rival_config`
# at import time instead.
_RIVAL_MODULES = {
    "doubao_outline_alpha.png": "doubao_engine",
    "jimeng_label_alpha.png": "jimeng_engine",
    "qwen_alpha.png": "qwen_engine",
    "samsung_ko_alpha.png": "samsung_engine",
    "samsung_en_alpha.png": "samsung_engine",
    "baidu_alpha.png": "baidu_engine",
    "kling_alpha.png": "kling_engine",
    "yuanbao_alpha.png": "yuanbao_engine",
    "generic_ai_label_alpha.png": "generic_ai_label_engine",
}

# Registered rival configs, keyed by asset name. Registration WINS over the lazy
# module lookup, so a vendor engine can pin its exact config object here.
_RIVAL_CONFIGS: dict[str, TextMarkConfig] = {}


def register_rival_config(asset_name: str, config: TextMarkConfig) -> None:
    """Register a mark's config so same-corner rivals can score it (engine-module API).

    Vendor engine modules call this at import time with their own ``_CONFIG``; the
    shared engine then never needs to know vendor module layout.
    """
    _RIVAL_CONFIGS[asset_name] = config


def _rival_config(asset_name: str) -> TextMarkConfig | None:
    """The rival mark's own config, for scoring its template on a shared blob.

    A rival's template geometry (``alpha_*_frac`` / ``min_gw``) is its own, and
    scoring it with this mark's geometry would compare a correctly-sized template
    against a mis-sized one and hand the margin a free win. Lazy because the engine
    modules import this one.
    """
    registered = _RIVAL_CONFIGS.get(asset_name)
    if registered is not None:
        return registered
    mod_path = _RIVAL_MODULES.get(asset_name)
    if mod_path is None:
        return None
    from importlib import import_module

    try:
        config = getattr(import_module(mod_path), "_CONFIG", None)
    except Exception:  # a missing/renamed engine must not break detection
        logger.debug("rival module %s unavailable; skipping its margin check.", mod_path)
        return None
    if isinstance(config, TextMarkConfig):
        return config
    return None


def _template_match_best(
    box_mask: NDArray[Any], scale_base: int, config: TextMarkConfig
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Best nominal-size template score and its inclusive ROI-local box.

    ``TM_CCOEFF_NORMED`` keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual glyph shape does.

    ``scale_base`` is the mark's own scaling dimension
    (:meth:`TextMarkEngine.scale_base`), not always the width: sizing the template on
    the wrong basis stretches it by the aspect ratio on landscape inputs and the
    correlation collapses. The template is clamped to the box: a miscalibrated
    ``alpha_*_frac`` (or a tiny locate box) must shrink the match, not allocate a
    template larger than the searched image.
    """
    sil = glyph_silhouette(config.asset_name)
    if sil is None or box_mask.size == 0:
        return (0.0, None)
    gw = min(box_mask.shape[1] - 1, max(config.min_gw, int(config.alpha_width_frac * scale_base)))
    gh = min(box_mask.shape[0] - 1, max(4, int(config.alpha_height_frac * scale_base)))
    if gw < config.min_gw or gh < 4:
        return (0.0, None)
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    result = cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(result)
    x, y = int(top_left[0]), int(top_left[1])
    return (float(score), (x, y, x + gw - 1, y + gh - 1))


def template_match_score(box_mask: NDArray[Any], scale_base: int, config: TextMarkConfig) -> float:
    """Zero-mean normalized correlation of the expected glyph against ``box_mask``."""
    return _template_match_best(box_mask, scale_base, config)[0]


class TextMarkEngine:
    """Visible text-mark detector/localizer (locate -> mask -> detect; mask feeds the fill)."""

    def __init__(self, config: TextMarkConfig) -> None:
        self.config = config

    # ── Templates (delegate to the asset-keyed module cache) ────────────

    def _alpha_template(self) -> NDArray[Any] | None:
        return load_alpha_template(self.config.asset_name)

    def _glyph_silhouette(self) -> NDArray[Any] | None:
        return glyph_silhouette(self.config.asset_name)

    def _template_match_score(self, box_mask: NDArray[Any], scale_base: int) -> float:
        return template_match_score(box_mask, scale_base, self.config)

    def _rival_margin_ok(self, score: float, box_mask: NDArray[Any], scale_base: int) -> bool:
        """Whether this mark's template beats every same-corner RIVAL's on the SAME blob.

        Detection was purely ABSOLUTE -- each engine scored its own template and
        compared against its own threshold, so nothing ever asked the discriminative
        question "does this blob look more like the neighbour's mark than like mine?".
        Two marks sharing a corner and a script (both ByteDance br CJK strips) survive
        binarization into very similar blobs, so an absolute gate cannot separate them.

        Upstream measured separability on hand-labeled examples, scoring BOTH
        templates against the same glyph blob: absolute ncc 0.96 vs
        ``ncc_mine - ncc_rival`` 0.99; a 0.10 margin separated the marks from the
        rival and clean examples without reducing recall. Marks with no same-corner
        rival declare ``rivals=()`` and are unaffected.
        """
        c = self.config
        if not c.rivals:
            return True
        for rival_asset in c.rivals:
            rival = _rival_config(rival_asset)
            if rival is None:
                continue
            if score - template_match_score(box_mask, scale_base, rival) < c.rival_margin:
                logger.debug("%s detect: loses the %s rival margin; rejecting.", c.name, rival_asset)
                return False
        return True

    # ── Locate ──────────────────────────────────────────────────────────

    def _roi_fields(
        self, image: NDArray[Any], loc: TextMarkLocation
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """``(luma, saturation, local_background)`` for the located box, all float32.

        The ROI is normalized to 3-channel BGR first (grayscale / BGRA would break
        ``axis=2``).

        Local background model: a strong Gaussian blur (sigma ~ box height); the white
        top-hat (``luma - local_bg``) lights up bright thin strokes regardless of the
        absolute background level.

        The callers each keep their own ``bh < 16 or bw < 16`` guard: they return three
        different sentinels for a degenerate ROI, so the check cannot move in here.
        """
        x, y, bw, bh = loc.bbox
        roi = image_io.to_bgr(image[y : y + bh, x : x + bw]).astype(np.float32)
        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        sigma = max(4.0, bh * 0.4)  # 0.4 factor and 4.0 floor are calibrated; do not retune
        return luma, sat, cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)

    def tophat_response(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any] | None:
        """The CONTINUOUS white top-hat in the located box -- the glyph signal, unbinarized.

        :meth:`extract_mask` thresholds this same response into a 0/255 glyph blob. That
        is fine for a mark stamped bold and opaque, and destructive for a faint one: a
        thin translucent overlay shatters into specks under the threshold, and no
        template can match a blob that is not there. The gates that were hard cuts in
        the binary path (saturation, absolute luma) become WEIGHTS here, so a faint
        stroke contributes in proportion to its strength instead of being dropped at a
        threshold. The response is max-normalized, which makes the score
        contrast-invariant.

        Kept per-mark (``detect_frontend``) rather than switched globally, because a
        front-end change must be measured per mark before it ships.
        """
        return self._residual_response(image, loc, absolute=False)

    def _residual_response(self, image: NDArray[Any], loc: TextMarkLocation, *, absolute: bool) -> NDArray[Any] | None:
        """Max-normalized uint8 local-luma residual in the located box, saturation-weighted.

        ``absolute=False`` keeps only the POSITIVE side -- the white top-hat, for a mark
        always rendered brighter than its background. ``absolute=True`` takes the
        magnitude, for a renderer that switches between light-on-dark and dark-on-light
        while preserving one silhouette; a one-polarity top-hat misses the latter.
        """
        c = self.config
        _x, _y, bw, bh = loc.bbox
        if bh < 16 or bw < 16:
            return None
        luma, sat, local_bg = self._roi_fields(image, loc)
        residual = luma - local_bg
        resp = (np.abs(residual) if absolute else np.clip(residual, 0, None)) * (sat < c.max_saturation)
        peak = float(resp.max())
        if peak <= 1e-6:
            return None
        return (resp / peak * 255).astype(np.uint8)

    def _detect_response(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any] | None:
        """The uint8 image the ladder sweep correlates the silhouette against, chosen by
        ``TextMarkConfig.detect_frontend``. ``binary`` runs no sweep and never reaches here."""
        frontend = self.config.detect_frontend
        if frontend == "gray":
            x, y, bw, bh = loc.bbox
            if bh < 16 or bw < 16:
                return None
            return cv2.cvtColor(image_io.to_bgr(image[y : y + bh, x : x + bw]), cv2.COLOR_BGR2GRAY)
        if frontend == "contrast":
            return self._residual_response(image, loc, absolute=True)
        if frontend == "tophat":
            return self._residual_response(image, loc, absolute=False)
        raise ValueError(f"{frontend!r} has no ladder response (binary runs no sweep)")

    def _ladder_best(
        self, image: NDArray[Any], loc: TextMarkLocation
    ) -> tuple[float, tuple[int, int, int, int] | None]:
        """Best TM_CCOEFF_NORMED of the mark's silhouette against its front-end response,
        and the ROI-local box (x0, y0, x1, y1) where that best match sits.

        Sweeps the mark's scale ladder: the nominal glyph size is derived from the
        mark's geometry, but a vendor re-rasterization shifts it by a few percent and
        the continuous response is sharp enough that an exact-size template would miss.
        The ladder is per-mark (``TextMarkConfig.ladder``), defaulting to the shipped
        3 rungs.

        Detection and the removal mask BOTH read this one method -- the score gates
        detection, the box bounds the fill. Sharing it is deliberate: the standing rule
        is that detection and the mask use the same front-end, and one method makes the
        drift impossible instead of merely discouraged.

        Tie-breaking is load-bearing: the ``0.0`` seed plus the STRICT ``>`` means the
        EARLIEST ladder rung wins a tie, and a sweep whose maximum is not above 0.0
        returns no box at all.
        """
        c = self.config
        resp = self._detect_response(image, loc)
        sil = self._glyph_silhouette()
        if resp is None or sil is None:
            return (0.0, None)
        base = self.scale_base(image)
        best_score = 0.0
        best_box: tuple[int, int, int, int] | None = None
        for scale in c.ladder:
            gw = max(c.min_gw, int(c.alpha_width_frac * base * scale))
            gh = max(4, int(c.alpha_height_frac * base * scale))
            if gw >= resp.shape[1] or gh >= resp.shape[0]:
                continue
            tmpl = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_AREA)
            if c.detect_frontend == "tophat" and c.template_blur > 0:
                tmpl = cv2.GaussianBlur(
                    tmpl.astype(np.float32), (0, 0), sigmaX=c.template_blur, sigmaY=c.template_blur
                ).astype(np.uint8)
            result = cv2.matchTemplate(resp, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, top_left = cv2.minMaxLoc(result)
            if score > best_score:
                tx, ty = int(top_left[0]), int(top_left[1])
                best_score, best_box = float(score), (tx, ty, tx + gw - 1, ty + gh - 1)
        return (best_score, best_box)

    def scale_base(self, image: NDArray[Any]) -> int:
        """The image dimension this mark's geometry scales with.

        Per-mark, and MEASURED -- a single shared basis is wrong. The tuned fractions
        were all calibrated on PORTRAIT captures, where width and short side coincide,
        so the basis was never exercised until landscape inputs were measured: a
        width-scaled box is inflated by the aspect ratio on a wide image and can miss
        the glyph, while a short-side basis broke marks whose wordmark tracks the
        width. "long" (max of the two sides) is the Microsoft badge's measured basis
        (the pill tracks the RENDER dimension).

        China's GB 45438-2025 clause 5.2(e) mandates glyph height >= 5% of "the
        shortest side" for CN marks, which is why a short-side basis is the natural
        prior -- but measurement wins over the standard's wording, hence per-mark.
        """
        if self.config.scale_basis == "short":
            return min(image.shape[:2])
        if self.config.scale_basis == "long":
            return max(image.shape[:2])
        return image.shape[1]

    def locate(self, image: NDArray[Any]) -> TextMarkLocation:
        """Anchor the watermark box in the configured corner, scaled by ``scale_basis``.

        Every fraction is taken against ``scale_base(image)`` -- see
        :data:`TextMarkConfig.scale_basis`, which is per-mark because the vendors
        genuinely differ. The box is clamped to the frame so a miscalibrated fraction
        or an extreme aspect ratio can never push it outside the image.
        """
        c = self.config
        h, w = image.shape[:2]
        base = self.scale_base(image)
        wm_w = max(40, int(base * c.width_frac))
        wm_h = max(16, int(base * c.height_frac))
        margin_x = max(c.margin_floor, int(base * c.margin_x_frac))
        margin_b = max(c.margin_floor, int(base * c.margin_bottom_frac))
        if c.corner == "br" or c.corner == "tr":
            x = max(0, w - margin_x - wm_w)
        elif c.corner in ("bc", "cc"):  # horizontally centered, margin_x unused
            x = max(0, (w - wm_w) // 2)
        else:
            x = min(margin_x, max(0, w - wm_w))
        # "tl"/"tr" anchor at the top instead: margin_bottom_frac is then the TOP margin.
        if c.corner in ("tl", "tr"):
            y = min(margin_b, max(0, h - wm_h))
        elif c.corner == "cc":  # frame-center: vertically centered, margin_bottom_frac unused
            y = max(0, (h - wm_h) // 2)
        else:
            y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return TextMarkLocation(x=x, y=y, w=wm_w, h=wm_h)

    # ── Mask ────────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any]:
        """Build a box-sized uint8 mask (255 = watermark glyph) for ``loc``.

        Returns just the glyph mask of the located box (shape ``(loc.h, loc.w)``),
        not a full-frame array: every caller immediately crops to ``loc.bbox``, so
        allocating a full ``(h, w)`` mask and embedding the box was O(image) work
        and memory for an O(box) result -- a wasted full-frame uint8 allocation on
        each detect (~12 MB on a 12 MP frame, recomputed per text-mark detector on
        the memory-tight identify path). The box mask is byte-identical to the old
        full-frame mask cropped to ``loc.bbox``.

        Polarity-aware: the mark is a light, low-saturation gray rendered brighter
        than the local background (white top-hat), so a white-paper document is left
        untouched (nothing brighter than its surroundings is masked there).
        """
        c = self.config
        x, y, bw, bh = loc.bbox
        # A degenerate ROI (a sliver from an extremely wide/short image) cannot hold
        # the mark and would feed cv2's GaussianBlur/morphology a ~1-px-tall array,
        # which can fault native code on some platforms. Skip the cv2 pipeline.
        if bh < 16 or bw < 16:
            return np.zeros((bh, bw), np.uint8)
        # Normalize the ROI to 3-channel BGR (grayscale / BGRA would break axis=2).
        roi = image_io.to_bgr(image[y : y + bh, x : x + bw]).astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < c.max_saturation

        # Local background model: a strong Gaussian blur (sigma ~ box height); the
        # white top-hat (luma - local_bg) lights up bright thin strokes regardless
        # of the absolute background level.
        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > c.tophat_delta) & (luma > c.logo_min_luma)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        k = c.morph_open_size
        return cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
