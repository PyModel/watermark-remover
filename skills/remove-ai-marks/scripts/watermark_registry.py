# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/watermark_registry.py, Apache-2.0; modified: scripts package imports are flattened to this repo's stdlib script package (region_eraser / tc260_verify imported top-level), the TC260 producer registry is tc260_verify's (this tree keeps one TC260 module pair), the aggregate no-mark status uses this repo's ``no_mark`` step-code vocabulary, the gemini engine is NOT registered — its slot reports pending-provenance (its upstream backgrounds are capture-derived; see assets/PROVENANCE.md) — and the wave-3 wan/openart rows are labeled provisional in registry output.
"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Localize -> fill -> validate.** A known mark is removed by LOCALIZING it (a
version-robust detector that returns a binary footprint MASK) and then handing
that mask to ONE shared, swappable fill backend (``region_eraser``: cv2 Telea/NS,
MI-GAN, or big-LaMa). The same detector then checks the result without changing
the mask or running a second fill. No mark carries a reverse-alpha step.

Entries (17 shipped):
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``qwen`` -- Alibaba Cloud Qwen "千问AI生成" text, bottom-right.
  - ``wan`` -- Alibaba Wan logo + wordmark, bottom-right. PROVISIONAL (one
    measured positive upstream; see ``wan_engine.py``).
  - ``kling`` -- Kuaishou Kling AI strip, bottom-right.
  - ``yuanbao`` -- Tencent Yuanbao two-line mark, bottom-right.
  - ``dola`` -- ByteDance Dola "Dola AI" image disclosure, bottom-right.
  - ``workbuddy`` -- Tencent WorkBuddy disclosure, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI Korean/English disclosure, bottom-left.
  - ``jimeng_pill`` -- Jimeng-basic "AI生成" pill or solid label, top-left.
  - ``runninghub`` -- RunningHub text mark, top-left (gray front-end).
  - ``baidu`` -- Baidu "百度 AI生成" text + white tag, bottom-right.
  - ``liblib`` -- LiblibAI wordmark, bottom-center.
  - ``liblib_pill`` -- LiblibAI compact "AI生成" pill, top-left.
  - ``microsoft`` -- measured Microsoft white AI badge or text pill, top-right.
  - ``generic_ai_label`` -- brand-less bare "AI生成" TC260 label, bottom-right.
  - ``openart`` -- OpenArt center-frame wordmark. PROVISIONAL / UNCALIBRATED
    (no measured corpus; see ``openart_engine.py``).

Deferred: ``gemini`` -- its slot reports pending-provenance
(:func:`pending_provenance_marks`); the upstream engine hard-loads
capture-derived backgrounds that are forbidden in this tree.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import tc260_verify

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# NOTE: region_eraser is imported LAZILY (inside fill/preferred_inpaint_backend),
# never at module scope: it pulls cv2, and this module is on the metadata-only
# import path too. The engine table below is names-only for the same reason.

logger = logging.getLogger(__name__)

Region = tuple[int, int, int, int]

# Fill backend for the shared removal path. ``auto`` resolves best-first to the
# highest quality installed model -- LaMa, else MI-GAN, else cv2 (see
# :func:`resolve_backend`); the others force a specific backend.
Backend = Literal["auto", "cv2", "migan", "lama"]

# Detection sensitivity for the removal path -- how much to trust a borderline mark.
#   * ``strict``: high-precision visual gate only; never relaxed, so a clean image
#     is never touched. Lowest recall on faint/moved marks.
#   * ``auto`` (default): relax a mark's gate ONLY when the image carries
#     same-product evidence the mark is there -- metadata provenance for that
#     vendor (TC260 ``ContentProducer`` codes mapped via ``tc260_verify``), or a
#     confidently detected sibling mark of the same product (see
#     :func:`resolve_trust`). No evidence -> stays strict.
Sensitivity = Literal["auto", "strict"]

# The trust level a mark's detection gate is resolved to (see ``resolve_trust``).
Trust = Literal["strict", "confirmed"]


def validate_sensitivity(value: str) -> Sensitivity:
    """Reject an unknown sensitivity LOUDLY instead of silently meaning ``auto``.

    ``Sensitivity`` is a ``Literal``, which is not enforced at runtime, so a
    caller upgrading with a stale value in hand would quietly get ``auto``
    behavior -- a silent semantic change on exactly the release where they most
    need to be told.
    """
    if value not in ("auto", "strict"):
        raise ValueError(f"unknown sensitivity {value!r}; expected 'auto' or 'strict'")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class MarkDetection:
    """Uniform detection result for a known mark (across heterogeneous engines)."""

    key: str
    label: str
    location: str
    detected: bool
    confidence: float
    region: Region
    # The engine's OWN detection object, threaded back to this mark's mask builder
    # so it does not re-run the detector. Opaque here. Excluded from eq/repr so the
    # uniform result stays comparable across engines.
    engine_detection: Any | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Localization:
    """A located mark: its detection verdict plus the full-frame removal mask.

    ``mask`` is a full-frame uint8 array (255 = remove) sized to the image, or None
    when nothing should be removed. ``region`` is the mark's bbox."""

    detected: bool
    confidence: float
    region: Region
    mask: NDArray[Any] | None


@dataclass(frozen=True)
class Context:
    """The evidence + policy the removal arbiter decides against (perception is
    kept separate from this decision). ``sensitivity`` is the caller's intent;
    ``provenance`` is the vendor keys local metadata confirms, the evidence that
    drives ``auto``."""

    sensitivity: Sensitivity = "auto"
    provenance: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        validate_sensitivity(self.sensitivity)


@dataclass(frozen=True)
class Candidate:
    """One mark's PERCEPTION output -- what the engine sees, with NO policy applied.

    Carries the mark's verdict at BOTH trust levels (``detected_strict`` = the
    conservative gate, ``detected_relaxed`` = the gate the engine relaxes to under
    provenance), so the arbiter can pick per mark without re-running detection.

    ``features`` is a generic bag of physical measurements a mark's gate may need
    (the mark owns which it reports via ``KnownMark.features``); e.g. the pill
    supplies ``footprint_flat`` (0/1)."""

    key: str
    label: str
    detected_strict: bool
    detected_relaxed: bool
    features: dict[str, float]
    # The exact perception outputs for the action stage: the first selected mark is
    # still localized on the unchanged input, so threading these avoids a duplicate
    # scan and makes room for the read-only post-removal validation. Optional for
    # policy-only candidates in tests.
    strict_detection: MarkDetection | None = field(default=None, compare=False, repr=False)
    relaxed_detection: MarkDetection | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Decision:
    """The arbiter's verdict for one fired mark: remove it, at the resolved trust
    level (``relax`` feeds the mark's mask build so the fill footprint matches the
    level the mark was accepted at)."""

    candidate: Candidate
    relax: bool


VisibleRemovalStatus = Literal["no_mark", "cleaned", "partial", "unvalidated", "error"]
MarkValidationStatus = Literal["cleaned", "partial", "unvalidated"]


@dataclass(frozen=True)
class MarkError:
    """One mark the auto pass could not process: its localize/fill raised.

    The pass contains the failure to this mark and continues; the aggregate
    status becomes ``error`` so the caller never mistakes a half-pass for a
    clean one."""

    key: str
    label: str
    stage: Literal["localize", "fill"]
    message: str


@dataclass(frozen=True)
class MarkRemovalResult:
    """One visible-mark fill and its read-only post-removal validation.

    ``partial`` means the same detector still accepts an overlapping region after
    the one fill. It is evidence about this engine, not an independent oracle.
    Validation never expands the mask and never runs a second fill."""

    key: str
    label: str
    location: str
    region: Region
    mask_bbox: Region
    backend: Literal["cv2", "migan", "lama"]
    confidence_before: float
    confidence_after: float | None
    status: MarkValidationStatus
    residual_region: Region | None = None
    # The full-frame mask that was actually filled. Not part of identity or repr
    # (an ndarray never is); carried so a pipeline caller can verify outside-mask
    # preservation against the exact pixels that changed, not a bbox proxy.
    mask: NDArray[Any] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class VisibleRemovalResult:
    """Detailed result of one automatic visible-mark pass."""

    image: NDArray[Any] = field(compare=False, repr=False)
    marks: tuple[MarkRemovalResult, ...]
    #: Marks whose localize/fill raised, contained per mark (see
    # :class:`MarkError`). Empty on a full pass.
    errors: tuple[MarkError, ...] = ()

    @property
    def labels(self) -> list[str]:
        """Labels whose masks were filled."""
        return [mark.label for mark in self.marks]

    @property
    def status(self) -> VisibleRemovalStatus:
        """Aggregate status; an engine failure outranks a known residual, which
        outranks unknown, and nothing found is ``no_mark`` (exit codes are the
        caller's business; this module introduces none)."""
        if not self.marks and not self.errors:
            return "no_mark"
        if self.errors:
            return "error"
        if any(mark.status == "partial" for mark in self.marks):
            return "partial"
        if any(mark.status == "unvalidated" for mark in self.marks):
            return "unvalidated"
        return "cleaned"


@dataclass(frozen=True)
class KnownMark:
    """A known visible watermark: where it lives, how to find and mask it.

    Removal is uniform (:meth:`remove`): localize the mark to a mask, then fill
    that mask with the chosen backend. Each mark supplies two cheap cv2/numpy
    callables -- ``_detect`` (verdict + bbox, no mask; the identify-style scan)
    and ``_mask`` (the full-frame footprint mask; removal)."""

    key: str
    label: str
    location: str  # usual place, human-readable ("bottom-right")
    in_auto: bool  # participate in `--mark auto` scanning
    # Product family, for the `auto` cross-mark corroboration: a confidently detected
    # mark relaxes only OTHER marks of the SAME product (different corners, one
    # product -- the Jimeng wordmark + the Jimeng pill). Doubao and Jimeng are BOTH
    # ByteDance but distinct products in the SAME bottom-right corner, so they must
    # NOT cross-relax. REQUIRED, deliberately: a defaulted empty product would let
    # two rows that forgot the field corroborate each other and silently bypass the
    # trust gate.
    product: str
    # The company responsible for the product, deliberately separate from the
    # labeling standard: several manufacturers all use TC260, but that does not make
    # their marks one product family.
    manufacturer: str
    # Which provenance regime this mark's vendor labels under, or None. "tc260" means
    # the vendor stamps the China AIGC label; it does not identify the manufacturer.
    # DECLARATIVE: this field documents and groups rows (tests and reports read
    # it); the trust decision itself only ever keys on evidence keys in
    # ``provenance`` (see ``resolve_trust``).
    label_regime: str | None
    # The sentence a caller reports when THIS mark is the strongest evidence. None
    # for a mark that never names a platform on its own (the weak pills).
    platform: str | None
    _detect: Callable[..., MarkDetection]
    _mask: Callable[..., NDArray[Any] | None]
    # Optional physical-feature probe: the mark's OWN measurements its gate needs
    # (e.g. the pill's footprint flatness). None = no extra evidence needed.
    _features: Callable[..., dict[str, float]] | None = None
    # Metadata signal names that confirm this mark's vendor, and platform substrings
    # that do. Empty means nothing confirms it. LIVE CONFIG:
    # :func:`metadata_evidence` applies exactly these probes to the file's
    # metadata, so a declared signal/token corroborates its mark through
    # ``resolve_trust``.
    provenance_signals: tuple[str, ...] = ()
    provenance_platform_tokens: tuple[str, ...] = ()
    # TC260 ``ContentProducer`` identities that name THIS mark's vendor (USCCs as
    # normalized by ``tc260_verify.uscc_of``, plus the bare product names a few
    # generators write). Derived from tc260_verify.TC260_PRODUCERS; the registry
    # coverage test prevents a TC260 mark from silently omitting its codes.
    # DECLARATIVE snapshot for reports/coverage tests: the runtime mapping rides
    # tc260_verify (``producer_for_code``), and this field must stay in sync
    # with it (the coverage test enforces that).
    tc260_producer_codes: tuple[str, ...] = ()
    # Optional single-pass dual verdict for the arbiter's perception stage. None =
    # fall back to two ``_detect`` calls.
    _detect_both: Callable[..., tuple[MarkDetection, MarkDetection]] | None = None
    # False for a weak detector that may be acted on after corroboration but must
    # not itself grant a sibling the relaxed trust level.
    can_corroborate: bool = True
    # True for a mark whose calibration is explicitly provisional (wave 3): the row
    # still participates, but every report surfaces the label.
    provisional: bool = False

    def features(self, image: NDArray[Any]) -> dict[str, float]:
        """Physical features the mark reports for the arbiter's gate (empty if none)."""
        return self._features(image) if self._features is not None else {}

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        """Detect the mark (verdict + bbox, no mask). ``provenance`` signals that
        external metadata already confirms this vendor, so the engine may relax its
        trust threshold."""
        return self._detect(image, provenance=provenance)

    def detect_both(self, image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
        """``(strict, relaxed)`` from ONE pass over the image.

        ``provenance`` never changes what a detector computes -- only the threshold
        it compares against -- so the expensive scan is shared. A mark without a
        ``_detect_both`` adapter falls back to two calls."""
        if self._detect_both is not None:
            return self._detect_both(image)
        return self.detect(image, provenance=False), self.detect(image, provenance=True)

    def localize(
        self,
        image: NDArray[Any],
        *,
        provenance: bool = False,
        force: bool = False,
        detection: MarkDetection | None = None,
    ) -> Localization:
        """Detect and build the removal mask in one call. ``detection`` lets a
        caller that ALREADY detected this mark at this trust level hand the result
        in rather than pay for the scan twice."""
        det = detection if detection is not None else self.detect(image, provenance=provenance)
        if not (det.detected or force):
            return Localization(det.detected, det.confidence, det.region, None)
        # Pass the (provenance-aware) detection to the mask builder so it does NOT
        # re-detect at a different trust level -- a relaxed detection must not be
        # re-demoted into a None mask (reported-removed-but-unchanged).
        mask = self._mask(image, force=force, detection=det)
        return Localization(det.detected, det.confidence, det.region, mask)

    def remove(
        self,
        image: NDArray[Any],
        *,
        backend: Backend = "auto",
        provenance: bool = False,
        force: bool = False,
        detection: MarkDetection | None = None,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by localize -> fill; returns ``(result, region)`` where
        ``region`` is the removed mark's bbox, or None if nothing was removed.

        NB: callers do NOT use ``region`` to clear alpha on save (upstream issue-#30
        white box).
        """
        loc = self.localize(image, provenance=provenance, force=force, detection=detection)
        if loc.mask is None or not loc.mask.any():
            return image.copy(), None
        return fill(image, loc.mask, backend=backend), (loc.region if loc.detected else None)


# ── Deferred slot ─────────────────────────────────────────────────────────────

# The 18th upstream mark. Its engine hard-loads capture-derived backgrounds
# (``gemini_bg_*.png``), which assets/PROVENANCE.md forbids in this tree, so the
# engine is deferred to a provenance-review follow-up. The slot stays visible so
# callers and reports can name WHY the sparkle is not detected here instead of
# the mark silently not existing.
_PENDING_PROVENANCE_MARKS: dict[str, str] = {
    "gemini": (
        "Google Gemini visible watermark (sparkle), bottom-right: engine deferred — "
        "capture-derived reference assets; pending provenance review"
    )
}


def pending_provenance_marks() -> dict[str, str]:
    """Marks with a registry slot that cannot ship yet, with the reason."""
    return dict(_PENDING_PROVENANCE_MARKS)


def provisional_marks() -> frozenset[str]:
    """Wave-3 marks whose calibration is explicitly provisional (registry rows)."""
    return frozenset(m.key for m in _REGISTRY if m.provisional)


# ── Engine adapters (lazy singletons; engines are cv2-only, no model load) ──

_engines: dict[str, Any] = {}

# key -> (module basename, class name). Only the NAMES live here: ``import_module``
# runs inside :func:`_engine` on first use, so importing this module never pulls
# cv2 through an engine. Same lazy-import shape as
# ``_text_mark_engine._rival_config`` and :func:`fill`.
_ENGINE_CLASS: dict[str, tuple[str, str]] = {
    "doubao": ("doubao_engine", "DoubaoEngine"),
    "jimeng": ("jimeng_engine", "JimengEngine"),
    "qwen": ("qwen_engine", "QwenEngine"),
    "wan": ("wan_engine", "WanEngine"),
    "kling": ("kling_engine", "KlingEngine"),
    "dola": ("dola_engine", "DolaEngine"),
    "workbuddy": ("workbuddy_engine", "WorkBuddyEngine"),
    "yuanbao": ("yuanbao_engine", "YuanbaoEngine"),
    "samsung": ("samsung_engine", "SamsungEngine"),
    "jimeng_pill": ("pill_engine", "PillEngine"),
    "runninghub": ("runninghub_engine", "RunningHubEngine"),
    "baidu": ("baidu_engine", "BaiduEngine"),
    "liblib": ("liblib_engine", "LibLibEngine"),
    "liblib_pill": ("liblib_engine", "LibLibPillEngine"),
    "microsoft": ("microsoft_engine", "MicrosoftEngine"),
    "generic_ai_label": ("generic_ai_label_engine", "GenericAiLabelEngine"),
    "openart": ("openart_engine", "OpenArtEngine"),
}


def _engine(key: str) -> Any:
    if key not in _engines:
        from importlib import import_module

        module_name, class_name = _ENGINE_CLASS[key]  # KeyError(key) for an unknown key
        _engines[key] = getattr(import_module(module_name), class_name)()
    return _engines[key]


def inpaint_model_available() -> bool:
    """True when any ONNX inpaint-model backend (MI-GAN or big-LaMa) can run."""
    import region_eraser

    return region_eraser.migan_available() or region_eraser.lama_available()


_warned_cv2_fallback = False


#: The registry's fill tiers as tiers of the SHARED backend stack
#: (``inpaint_backends.FILL_RESOLVE_ORDER``) -- the single source of order.
#: A stack tier absent here (the pure texture/simple backends) is not a fill
#: this registry executes; the cv2 floor precedes it.
_FILL_STACK_TIERS: dict[str, Literal["lama", "migan", "cv2"]] = {
    "lama": "lama",
    "migan": "migan",
    "cv2-telea": "cv2",
    "cv2-ns": "cv2",
}


def preferred_inpaint_backend() -> Literal["lama", "migan", "cv2"]:
    """Backend the ``auto`` fill resolves to: the shared backend-stack order,
    filtered to the tiers this registry executes (LaMa > MI-GAN > cv2).

    The order comes from ``inpaint_backends.FILL_RESOLVE_ORDER`` projected
    through :data:`_FILL_STACK_TIERS` -- there is no second hardcoded ladder.
    LaMa is the highest-quality inpaint; MI-GAN the lighter learned model; cv2
    the classical no-deps floor and last resort. A one-time quality warning
    fires when ``auto`` falls back to cv2."""
    global _warned_cv2_fallback  # noqa: PLW0603 - deliberate one-shot warning latch (upstream parity)
    import inpaint_backends
    import region_eraser

    for tier in inpaint_backends.FILL_RESOLVE_ORDER:
        mapped = _FILL_STACK_TIERS.get(tier)
        if mapped is None:
            continue
        if mapped == "lama" and region_eraser.lama_available():
            return "lama"
        if mapped == "migan" and region_eraser.migan_available():
            return "migan"
        if mapped == "cv2":
            break  # the classical floor: the stack's first cv2 slot decides
    if not _warned_cv2_fallback:
        _warned_cv2_fallback = True
        logger.warning(
            "No learned-inpaint backend available (onnxruntime not installed); falling back "
            "to the cv2 classical inpaint, which can smear textured or structured backgrounds. "
            "Install the 'lama' (best) or 'migan' (lighter) extra for higher-quality fills."
        )
    return "cv2"


def resolve_backend(backend: Backend) -> Literal["cv2", "migan", "lama"]:
    """Resolve ``auto`` to the preferred installed backend; pass the rest through."""
    if backend == "auto":
        return preferred_inpaint_backend()
    return backend


def fill(image: NDArray[Any], mask: NDArray[Any], *, backend: Backend = "auto") -> NDArray[Any]:
    """The ONE shared, mark-agnostic removal: erase ``mask`` (255 = remove) via the
    chosen inpaint backend. ``auto`` resolves in quality order (see
    :func:`resolve_backend`)."""
    chosen = resolve_backend(backend)
    import region_eraser

    return region_eraser.erase(image, mask=mask, backend=chosen)


# ── Detection adapters (verdict + bbox; no mask work on this path) ──


def _engine_mark_detect(key: str, label: str, location: str) -> Callable[..., MarkDetection]:
    def detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        d = _engine(key).detect(image, provenance=provenance)
        return MarkDetection(
            key, label, location, d.detected, d.confidence, d.region, engine_detection=d
        )

    return detect


def _engine_mark_detect_both(
    key: str, label: str, location: str
) -> Callable[..., tuple[MarkDetection, MarkDetection]]:
    def detect_both(image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
        strict, relaxed = _engine(key).detect_both(image)
        return (
            MarkDetection(
                key,
                label,
                location,
                strict.detected,
                strict.confidence,
                strict.region,
                engine_detection=strict,
            ),
            MarkDetection(
                key,
                label,
                location,
                relaxed.detected,
                relaxed.confidence,
                relaxed.region,
                engine_detection=relaxed,
            ),
        )

    return detect_both


def _engine_mark_mask(key: str) -> Callable[..., NDArray[Any] | None]:
    def mask(
        image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
    ) -> NDArray[Any] | None:
        # Thread the engine's OWN accepted detection into the mask builder: no
        # repeated scan, and the trust verdict selected by the arbiter is preserved.
        return _engine(key).footprint_mask(
            image,
            force=force,
            detection=detection.engine_detection if detection is not None else None,
        )

    return mask


def _text_mark(
    key: str,
    label: str,
    location: str,
    *,
    platform: str,
    manufacturer: str,
    product: str | None = None,
    label_regime: str | None = "tc260",
    provenance_signals: tuple[str, ...] = ("aigc",),
    provenance_platform_tokens: tuple[str, ...] = (),
    provisional: bool = False,
) -> KnownMark:
    """Build a text-mark registry row from its shared detector and mask adapters.

    ``product`` defaults to the key (one mark, one product); pass it only when two
    marks share a product. ``label_regime`` and ``provenance_signals`` default to
    the China-AIGC label because every text mark registered here except Samsung,
    Microsoft, generic_ai_label and OpenArt uses it."""
    return KnownMark(
        key,
        label,
        location,
        True,
        product or key,
        manufacturer,
        label_regime,
        platform,
        _engine_mark_detect(key, label, location),
        _engine_mark_mask(key),
        provenance_signals=provenance_signals,
        tc260_producer_codes=tc260_verify.producer_codes_for_mark(key),
        provenance_platform_tokens=provenance_platform_tokens,
        _detect_both=_engine_mark_detect_both(key, label, location),
        provisional=provisional,
    )


# ── Jimeng-basic "AI生成" pill or solid label (top-left) ──
def _pill_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    d = _engine("jimeng_pill").detect(image, provenance=provenance)
    return MarkDetection(
        "jimeng_pill",
        "Jimeng AI生成 pill",
        "top-left",
        d.detected,
        d.confidence,
        d.region,
        engine_detection=d,
    )


def _pill_detect_both(image: NDArray[Any]) -> tuple[MarkDetection, MarkDetection]:
    strict, relaxed = _engine("jimeng_pill").detect_both(image)

    def wrap(detection: Any) -> MarkDetection:
        return MarkDetection(
            "jimeng_pill",
            "Jimeng AI生成 pill",
            "top-left",
            detection.detected,
            detection.confidence,
            detection.region,
            engine_detection=detection,
        )

    return wrap(strict), wrap(relaxed)


def _pill_mask(
    image: NDArray[Any], *, force: bool = False, detection: MarkDetection | None = None
) -> NDArray[Any] | None:
    engine_detection = detection.engine_detection if detection is not None else None
    return _engine("jimeng_pill").footprint_mask(image, force=force, detection=engine_detection)


def _pill_features(image: NDArray[Any]) -> dict[str, float]:
    """The pill's own gate feature: top-left footprint flatness (1.0 = flat enough
    for an invisible fill), read by the metadata arm of :func:`_keep_pill`."""
    return {"footprint_flat": float(_engine("jimeng_pill").footprint_is_flat(image))}


_REGISTRY: tuple[KnownMark, ...] = (
    _text_mark(
        "doubao",
        "Doubao 豆包AI生成 text",
        "bottom-right",
        platform="ByteDance Doubao (visible 豆包AI生成 mark detected)",
        manufacturer="bytedance",
    ),
    _text_mark(
        "jimeng",
        "Jimeng 即梦AI wordmark",
        "bottom-right",
        platform="ByteDance Jimeng / Dreamina (visible 即梦AI mark detected)",
        manufacturer="bytedance",
    ),
    _text_mark(
        "qwen",
        "Qwen 千问AI生成 text",
        "bottom-right",
        platform="Alibaba Cloud Qwen (visible 千问AI生成 mark detected)",
        manufacturer="alibaba",
    ),
    _text_mark(
        "wan",
        "Wan logo and wordmark",
        "bottom-right",
        platform="Alibaba Wan (visible Wan mark detected)",
        manufacturer="alibaba",
        provisional=True,  # wave 3: one measured positive; see wan_engine.py
    ),
    _text_mark(
        "kling",
        "Kling AI 可灵AI / KlingAI 3.0 text",
        "bottom-right",
        platform="Kuaishou Kling AI (visible 可灵AI mark detected)",
        manufacturer="kuaishou",
    ),
    _text_mark(
        "yuanbao",
        "Tencent Yuanbao 元宝 / AI生成 mark",
        "bottom-right",
        platform="Tencent Yuanbao (visible 元宝 / AI生成 mark detected)",
        manufacturer="tencent",
    ),
    _text_mark(
        "dola",
        "Dola AI image wordmark",
        "bottom-right",
        platform="Dola AI (visible Dola AI disclosure)",
        manufacturer="bytedance",
        label_regime=None,
        provenance_signals=(),
    ),
    _text_mark(
        "workbuddy",
        "AI生成 / WORKBUDDY disclosure",
        "bottom-right",
        platform="WorkBuddy (visible AI content disclosure)",
        manufacturer="tencent",
        label_regime=None,
        provenance_signals=(),
    ),
    # Samsung Galaxy AI is a device editing marker, not a TC260 label.
    _text_mark(
        "samsung",
        "Samsung Galaxy AI text",
        "bottom-left",
        manufacturer="samsung",
        label_regime=None,
        provenance_signals=("samsung_genai",),
        platform="Samsung Galaxy AI (visible AI-generated-content mark detected)",
    ),
    _text_mark(
        "runninghub",
        "RunningHub AI生成 text",
        "top-left",
        platform="RunningHub (visible RunningHub AI生成 mark detected)",
        manufacturer="runninghub",
    ),
    _text_mark(
        "baidu",
        "Baidu 百度 AI生成 text",
        "bottom-right",
        platform="Baidu (visible 百度 AI生成 mark detected)",
        manufacturer="baidu",
    ),
    _text_mark(
        "liblib",
        "LiblibAI wordmark",
        "bottom-center",
        platform="LiblibAI (visible LiblibAI mark detected)",
        manufacturer="liblib",
    ),
    KnownMark(
        "liblib_pill",
        "LiblibAI AI生成 pill",
        "top-left",
        True,
        "liblib",
        "liblib",
        "tc260",
        # The compact pill is generic on its own; the bottom wordmark or LiblibAI
        # metadata is what makes the relaxed verdict actionable.
        None,
        _engine_mark_detect("liblib_pill", "LiblibAI AI生成 pill", "top-left"),
        _engine_mark_mask("liblib_pill"),
        provenance_signals=("aigc",),
        _detect_both=_engine_mark_detect_both("liblib_pill", "LiblibAI AI生成 pill", "top-left"),
        can_corroborate=False,
    ),
    # One measured Microsoft visible-mark variant: a white top-right pill with
    # dark internal shapes. Microsoft's documented feature also permits other
    # variants this detector does not cover.
    _text_mark(
        "microsoft",
        "Microsoft top-right AI badge",
        "top-right",
        manufacturer="microsoft",
        label_regime=None,
        provenance_signals=(),
        platform="Microsoft (visible top-right AI badge detected)",
        provenance_platform_tokens=("microsoft",),
    ),
    # Brand-less fallback: a bare "AI生成" TC260 label with no vendor wordmark
    # (gallery-level AI-edit compliance stamps). label_regime=None and
    # provenance_signals=(): there is no one producer to name, so it opts out of
    # the "tc260" regime tag and out of provenance relaxation (mirrors the
    # engine's provenance_ncc_factor=1.0: never relaxed).
    _text_mark(
        "generic_ai_label",
        "Generic bare AI生成 text (no vendor wordmark)",
        "bottom-right",
        platform="Generic AI-generated content label (bare 'AI生成' mark; specific vendor undetermined)",
        manufacturer="unknown",
        label_regime=None,
        provenance_signals=(),
    ),
    # Frame-center, not a corner; UNCALIBRATED (no measured corpus) and no known
    # metadata signal: label_regime=None, provenance_signals=() mirror Microsoft's
    # non-attributable shape rather than the TC260 default.
    _text_mark(
        "openart",
        "OpenArt wordmark",
        "frame-center",
        platform="OpenArt (visible OpenArt mark detected)",
        manufacturer="openart",
        label_regime=None,
        provenance_signals=(),
        provisional=True,  # wave 3: uncalibrated; see openart_engine.py
    ),
    # Same product as the Jimeng wordmark, so a strict wordmark relaxes it.
    KnownMark(
        "jimeng_pill",
        "Jimeng AI生成 pill",
        "top-left",
        True,
        "jimeng",
        "bytedance",
        "tc260",
        # The capture-less pill is too weak a detector to attribute a platform on
        # its own; the Jimeng wordmark is what names ByteDance.
        None,
        _pill_detect,
        _pill_mask,
        _pill_features,
        _detect_both=_pill_detect_both,
        can_corroborate=False,
    ),
)

# Product family per mark, derived from the registry rows so registering a mark is
# one edit.
_PRODUCT_OF: dict[str, str] = {m.key: m.product for m in _REGISTRY}

# Weak pill detectors cannot vouch for sibling marks. Letting a pill do so produced
# a closed false-positive loop upstream: a clean-image pill fire relaxed the
# wordmark, and that wordmark then bypassed the pill's flatness guard. Keep this
# policy on each registry row so adding another weak companion cannot silently omit
# the safeguard.
_CANNOT_CORROBORATE: frozenset[str] = frozenset(m.key for m in _REGISTRY if not m.can_corroborate)


def _provenance_confirms_product(product: str, provenance: frozenset[str]) -> bool:
    """Whether external provenance names any registered mark of ``product``."""
    return any(_PRODUCT_OF.get(key) == product for key in provenance)


def known_marks() -> tuple[KnownMark, ...]:
    """All registered known visible watermarks (the shipped 17)."""
    return _REGISTRY


def mark_keys() -> list[str]:
    """Keys of all registered marks (for CLI choices)."""
    return [m.key for m in _REGISTRY]


def get_mark(key: str) -> KnownMark:
    """Look up a known mark by key (raises KeyError if unknown)."""
    for m in _REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def detect_marks(
    image: NDArray[Any],
    *,
    include_explicit: bool = True,
    provenance: frozenset[str] = frozenset(),
) -> list[MarkDetection]:
    """Detect every known mark in its usual place.

    ``include_explicit=False`` scans only the ``in_auto`` marks. ``provenance``
    names the vendor keys that external metadata already confirms (TC260 producer
    keys mapped via ``tc260_verify``), so every mark belonging to that product may
    relax its trust gate."""
    return [
        m.detect(image, provenance=_provenance_confirms_product(m.product, provenance))
        for m in _REGISTRY
        if include_explicit or m.in_auto
    ]


def resolve_trust(
    key: str,
    *,
    sensitivity: Sensitivity,
    provenance: frozenset[str],
    strict_keys: set[str],
) -> Trust:
    """The trust level mark ``key``'s detection gate is resolved to.

    The single place that turns the ``sensitivity`` policy + evidence into a
    per-mark level (which the engines consume as ``provenance = level != "strict"``).
    ``strict`` never relaxes. A mark is ``confirmed`` only on same-product evidence
    -- metadata naming any mark of that product, or a confidently strict-detected
    sibling (minus the marks too weak to vouch, :data:`_CANNOT_CORROBORATE`).
    ``provenance`` carries mark KEYS, which come from two collectors: TC260
    producer codes (``tc260_label`` + ``tc260_verify``) and the non-TC260
    metadata evidence of :func:`metadata_evidence` (the rows' own
    ``provenance_signals`` / ``provenance_platform_tokens`` declarations).
    Without that evidence a mark stays ``strict``: there is no path that relaxes a
    gate on anything less than same-product evidence."""
    if sensitivity == "strict":
        return "strict"
    product = _PRODUCT_OF[key]
    confirmed = _provenance_confirms_product(product, provenance) or any(
        _PRODUCT_OF[k] == product for k in strict_keys if k != key and k not in _CANNOT_CORROBORATE
    )
    return "confirmed" if confirmed else "strict"


_XMP_PACKET_RE = re.compile(rb"<x:xmpmeta[\s\S]{0,262144}?</x:xmpmeta>", re.IGNORECASE)


def _metadata_generator_text(data: bytes) -> str:
    """Generator-bearing metadata text for platform-token matching (bounded).

    Only structured metadata fields are scanned -- the values of PNG
    tEXt/iTXt/zTXt generator keys (Software/Creator/parameters) and the XMP
    packet when present -- never the pixel payload or free-text comments, so a
    caption mentioning a vendor cannot corroborate its mark. Malformed PNG
    structures contribute nothing instead of raising.
    """
    parts: list[str] = []
    try:
        import image_meta
        import png_chunks

        if data[:8] == png_chunks.PNG_SIGNATURE:
            for chunk in png_chunks.iter_png_chunks(data):
                if chunk.kind not in (b"tEXt", b"zTXt", b"iTXt"):
                    continue
                for key, value in image_meta._png_text_entries(bytes(chunk.payload), chunk.kind):
                    if key.strip().lower() in image_meta._GENERATOR_TEXT_KEYS:
                        parts.append(value)
    except ValueError:
        pass  # malformed PNG: no generator text to match on
    packet = _XMP_PACKET_RE.search(data)
    if packet is not None:
        parts.append(packet.group(0).decode("latin-1", errors="replace"))
    return "\n".join(parts)


def metadata_evidence(data: bytes) -> frozenset[str]:
    """Mark keys corroborated by the FILE's own metadata (non-TC260 signals).

    This is the runtime consumer of the rows' ``provenance_signals`` and
    ``provenance_platform_tokens`` declarations: the fields name the metadata
    that confirms each vendor, and this function applies exactly those probes.

      * ``samsung_genai`` -- Samsung's genAIType container metadata present
        (:func:`image_meta.samsung_genai_in`);
      * platform tokens -- a case-insensitive hit in the generator-bearing
        metadata text (:func:`_metadata_generator_text`).

    TC260 producer keys are deliberately NOT re-derived here; they ride
    ``tc260_label`` + ``tc260_verify`` in the caller. The returned keys merge
    with those to form the ``provenance`` context ``resolve_trust`` decides
    on, so non-TC260 evidence corroborates through the same trust gate.
    """
    if not data:
        return frozenset()
    keys: set[str] = set()
    signals = {mark.key: mark.provenance_signals for mark in _REGISTRY if mark.provenance_signals}
    if any("samsung_genai" in row_signals for row_signals in signals.values()):
        import image_meta

        if image_meta.samsung_genai_in(data) is not None:
            keys.update(
                key for key, row_signals in signals.items() if "samsung_genai" in row_signals
            )
    tokens = {token.casefold() for mark in _REGISTRY for token in mark.provenance_platform_tokens}
    if tokens:
        haystack = _metadata_generator_text(data).casefold()
        if haystack:
            for mark in _REGISTRY:
                if any(token.casefold() in haystack for token in mark.provenance_platform_tokens):
                    keys.add(mark.key)
    return frozenset(keys)


def tc260_producer_mark(code: str) -> KnownMark | None:
    """Return the registry row for one TC260 ``ContentProducer`` identity.

    Only MAPPED producers resolve (``tc260_verify.producer_for_code``): a label
    written by an unregistered producer names no mark here, so it never relaxes a
    gate."""
    producer = tc260_verify.producer_for_code(code)
    if producer is None or producer.image_mark is None:
        return None
    try:
        return get_mark(producer.image_mark)
    except KeyError:
        return None


def _pill_suppressors() -> set[str]:
    """Other products from the pill's manufacturer whose detection vetoes it."""
    pill = get_mark("jimeng_pill")
    return {
        m.key
        for m in _REGISTRY
        if m.manufacturer == pill.manufacturer and m.product != pill.product
    }


def _keep_pill(
    keys: set[str],
    *,
    provenance: frozenset[str],
    footprint_flat: bool,
    verified_label: bool = False,
) -> bool:
    """Whether to auto-remove the ``AI生成`` pill or verified label.

    Pure decision (the flatness feature is precomputed at perception time and passed
    in). The pill detector is weak and metadata/intent confirms the platform, not
    pill presence, so a naive confirmation-OR gate over-fires on textured ceilings
    and walls that the fill visibly smears. Arms:
      * bottom-right "★ 即梦AI" wordmark fired -> remove the pill unrestricted;
      * TC260 metadata confirms Jimeng (``"jimeng" in provenance``, no wordmark) ->
        remove ONLY when the solid label template verifies the text, or when the
        weaker outline's top-left footprint is flat enough for an invisible fill.
    A Doubao image is TC260 too but is not Jimeng-basic, so its mark suppresses the
    pill. The independently attributed LiblibAI pill also wins over the generic
    ``AI生成`` label shape. Marks from other TC260 manufacturers are unrelated and
    do not participate in this ByteDance-family decision."""
    if "liblib_pill" in keys or _pill_suppressors() & keys:
        return False
    if "jimeng" in keys:
        return True
    if "jimeng" in provenance:
        return footprint_flat or verified_label
    return False


def _build_candidates(image: NDArray[Any]) -> list[Candidate]:
    """PERCEPTION pass: run every ``in_auto`` mark's detector at both trust levels
    and package the raw verdicts + physical features into :class:`Candidate`
    objects. No policy here -- the arbiter (:func:`decide`) makes every call.

    Each mark is detected at the strict AND the relaxed (``provenance=True``) level
    from ONE scan per mark (:meth:`KnownMark.detect_both`): a relaxed gate is
    monotonically more permissive, so this reproduces a strict-then-relax pass
    exactly. Each mark reports its own gate features via :meth:`KnownMark.features`
    (computed only when the mark is detected, so a clean image pays nothing extra)."""
    cands: list[Candidate] = []
    for m in _REGISTRY:
        if not m.in_auto:
            continue
        strict, relaxed = m.detect_both(image)
        feats = m.features(image) if (strict.detected or relaxed.detected) else {}
        if relaxed.engine_detection is not None and getattr(
            relaxed.engine_detection, "verified_label", False
        ):
            feats["verified_label"] = 1.0
        cands.append(
            Candidate(
                m.key,
                m.label,
                strict.detected,
                relaxed.detected,
                feats,
                strict_detection=strict,
                relaxed_detection=relaxed,
            )
        )
    return cands


def decide(candidates: list[Candidate], context: Context) -> list[Decision]:
    """The removal ARBITER: a pure function turning perception + context into the
    ordered list of marks to remove (and the trust level each was accepted at).

    All policy lives here: per-mark trust resolution (:func:`resolve_trust`) and the
    Jimeng pill/label gate (:func:`_keep_pill`). No image, no I/O -- unit-testable
    in isolation, and the same decision drives every caller."""
    strict_keys = {c.key for c in candidates if c.detected_strict}
    fired: list[Decision] = []
    for c in candidates:
        trust = resolve_trust(
            c.key,
            sensitivity=context.sensitivity,
            provenance=context.provenance,
            strict_keys=strict_keys,
        )
        relax = trust != "strict"
        ok = c.detected_relaxed if relax else c.detected_strict
        if ok:
            fired.append(Decision(c, relax))
    keys = {d.candidate.key for d in fired}
    if "jimeng_pill" in keys:
        pill = next(d for d in fired if d.candidate.key == "jimeng_pill")
        flat = bool(pill.candidate.features.get("footprint_flat", 0.0))
        verified_label = bool(pill.candidate.features.get("verified_label", 0.0))
        if not _keep_pill(
            keys,
            provenance=context.provenance,
            footprint_flat=flat,
            verified_label=verified_label,
        ):
            fired = [d for d in fired if d.candidate.key != "jimeng_pill"]
    return fired


def _mask_bbox(mask: NDArray[Any]) -> Region | None:
    """Smallest ``(x, y, w, h)`` box containing a mask, or None when empty."""
    import cv2

    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, w, h) if w > 0 and h > 0 else None


def _regions_overlap(left: Region, right: Region) -> bool:
    """Whether two positive-area ``(x, y, w, h)`` regions overlap."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    if min(lw, lh, rw, rh) <= 0:
        return False
    return lx < rx + rw and rx < lx + lw and ly < ry + rh and ry < ly + lh


def remove_auto_marks_detailed(
    image: NDArray[Any],
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> VisibleRemovalResult:
    """Fill every decided mark once, then validate each result without editing it.

    The stages are separated: PERCEPTION (:func:`_build_candidates`), DECISION
    (:func:`decide`), ACTION (localize -> :func:`fill` per winner). Marks coexist in
    different corners, so every decided mark is removed, chained on the
    progressively-cleaned image.

    Each successful action has a fourth, read-only stage: the same mark detector
    runs once on the filled image. An accepted region counts as a residual only when
    it overlaps the mask that was actually filled. The result is ``partial`` in that
    case, ``unvalidated`` if the detector fails, and ``cleaned`` otherwise. This is
    a self-consistency check, not an independent removal oracle; it never changes
    the mask, retries the fill, or hides the candidate image.

    The first selected mark reuses the exact detection from perception; later marks
    re-detect on the progressively modified image, so overlapping/co-firing marks
    cannot act on stale evidence.

    Failure containment is per mark: a localize/fill raise yields a
    :class:`MarkError` row and the pass continues with the remaining marks; the
    aggregate status is ``error`` whenever any mark was contained. The returned
    image is always a fresh array (a no-mark pass never aliases the caller's
    input)."""
    context = Context(sensitivity=sensitivity, provenance=provenance)
    result = image.copy()
    removals: list[MarkRemovalResult] = []
    errors: list[MarkError] = []
    resolved_backend: Literal["cv2", "migan", "lama"] | None = None
    for d in decide(_build_candidates(image), context):
        mark = get_mark(d.candidate.key)
        cached_detection = None
        if not removals:
            cached_detection = (
                d.candidate.relaxed_detection if d.relax else d.candidate.strict_detection
            )
        try:
            localization = mark.localize(
                result,
                provenance=d.relax,
                force=False,
                detection=cached_detection,
            )
            mask = localization.mask
            bbox = _mask_bbox(mask) if mask is not None else None
            if mask is None or bbox is None:
                continue
        except Exception as exc:
            logger.warning(
                "Auto-pass localize failed for %s; the mark is skipped and the pass continues: %s",
                mark.key,
                exc,
                exc_info=True,
            )
            errors.append(MarkError(mark.key, mark.label, "localize", str(exc)))
            continue

        if resolved_backend is None:
            resolved_backend = resolve_backend(backend)
        region = localization.region
        confidence_before = localization.confidence
        try:
            result = fill(result, mask, backend=resolved_backend)
        except Exception as exc:
            logger.warning(
                "Auto-pass fill failed for %s (%s); the mark is skipped and the pass continues: %s",
                mark.key,
                resolved_backend,
                exc,
                exc_info=True,
            )
            errors.append(MarkError(mark.key, mark.label, "fill", str(exc)))
            continue
        filled_mask = mask  # carried on the result for outside-mask verification
        del localization, mask

        confidence_after: float | None = None
        residual_region: Region | None = None
        validation_status: MarkValidationStatus
        try:
            after = mark.detect(result, provenance=d.relax)
            confidence_after = after.confidence
            if after.detected and _regions_overlap(after.region, bbox):
                validation_status = "partial"
                residual_region = after.region
            else:
                validation_status = "cleaned"
        except Exception as exc:
            validation_status = "unvalidated"
            logger.warning(
                "Post-removal validation failed for %s after %s fill in mask %s: %s",
                mark.key,
                resolved_backend,
                bbox,
                exc,
                exc_info=True,
            )

        removals.append(
            MarkRemovalResult(
                key=mark.key,
                label=mark.label,
                location=mark.location,
                region=region,
                mask_bbox=bbox,
                backend=resolved_backend,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                status=validation_status,
                residual_region=residual_region,
                mask=filled_mask,
            )
        )

    return VisibleRemovalResult(result, tuple(removals), tuple(errors))


def remove_auto_marks(
    image: NDArray[Any],
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> tuple[NDArray[Any], list[str]]:
    """Compatibility wrapper returning the filled image and removed labels.

    The action still performs the read-only post-removal check. Call
    :func:`remove_auto_marks_detailed` when the ``cleaned`` / ``partial`` /
    ``unvalidated`` / ``no_mark`` result is needed."""
    report = remove_auto_marks_detailed(
        image,
        sensitivity=sensitivity,
        provenance=provenance,
        backend=backend,
    )
    return report.image, report.labels


def remove_mark_detailed(
    image: NDArray[Any],
    key: str,
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> VisibleRemovalResult:
    """Remove ONE named mark via localize -> fill -> validate (the ``--mark <key>``
    path). Nothing else in the registry is scanned, so a co-firing sibling never
    rides along on an explicitly named removal.

    The trust level is resolved exactly as in the auto pass (:func:`resolve_trust`):
    ``strict`` never relaxes; ``auto`` relaxes only when the provenance evidence
    names this mark's product. Nothing found is a ``no_mark`` result (empty marks
    tuple) with the image returned unchanged — an idempotent pass, no error."""
    validate_sensitivity(sensitivity)
    mark = get_mark(key)
    relax = (
        resolve_trust(key, sensitivity=sensitivity, provenance=provenance, strict_keys=set())
        != "strict"
    )
    localization = mark.localize(image, provenance=relax)
    if localization.mask is None or not localization.mask.any():
        return VisibleRemovalResult(image.copy(), ())
    resolved_backend = resolve_backend(backend)
    bbox = _mask_bbox(localization.mask)
    filled_mask = localization.mask
    result = fill(image, localization.mask, backend=resolved_backend)

    confidence_after: float | None = None
    residual_region: Region | None = None
    validation_status: MarkValidationStatus
    try:
        after = mark.detect(result, provenance=relax)
        confidence_after = after.confidence
        if after.detected and _regions_overlap(after.region, bbox):
            validation_status = "partial"
            residual_region = after.region
        else:
            validation_status = "cleaned"
    except Exception as exc:
        validation_status = "unvalidated"
        logger.warning(
            "Post-removal validation failed for %s after %s fill in mask %s: %s",
            mark.key,
            resolved_backend,
            bbox,
            exc,
            exc_info=True,
        )
    removal = MarkRemovalResult(
        key=mark.key,
        label=mark.label,
        location=mark.location,
        region=localization.region if localization.detected else None,
        mask_bbox=bbox,
        backend=resolved_backend,
        confidence_before=localization.confidence,
        confidence_after=confidence_after,
        status=validation_status,
        residual_region=residual_region,
        mask=filled_mask,
    )
    return VisibleRemovalResult(result, (removal,))
