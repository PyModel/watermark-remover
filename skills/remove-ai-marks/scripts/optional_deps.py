"""Graceful optional dependency detection for watermark-remover.

Every optional capability lives behind a try/import that never raises
ModuleNotFoundError at import time.  Callers inspect the Availability
dataclass and either skip a feature or present a clean error message.

Usage
-----
    from optional_deps import BackendAvailability, check_optional
    v = check_optional("visible")
    if v.available:
        import cv2
    else:
        eprint(v.hint)        # "Install watermark-remover[visible] …"

Raises
------
    Never.  Import-time failures are caught and returned as
    BackendAvailability(available=False, …).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Result of checking whether an optional backend is installed."""

    #: ``True`` when all required sub-dependencies are importable.
    available: bool
    #: Canonical extra name, e.g. ``"visible"``.
    extra: str
    #: Human-readable reason when unavailable.
    reason: str = ""
    #: Human-readable install hint for the user.
    hint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.available:
            object.__setattr__(self, "hint", f"{self.extra} extras are installed")
        else:
            object.__setattr__(
                self,
                "hint",
                f"Install watermark-remover[{self.extra}] to enable this backend. "
                f"Reason: {self.reason}",
            )


def module_available(*names: str) -> bool:
    """True when every named module imports right now.

    The probe optional decoders use to decide "can I run": unlike
    :func:`check_optional` it names modules, not extras, so a caller can gate
    on the exact set it will import next (and callers that prefer the
    user-facing vocabulary still route through :func:`check_optional`).
    """
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


def check_optional(extra: str) -> BackendAvailability:
    """Return availability for a single named extra.

    Dispatches to *_EXTRAS dicts defined below.  An extra may probe several
    modules; it is available only when every probed module imports.
    """
    _checkers: dict[str, tuple[str, ...]] = {
        "visible": ("cv2",),
        "quality": ("skimage",),
        "ai": ("torch",),
        "provenance": ("c2pa",),
        # The pixel extra (numpy + headless OpenCV + Pillow) is the declared
        # home for pixel-tier work; `visible` stays its superset alias.
        "pixel": ("numpy", "cv2", "PIL"),
        # The learned ONNX inpaint tiers (region_eraser): inference runtime plus
        # the hub client that fetches the model on first use.
        "migan": ("onnxruntime", "huggingface_hub"),
        "lama": ("onnxruntime", "huggingface_hub"),
        # The open DWT-DCT detector (dwt_dct.py + invisible_watermark.py):
        # the exact module set its is_available() probe names.
        "detect": ("numpy", "cv2", "pywt"),
        # The Adobe TrustMark soft-binding decoder (trustmark_detector.py): the
        # pip package pulls torch itself and downloads model weights on first
        # use. Standalone by design -- it never joins the `all` aggregate
        # (torch stays opt-in, per the extras-surface ruling).
        "trustmark": ("trustmark",),
        # The aggregate extra joins the registry with a real checker: every
        # member declared in pyproject at this commit (visible + quality + ai
        # + provenance + detect; source-classify joins when it lands). Probe
        # order is irrelevant; the union is what the aggregate promises.
        "all": ("cv2", "skimage", "torch", "c2pa", "numpy", "pywt"),
    }
    pkgs = _checkers.get(extra)
    if pkgs is None:
        return BackendAvailability(
            available=False,
            extra=extra,
            reason=f"unknown extra: {extra}",
        )
    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
        except Exception as exc:
            return BackendAvailability(
                available=False,
                extra=extra,
                reason=str(exc).split("\n")[0][:120],
            )
    return BackendAvailability(available=True, extra=extra)


# ---------------------------------------------------------------------------
# Callable check_* helpers (convenience)
# ---------------------------------------------------------------------------


def has_visible() -> bool:
    return check_optional("visible").available


def has_quality() -> bool:
    return check_optional("quality").available


def has_ai() -> bool:
    return check_optional("ai").available


def has_provenance() -> bool:
    return check_optional("provenance").available


def has_pixel() -> bool:
    return check_optional("pixel").available


#: Every extra `check_optional` knows about, for capability reporting.
KNOWN_EXTRAS = (
    "visible",
    "quality",
    "ai",
    "provenance",
    "pixel",
    "migan",
    "lama",
    "detect",
    "trustmark",
    "all",
)


def extras_status() -> dict[str, dict[str, object]]:
    """Availability of every optional extra, with its install hint."""
    status: dict[str, dict[str, object]] = {}
    for extra in KNOWN_EXTRAS:
        availability = check_optional(extra)
        status[extra] = {
            "available": availability.available,
            "hint": availability.hint,
        }
    return status


# ---------------------------------------------------------------------------
# Safe optional imports — always return a module or None
# ---------------------------------------------------------------------------


def _import_safe(name: str) -> Any | None:
    """Import *name* or return None (never raises)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


#: ``cv2`` or None — the OpenCV package.
cv2: Any = _import_safe("cv2")
#: ``PIL`` or None — Pillow library.
PIL: Any = _import_safe("PIL")
#: ``skimage`` or None — scikit-image.
skimage: Any = _import_safe("skimage")
#: ``torch`` or None — PyTorch.
torch: Any = _import_safe("torch")
#: ``c2pa`` or None — C2PA library.
c2pa: Any = _import_safe("c2pa")
