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


def check_optional(extra: str) -> BackendAvailability:
    """Return availability for a single named extra.

    Dispatches to *_EXTRAS dicts defined below.
    """
    _checkers: dict[str, str] = {
        "visible": "cv2",
        "quality": "skimage",
        "ai": "torch",
        "provenance": "c2pa",
    }
    pkg = _checkers.get(extra)
    if pkg is None:
        return BackendAvailability(
            available=False,
            extra=extra,
            reason=f"unknown extra: {extra}",
        )
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
