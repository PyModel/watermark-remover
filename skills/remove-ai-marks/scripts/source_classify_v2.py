# Ported from wiltodelta/remove-ai-watermarks
# src/remove_ai_watermarks/source_classify_v2.py, Apache-2.0; modified: the
# hub-install hint names this repo's distribution
# (watermark-remover[source-classify]) and an ``is_available()`` probe was
# added so callers (the identify signal collector) can gate on the exact
# module set the classify path imports before running. The spectral template
# scoring, threshold routing, conflict handling, and abstention logic are
# verbatim.
"""Opt-in OpenAI/Google export-source patterns in decoded image pixels.

This classifier does not decode SynthID or establish watermark presence.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from hashlib import file_digest
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

WEIGHTS_REPO = "wiltodelta/openai-google-image-source-classifier"
WEIGHTS_REVISION = "4563fb97e5572030ad8d6acbb6f20eaab7fc3009"
WEIGHTS_ENV = "RAIW_IMAGE_SOURCE_WEIGHTS"
MODEL_FILE = "openai-google-source-v1.npz"
MODEL_SHA256 = "0e94e565c181416e4f63a2bfc06cd5179f4201befda7481ac510f6a168c612f4"
MODEL_SCHEMA = "openai-google-source-v1"
GROUPS = ("square", "portrait", "landscape")
ROUTES = ("qwen", "mark", "direct")
_FILTER_NAMES = (
    "old_google_default_photo_fit",
    "old_google_sigma24",
    "r152_google",
    "r152_openai",
    "r161_google",
    "r161_openai",
    *(f"r163_{group}" for group in GROUPS),
)
_THRESHOLD_NAMES = (
    "old_google_default_parent",
    "old_google_sigma24_parent",
    "r152_google",
    "r152_openai",
    "r161_google",
    "r161_openai",
    "r183_openai_gate",
    *(f"r163_{group}" for group in GROUPS),
    *(f"openai_{route}" for route in ROUTES),
)
CLAIM = "likely image source/export pattern; not SynthID detection"

SourceLabel = Literal["openai", "google", "unknown"]
SourceReason = Literal["classified", "abstained", "conflict", "feature_unavailable"]
PROVIDERS = tuple(label for label in get_args(SourceLabel) if label != "unknown")


@dataclass(frozen=True)
class ImageSourceClassification:
    """Narrow source attribution with honest abstention and watermark unknown."""

    label: SourceLabel
    reason: SourceReason
    scores: dict[str, float]
    watermark_truth: Literal["unknown"] = "unknown"
    claim: str = CLAIM
    model_revision: str = WEIGHTS_REVISION

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reason": self.reason,
            "scores": self.scores,
            "watermark_truth": self.watermark_truth,
            "claim": self.claim,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class _Model:
    filters: dict[str, tuple[NDArray[Any], NDArray[Any]]]
    templates: dict[str, NDArray[Any]]
    thresholds: dict[str, float]


def is_available() -> bool:
    """True when the numeric stack and hub client this classifier imports exist."""
    from optional_deps import module_available

    return module_available("numpy", "cv2", "PIL", "huggingface_hub")


def _model_path() -> Path:
    override = os.environ.get(WEIGHTS_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        candidate = path / MODEL_FILE if path.is_dir() else path
        if not candidate.is_file():
            raise RuntimeError(f"{WEIGHTS_ENV} does not contain {MODEL_FILE}: {path}")
        return candidate
    try:
        hub: Any = import_module("huggingface_hub")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install 'watermark-remover[source-classify]' for image-source weights"
        ) from error
    return Path(
        hub.hf_hub_download(repo_id=WEIGHTS_REPO, filename=MODEL_FILE, revision=WEIGHTS_REVISION)
    )


@lru_cache(maxsize=2)
def _load_model(path: Path, expected_sha256: str) -> _Model:
    import numpy as np

    with path.open("rb") as source:
        actual_sha256 = file_digest(source, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("image-source model SHA-256 mismatch")
    required = {
        "schema",
        *(f"{name}_{field}" for name in _FILTER_NAMES for field in ("mean", "precision")),
        *(f"openai_{route}_unit" for route in ROUTES),
        *(f"threshold_{name}" for name in _THRESHOLD_NAMES),
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required or str(payload["schema"]) != MODEL_SCHEMA:
            raise ValueError("image-source model schema mismatch")
        loaded_filters: dict[str, tuple[NDArray[Any], NDArray[Any]]] = {}
        for name in _FILTER_NAMES:
            mean = payload[f"{name}_mean"]
            precision = payload[f"{name}_precision"]
            if mean.shape != (256, 256, 3) or mean.dtype != np.complex64:
                raise ValueError(f"invalid {name} mean")
            if precision.shape != mean.shape or precision.dtype != np.float32:
                raise ValueError(f"invalid {name} precision")
            if (
                not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(precision))
                or np.any(precision < 0)
            ):
                raise ValueError(f"nonfinite or negative {name} filter")
            loaded_filters[name] = (mean, precision)
        templates: dict[str, NDArray[Any]] = {}
        for route in ROUTES:
            unit = payload[f"openai_{route}_unit"]
            if (
                unit.shape != (992, 992)
                or unit.dtype != np.float32
                or not np.all(np.isfinite(unit))
            ):
                raise ValueError(f"invalid {route} template")
            if not np.isclose(np.linalg.norm(unit), 1.0, rtol=1e-5):
                raise ValueError(f"nonunit {route} template")
            templates[route] = unit.ravel()
        limits: dict[str, float] = {}
        for name in _THRESHOLD_NAMES:
            value = payload[f"threshold_{name}"]
            if value.shape != () or value.dtype != np.float64 or not np.isfinite(value):
                raise ValueError(f"invalid {name} threshold")
            limits[name] = float(value)
    return _Model(loaded_filters, templates, limits)


def _spectrum(rgb: NDArray[Any], sigma: float) -> NDArray[Any]:
    import cv2
    import numpy as np
    from PIL import Image

    small = np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))
    values = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_RGB2YCrCb)
    high = values - cv2.GaussianBlur(values, (0, 0), sigma)
    energy = cv2.GaussianBlur(high * high, (0, 0), 3)
    high = np.clip(high / np.sqrt(energy + (1 / 255) ** 2), -4, 4)
    if sigma == 1.2:
        high -= high.mean(axis=(0, 1), keepdims=True)
    else:
        high -= high.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    high /= np.maximum(np.sqrt(np.sum(high * high, axis=(0, 1), keepdims=True)), 1e-12)
    return np.fft.fft2(high, axes=(0, 1), norm="ortho").astype(np.complex64)


def _template_score(values: NDArray[Any], weights: tuple[NDArray[Any], NDArray[Any]]) -> float:
    import numpy as np

    mean, precision = weights
    numerator = np.real(values * np.conj(mean) * precision).sum(axis=(0, 1))
    denominator = max(
        float(
            np.sqrt((np.abs(values) ** 2 * precision).sum() * (np.abs(mean) ** 2 * precision).sum())
        ),
        1e-12,
    )
    return float(numerator.sum() / denominator)


def _openai_template_scores(rgb: NDArray[Any], model: _Model) -> dict[str, float]:
    import cv2
    import numpy as np

    if rgb.shape[:2] != (1024, 1024):
        return {}
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    high = luminance - cv2.GaussianBlur(luminance, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT)
    energy = cv2.GaussianBlur(high * high, (0, 0), 3.0, borderType=cv2.BORDER_REFLECT)
    residual = np.clip(high / np.sqrt(energy + 1), -4, 4)[8:1000, 8:1000].ravel().astype(np.float32)
    residual -= residual.mean()
    norm = np.linalg.norm(residual)
    if norm <= 0:
        return {}
    return {route: float(np.dot(residual, unit) / norm) for route, unit in model.templates.items()}


def _aspect(width: int, height: int) -> str:
    ratio = width / height
    if 0.87 <= ratio <= 1.15:
        return "square"
    if 0.55 <= ratio < 0.87:
        return "portrait"
    if 1.15 < ratio <= 1.82:
        return "landscape"
    return "other"


def classify_image_source(source: str | os.PathLike[str]) -> ImageSourceClassification:
    """Return likely OpenAI/Google source or abstain; never verify SynthID.

    Weights download from the pinned hub revision on first use (or come from
    ``RAIW_IMAGE_SOURCE_WEIGHTS``). The spectral half is a fixed 256px
    resize; the spatial-template half only engages at exactly 1024x1024 —
    the classifier's own size bounds.
    """
    import numpy as np
    from PIL import Image

    with Image.open(source) as opened:
        if min(opened.size) < 256:
            return ImageSourceClassification("unknown", "feature_unavailable", {})
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    model = _load_model(_model_path(), MODEL_SHA256)
    height, width = rgb.shape[:2]
    regular = _spectrum(rgb, 1.2)
    old_default = _template_score(regular, model.filters["old_google_default_photo_fit"])
    old_sigma24 = _template_score(_spectrum(rgb, 2.4), model.filters["old_google_sigma24"])
    limits = model.thresholds
    old_google = (
        old_default / limits["old_google_default_parent"]
        + old_sigma24 / limits["old_google_sigma24_parent"]
    ) / 2
    r152_google = _template_score(regular, model.filters["r152_google"])
    r152_openai = _template_score(regular, model.filters["r152_openai"])
    r161_google = _template_score(regular, model.filters["r161_google"])
    r161_openai = _template_score(regular, model.filters["r161_openai"])
    aspect = _aspect(width, height)
    r163_openai = (
        _template_score(regular, model.filters[f"r163_{aspect}"]) if aspect in GROUPS else None
    )
    old_openai_scores = _openai_template_scores(rgb, model)
    google = (
        old_google >= 1.0
        or r152_google >= limits["r152_google"]
        or r161_google >= limits["r161_google"]
    )
    openai = (
        any(score >= limits[f"openai_{route}"] for route, score in old_openai_scores.items())
        or r152_openai >= limits["r152_openai"]
        or r161_openai >= limits["r161_openai"]
        or (
            r163_openai is not None
            and r163_openai >= limits[f"r163_{aspect}"]
            and r152_openai >= limits["r183_openai_gate"]
        )
    )
    scores = {
        "old_google": old_google,
        "r152_google": r152_google,
        "r152_openai": r152_openai,
        "r161_google": r161_google,
        "r161_openai": r161_openai,
    }
    if r163_openai is not None:
        scores["r163_openai"] = r163_openai
    scores.update({f"openai_{route}": score for route, score in old_openai_scores.items()})
    if google and openai:
        return ImageSourceClassification("unknown", "conflict", scores)
    if google:
        return ImageSourceClassification("google", "classified", scores)
    if openai:
        return ImageSourceClassification("openai", "classified", scores)
    return ImageSourceClassification("unknown", "abstained", scores)
