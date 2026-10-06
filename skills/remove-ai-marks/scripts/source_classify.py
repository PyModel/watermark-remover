# Ported from wiltodelta/remove-ai-watermarks
# src/remove_ai_watermarks/source_classify.py, Apache-2.0; modified: the
# intra-package imports are adapted to this repo's flat scripts layout
# (image_io / forensic_124d / source_spectral / optional_deps), the extra hint
# names this repo's distribution (watermark-remover[source-classify]),
# is_available() also probes Pillow (part of the exact set the classify path
# imports), and a failed hub download raises a RuntimeError carrying the
# pinned repo/revision context instead of the bare transport error. The
# feature pipeline, model schema validation, threshold tolerance, and
# abstention logic are verbatim.
"""Abstaining source/export-pipeline classification from image pixels.

This module does not detect or decode SynthID. It labels original-export-looking
pixel pipelines as OpenAI, Google, or unknown after metadata may have been
removed. Call it explicitly; ``identify`` never imports or runs it as part of
its metadata judge.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

SOURCE_CLASSIFY_EXTRA = "'watermark-remover[source-classify]'"
WEIGHTS_ENV = "RAIW_SOURCE_CLASSIFY_WEIGHTS"
WEIGHTS_REPO = "wiltodelta/raiw-source-classify"
WEIGHTS_REVISION = "4b8b408d03dac20231d46217977fd164414b6b9e"
MODEL_FILE = "source-pipeline-mlp.npz"
MODEL_SHA256 = "6430b0738869ff07b6845b7c6149c2a83bc3f3785878082886c241c6f7ca0c4a"
CLASSES = ("openai", "google", "unknown")
PROVIDERS = CLASSES[:2]
FEATURE_WIDTH = 892
SPECTRAL_GRID = 16
CLAIM = "source/export pipeline classification; not SynthID detection"
# The identify/detect-side decode bound for the auto-run surface below: the
# classifiers are O(image pixels) on their forensic half (the caller owns the
# decode bound), so above this pixel count the signal abstains with an
# explicit note instead of holding a huge float buffer.
MAX_SOURCE_CLASSIFY_PIXELS = 4096 * 4096
# NumPy and the training runtime can differ by a few float32 ULPs at a matrix
# boundary. Treat values within this tolerance as the stored threshold so the
# exported runtime preserves the training runtime's >= decision.
THRESHOLD_TOLERANCE = 1e-5

SourceLabel = Literal["openai", "google", "unknown"]
SourceReason = Literal["classified", "abstained", "feature_unavailable"]


@dataclass(frozen=True)
class SourceClassification:
    """One narrow source-pipeline result with an explicit abstention."""

    label: SourceLabel
    reason: SourceReason
    scores: dict[str, float]
    margins: dict[str, float]
    thresholds: dict[str, float]
    claim: str = CLAIM
    model_revision: str = WEIGHTS_REVISION

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reason": self.reason,
            "scores": self.scores,
            "margins": self.margins,
            "thresholds": self.thresholds,
            "claim": self.claim,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class _Model:
    mean: NDArray[Any]
    scale: NDArray[Any]
    margins: NDArray[Any]
    layer0_weight: NDArray[Any]
    layer0_bias: NDArray[Any]
    layer1_weight: NDArray[Any]
    layer1_bias: NDArray[Any]
    output_weight: NDArray[Any]
    output_bias: NDArray[Any]


def is_available() -> bool:
    """Return whether the lightweight source-classify runtime is installed."""
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
            f"Install {SOURCE_CLASSIFY_EXTRA} to download the source classifier"
        ) from error
    try:
        return Path(
            hub.hf_hub_download(
                repo_id=WEIGHTS_REPO,
                filename=MODEL_FILE,
                revision=WEIGHTS_REVISION,
            )
        )
    except Exception as exc:
        log.warning(
            "Downloading %s from %s@%s failed: %s", MODEL_FILE, WEIGHTS_REPO, WEIGHTS_REVISION, exc
        )
        raise RuntimeError(
            f"Downloading {MODEL_FILE} from {WEIGHTS_REPO}@{WEIGHTS_REVISION} failed: {exc}"
        ) from exc


@lru_cache(maxsize=2)
def _load_model(path: Path, expected_sha256: str | None = MODEL_SHA256) -> _Model:
    import numpy as np

    if expected_sha256 is not None:
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"model SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    required = {
        "classes",
        "kind",
        "mean",
        "scale",
        "margins",
        "layer0_weight",
        "layer0_bias",
        "layer1_weight",
        "layer1_bias",
        "output_weight",
        "output_bias",
        "activation",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise ValueError(f"model fields must be exactly {sorted(required)}")
        if tuple(str(value) for value in payload["classes"]) != CLASSES:
            raise ValueError(f"model classes must be {CLASSES}")
        if str(payload["kind"]) != "fused":
            raise ValueError("model kind must be fused")
        if str(payload["activation"]) != "gelu_exact":
            raise ValueError("model activation must be gelu_exact")
        arrays = {
            name: np.asarray(
                payload[name],
                dtype=np.float64 if name in {"mean", "scale", "margins"} else np.float32,
            )
            for name in required - {"classes", "kind", "activation"}
        }
    shapes = {
        "mean": (FEATURE_WIDTH,),
        "scale": (FEATURE_WIDTH,),
        "margins": (len(PROVIDERS),),
        "layer0_weight": (128, FEATURE_WIDTH),
        "layer0_bias": (128,),
        "layer1_weight": (64, 128),
        "layer1_bias": (64,),
        "output_weight": (len(CLASSES), 64),
        "output_bias": (len(CLASSES),),
    }
    for name, expected in shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"model field {name} must have shape {expected}")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"model field {name} must be finite")
    if np.any(arrays["scale"] <= 0) or np.any(arrays["margins"] < 0):
        raise ValueError("model scales must be positive and margins non-negative")
    return _Model(**arrays)


def _gelu_exact(values: NDArray[Any]) -> NDArray[Any]:
    import numpy as np

    flat = values.ravel()
    errors = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in flat), dtype=np.float32
    )
    return (0.5 * flat * (1.0 + errors)).reshape(values.shape).astype(np.float32)


def _scores(features: NDArray[Any], model: _Model) -> NDArray[Any]:
    import numpy as np

    values = np.asarray((features - model.mean) / model.scale, dtype=np.float32)
    hidden0 = _gelu_exact(model.layer0_weight @ values + model.layer0_bias)
    hidden1 = _gelu_exact(model.layer1_weight @ hidden0 + model.layer1_bias)
    return np.asarray(model.output_weight @ hidden1 + model.output_bias, dtype=np.float64)


def _result(raw_scores: NDArray[Any], thresholds: NDArray[Any]) -> SourceClassification:
    import numpy as np

    margins = np.empty(len(PROVIDERS), dtype=np.float64)
    for index in range(len(PROVIDERS)):
        competitors = np.delete(raw_scores, index)
        margins[index] = raw_scores[index] - float(np.max(competitors))
    excess = margins - thresholds
    best = int(np.argmax(excess))
    accepted = bool(excess[best] >= -THRESHOLD_TOLERANCE)
    label: SourceLabel = PROVIDERS[best] if accepted else "unknown"
    return SourceClassification(
        label=label,
        reason="classified" if accepted else "abstained",
        scores={name: float(raw_scores[index]) for index, name in enumerate(CLASSES)},
        margins={name: float(margins[index]) for index, name in enumerate(PROVIDERS)},
        thresholds={name: float(thresholds[index]) for index, name in enumerate(PROVIDERS)},
    )


def _unavailable(model: _Model) -> SourceClassification:
    return SourceClassification(
        label="unknown",
        reason="feature_unavailable",
        scores={},
        margins={},
        thresholds={name: float(model.margins[index]) for index, name in enumerate(PROVIDERS)},
    )


def classify_source(source: str | os.PathLike[str]) -> SourceClassification:
    """Classify an original-looking image export, abstaining as ``unknown``.

    The result describes a complete source/export pipeline. It is not evidence
    that SynthID is present or absent. Weights download from the pinned hub
    revision on first use (or come from ``RAIW_SOURCE_CLASSIFY_WEIGHTS``); the
    caller owns the decode size — the pixel pipeline is O(image pixels) for the
    124-d extractor and a fixed canonical resize for the spectral half.
    """
    import image_io
    import numpy as np
    from forensic_124d import image_features
    from PIL import Image
    from source_spectral import feature_from_image, make_geometry

    model = _load_model(_model_path(), expected_sha256=MODEL_SHA256)
    image_io._register_heif()  # pyright: ignore[reportPrivateUsage]
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    pixels = np.asarray(image, dtype=np.uint8)
    forensic = image_features(pixels)
    if forensic is None:
        return _unavailable(model)
    spectral = feature_from_image(image, make_geometry(grid=SPECTRAL_GRID))
    if spectral is None:
        return _unavailable(model)
    features = np.concatenate((forensic, spectral))
    if features.shape != (FEATURE_WIDTH,) or not np.all(np.isfinite(features)):
        return _unavailable(model)
    return _result(_scores(features, model), model.margins)


def _oversize_note(width: int, height: int) -> str:
    pixels = width * height
    return (
        f"image exceeds the identify decode bound ({pixels} > "
        f"{MAX_SOURCE_CLASSIFY_PIXELS} px); the classifier was skipped — "
        "absence is never clean"
    )


def _source_pipeline_entry(path: Path) -> dict[str, Any]:
    """The E2 half of :func:`source_classify_signals` (fail-soft)."""
    entry: dict[str, Any] = {"detector": "source_classify"}
    try:
        if not is_available():
            entry["available"] = False
            entry["error"] = (
                "source-classify extra not installed (numpy/cv2/Pillow/"
                "huggingface_hub); install watermark-remover[source-classify] "
                "to run the source-export-pipeline classifier"
            )
            return entry
        from identify import source_classify_signal
        from PIL import Image

        entry["available"] = True
        with Image.open(path) as opened:
            width, height = opened.size
        if width * height > MAX_SOURCE_CLASSIFY_PIXELS:
            entry.update(source_classify_signal(None))
            entry["note"] = _oversize_note(width, height)
            return entry
        classification = classify_source(path)
        entry.update(source_classify_signal(classification))
        entry["classification"] = classification.to_dict()
    except Exception as exc:
        entry["available"] = True
        entry["error"] = f"source-pipeline classification failed: {exc!r}"
    return entry


def _image_source_entry(path: Path) -> dict[str, Any]:
    """The E3 half of :func:`source_classify_signals` (fail-soft)."""
    import source_classify_v2

    entry: dict[str, Any] = {"detector": "image_source"}
    try:
        if not source_classify_v2.is_available():
            entry["available"] = False
            entry["error"] = (
                "source-classify extra not installed (numpy/cv2/Pillow/"
                "huggingface_hub); install watermark-remover[source-classify] "
                "to run the image-source classifier"
            )
            return entry
        from identify import image_source_signal
        from PIL import Image

        entry["available"] = True
        with Image.open(path) as opened:
            width, height = opened.size
        if width * height > MAX_SOURCE_CLASSIFY_PIXELS:
            entry.update(image_source_signal(None))
            entry["note"] = _oversize_note(width, height)
            return entry
        classification = source_classify_v2.classify_image_source(path)
        entry.update(image_source_signal(classification))
        entry["classification"] = classification.to_dict()
    except Exception as exc:
        entry["available"] = True
        entry["error"] = f"image-source classification failed: {exc!r}"
    return entry


def source_classify_signals(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Run both source classifiers when the ``source-classify`` extra is there.

    The bounded evidence-only auto-run surface for /detect-style callers:
    each classifier executes only when its full module set imports, the decode
    is bounded by :data:`MAX_SOURCE_CLASSIFY_PIXELS` (larger images abstain
    with an explicit note), weights download from their pinned revisions on
    first use, and every failure mode is an explicit entry. Nothing here
    asserts AI provenance and nothing is ever a clean verdict — an unavailable
    extra, a crash, an abstention and a skipped oversize are all shades of
    unknown. The identify-side vocabulary mapping for the results lives in
    ``identify.source_classify_signal`` / ``identify.image_source_signal``;
    the identify judge itself never runs this (it stays metadata-only).
    """
    target = Path(path)
    return {
        "source_pipeline": _source_pipeline_entry(target),
        "image_source": _image_source_entry(target),
    }
