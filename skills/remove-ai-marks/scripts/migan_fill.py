# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/region_eraser.py (MI-GAN ONNX backend: session/cache, preprocess, session run, postprocess), Apache-2.0; modified: flattened into this repo's script package as an InpaintBackend adapter over morphomod Raster/Mask instead of raw ndarrays, the model download is pinned to an audited commit revision (upstream carries none) and fails soft with an actionable message, and the availability gate is a module-level probe seam.

"""MI-GAN ONNX fill backend — the ``migan`` extra's InpaintBackend adapter.

Runs the ``andraniksargsyan/migan`` ONNX model (MIT, ~28 MB) on CPU through
onnxruntime. The model is downloaded on first use, pinned to an audited
commit revision, into the Hugging Face cache (``HF_HOME``/``~/.cache/huggingface``)
— it is never bundled with this package, and nothing is written inside the
repository.

Like upstream, inference runs on a padded crop around the mask (at native
resolution — MI-GAN takes arbitrary spatial dims), so the ONNX working set is
bounded by the MARK size rather than scaling with the image; only the masked
pixels of the crop are pasted back, so everything outside the mask stays
pixel-exact. The shipped model expects the mask inverted (255 = known/keep,
0 = hole/erase — the inverse of this package's 255-erase convention), so the
mask is inverted before feeding the model.

The adapter converts morphomod ``Raster``/``Mask`` to numpy and back; the
result always matches the source's dimensions and channel count (grayscale is
promoted for the model and demoted back; an alpha plane is split off and
re-attached untouched).

onnxruntime, huggingface_hub, numpy and cv2 stay lazy: they load only inside
the probes and the inpaint path, so importing this module never pulls a pixel
dependency into the stdlib core import graph.
"""

from __future__ import annotations

import logging
from typing import Any

from domain_types import InpaintRequest, InpaintResult

logger = logging.getLogger(__name__)

_MODEL_REPO = "andraniksargsyan/migan"
_MODEL_FILE = "migan.onnx"
#: Pinned commit of the model repo; the download never floats to a newer head.
_MODEL_REVISION = "406830d0fa60666da0071c342ad2fbc8f30c5c64"
_MODEL_LABEL = "MI-GAN"

#: Cached onnxruntime sessions, keyed by backend name (loading is expensive;
#: reuse across calls). Replaced (never mutated) in tests.
_sessions: dict[str, Any] = {}


def onnxruntime_available() -> bool:
    """True when onnxruntime imports — this tier's capability probe.

    Module-level on purpose (the region_eraser probe pattern): tests force
    either side deterministically by patching this name, so the fail-soft and
    registered paths never depend on the ambient environment.
    """
    import optional_deps

    return optional_deps._import_safe("onnxruntime") is not None


class MiganFillBackend:
    """MI-GAN ONNX fill behind the InpaintBackend protocol."""

    name = "migan"

    @property
    def available(self) -> bool:
        return onnxruntime_available()

    def supports(self, request: InpaintRequest) -> bool:
        # The crop bounding keeps memory flat in the image size, so unlike the
        # cv2 tier there is no mask-ratio ceiling — but an empty mask has
        # nothing to fill and the adapter only speaks 1/3/4-channel rasters.
        if request.source.channels not in (1, 3, 4):
            return False
        return request.mask.marked > 0

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        raise NotImplementedError("MI-GAN fill inference is wired in ticket 04")


def get_migan_backends() -> list[MiganFillBackend]:
    """The MI-GAN backends to register: empty unless onnxruntime imports."""
    if not onnxruntime_available():
        return []
    return [MiganFillBackend()]
