# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/region_eraser.py (big-LaMa ONNX backend: session/cache, preprocess, session run, postprocess), Apache-2.0; modified: flattened into this repo's script package as an InpaintBackend adapter over morphomod Raster/Mask instead of raw ndarrays, the model download is pinned to an audited commit revision (upstream carries none) and fails soft with an actionable message, and the availability gate is a module-level probe seam.

"""big-LaMa ONNX fill backend — the ``lama`` extra's InpaintBackend adapter.

Runs the ``Carve/LaMa-ONNX`` big-LaMa model (Apache-2.0, ~200 MB, ``lama_fp32.onnx``)
on CPU through onnxruntime. The model is downloaded on first use, pinned to an
audited commit revision, into the Hugging Face cache (``HF_HOME``/``~/.cache/huggingface``)
— it is never bundled with this package, and nothing is written inside the
repository.

LaMa runs at a fixed square input size. Like upstream, the adapter crops a
padded region around the mask, inpaints that crop at the model's size, and
pastes only the masked pixels back — untouched areas stay pixel-exact and the
float32 working set stays at the model's fixed square regardless of the image
or mark size.

The adapter converts morphomod ``Raster``/``Mask`` to numpy and back; the
result always matches the source's dimensions and channel count (grayscale is
promoted for the model and demoted back; an alpha plane is split off and
re-attached untouched). The registry path always hands the adapter 8-bit
``Raster`` data (PNG decode is uint8), so the adapter is uint8-only by
construction.

onnxruntime, huggingface_hub, numpy and cv2 stay lazy: they load only inside
the probes and the inpaint path, so importing this module never pulls a pixel
dependency into the stdlib core import graph.
"""

from __future__ import annotations

import logging
from typing import Any

from domain_types import InpaintRequest, InpaintResult

logger = logging.getLogger(__name__)

_MODEL_REPO = "Carve/LaMa-ONNX"
_MODEL_FILE = "lama_fp32.onnx"
#: Pinned commit of the model repo; the download never floats to a newer head.
_MODEL_REVISION = "c3c0c9e468934d62e79c329e35d82dd09ff8c444"
_MODEL_LABEL = "LaMa-ONNX"

#: Cached onnxruntime sessions, keyed by backend name (loading is expensive;
#: reuse across calls). Replaced (never mutated) in tests.
_sessions: dict[str, Any] = {}

#: Fallback spatial input size when the ONNX metadata does not declare one.
_FALLBACK_SIZE = 512


def onnxruntime_available() -> bool:
    """True when onnxruntime imports — this tier's capability probe.

    Module-level on purpose (the region_eraser probe pattern): tests force
    either side deterministically by patching this name, so the fail-soft and
    registered paths never depend on the ambient environment.
    """
    import optional_deps

    return optional_deps._import_safe("onnxruntime") is not None


class LamaFillBackend:
    """big-LaMa ONNX fill behind the InpaintBackend protocol."""

    name = "lama"

    @property
    def available(self) -> bool:
        return onnxruntime_available()

    def supports(self, request: InpaintRequest) -> bool:
        # The fixed-square resize keeps memory flat in the image size, so
        # unlike the cv2 tier there is no mask-ratio ceiling — but an empty
        # mask has nothing to fill and the adapter only speaks 1/3/4-channel
        # rasters.
        if request.source.channels not in (1, 3, 4):
            return False
        return request.mask.marked > 0

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        raise NotImplementedError("LaMa fill inference is wired in ticket 04")


def get_lama_backends() -> list[LamaFillBackend]:
    """The LaMa backends to register: empty unless onnxruntime imports."""
    if not onnxruntime_available():
        return []
    return [LamaFillBackend()]
