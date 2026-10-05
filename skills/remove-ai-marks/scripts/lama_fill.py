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


def _validate_request(request: InpaintRequest) -> None:
    """Reject requests the adapter cannot fill meaningfully.

    Unlike the cv2 tier there is no pixel-count ceiling: the model's fixed
    square input bounds the float32 working set regardless of image size.
    """
    w, h = request.source.width, request.source.height
    if (request.mask.width, request.mask.height) != (w, h):
        raise ValueError("mask dimensions do not match the source raster")
    if request.mask.marked == 0:
        raise ValueError("mask contains no marked pixels")
    if request.mask.marked >= w * h:
        raise ValueError("mask covers the entire image")


def _raster_planes(source: Any) -> tuple[Any, Any | None, bool]:
    """Raster -> (h,w,3) uint8 BGR plus its split-off alpha plane (or None).

    Grayscale is promoted to 3 identical channels (``was_gray`` tells the
    postprocess to demote again); the alpha plane always bypasses the model.
    """
    import numpy as np

    arr = np.frombuffer(bytes(source.data), dtype=np.uint8).reshape(
        source.height, source.width, source.channels
    )
    alpha = np.ascontiguousarray(arr[:, :, 3]) if source.channels == 4 else None
    bgr = arr[:, :, :3] if source.channels == 4 else arr
    if source.channels == 1:
        bgr = np.repeat(bgr, 3, axis=2)
    return np.ascontiguousarray(bgr), alpha, source.channels == 1


def _to_raster(bgr: Any, alpha: Any | None, was_gray: bool, original: Any) -> Any:
    """Rebuild a morphomod Raster with the ORIGINAL's dims/channels."""
    import numpy as np
    from morphomod import Raster

    if was_gray:
        import cv2

        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[:, :, None]
    elif alpha is not None:
        out = np.dstack([bgr, alpha])
    else:
        out = bgr
    data = bytearray(np.ascontiguousarray(out).tobytes())
    return Raster(
        width=original.width, height=original.height, channels=original.channels, data=data
    )


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
        """Fill the marked region via big-LaMa; everything else stays byte-exact."""
        import cv2
        import numpy as np

        _validate_request(request)
        bgr, alpha, was_gray = _raster_planes(request.source)
        mask = np.frombuffer(bytes(request.mask.data), dtype=np.uint8).reshape(
            request.source.height, request.source.width
        )
        # The padded-crop helper is region_eraser's bounded-box control
        # (clamped, single-sourced); importing it lazily keeps this module's
        # import graph free of region_eraser's top-level cv2/numpy.
        import region_eraser

        h, w = bgr.shape[:2]
        box = region_eraser._padded_crop_box(mask, h, w, pad_frac=0.4, pad_min=16)
        if box is None:  # defensive: supports()/validate already reject empty masks
            return InpaintResult(
                inpainted=request.source,
                backend_name=self.name,
                quality_estimate=0.0,
                diagnostics={"note": "empty mask"},
            )
        cx0, cy0, cx1, cy1 = box
        crop = np.ascontiguousarray(bgr[cy0:cy1, cx0:cx1])
        crop_mask = np.ascontiguousarray(mask[cy0:cy1, cx0:cx1])
        ch, cw = crop.shape[:2]

        session = _load_session()
        size = _model_size(session)

        # Preprocess: resize crop + mask to the model square, normalize to
        # [0,1] RGB CHW float32 (mask polarity: 1 = hole).
        crop_rs = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        mask_rs = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        img_in = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))[None]  # (1,3,size,size)
        mask_in = (mask_rs > 127).astype(np.float32)[None, None]  # (1,1,size,size)

        inputs = session.get_inputs()
        out = session.run(None, {inputs[0].name: img_in, inputs[1].name: mask_in})[0]
        out = np.asarray(out)[0]  # (3,size,size)
        out = np.transpose(out, (1, 2, 0))
        if float(out.max()) <= 1.5:  # model emits [0,1]
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Resize back to crop size and paste only the masked pixels.
        out_crop = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)
        result = bgr.copy()
        region = result[cy0:cy1, cx0:cx1]
        paste = crop_mask > 127
        region[paste] = out_crop[paste]
        result[cy0:cy1, cx0:cx1] = region
        return InpaintResult(
            inpainted=_to_raster(result, alpha, was_gray, request.source),
            backend_name=self.name,
            quality_estimate=0.9,
            diagnostics={
                "crop": [cx0, cy0, cx1, cy1],
                "model_size": size,
                "model": f"{_MODEL_REPO}@{_MODEL_REVISION[:12]}",
            },
        )


def get_lama_backends() -> list[LamaFillBackend]:
    """The LaMa backends to register: empty unless onnxruntime imports."""
    if not onnxruntime_available():
        return []
    return [LamaFillBackend()]


def _download_model() -> str:
    raise NotImplementedError("download seam is wired in ticket 04")


def _model_size(session: Any) -> int:
    """The model's declared square spatial input size (fallback 512)."""
    dims = session.get_inputs()[0].shape
    return next((d for d in reversed(dims) if isinstance(d, int) and d > 1), _FALLBACK_SIZE)


def _load_session() -> Any:
    """Load (once) the big-LaMa ONNX session, downloading pinned on first use."""
    cached = _sessions.get("lama")
    if cached is not None:
        return cached

    import onnxruntime as ort

    model_path = _download_model()
    logger.info("Loading %s model: %s", _MODEL_LABEL, model_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _sessions["lama"] = session
    return session
