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


def _validate_request(request: InpaintRequest) -> None:
    """Reject requests the adapter cannot fill meaningfully.

    Unlike the cv2 tier there is no pixel-count ceiling: the padded crop (not
    the image size) bounds the ONNX working set, which is the point of
    cropping.
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
        """Fill the marked region via MI-GAN; everything else stays byte-exact."""
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
        box = region_eraser._padded_crop_box(mask, h, w, pad_frac=2.0, pad_min=256)
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

        filled = _run_migan_crop(_load_session(), crop, crop_mask)

        result = bgr.copy()
        region = result[cy0:cy1, cx0:cx1]
        hole = crop_mask > 127
        region[hole] = filled[hole]
        result[cy0:cy1, cx0:cx1] = region
        return InpaintResult(
            inpainted=_to_raster(result, alpha, was_gray, request.source),
            backend_name=self.name,
            quality_estimate=0.85,
            diagnostics={
                "crop": [cx0, cy0, cx1, cy1],
                "model": f"{_MODEL_REPO}@{_MODEL_REVISION[:12]}",
            },
        )


def get_migan_backends() -> list[MiganFillBackend]:
    """The MI-GAN backends to register: empty unless onnxruntime imports."""
    if not onnxruntime_available():
        return []
    return [MiganFillBackend()]


def _download_model() -> str:
    """hf_hub_download pinned to the audited commit; fails soft with next steps.

    No local_dir/cache_dir is passed: the model lands in the Hugging Face
    cache (``HF_HOME``/``~/.cache/huggingface``), never inside this
    repository.
    """
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE, revision=_MODEL_REVISION)
    except Exception as exc:
        raise RuntimeError(
            f"Downloading the {_MODEL_LABEL} model "
            f"(https://huggingface.co/{_MODEL_REPO} / {_MODEL_FILE} @ "
            f"{_MODEL_REVISION[:12]}) failed: {exc}. The model downloads on "
            "first use into the Hugging Face cache (set HF_HOME to relocate "
            "it); check network access and free disk space, then retry. "
            "Nothing is bundled with the package, and nothing was written "
            "inside the installation directory."
        ) from exc


def _load_session() -> Any:
    """Load (once) the MI-GAN ONNX session, downloading pinned on first use."""
    cached = _sessions.get("migan")
    if cached is not None:
        return cached

    import onnxruntime as ort

    model_path = _download_model()
    logger.info("Loading %s model: %s", _MODEL_LABEL, model_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _sessions["migan"] = session
    return session


def _run_migan_crop(session: Any, crop_bgr: Any, crop_mask: Any) -> Any:
    """Run MI-GAN on one already-bounded uint8 BGR crop; returns filled BGR.

    Preprocess: BGR->RGB, HWC->CHW uint8; the mask is inverted to the shipped
    model's polarity (255 = known/keep, 0 = hole/erase). Postprocess maps the
    output back to BGR HWC, resizing if the model returned a different spatial
    size than it was fed.
    """
    import cv2
    import numpy as np

    ch, cw = crop_bgr.shape[:2]
    inputs = session.get_inputs()
    img_name, mask_name = inputs[0].name, inputs[1].name

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)  # (1,3,ch,cw)
    known = (crop_mask <= 127).astype(np.uint8) * 255  # model polarity
    mask_in = known[None, None]  # (1,1,ch,cw)

    out = session.run(None, {img_name: img_in, mask_name: mask_in})[0]
    res = np.transpose(np.asarray(out)[0], (1, 2, 0)).astype(np.uint8)  # (ch',cw',3) RGB
    if res.shape[:2] != (ch, cw):
        res = cv2.resize(res, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
