"""Tests for the ONNX learned fill backends (``migan`` / ``lama`` extras).

Seams (ticket 04 contract): the registry backend API (inpaint_backends
bootstrap + resolve_backend) and the extras status surface (optional_deps /
the /capabilities data). The fail-soft paths are forced deterministically
(the per-module onnxruntime gate is monkeypatched by name, never assumed
from the ambient environment); the real-model tier is additionally gated
behind WATERMARK_ONNX_FILL_TEST=1 plus an importable onnxruntime and never
downloads in the default suite.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from domain_types import InpaintRequest
from morphomod import Mask, Raster


def _make_raster(w: int = 32, h: int = 32, ch: int = 3, fill: int = 128) -> Raster:
    return Raster(width=w, height=h, channels=ch, data=bytearray([fill] * w * h * ch))


def _make_mask(w: int = 32, h: int = 32, box: tuple[int, int, int, int] | None = None) -> Mask:
    data = bytearray(w * h)
    if box is not None:
        x, y, bw, bh = box
        for yy in range(y, y + bh):
            for xx in range(x, x + bw):
                data[yy * w + xx] = 255
    return Mask(width=w, height=h, data=data)


#: (module name, backend attribute, backend name, list getter) for both tiers.
_TIERS = [
    ("migan_fill", "MiganFillBackend", "migan", "get_migan_backends"),
    ("lama_fill", "LamaFillBackend", "lama", "get_lama_backends"),
]


@pytest.fixture
def fill_modules(monkeypatch: pytest.MonkeyPatch):
    """Import both fill modules with empty session caches."""
    import lama_fill
    import migan_fill

    monkeypatch.setattr(migan_fill, "_sessions", {})
    monkeypatch.setattr(lama_fill, "_sessions", {})
    return migan_fill, lama_fill


def _force_runtime(monkeypatch: pytest.MonkeyPatch, modules, *, present: bool) -> None:
    """Force every fill module's onnxruntime gate to a known state, by name.

    The gates are module-level functions (the region_eraser probe pattern) so
    the fail-soft and registered paths are both testable on any machine, with
    or without onnxruntime installed.
    """
    for module in modules:
        monkeypatch.setattr(module, "onnxruntime_available", lambda: present)


def _force_session(monkeypatch: pytest.MonkeyPatch, module, session) -> None:
    """Inject a fake session at the module's _load_session seam (by name)."""
    monkeypatch.setattr(module, "_load_session", lambda: session)


class _FakeMiganSession:
    """MI-GAN ONNX contract: uint8 CHW image + inverted known-mask (255=keep).

    Fills every hole pixel with a constant color so paste behaviour is
    observable, and records each crop's spatial dims for the bounded-crop
    assertions.
    """

    def __init__(self, fill: tuple[int, int, int] = (10, 20, 30)) -> None:
        self.fill = fill
        self.crop_dims: list[tuple[int, int]] = []

    def get_inputs(self):
        return [SimpleNamespace(name="image"), SimpleNamespace(name="mask")]

    def run(self, _outputs, feeds):
        img = feeds["image"]  # (1,3,h,w) uint8
        known = feeds["mask"][0, 0]  # (h,w) uint8, 255=keep / 0=hole
        h, w = known.shape
        self.crop_dims.append((h, w))
        rgb = np.transpose(img[0], (1, 2, 0)).copy()
        rgb[known == 0] = self.fill
        return [np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)]


class _FakeLamaSession:
    """LaMa ONNX contract: fixed-square float32 inputs, mask 1=hole, [0,1] RGB."""

    size = 512

    def __init__(self, fill: tuple[float, float, float] = (0.04, 0.08, 0.12)) -> None:
        self.fill = fill
        self.crop_dims: list[tuple[int, int]] = []

    def get_inputs(self):
        s = self.size
        return [
            SimpleNamespace(name="image", shape=[1, 3, s, s]),
            SimpleNamespace(name="mask", shape=[1, 1, s, s]),
        ]

    def run(self, _outputs, feeds):
        img = feeds["image"]  # (1,3,S,S) float32 [0,1]
        hole = feeds["mask"][0, 0] > 0.5  # (S,S)
        s = hole.shape[0]
        self.crop_dims.append((s, s))
        rgb = np.transpose(img[0], (1, 2, 0)).copy()
        rgb[hole] = self.fill
        return [np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)]


def _pixels(raster: Raster) -> np.ndarray:
    return np.frombuffer(bytes(raster.data), dtype=np.uint8).reshape(
        raster.height, raster.width, raster.channels
    )


def _mask_boolean(mask: Mask) -> np.ndarray:
    """The mask as a boolean (h,w) array: True where marked (255=erase)."""
    return np.frombuffer(bytes(mask.data), dtype=np.uint8).reshape(mask.height, mask.width) > 0


class TestLearnedBackendLists:
    def test_lists_are_empty_without_onnxruntime(self, monkeypatch, fill_modules) -> None:
        _force_runtime(monkeypatch, fill_modules, present=False)

        migan_fill, lama_fill = fill_modules
        assert migan_fill.get_migan_backends() == []
        assert lama_fill.get_lama_backends() == []

    @pytest.mark.parametrize(("module_name", "backend_attr", "expected_name", "getter"), _TIERS)
    def test_one_backend_registers_when_runtime_present(
        self, monkeypatch, fill_modules, module_name, backend_attr, expected_name, getter
    ) -> None:
        _force_runtime(monkeypatch, fill_modules, present=True)
        module = importlib.import_module(module_name)

        backends = getattr(module, getter)()

        assert len(backends) == 1
        assert backends[0].name == expected_name
        assert backends[0].available is True


class TestAdapterContract:
    @pytest.mark.parametrize(("module_name", "backend_attr", "expected_name", "getter"), _TIERS)
    def test_available_tracks_the_runtime_gate(
        self, monkeypatch, fill_modules, module_name, backend_attr, expected_name, getter
    ) -> None:
        module = importlib.import_module(module_name)
        backend = getattr(module, backend_attr)()

        _force_runtime(monkeypatch, fill_modules, present=False)
        assert backend.available is False
        _force_runtime(monkeypatch, fill_modules, present=True)
        assert backend.available is True

    @pytest.mark.parametrize(("module_name", "backend_attr", "expected_name", "getter"), _TIERS)
    def test_supports_nonempty_masks_on_supported_channel_counts(
        self, fill_modules, module_name, backend_attr, expected_name, getter
    ) -> None:
        module = importlib.import_module(module_name)
        backend = getattr(module, backend_attr)()

        for channels in (1, 3, 4):
            request = InpaintRequest(
                source=_make_raster(ch=channels), mask=_make_mask(box=(4, 4, 2, 2))
            )
            assert backend.supports(request) is True, f"channels={channels}"

        assert (
            backend.supports(InpaintRequest(source=_make_raster(), mask=_make_mask())) is False
        ), "empty mask"

        # A channel count the domain type itself forbids, seen through the
        # adapter's attribute seam (Raster only constructs 1/3/4).
        odd = SimpleNamespace(width=32, height=32, channels=2)
        request = InpaintRequest(source=odd, mask=_make_mask(box=(4, 4, 2, 2)))
        assert backend.supports(request) is False, "unsupported channel count"


class TestInpaintThroughFakeSession:
    """The Raster/Mask <-> numpy adapter seam, driven by fake ONNX sessions.

    No network, no real model: the fakes pin each model's I/O contract
    (dtype, channel order, mask polarity) and let the crop/paste/channel
    behaviour be asserted exactly. Real-model coverage is the env-gated tier.
    """

    def test_migan_fills_only_masked_pixels(self, monkeypatch, fill_modules) -> None:
        migan_fill = fill_modules[0]
        session = _FakeMiganSession()
        _force_session(monkeypatch, migan_fill, session)
        raster = _make_raster(fill=200)
        mask = _make_mask(box=(8, 8, 4, 4))

        result = migan_fill.MiganFillBackend().inpaint(InpaintRequest(source=raster, mask=mask))

        assert result.backend_name == "migan"
        out = _pixels(result.inpainted)
        src = _pixels(raster)
        assert out.shape == src.shape
        hole = _mask_boolean(mask)
        # The fake fills holes with RGB (10,20,30); the adapter's RGB->BGR
        # postprocess hands the caller (30,20,10) -- the asymmetric fill pins
        # the channel-order conversion.
        assert (out[hole] == np.array([30, 20, 10], dtype=np.uint8)).all(), "hole filled"
        assert (out[~hole] == src[~hole]).all(), "outside mask pixel-exact"

    def test_lama_fills_only_masked_pixels(self, monkeypatch, fill_modules) -> None:
        lama_fill = fill_modules[1]
        session = _FakeLamaSession()
        _force_session(monkeypatch, lama_fill, session)
        raster = _make_raster(fill=200)
        mask = _make_mask(box=(12, 12, 4, 4))  # 16-aligned in the 512 square

        result = lama_fill.LamaFillBackend().inpaint(InpaintRequest(source=raster, mask=mask))

        assert result.backend_name == "lama"
        out = _pixels(result.inpainted)
        src = _pixels(raster)
        hole = _mask_boolean(mask)
        # [0,1] RGB fill (0.04,0.08,0.12) -> *255 -> uint8 -> BGR (30,20,10).
        assert (out[hole] == np.array([30, 20, 10], dtype=np.uint8)).all(), "hole filled"
        assert (out[~hole] == src[~hole]).all(), "outside mask pixel-exact"
        assert session.crop_dims == [(512, 512)], "fixed-square model input"

    def test_migan_preserves_alpha_and_demotes_grayscale(self, monkeypatch, fill_modules) -> None:
        migan_fill = fill_modules[0]
        session = _FakeMiganSession()
        _force_session(monkeypatch, migan_fill, session)
        rgba = _make_raster(ch=4, fill=200)
        for i in range(3, len(rgba.data), 4):
            rgba.data[i] = 77  # constant alpha plane
        gray = _make_raster(ch=1, fill=90)

        rgba_out = _pixels(
            migan_fill.MiganFillBackend()
            .inpaint(InpaintRequest(source=rgba, mask=_make_mask(box=(8, 8, 4, 4))))
            .inpainted
        )
        assert (rgba_out[:, :, 3] == 77).all(), "alpha untouched everywhere"
        assert (rgba_out[8:12, 8:12, :3] == np.array([30, 20, 10], dtype=np.uint8)).all()

        gray_out = _pixels(
            migan_fill.MiganFillBackend()
            .inpaint(InpaintRequest(source=gray, mask=_make_mask(box=(8, 8, 4, 4))))
            .inpainted
        )
        assert gray_out.shape == (32, 32, 1)
        # cv2 BT.601 fixed point on BGR fill (30,20,10):
        # (10*4899 + 20*9617 + 30*1868 + 8192) >> 14 = 18 -- a hand-computed
        # literal, not a recompute.
        assert (gray_out[8:12, 8:12, 0] == 18).all(), "gray demoted through BGR2GRAY"
        assert (gray_out[~_mask_boolean(_make_mask(box=(8, 8, 4, 4)))] == 90).all()

    def test_lama_preserves_alpha_and_demotes_grayscale(self, monkeypatch, fill_modules) -> None:
        lama_fill = fill_modules[1]
        session = _FakeLamaSession()
        _force_session(monkeypatch, lama_fill, session)
        rgba = _make_raster(ch=4, fill=200)
        for i in range(3, len(rgba.data), 4):
            rgba.data[i] = 77
        gray = _make_raster(ch=1, fill=90)

        rgba_out = _pixels(
            lama_fill.LamaFillBackend()
            .inpaint(InpaintRequest(source=rgba, mask=_make_mask(box=(12, 12, 4, 4))))
            .inpainted
        )
        assert (rgba_out[:, :, 3] == 77).all(), "alpha untouched everywhere"

        gray_out = _pixels(
            lama_fill.LamaFillBackend()
            .inpaint(InpaintRequest(source=gray, mask=_make_mask(box=(12, 12, 4, 4))))
            .inpainted
        )
        assert gray_out.shape == (32, 32, 1)
        assert (gray_out[12:16, 12:16, 0] == 18).all(), "gray demoted through BGR2GRAY"

    def test_migan_crop_is_bounded_by_the_mark_not_the_image(
        self, monkeypatch, fill_modules
    ) -> None:
        # Tier-1 bounded memory: a small mark on a 1 MP frame must not hand
        # the model the whole image. pad_frac=2.0/pad_min=256 around the 8px
        # mark gives a 520px square crop; the rest of the frame never reaches
        # ONNX.
        migan_fill = fill_modules[0]
        session = _FakeMiganSession()
        _force_session(monkeypatch, migan_fill, session)
        raster = _make_raster(w=1024, h=1024, fill=200)
        mask = _make_mask(w=1024, h=1024, box=(500, 500, 8, 8))

        migan_fill.MiganFillBackend().inpaint(InpaintRequest(source=raster, mask=mask))

        assert session.crop_dims == [(520, 520)]

    def test_lama_working_set_is_the_fixed_square(self, monkeypatch, fill_modules) -> None:
        # Tier-1 bounded memory: regardless of image/mark size, the model only
        # ever sees its fixed square (512 here, from the session metadata).
        lama_fill = fill_modules[1]
        session = _FakeLamaSession()
        _force_session(monkeypatch, lama_fill, session)
        raster = _make_raster(w=1024, h=1024, fill=200)
        mask = _make_mask(w=1024, h=1024, box=(500, 500, 8, 8))

        lama_fill.LamaFillBackend().inpaint(InpaintRequest(source=raster, mask=mask))

        assert session.crop_dims == [(512, 512)]

    def test_inpaint_rejects_empty_and_full_masks(self, fill_modules) -> None:
        raster = _make_raster()
        migan = fill_modules[0].MiganFillBackend()
        lama = fill_modules[1].LamaFillBackend()
        for backend in (migan, lama):
            with pytest.raises(ValueError, match="no marked pixels"):
                backend.inpaint(InpaintRequest(source=raster, mask=_make_mask()))
            full = _make_mask()
            full.data[:] = bytes([255]) * 32 * 32
            with pytest.raises(ValueError, match="entire image"):
                backend.inpaint(InpaintRequest(source=raster, mask=full))
