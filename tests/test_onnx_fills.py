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
