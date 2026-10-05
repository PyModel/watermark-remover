"""Tests for inpaint_backends module."""

from __future__ import annotations

import shlex
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import optional_deps
import pytest
from domain_types import InpaintRequest
from inpaint_backends import (
    ExternalCommandBackend,
    SimpleFillBackend,
    TexturePatchBackend,
    get_backends,
)
from morphomod import Mask, Raster


def _make_raster(w: int = 20, h: int = 20, ch: int = 3, fill: int = 128) -> Raster:
    return Raster(width=w, height=h, channels=ch, data=bytearray([fill] * w * h * ch))


def _make_mask(w: int = 20, h: int = 20, fill: int = 0) -> Mask:
    return Mask(width=w, height=h, data=bytearray([fill] * w * h))


class TestGetBackends:
    def test_returns_list(self) -> None:
        backends = get_backends()
        assert isinstance(backends, list)
        assert len(backends) >= 2

    def test_contains_expected_backends(self) -> None:
        names = [b.name for b in get_backends()]
        assert "texture-patch" in names
        assert "simple" in names


class TestTexturePatchBackend:
    def test_available(self) -> None:
        b = TexturePatchBackend()
        assert b.available is True
        assert b.name == "texture-patch"

    def test_supports_rgb(self) -> None:
        b = TexturePatchBackend()
        raster = _make_raster(ch=3)
        mask = _make_mask()  # empty mask by default
        assert b.supports(InpaintRequest(source=raster, mask=mask))

    def test_supports_small_mask(self) -> None:
        b = TexturePatchBackend()
        # Small mask ratio
        raster = _make_raster(w=100, h=100, ch=3)
        mask = _make_mask(w=100, h=100)
        # Mark a small region
        for y in range(5):
            for x in range(5):
                mask.data[y * 100 + x] = 255
        assert b.supports(InpaintRequest(source=raster, mask=mask))

    def test_rejects_large_mask(self) -> None:
        b = TexturePatchBackend()
        raster = _make_raster(w=100, h=100, ch=3)
        mask = _make_mask(w=100, h=100)
        # Mark more than 50% of pixels
        for y in range(51):
            for x in range(100):
                mask.data[y * 100 + x] = 255
        assert not b.supports(InpaintRequest(source=raster, mask=mask))

    def test_rejects_single_channel(self) -> None:
        b = TexturePatchBackend()
        raster = _make_raster(ch=1)
        mask = _make_mask()
        assert not b.supports(InpaintRequest(source=raster, mask=mask))

    def test_inpaint(self) -> None:
        b = TexturePatchBackend()
        raster = _make_raster(w=20, h=20, ch=3)
        mask = _make_mask(w=20, h=20)
        # Mark a small region
        for y in range(5, 10):
            for x in range(5, 10):
                mask.data[y * 20 + x] = 255
        result = b.inpaint(InpaintRequest(source=raster, mask=mask))
        assert result.backend_name == "texture-patch"
        assert result.quality_estimate > 0
        assert result.diagnostics.get("texture_match") is not None


class TestSimpleFillBackend:
    def test_available(self) -> None:
        b = SimpleFillBackend()
        assert b.available is True
        assert b.name == "simple"

    def test_supports_rgb(self) -> None:
        b = SimpleFillBackend()
        raster = _make_raster(ch=3)
        mask = _make_mask()  # empty mask by default
        assert b.supports(InpaintRequest(source=raster, mask=mask))

    def test_supports_grayscale(self) -> None:
        b = SimpleFillBackend()
        raster = _make_raster(ch=1)
        mask = _make_mask()
        assert b.supports(InpaintRequest(source=raster, mask=mask))

    def test_supports_rgba(self) -> None:
        b = SimpleFillBackend()
        raster = _make_raster(ch=4)
        mask = _make_mask()
        # Simple fill supports alpha (4 channels)
        assert b.supports(InpaintRequest(source=raster, mask=mask))

    def test_inpaint(self) -> None:
        b = SimpleFillBackend()
        raster = _make_raster(w=20, h=20, ch=3)
        mask = _make_mask(w=20, h=20)
        for y in range(5, 10):
            for x in range(5, 10):
                mask.data[y * 20 + x] = 255
        result = b.inpaint(InpaintRequest(source=raster, mask=mask))
        assert result.backend_name == "simple"
        assert result.quality_estimate > 0


class TestExternalCommandBackend:
    def test_available_with_executable(self) -> None:
        b = ExternalCommandBackend("echo hello")
        assert b.available is True

    def test_unavailable_with_missing_executable(self) -> None:
        b = ExternalCommandBackend("/nonexistent/cmd arg1 arg2")
        assert b.available is False

    def test_supports_any(self) -> None:
        b = ExternalCommandBackend("echo hello")
        raster = _make_raster()
        mask = _make_mask()
        assert b.supports(InpaintRequest(source=raster, mask=mask))

    def test_placeholder_command_is_available(self) -> None:
        command = f'{shlex.quote(sys.executable)} -c "print(1)" {{input}} {{mask}} {{output}}'
        assert ExternalCommandBackend(command).available

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"timeout": 0}, ValueError),
            ({"timeout": float("nan")}, ValueError),
            ({"output_limit": 0}, ValueError),
            ({"output_limit": True}, TypeError),
        ],
    )
    def test_rejects_invalid_limits(self, kwargs, error) -> None:
        with pytest.raises(error):
            ExternalCommandBackend("echo hello", **kwargs)

    def test_end_to_end_command_writes_valid_output(self, tmp_path: Path) -> None:
        copier = tmp_path / "copy_output.py"
        copier.write_text(
            "import shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[3])\n",
            encoding="utf-8",
        )
        command = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(copier))} "
            '"{input}" "{mask}" "{output}"'
        )
        raster = _make_raster(w=4, h=3, ch=3, fill=77)
        mask = _make_mask(w=4, h=3)
        mask.data[5] = 255

        result = ExternalCommandBackend(command, timeout=5).inpaint(
            InpaintRequest(source=raster, mask=mask)
        )

        assert result.backend_name == "external"
        assert result.inpainted == raster
        assert result.diagnostics["returncode"] == 0

    def test_frozen(self) -> None:
        backend = ExternalCommandBackend("echo hello")
        with pytest.raises(FrozenInstanceError):
            backend.command = "other"


# ---------------------------------------------------------------------------
# Registry bootstrap: cv2 fill backends join and leave with OpenCV itself
# ---------------------------------------------------------------------------


def _install_stub_opencv(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Make OpenCV importable process-wide with a minimal cv2 stand-in.

    Mirrors the stub-module prior art (test_stability_server.py ships a real
    cv2.py; test_opencv_inpaint.py fakes numpy via sys.modules).  Both patch
    points are needed: ``optional_deps.cv2`` gates registry bootstrap while
    ``has_visible()`` resolves through ``importlib.import_module``.
    """
    stub = types.ModuleType("cv2")
    stub.INPAINT_TELEA = 1
    stub.INPAINT_NS = 2
    monkeypatch.setitem(sys.modules, "cv2", stub)
    monkeypatch.setattr(optional_deps, "cv2", stub)
    return stub


@pytest.fixture
def fresh_registry(monkeypatch: pytest.MonkeyPatch):
    """Rebuild the backend registry from scratch, restoring it afterwards."""
    import inpaint_backends

    saved = list(inpaint_backends._REGISTRY)
    saved_flag = inpaint_backends._bootstrapped
    inpaint_backends._REGISTRY.clear()
    inpaint_backends._bootstrapped = False
    try:
        yield inpaint_backends
    finally:
        inpaint_backends._REGISTRY.clear()
        inpaint_backends._REGISTRY.extend(saved)
        inpaint_backends._bootstrapped = saved_flag


def test_bootstrap_registers_cv2_backends_when_opencv_present(
    monkeypatch: pytest.MonkeyPatch, fresh_registry
) -> None:
    _install_stub_opencv(monkeypatch)

    names = [backend.name for backend in fresh_registry.get_backends()]

    assert "cv2-telea" in names
    assert "cv2-ns" in names


def test_bootstrap_excludes_cv2_backends_without_opencv(
    monkeypatch: pytest.MonkeyPatch, fresh_registry
) -> None:
    # Forced-absent regardless of the ambient environment: the incumbent
    # pair must stay intact and no half-registered cv2 backend may appear.
    monkeypatch.setattr(optional_deps, "cv2", None)

    names = [backend.name for backend in fresh_registry.get_backends()]

    assert names == ["texture-patch", "simple"]
