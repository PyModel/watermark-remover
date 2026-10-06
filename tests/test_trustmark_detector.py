"""Tests for the optional Adobe TrustMark detector.

TrustMark is an optional dependency (extra ``trustmark``) that downloads model
weights on first use, so the decode path is only exercised when it is installed
(mirrors the imwatermark handling). The always-on tests pin the graceful
absent/error behavior: detect must return None, never raise, and importing the
module must not pull torch/trustmark/PIL (weights and frames are lazy).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import load_toml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import trustmark_detector as td


@pytest.fixture
def tmp_clean_png(tmp_path: Path) -> Path:
    """A small synthetic gradient photo-shaped PNG with no watermark."""
    path = tmp_path / "clean.png"
    width, height = 64, 48
    img = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), (x * 4 % 256, y * 5 % 256, (x + y) * 2 % 256))
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Fail-soft protocol (always-on: these run with or without the extra)
# ---------------------------------------------------------------------------


def test_detect_never_raises(tmp_clean_png: Path):
    # Whether or not trustmark is installed, a clean image must yield None
    # (no watermark) without raising. When absent, the import guard returns None.
    assert td.detect_trustmark(tmp_clean_png) is None


def test_unreadable_file_returns_none(tmp_path: Path):
    bad = tmp_path / "not_an_image.txt"
    bad.write_bytes(b"not an image")
    assert td.detect_trustmark(bad) is None


def test_detect_trustmark_image_never_raises(tmp_clean_png: Path):
    with Image.open(tmp_clean_png) as img:
        assert td.detect_trustmark_image(img) is None


def test_is_available_follows_the_trustmark_module(monkeypatch: pytest.MonkeyPatch):
    import optional_deps

    assert td.is_available() == optional_deps.module_available("trustmark")
    monkeypatch.setattr(optional_deps, "module_available", lambda *names: False)
    assert td.is_available() is False
    monkeypatch.setattr(optional_deps, "module_available", lambda *names: True)
    assert td.is_available() is True


def test_module_import_does_not_pull_torch_trustmark_or_pillow():
    """No torch/trustmark/PIL import at module import time (weights stay lazy)."""
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
        "import trustmark_detector; "
        "assert 'torch' not in sys.modules, 'torch imported at module import'; "
        "assert 'trustmark' not in sys.modules, 'trustmark imported at module import'; "
        "assert 'PIL' not in sys.modules, 'PIL imported at module import'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Live decode (availability-gated: needs the trustmark extra; first use
# downloads the model weights, so it stays behind the installed-extra gate
# exactly like the oracle-gated imwatermark vectors).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not td.is_available(), reason="trustmark extra not installed")
def test_clean_image_reports_no_watermark(tmp_clean_png: Path):
    # With the decoder present, an un-watermarked image must report absent.
    assert td.detect_trustmark(tmp_clean_png) is None


# ---------------------------------------------------------------------------
# False-positive durability gate (mocked decoder: no network, no weights).
# The re-encode gate keeps real (durable) TrustMarks and drops BCH false
# positives that collapse under a mild JPEG round-trip.
# ---------------------------------------------------------------------------


class _FakeDecoder:
    """A TrustMark decoder whose successive ``decode`` calls return scripted
    ``(secret, present, schema)`` tuples -- the first for the original image, the
    second for the re-encoded copy used by the false-positive durability gate."""

    def __init__(self, *results: tuple[str, bool, int]):
        self._results = list(results)
        self.calls = 0

    def decode(self, _img: object, mode: str = "binary") -> tuple[str, bool, int]:
        assert mode == "binary"
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


class TestFalsePositiveGate:
    @pytest.fixture(autouse=True)
    def _force_available(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(td, "is_available", lambda: True)

    def _patch_decoder(self, monkeypatch: pytest.MonkeyPatch, decoder: _FakeDecoder) -> None:
        monkeypatch.setattr(td, "_decoder", lambda: decoder)

    def test_durable_watermark_survives_and_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
    ):
        decoder = _FakeDecoder(("secret", True, 2), ("secret", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        result = td.detect_trustmark(tmp_clean_png)
        assert result == "Adobe TrustMark (variant P, schema 2)"
        assert decoder.calls == 2  # original + re-encode

    def test_already_decoded_image_uses_the_same_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
    ):
        decoder = _FakeDecoder(("secret", True, 2), ("secret", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        with Image.open(tmp_clean_png) as img:
            result = td.detect_trustmark_image(img, source="test fixture")
        assert result == "Adobe TrustMark (variant P, schema 2)"
        assert decoder.calls == 2

    def test_false_positive_collapsing_on_reencode_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
    ):
        # Present on the original, absent after re-encode -> content-noise FP.
        decoder = _FakeDecoder(("01", True, 2), ("", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert td.detect_trustmark(tmp_clean_png) is None

    def test_schema_drift_on_reencode_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
    ):
        # Present both times but the schema changes -> not a stable watermark.
        decoder = _FakeDecoder(("01", True, 2), ("01", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert td.detect_trustmark(tmp_clean_png) is None

    def test_payload_drift_on_reencode_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
    ):
        decoder = _FakeDecoder(("first", True, 2), ("second", True, 2))
        self._patch_decoder(monkeypatch, decoder)
        assert td.detect_trustmark(tmp_clean_png) is None

    def test_weak_schema_three_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
    ):
        decoder = _FakeDecoder(("secret", True, 3))
        self._patch_decoder(monkeypatch, decoder)
        assert td.detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1

    def test_absent_skips_reencode(self, monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path):
        decoder = _FakeDecoder(("", False, -1))
        self._patch_decoder(monkeypatch, decoder)
        assert td.detect_trustmark(tmp_clean_png) is None
        assert decoder.calls == 1  # no second decode when the first is absent


@pytest.mark.skipif(not td.is_available(), reason="trustmark extra not installed")
def test_unavailable_decoder_yields_no_finding_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_clean_png: Path
):
    import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *names: False)
    assert td.detect_trustmark(tmp_clean_png) is None


# ---------------------------------------------------------------------------
# `trustmark` extra declaration + checker (this is the /capabilities surface:
# server.py's capabilities() reports extras_status(), which lists KNOWN_EXTRAS).
# ---------------------------------------------------------------------------


def test_pyproject_declares_the_trustmark_extra() -> None:
    project = load_toml("pyproject.toml")["project"]
    extras = project["optional-dependencies"]
    trustmark = extras["trustmark"]
    assert any(pkg.split(">=")[0].strip().lower() == "trustmark" for pkg in trustmark), trustmark
    # R1: torch extras stay standalone -- the aggregate stays on the numpy tier.
    assert "watermark-remover[trustmark]" not in extras["all"]


def test_check_optional_probes_trustmark(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import optional_deps

    probed: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: probed.append(name) or object())
    availability = optional_deps.check_optional("trustmark")
    assert availability.available is True
    assert availability.extra == "trustmark"
    assert probed == ["trustmark"]


def test_extras_status_lists_trustmark() -> None:
    import optional_deps

    assert "trustmark" in optional_deps.KNOWN_EXTRAS
    status = optional_deps.extras_status()
    assert "trustmark" in status
    assert set(status["trustmark"]) == {"available", "hint"}


# ---------------------------------------------------------------------------
# Soft-binding finding: inspect_soft_binding gains the TrustMark entry (seam
# agreed for this ticket). Vectors mock the detector -- no network, no weights.
# ---------------------------------------------------------------------------


class TestSoftBindingFinding:
    @pytest.fixture
    def report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from inspect_soft_binding import inspect_soft_binding

        def _run(scheme: str | None, *, available: bool = True, crash: Exception | None = None):
            monkeypatch.setattr(td, "is_available", lambda: available)

            def _detect(_path: Path) -> str | None:
                if crash is not None:
                    raise crash
                return scheme

            monkeypatch.setattr(td, "detect_trustmark", _detect)
            asset = tmp_path / "asset.txt"
            asset.write_text("ordinary prose", encoding="utf-8")
            return inspect_soft_binding(asset)

        return _run

    def test_durable_trustmark_is_a_soft_binding_finding(self, report):
        result = report("Adobe TrustMark (variant P, schema 1)")
        assert result["trustmark"]["available"] is True
        assert result["trustmark"]["scheme"] == "Adobe TrustMark (variant P, schema 1)"
        assert result["soft_binding"]["found"] is True
        assert result["warning"] is not None
        assert "TrustMark" in result["warning"]
        assert "re-link" in result["warning"]

    def test_absent_trustmark_is_not_a_finding(self, report):
        result = report(None)
        assert result["trustmark"] == {"available": True, "scheme": None}
        assert result["soft_binding"]["found"] is False
        assert result["warning"] is None

    def test_extra_absent_reports_unavailable_not_clean(self, report):
        result = report(None, available=False)
        assert result["trustmark"] == {"available": False, "scheme": None}
        assert result["soft_binding"]["found"] is False
        assert result["warning"] is None

    def test_decoder_crash_is_fail_soft(self, report):
        result = report(None, crash=RuntimeError("model download failed"))
        assert result["trustmark"]["available"] is True
        assert result["trustmark"]["scheme"] is None
        assert "model download failed" in result["trustmark"]["error"]
        assert result["soft_binding"]["found"] is False


def test_capabilities_surfacing_lists_the_trustmark_extra() -> None:
    """server.capabilities() reports extras_status(), so the extra surfaces there."""
    import server

    extras = server.capabilities()["extras"]
    assert "trustmark" in extras
    assert set(extras["trustmark"]) == {"available", "hint"}
