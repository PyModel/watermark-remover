"""Wiring tests for the ``detect`` extra: declaration, checker, and HTTP surface.

The extra is the delivery contract for the open DWT-DCT detector: declared in
pyproject, answerable by ``check_optional`` (probing exactly numpy + cv2 +
pywt), listed in ``extras_status`` / ``/capabilities``, and carried by the
``/detect`` image path fail-soft — an absent extra is reported as
unavailable/unknown, never as "no watermark found".
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from conftest import load_toml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import optional_deps


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _post(conn, path: str, payload: dict):
    import json

    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


@pytest.fixture(scope="module")
def conn():
    import http.client
    import threading

    import server

    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _pyproject() -> dict:
    return load_toml("pyproject.toml")


def test_detect_extra_is_declared_and_checkable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`detect` joins the registry and probes exactly numpy + cv2 + pywt."""
    assert "detect" in optional_deps.KNOWN_EXTRAS

    probed: list[str] = []

    def recording_import(name: str):
        probed.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", recording_import)
    availability = optional_deps.check_optional("detect")
    assert availability.available is True
    assert availability.extra == "detect"
    assert probed == ["numpy", "cv2", "pywt"]


def test_detect_extra_missing_deps_report_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", missing)
    availability = optional_deps.check_optional("detect")
    assert availability.available is False
    assert "Install watermark-remover[detect]" in availability.hint


def test_aggregate_all_extra_joins_the_registry_with_a_real_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`all` stops being a pyproject-only name: it is checkable, honestly."""
    assert "all" in optional_deps.KNOWN_EXTRAS
    assert "all" in optional_deps.extras_status()

    probed: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: probed.append(name) or object())
    availability = optional_deps.check_optional("all")
    assert availability.available is True
    # The checker covers every aggregate member declared at this commit:
    # visible+quality+ai+provenance+detect (source-classify joins later).
    assert {"cv2", "skimage", "torch", "c2pa", "numpy", "pywt"} <= set(probed)


def test_pyproject_declares_detect_and_all_gains_it() -> None:
    """The manifest is the source of truth for what the checker probes."""
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]
    detect = extras["detect"]
    assert any(pkg.startswith("PyWavelets") or pkg == "pywavelets" for pkg in detect), detect
    assert any(pkg.startswith("opencv") for pkg in detect), detect
    assert any(pkg.lower() == "numpy" for pkg in detect), detect
    assert "watermark-remover[detect]" in extras["all"]


def test_the_dev_only_oracle_is_never_a_declared_dependency() -> None:
    """invisible-watermark stays a fixture-generation tool, never a dependency."""
    project = _pyproject()["project"]
    declared: list[str] = list(project.get("dependencies") or [])
    for extra_deps in (project.get("optional-dependencies") or {}).values():
        declared.extend(extra_deps)
    for requirement in declared:
        name = requirement.split(";")[0].split("[")[0].split("==")[0].split(">=")[0].strip().lower()
        assert name not in {"invisible-watermark", "imwatermark"}, requirement
    test_requirements = (ROOT / "requirements-test.txt").read_text(encoding="utf-8").lower()
    assert "invisible-watermark" not in test_requirements


def test_extras_status_reports_detect() -> None:
    status = optional_deps.extras_status()
    assert "detect" in status
    assert set(status["detect"]) == {"available", "hint"}


def test_capabilities_report_lists_detect() -> None:
    import server

    extras = server.capabilities()["extras"]
    assert "detect" in extras


# ---------------------------------------------------------------------------
# identify signal + /detect image wiring


def test_invisible_watermark_signal_maps_a_scheme_into_the_evidence_vocabulary() -> None:
    import identify

    signal = identify.invisible_watermark_signal("Stable Diffusion XL")
    assert signal["scheme"] == "Stable Diffusion XL"
    assert signal["asserts_ai"] is True
    assert "Stable Diffusion XL" in signal["finding"]
    # Honesty contract: the fragility caveat rides along — absence is never clean.
    assert "absence is never" in signal["note"]


def test_invisible_watermark_signal_absence_is_unknown_never_clean() -> None:
    import identify

    signal = identify.invisible_watermark_signal(None)
    assert signal["scheme"] is None
    assert signal["asserts_ai"] is False
    assert signal["finding"] is None
    assert "absence is never" in signal["note"]


def test_detect_image_reports_the_invisible_watermark_entry(conn) -> None:
    """A plain tiny PNG still carries the invisible_watermark detector entry."""
    import struct
    import zlib

    def chunk(ctype: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(ctype)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )
    status, body = _post(conn, "/detect", {"file": _b64(png), "name": "shot.png"})
    assert status == 200
    entries = [d for d in body["detections"] if d.get("detector") == "invisible_watermark"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["available"] is True
    assert entry["found"] is False
    # Honesty contract: a miss is never reported as clean.
    assert "absence is never" in entry["note"]


def test_detect_image_carries_the_scheme_when_decoded(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    import invisible_watermark

    monkeypatch.setattr(
        invisible_watermark, "detect_invisible_watermark", lambda *_a, **_k: "Stable Diffusion XL"
    )
    data = (ROOT / "tests" / "fixtures" / "sample_meta.svg").read_bytes()
    status, body = _post(conn, "/detect", {"file": _b64(data), "name": "shot.png"})
    assert status == 200
    entry = next(d for d in body["detections"] if d.get("detector") == "invisible_watermark")
    assert entry["found"] is True
    assert entry["scheme"] == "Stable Diffusion XL"
    assert entry["asserts_ai"] is True


def test_detect_image_without_the_extra_is_unavailable_not_clean(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    import invisible_watermark

    monkeypatch.setattr(invisible_watermark, "is_available", lambda: False)
    data = (ROOT / "tests" / "fixtures" / "sample_meta.svg").read_bytes()
    status, body = _post(conn, "/detect", {"file": _b64(data), "name": "shot.png"})
    assert status == 200
    entry = next(d for d in body["detections"] if d.get("detector") == "invisible_watermark")
    assert entry["available"] is False
    assert "detect" in entry["error"]
    assert "found" not in entry  # no detector ran: no negative finding exists


def test_detect_image_detector_crash_is_fail_soft(conn, monkeypatch: pytest.MonkeyPatch) -> None:
    import invisible_watermark

    def explode(*_a, **_k):
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(invisible_watermark, "detect_invisible_watermark", explode)
    data = (ROOT / "tests" / "fixtures" / "sample_meta.svg").read_bytes()
    status, body = _post(conn, "/detect", {"file": _b64(data), "name": "shot.png"})
    assert status == 200
    entry = next(d for d in body["detections"] if d.get("detector") == "invisible_watermark")
    assert entry["available"] is True
    assert "error" in entry
    assert "found" not in entry  # a failed detector reports no negative finding
