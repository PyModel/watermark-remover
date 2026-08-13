"""Tests for PNG/JPEG metadata strip."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
from image_meta import clean_image, inspect_jpeg, strip_jpeg, strip_png


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _minimal_png_with_text() -> bytes:
    """1x1 IHDR + tEXt with c2pa marker + IDAT + IEND (not a valid image decode, structure OK)."""
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 1x1 RGB 8bit
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    # minimal empty IDAT may fail decoders; enough for our chunk walker
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa test contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _minimal_jpeg_with_app11() -> bytes:
    """Minimal JPEG: SOI, APP0 JFIF, APP11 with c2pa, SOS stub, EOI."""
    app0 = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    app0_seg = b"\xff\xe0" + struct.pack(">H", len(app0) + 2) + app0
    app11 = b"JUMB" + b"c2pa-manifest-fake"
    app11_seg = b"\xff\xeb" + struct.pack(">H", len(app11) + 2) + app11
    sos_payload = b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    sos = b"\xff\xda" + struct.pack(">H", len(sos_payload) + 2) + sos_payload
    entropy = b"\x00\x00"  # dummy
    return b"\xff\xd8" + app0_seg + app11_seg + sos + entropy + b"\xff\xd9"


def test_strip_png_removes_text_c2pa(tmp_path: Path):
    data = _minimal_png_with_text()
    cleaned, actions = strip_png(data)
    assert b"c2pa" not in cleaned.lower() or b"tEXt" not in cleaned
    assert any("drop" in a for a in actions)
    # structural: still starts with PNG sig and has IEND
    assert cleaned.startswith(b"\x89PNG")
    assert b"IEND" in cleaned


def test_strip_jpeg_removes_app11():
    data = _minimal_jpeg_with_app11()
    cleaned, actions = strip_jpeg(data)
    assert b"c2pa-manifest-fake" not in cleaned
    assert any("APP11" in a or "drop" in a for a in actions)
    assert cleaned.startswith(b"\xff\xd8")


def test_benign_app11_is_not_reported_as_c2pa_and_is_preserved_selectively():
    payload = b"vendor-private-data"
    segment = b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload
    data = b"\xff\xd8" + segment + b"\xff\xd9"
    has_c2pa, has_ai, findings = inspect_jpeg(data)
    assert not has_c2pa and not has_ai
    assert not findings
    cleaned, _ = strip_jpeg(data, strip_all_app=False)
    assert payload in cleaned


def test_keep_non_ai_png_metadata_preserves_benign_exif():
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    data = (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"eXIf", b"Camera=Canon")
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )
    selective, _ = strip_png(data, strip_all_text=False)
    aggressive, _ = strip_png(data, strip_all_text=True)
    assert b"Camera=Canon" in selective
    assert b"Camera=Canon" not in aggressive


def test_cleaners_fail_closed_on_truncated_binary():
    truncated_png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 50) + b"tEXt" + b"short"
    try:
        strip_png(truncated_png)
    except ValueError as error:
        assert "truncated" in str(error) or "IEND" in str(error)
    else:
        raise AssertionError("expected truncated PNG rejection")

    valid_png = bytearray(_minimal_png_with_text())
    valid_png[-1] ^= 0xFF  # corrupt IEND CRC
    try:
        strip_png(bytes(valid_png))
    except ValueError as error:
        assert "CRC mismatch" in str(error)
    else:
        raise AssertionError("expected CRC rejection")

    truncated_jpeg = b"\xff\xd8\xff\xe1\x00\x20short"
    try:
        strip_jpeg(truncated_jpeg)
    except ValueError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("expected truncated JPEG rejection")


def test_exiftool_nonzero_is_logged_as_failure(tmp_path: Path, monkeypatch):
    src = tmp_path / "input.png"
    src.write_bytes(_minimal_png_with_text())
    dest = tmp_path / "out.png"
    monkeypatch.setattr(
        image_meta,
        "which",
        lambda name: "/fake/exiftool" if name == "exiftool" else None,
    )
    monkeypatch.setattr(
        image_meta.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="denied"),
    )
    report = clean_image(src, dest)
    assert any("exiftool failed (rc=7)" in action for action in report["actions"])
    assert not any(action == "exiftool -all= pass" for action in report["actions"])


def test_clean_image_json_preserves_residual_exit_code(tmp_path: Path):
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    # Literal marker in critical IDAT is preserved by design and remains residual.
    data = sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", b"c2pa") + _png_chunk(b"IEND", b"")
    src = tmp_path / "residual.png"
    src.write_bytes(data)
    script = SCRIPTS / "clean_image.py"
    result = subprocess.run(
        [sys.executable, str(script), str(src), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["still_has_c2pa"] is True


def test_clean_image_roundtrip(tmp_path: Path):
    src = tmp_path / "t.png"
    src.write_bytes(_minimal_png_with_text())
    dest = tmp_path / "t.cleaned.png"
    result = clean_image(src, dest)
    assert dest.is_file()
    assert result["bytes_out"] > 0
