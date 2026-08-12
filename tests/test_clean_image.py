"""Tests for PNG/JPEG metadata strip."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from image_meta import clean_image, inspect_image, strip_jpeg, strip_png  # noqa: E402


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


def test_clean_image_roundtrip(tmp_path: Path):
    src = tmp_path / "t.png"
    src.write_bytes(_minimal_png_with_text())
    dest = tmp_path / "t.cleaned.png"
    result = clean_image(src, dest)
    assert dest.is_file()
    assert result["bytes_out"] > 0
