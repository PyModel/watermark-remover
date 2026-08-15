"""F4: truthful wmCt provenance-replacement marker (optional; strip-without-replacement stays default)."""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

from conftest import fake_command_result  # noqa: F401  (registered fixtures)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from image_meta import WMCT_KEYWORD, add_wmct_marker, clean_image
from png_chunks import iter_png_chunks

CLEAN_FILE = SCRIPTS / "clean_file.py"


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _minimal_png_with_text() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa test contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _wmct_chunks(data: bytes) -> list[bytes]:
    found = []
    keyword = WMCT_KEYWORD.encode("latin-1")
    for chunk in iter_png_chunks(data):
        if chunk.kind == keyword:
            found.append(bytes(chunk.payload))
    return found


def _write_png(tmp_path: Path, name: str = "input.png") -> Path:
    p = tmp_path / name
    p.write_bytes(_minimal_png_with_text())
    return p


def test_add_wmct_marker_inserts_valid_chunk() -> None:
    src = _minimal_png_with_text()
    marked = add_wmct_marker(src)
    # structurally valid: CRC-validated iteration succeeds, marker present before IEND
    chunks = list(iter_png_chunks(marked))
    kinds = [c.kind for c in chunks]
    assert WMCT_KEYWORD.encode("latin-1") in kinds
    assert kinds[-1] == b"IEND"
    marker = next(c for c in chunks if c.kind == WMCT_KEYWORD.encode("latin-1"))
    payload = bytes(marker.payload)
    assert WMCT_KEYWORD.encode("latin-1") in payload
    assert b"cleaned" in payload


def test_add_wmct_marker_rejects_non_png() -> None:
    import pytest

    with pytest.raises(ValueError):
        add_wmct_marker(b"\xff\xd8\xff\xd9")


def test_clean_image_marker_default_absent(tmp_path: Path) -> None:
    # Frictionless default: strip-without-replacement — no wmCt chunk written.
    src = _write_png(tmp_path)
    dest = tmp_path / "cleaned.png"
    report = clean_image(src, dest)
    assert report["wmct_marker"] is False
    assert _wmct_chunks(dest.read_bytes()) == []
    assert report["still_has_c2pa"] is False


def test_clean_image_marker_injected(tmp_path: Path) -> None:
    src = _write_png(tmp_path)
    dest = tmp_path / "cleaned.png"
    report = clean_image(src, dest, wmct_marker=True)
    assert report["wmct_marker"] is True
    markers = _wmct_chunks(dest.read_bytes())
    assert len(markers) == 1
    assert b"cleaned" in markers[0]
    # marker must not re-flag the image as AI metadata (would force residual exit)
    assert report["still_has_c2pa"] is False
    assert report["still_has_ai_metadata"] is False


def test_clean_image_marker_skipped_for_jpeg(tmp_path: Path) -> None:
    # Minimal JPEG: SOI, APP0, APP11 c2pa, SOS, EOI.
    app0 = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    app0_seg = b"\xff\xe0" + struct.pack(">H", len(app0) + 2) + app0
    app11 = b"JUMB" + b"c2pa-manifest-fake"
    app11_seg = b"\xff\xeb" + struct.pack(">H", len(app11) + 2) + app11
    sos_payload = b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    sos = b"\xff\xda" + struct.pack(">H", len(sos_payload) + 2) + sos_payload
    jpeg = b"\xff\xd8" + app0_seg + app11_seg + sos + b"\x00\x00" + b"\xff\xd9"
    src = tmp_path / "input.jpg"
    src.write_bytes(jpeg)
    dest = tmp_path / "cleaned.jpg"
    report = clean_image(src, dest, wmct_marker=True)
    assert report["wmct_marker"] is False
    assert any("wmCt" in a for a in report["actions"])


def test_clean_file_cli_wmct_marker(tmp_path: Path) -> None:
    src = _write_png(tmp_path)
    dest = tmp_path / "cleaned.png"
    result = subprocess.run(
        [sys.executable, str(CLEAN_FILE), str(src), "-o", str(dest), "--wmct-marker", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert _wmct_chunks(dest.read_bytes()) and WMCT_KEYWORD.encode("latin-1") in dest.read_bytes()


def test_clean_file_cli_no_marker_by_default(tmp_path: Path) -> None:
    src = _write_png(tmp_path)
    dest = tmp_path / "cleaned.png"
    result = subprocess.run(
        [sys.executable, str(CLEAN_FILE), str(src), "-o", str(dest), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert WMCT_KEYWORD.encode("latin-1") not in dest.read_bytes()
