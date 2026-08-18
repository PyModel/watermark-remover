"""Decompression-bomb defense and unsupported-format notes for image metadata.

- PNG zTXt/iTXt compressed text chunks are decompressed through a bounded
  zlib decompressor capped at MAX_PNG_TEXT_BYTES; a chunk whose output would
  exceed the cap fails closed (no text entry, no AI finding) instead of
  allocating unbounded memory.
- inspect_image annotates ImageInspectReport.notes when the detected format
  has no inspection support (the 'unknown' fallback).
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from image_meta import MAX_PNG_TEXT_BYTES, inspect_image, inspect_png

LARGE_BUT_OK_BYTES = 1024 * 1024  # well under the cap, far larger than a normal chunk


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _minimal_png_with_text_chunk(ctype: bytes, payload: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(ctype, payload)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _ztext_payload(value: bytes) -> bytes:
    # keyword \0 compression-method(0) compressed text
    return b"Software\x00\x00" + zlib.compress(value)


def _itext_compressed_payload(value: bytes) -> bytes:
    # keyword \0 comp-flag(1) comp-method(0) lang \0 tkey \0 compressed text
    return b"Software\x00\x01\x00\x00\x00" + zlib.compress(value)


def _bomb_value() -> bytes:
    """Decompressed text that would exceed the cap while still naming a generator."""
    return b"ChatGPT" + b"\x00" * (MAX_PNG_TEXT_BYTES + 1024 * 1024)


# ---------------------------------------------------------------------------
# Decompression-bomb defense (zTXt / iTXt)
# ---------------------------------------------------------------------------


def test_ztext_bomb_fails_closed():
    """A zTXt chunk expanding past the cap yields no AI finding (fail closed)."""
    data = _minimal_png_with_text_chunk(b"zTXt", _ztext_payload(_bomb_value()))
    has_c2pa, has_ai, findings = inspect_png(data)
    assert has_c2pa is False
    assert has_ai is False
    assert not any("AI generator" in f for f in findings)


def test_itext_compressed_bomb_fails_closed():
    """A compressed iTXt chunk expanding past the cap also fails closed."""
    data = _minimal_png_with_text_chunk(b"iTXt", _itext_compressed_payload(_bomb_value()))
    has_c2pa, has_ai, findings = inspect_png(data)
    assert has_c2pa is False
    assert has_ai is False
    assert not any("AI generator" in f for f in findings)


def test_large_under_cap_ztext_still_detected():
    """Legitimate large compressed text under the cap still flags the generator."""
    value = b"ChatGPT" + b"x" * (LARGE_BUT_OK_BYTES - len(b"ChatGPT"))
    data = _minimal_png_with_text_chunk(b"zTXt", _ztext_payload(value))
    has_c2pa, has_ai, findings = inspect_png(data)
    assert has_c2pa is False
    assert has_ai is True
    assert any("AI generator" in f and "ChatGPT" in f for f in findings)


def test_compressed_itext_still_detected():
    """The compressed iTXt path (comp-flag=1) still flags the generator."""
    data = _minimal_png_with_text_chunk(b"iTXt", _itext_compressed_payload(b"ChatGPT"))
    has_c2pa, has_ai, findings = inspect_png(data)
    assert has_c2pa is False
    assert has_ai is True
    assert any("AI generator" in f and "ChatGPT" in f for f in findings)


# ---------------------------------------------------------------------------
# Unsupported-format notes on ImageInspectReport
# ---------------------------------------------------------------------------


def test_unknown_format_gets_not_inspected_note(tmp_path: Path):
    src = tmp_path / "unknown.bin"
    src.write_bytes(b"no magic bytes here")
    report = inspect_image(src)
    assert report.format == "unknown"
    assert report.findings == ["unsupported format"]
    assert report.has_c2pa is False
    assert report.has_ai_metadata is False
    assert "format 'unknown' is not inspected" in report.notes


def test_supported_format_has_no_not_inspected_note(tmp_path: Path):
    src = tmp_path / "ok.png"
    src.write_bytes(_minimal_png_with_text_chunk(b"tEXt", b"Software\x00ChatGPT"))
    report = inspect_image(src)
    assert report.format == "png"
    assert not any("not inspected" in note for note in report.notes)
