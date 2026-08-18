"""Tests for format routing: unknown kind + header-only classification."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from asset_kind import classify, classify_bytes


def _boom(*_args, **_kwargs):
    raise AssertionError("full read_bytes() was not expected here")


def test_classify_extension_wins_without_reading(monkeypatch, tmp_path):
    f = tmp_path / "note.txt"
    f.write_bytes(b"anything, the extension decides")
    monkeypatch.setattr(Path, "read_bytes", _boom)
    assert classify(f) == "text"


def test_classify_header_sniffs_image_without_full_read(monkeypatch, tmp_path):
    f = tmp_path / "no_extension"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    monkeypatch.setattr(Path, "read_bytes", _boom)
    assert classify(f) == "image"


def test_classify_header_sniffs_pdf_without_full_read(monkeypatch, tmp_path):
    f = tmp_path / "no_extension"
    f.write_bytes(b"%PDF-1.7\n" + b"0" * 100)
    monkeypatch.setattr(Path, "read_bytes", _boom)
    assert classify(f) == "container"


def test_classify_zip_needs_full_read_for_central_directory(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    f = tmp_path / "no_extension"
    f.write_bytes(buf.getvalue())
    assert classify(f) == "container"


def test_classify_truncated_zip_header_is_unknown(tmp_path):
    f = tmp_path / "no_extension"
    f.write_bytes(b"PK\x03\x04this is not a real zip")
    assert classify(f) == "unknown"


def test_classify_unknown_for_unrecognized_bytes(tmp_path):
    f = tmp_path / "no_extension"
    f.write_bytes(b"just plain text with no magic and no extension")
    assert classify(f) == "unknown"


def test_classify_bytes_unknown_fallback():
    assert classify_bytes(b"random bytes \x00\xff", None) == "unknown"
    assert classify_bytes(b"\x89PNG\r\n\x1a\nrest", None) == "image"
    assert classify_bytes(b"PK\x03\x04rest", None) == "unknown"  # not a full zip


def _ftyp(brand: bytes) -> bytes:
    payload = brand + b"\x00\x00\x00\x00" + brand
    return (len(payload) + 8).to_bytes(4, "big") + b"ftyp" + payload


def test_classify_bytes_heif_magic_is_image():
    # detect_format reports the whole HEIF/HEIC family as "heif" (never
    # "heic"); routing must accept that token or HEIF bytes land in "unknown".
    assert classify_bytes(_ftyp(b"heic"), None) == "image"
    assert classify_bytes(_ftyp(b"heic"), "") == "image"
    assert classify_bytes(_ftyp(b"heic"), ".bin") == "image"


def test_classify_heif_file_without_known_extension(tmp_path):
    f = tmp_path / "photo.blob"
    f.write_bytes(_ftyp(b"heic") + b"\x00" * 64)
    assert classify(f) == "image"
