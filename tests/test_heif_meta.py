"""Tests for HEIF/AVIF (ISO-BMFF) provenance neutralization."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from heif_meta import clean_heif, detect_heif, inspect_heif, neutralize_heif
from image_meta import (
    clean_image,
    detect_format,
    inspect_avif,
    inspect_heic,
    strip_avif,
    strip_heic,
)

EXIF_PAYLOAD = b"Exif\x00\x00Canon EOS R5 c2pa.jumb AIGC digitalSourceType trainedAlgorithmicMedia"
PIXEL_BYTES = b"\x89\x00\xff\x13PIXELDATA\x37"


def _box(btype: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + btype + payload


def _fullbox(btype: bytes, version: int, payload: bytes) -> bytes:
    return _box(btype, bytes([version]) + b"\x00\x00\x00" + payload)


def make_heif(
    major: bytes = b"heic",
    with_jumb: bool = True,
    *,
    item_type: bytes = b"Exif",
    item_payload: bytes = EXIF_PAYLOAD,
    data_reference_index: int = 0,
) -> bytes:
    ftyp = _box(b"ftyp", major + b"\x00\x00\x00\x00" + b"mif1" + major)
    infe = _fullbox(b"infe", 2, struct.pack(">HH", 1, 0) + item_type + item_type + b"\x00")
    iinf = _fullbox(b"iinf", 0, struct.pack(">H", 1) + infe)

    def iloc(off: int) -> bytes:
        body = (
            bytes([0x44, 0x00])  # offset_size=4, length_size=4, base_offset_size=0
            + struct.pack(">H", 1)  # item_count
            + struct.pack(">HHH", 1, data_reference_index, 1)  # item_ID, data_ref, extent_count
            + struct.pack(">II", off, len(item_payload))
        )
        return _fullbox(b"iloc", 0, body)

    hdlr = _fullbox(b"hdlr", 0, b"\x00" * 4 + b"pict" + b"\x00" * 12 + b"Picture\x00")
    jumb = _box(b"jumb", b"c2pa contentcredentials manifest") if with_jumb else b""

    meta0 = _fullbox(b"meta", 0, hdlr + iinf + iloc(0) + jumb)
    exif_off = len(ftyp) + len(meta0) + 8  # after mdat header
    meta = _fullbox(b"meta", 0, hdlr + iinf + iloc(exif_off) + jumb)
    assert len(meta) == len(meta0)
    return ftyp + meta + _box(b"mdat", item_payload + PIXEL_BYTES)


def test_detect_heif_brands():
    assert detect_heif(make_heif(b"heic")) == "heif"
    assert detect_heif(make_heif(b"avif")) == "avif"
    assert detect_heif(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) == "unknown"
    assert detect_format(make_heif(b"heic")) == "heif"
    assert detect_format(make_heif(b"avif")) == "avif"


def test_inspect_flags_c2pa_and_ai():
    has_c2pa, has_ai, findings, details = inspect_heif(make_heif())
    assert has_c2pa and has_ai
    assert details["format"] == "heif"
    assert any("jumb" in f for f in findings)
    assert any("Exif item" in f for f in findings)


def test_neutralize_keep_non_ai_preserves_camera_exif_and_pixels():
    data = make_heif()
    cleaned, actions = neutralize_heif(data, strip_all_metadata=False)
    assert len(cleaned) == len(data)  # in-place: offsets preserved
    assert b"Canon EOS R5" in cleaned  # camera EXIF preserved
    assert PIXEL_BYTES in cleaned  # pixel data untouched
    for marker in (b"c2pa", b"jumb", b"AIGC", b"digitalSourceType"):
        assert marker not in cleaned.lower() or marker == b"jumb"  # type retyped to free
    assert b"jumb" not in cleaned  # box type overwritten
    assert b"free" in cleaned
    has_c2pa, has_ai, _, _ = inspect_heif(cleaned)
    assert not has_c2pa and not has_ai
    assert any("neutralized" in a for a in actions)


def test_neutralize_strip_all_zeroes_exif_extent():
    data = make_heif()
    cleaned, _ = neutralize_heif(data, strip_all_metadata=True)
    assert len(cleaned) == len(data)
    assert b"Canon EOS R5" not in cleaned
    assert PIXEL_BYTES in cleaned
    has_c2pa, has_ai, _, _ = inspect_heif(cleaned)
    assert not has_c2pa and not has_ai


def test_image_meta_delegates():
    data = make_heif(b"avif")
    has_c2pa, has_ai, _ = inspect_avif(data)
    assert has_c2pa and has_ai
    cleaned, _actions = strip_avif(data, strip_all=True)
    assert detect_heif(cleaned) == "avif"
    assert b"Canon" not in cleaned
    cleaned2, _ = strip_heic(make_heif(b"heic"), strip_all=False)
    assert b"Canon EOS R5" in cleaned2
    has_c2pa2, has_ai2, _ = inspect_heic(cleaned2)
    assert not has_c2pa2 and not has_ai2


def test_external_data_reference_is_never_edited_as_local_extent():
    data = make_heif(with_jumb=False, data_reference_index=1)
    cleaned, _ = neutralize_heif(data, strip_all_metadata=True)
    assert EXIF_PAYLOAD in cleaned
    assert PIXEL_BYTES in cleaned


def test_strip_all_preserves_non_xmp_mime_item():
    payload = b"application/octet-stream\x00BINARY-AUXILIARY-DATA"
    data = make_heif(with_jumb=False, item_type=b"mime", item_payload=payload)
    cleaned, _ = neutralize_heif(data, strip_all_metadata=True)
    assert payload in cleaned
    assert PIXEL_BYTES in cleaned


def test_clean_image_routes_heif(tmp_path: Path):
    src = tmp_path / "photo.heic"
    src.write_bytes(make_heif())
    dest = tmp_path / "photo.cleaned.heic"
    result = clean_image(src, dest)
    assert result["format"] == "heif"
    assert result["bytes_out"] == result["bytes_in"]
    assert not result["still_has_c2pa"]
    assert not result["still_has_ai_metadata"]
    assert detect_heif(dest.read_bytes()) == "heif"  # still a valid HEIF


def test_malformed_iloc_is_reported_and_cleaning_fails_closed():
    ftyp = _box(b"ftyp", b"heic" + b"\x00\x00\x00\x00" + b"mif1heic")
    # iloc v0 declares one item but contains no item record.
    iloc = _fullbox(b"iloc", 0, bytes([0x44, 0x00]) + struct.pack(">H", 1))
    malformed = ftyp + _fullbox(b"meta", 0, iloc)
    has_c2pa, has_ai, findings, details = inspect_heif(malformed)
    assert not has_c2pa and has_ai
    assert details["malformed"] is True
    assert any("malformed HEIF metadata table" in finding for finding in findings)
    try:
        neutralize_heif(malformed)
    except ValueError as error:
        assert "truncated iloc" in str(error)
    else:
        raise AssertionError("expected malformed iloc rejection")


def test_clean_heif_rejects_non_bmff(tmp_path: Path):
    src = tmp_path / "x.heic"
    src.write_bytes(b"not a bmff file at all")
    try:
        clean_heif(src, tmp_path / "out.heic")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
