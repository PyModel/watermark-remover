"""Tests for HEIF/AVIF (ISO-BMFF) provenance neutralization."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from heif_meta import (
    C2PA_BMFF_UUID,
    XMP_UUID,
    clean_heif,
    detect_heif,
    inspect_heif,
    neutralize_heif,
)
from image_meta import (
    clean_image,
    detect_format,
    inspect_avif,
    inspect_heic,
    inspect_image,
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
    item_info = struct.pack(">HH", 1, 0) + item_type + item_type + b"\x00"
    if item_type == b"mime":
        content_type = (
            b"application/rdf+xml"
            if item_payload.lstrip().startswith(b"<x:xmpmeta")
            else b"application/octet-stream"
        )
        item_info += content_type + b"\x00"
    infe = _fullbox(b"infe", 2, item_info)
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


def test_c2pa_uuid_box_is_detected_and_neutralized():
    ftyp = _box(b"ftyp", b"heic" + b"\x00\x00\x00\x00" + b"mif1heic")
    data = ftyp + _box(b"uuid", C2PA_BMFF_UUID + b"manifest payload")
    has_c2pa, has_ai, findings, _ = inspect_heif(data)
    assert has_c2pa and has_ai
    assert any("uuid" in finding for finding in findings)
    cleaned, actions = neutralize_heif(data)
    assert C2PA_BMFF_UUID not in cleaned
    assert b"free" in cleaned
    assert any("uuid" in action for action in actions)


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


def test_external_data_reference_is_reported_and_cleaning_fails_closed():
    data = make_heif(with_jumb=False, data_reference_index=1)
    has_c2pa, has_ai, findings, _ = inspect_heif(data)
    assert has_ai and has_c2pa  # byte-scan still sees C2PA tokens in the external item
    assert any("unsupported external/idat" in finding for finding in findings)
    try:
        neutralize_heif(data, strip_all_metadata=True)
    except ValueError as error:
        assert "unsupported HEIF metadata extent layout" in str(error)
    else:
        raise AssertionError("expected unsupported extent rejection")
    assert EXIF_PAYLOAD in data
    assert PIXEL_BYTES in data


def test_strip_all_preserves_xml_mime_that_is_not_declared_xmp():
    payload = b"<svg xmlns='http://www.w3.org/2000/svg'><text>OpenAI c2pa</text></svg>"
    data = make_heif(with_jumb=False, item_type=b"mime", item_payload=payload)
    cleaned, _ = neutralize_heif(data, strip_all_metadata=True)
    assert payload in cleaned
    assert PIXEL_BYTES in cleaned


def test_strip_all_zeroes_declared_xmp_mime_item():
    payload = b"<x:xmpmeta>OpenAI c2pa</x:xmpmeta>"
    data = make_heif(with_jumb=False, item_type=b"mime", item_payload=payload)
    cleaned, _ = neutralize_heif(data, strip_all_metadata=True)
    assert payload not in cleaned
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


# ---- Ported from THEIRS test_avif_heic.py (adapted to OURS naming) ----


def _build_box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + fourcc + payload


def _build_full_box(fourcc: bytes, version: int, flags: int, payload: bytes) -> bytes:
    vf = struct.pack(">I", (version << 24) | (flags & 0xFFFFFF))
    return _build_box(fourcc, vf + payload)


def _minimal_avif_with_c2pa_and_xmp() -> bytes:
    """Minimal ISOBMFF AVIF with ftyp, meta (with XMP uuid + jumb sub-box), top-level jumb, and mdat."""
    ftyp = _build_box(b"ftyp", b"avif\x00\x00\x00\x00avifmif1")
    jumb_sub = _build_box(b"jumb", b"c2pa.manifest.store.v1")
    xmp_uuid_sub = _build_box(
        b"uuid",
        XMP_UUID
        + b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><rdf:Description rdf:about='' xmlns:ai='http://ns.adobe.com/ai/'><ai:GeneratedBy>Midjourney</ai:GeneratedBy></rdf:Description></rdf:RDF></x:xmpmeta>",
    )
    hdlr = _build_full_box(
        b"hdlr",
        0,
        0,
        b"\x00\x00\x00\x00pict\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00PictureHandler\x00",
    )
    meta = _build_full_box(b"meta", 0, 0, hdlr + jumb_sub + xmp_uuid_sub)
    top_jumb = _build_box(b"jumb", b"c2pa.claim.v1 contentcredentials")
    mdat = _build_box(b"mdat", b"\x00\x01\x02\x03\x04\x05image_pixel_data")
    return ftyp + meta + top_jumb + mdat


def _minimal_heic_with_xmp() -> bytes:
    """Minimal ISOBMFF HEIC with ftyp, meta, and XMP uuid box."""
    ftyp = _build_box(b"ftyp", b"heic\x00\x00\x00\x00mif1heic")
    hdlr = _build_full_box(
        b"hdlr",
        0,
        0,
        b"\x00\x00\x00\x00pict\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00PictureHandler\x00",
    )
    xmp_uuid = _build_box(
        b"uuid",
        XMP_UUID
        + b"<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF><rdf:Description digitalSourceType='trainedAlgorithmicMedia'/></rdf:RDF></x:xmpmeta>",
    )
    meta = _build_full_box(b"meta", 0, 0, hdlr)
    mdat = _build_box(b"mdat", b"\xaa\xbb\xcc\xddhevc_stream")
    return ftyp + meta + xmp_uuid + mdat


def test_detect_format_avif_and_heif():
    avif_bytes = _minimal_avif_with_c2pa_and_xmp()
    assert detect_format(avif_bytes) == "avif"

    heif_bytes = _minimal_heic_with_xmp()
    assert detect_format(heif_bytes) == "heif"

    avis_bytes = _build_box(b"ftyp", b"avis\x00\x00\x00\x00avis") + b"data"
    assert detect_format(avis_bytes) == "avif"

    mif1_bytes = _build_box(b"ftyp", b"mif1\x00\x00\x00\x00mif1heic") + b"data"
    assert detect_format(mif1_bytes) == "heif"


def test_inspect_heif_detects_c2pa_and_xmp():
    avif_bytes = _minimal_avif_with_c2pa_and_xmp()
    has_c2pa, has_ai, findings, _details = inspect_heif(avif_bytes)
    assert has_c2pa is True
    assert has_ai is True
    assert any("C2PA" in f or "jumb" in f.lower() for f in findings)
    assert any("uuid" in f.lower() or "XMP" in f for f in findings)


def test_inspect_heif_heic_ai_metadata():
    heic_bytes = _minimal_heic_with_xmp()
    _has_c2pa, has_ai, findings, _details = inspect_heif(heic_bytes)
    assert has_ai is True
    assert any("trainedAlgorithmicMedia" in f or "XMP" in f for f in findings)


def test_neutralize_heif_removes_c2pa_and_xmp():
    avif_bytes = _minimal_avif_with_c2pa_and_xmp()
    cleaned, actions = neutralize_heif(avif_bytes)

    assert any("jumb" in a.lower() for a in actions)
    assert any("xmp" in a.lower() or "uuid" in a.lower() for a in actions)

    # Re-inspect cleaned bytes
    has_c2pa, has_ai, _findings, _details = inspect_heif(cleaned)
    assert has_c2pa is False
    assert has_ai is False

    # Structure still valid ISOBMFF and starts with ftyp
    assert detect_format(cleaned) == "avif"
    assert b"mdat" in cleaned
    assert b"jumb" not in cleaned.lower()


def test_clean_image_avif_roundtrip(tmp_path: Path):
    src = tmp_path / "photo.avif"
    src.write_bytes(_minimal_avif_with_c2pa_and_xmp())
    dest = tmp_path / "photo.cleaned.avif"

    report = clean_image(src, dest)
    assert dest.is_file()
    assert report["format"] == "avif"
    assert report["still_has_c2pa"] is False
    assert report["still_has_ai_metadata"] is False
    assert any("jumb" in a.lower() for a in report["actions"])

    inspect_rep = inspect_image(dest)
    assert inspect_rep.format == "avif"
    assert inspect_rep.has_c2pa is False
    assert inspect_rep.has_ai_metadata is False


def test_clean_image_heic_roundtrip(tmp_path: Path):
    src = tmp_path / "camera.heic"
    src.write_bytes(_minimal_heic_with_xmp())
    dest = tmp_path / "camera.cleaned.heic"

    report = clean_image(src, dest)
    assert dest.is_file()
    assert report["format"] == "heif"
    assert report["still_has_ai_metadata"] is False
    assert any("uuid" in a.lower() or "xmp" in a.lower() for a in report["actions"])


def test_fixture_avif_and_heic_are_detected_and_cleaned(tmp_path: Path):
    """The ported sample C2PA fixtures must round-trip through clean_image."""
    for name, expect in (("sample_c2pa.avif", "avif"), ("sample_c2pa.heic", "heif")):
        fixture = Path(__file__).resolve().parent / "fixtures" / name
        data = fixture.read_bytes()
        assert detect_format(data) == expect
        src = tmp_path / name
        src.write_bytes(data)
        dest = tmp_path / ("cleaned-" + name)
        report = clean_image(src, dest)
        assert report["format"] == expect
        assert report["still_has_c2pa"] is False
        assert report["still_has_ai_metadata"] is False
