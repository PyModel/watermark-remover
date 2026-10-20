"""Bounded TC260 AIGC label parsing and its detection seam.

Parser vectors adapted from the upstream metadata tests. The image seam tests
(fixture file in → inspect findings out) cover the EXIF/XMP/APPn placements in
``image_meta`` and the ISO-BMFF AIGC tag in ``heif_meta``.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import heif_meta
import image_meta
from tc260_label import (
    MAX_TC260_SCAN_WINDOW_BYTES,
    MAX_TC260_VALUE_BYTES,
    label_from_metadata,
    label_span,
    parse_tc260_aigc_json,
)

FIXTURES = ROOT / "tests" / "fixtures" / "tc260"
DOUBAO_USCC = "91110102MACQD9K640"
PRODUCE_ID = "c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"


def _label_json(**overrides: object) -> str:
    label = {"Label": "1", "ContentProducer": DOUBAO_USCC, "ProduceID": PRODUCE_ID}
    label.update(overrides)  # type: ignore[arg-type]
    return json.dumps(label, separators=(",", ":"))


class TestParseTc260AigcJson:
    def test_a_producer_label_parses_field_aware(self) -> None:
        parsed = parse_tc260_aigc_json(_label_json().encode())
        assert parsed == {"Label": "1", "ContentProducer": DOUBAO_USCC, "ProduceID": PRODUCE_ID}

    def test_structured_reserved_fields_survive_as_json(self) -> None:
        security = {"SecurityData": {"Type": "TC260PG"}}
        parsed = parse_tc260_aigc_json(_label_json(ReservedCode1=security).encode())
        assert parsed is not None
        assert json.loads(parsed["ReservedCode1"]) == security

    def test_service_provider_schema_is_accepted(self) -> None:
        label = {"ServiceProvider": "91110102MACQD9K640", "ServiceUser": "user-1"}
        assert parse_tc260_aigc_json(json.dumps(label).encode()) == label

    def test_field_names_canonicalize_case_insensitively(self) -> None:
        parsed = parse_tc260_aigc_json(b'{"label":"1","contentproducer":"x"}')
        assert parsed == {"Label": "1", "ContentProducer": "x"}

    def test_a_json_object_without_tc260_fields_is_rejected(self) -> None:
        assert parse_tc260_aigc_json(b'{"weather":"sunny"}') is None

    def test_non_object_json_is_rejected(self) -> None:
        assert parse_tc260_aigc_json(b'["Label","1"]') is None
        assert parse_tc260_aigc_json(b'"Label"') is None

    def test_malformed_json_is_a_stable_rejection(self) -> None:
        assert parse_tc260_aigc_json(b'{"Label":') is None
        assert parse_tc260_aigc_json(b"") is None

    def test_oversized_payload_is_a_bounded_refusal(self) -> None:
        # A label just over the cap is refused before any JSON work, findings-free.
        oversized = b'{"Label":"' + b"x" * MAX_TC260_VALUE_BYTES + b'"}'
        assert len(oversized) > MAX_TC260_VALUE_BYTES
        assert parse_tc260_aigc_json(oversized) is None

    def test_non_utf8_payload_is_a_stable_rejection(self) -> None:
        assert parse_tc260_aigc_json(b'{"Label":"\xff\xfe\xfd"}') is None

    def test_nul_and_space_padding_is_tolerated(self) -> None:
        assert parse_tc260_aigc_json((_label_json() + "\x00\x00 ").encode()) is not None


class TestLabelFromMetadata:
    def test_a_raw_label_json_candidate_parses(self) -> None:
        # Doubao's PNG tEXt value: the bare label JSON itself.
        assert label_from_metadata(b"", (_label_json(),)) is not None

    def test_a_candidate_without_tc260_fields_is_rejected(self) -> None:
        assert label_from_metadata(b"", ('{"weather":"sunny"}',)) is None

    def test_an_xmp_namespaced_element_parses_any_json_object(self) -> None:
        xmp = (
            b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
            b'<TC260:AIGC>{"Label":"1","ContentProducer":"'
            + DOUBAO_USCC.encode()
            + b'"}</TC260:AIGC>'
            b"</x:xmpmeta>"
        )
        parsed = label_from_metadata(xmp)
        assert parsed is not None
        assert parsed["ContentProducer"] == DOUBAO_USCC

    def test_an_xmp_attribute_form_parses(self) -> None:
        # Attribute values entity-encode the embedded quotes.
        body = _label_json().replace('"', "&quot;").encode()
        xmp = b'<rdf:Description TC260:AIGC="' + body + b'" />'
        parsed = label_from_metadata(xmp)
        assert parsed is not None
        assert parsed["Label"] == "1"

    def test_an_html_entity_encoded_xmp_body_parses(self) -> None:
        xmp = b"<TC260:AIGC>{&#34;Label&#34;:&#34;1&#34;}</TC260:AIGC>"
        parsed = label_from_metadata(xmp)
        assert parsed is not None
        assert parsed["Label"] == "1"

    def test_a_wrapped_aigc_object_parses_the_inner_label(self) -> None:
        # EXIF UserComment form: {"AIGC":{"Label":...}} — brace-matched inward.
        wrapped = b'{"AIGC":' + _label_json().encode() + b"}"
        parsed = label_from_metadata(wrapped)
        assert parsed is not None
        assert parsed["ProduceID"] == PRODUCE_ID

    def test_a_bare_aigc_blob_parses_the_glued_label(self) -> None:
        blob = b"AIGC" + _label_json().encode() + b"trailing"
        parsed = label_from_metadata(blob)
        assert parsed is not None
        assert parsed["Label"] == "1"

    def test_metadata_without_a_label_returns_none(self) -> None:
        assert label_from_metadata(b"<exif>Canon EOS R5</exif>") is None
        assert label_from_metadata(b"") is None

    def test_an_oversized_xmp_body_is_a_bounded_refusal(self) -> None:
        big = b"<TC260:AIGC>" + b"x" * (MAX_TC260_VALUE_BYTES + 1) + b"</TC260:AIGC>"
        assert label_from_metadata(big) is None

    def test_an_oversized_xmp_label_is_refused_before_parsing(self) -> None:
        """A valid-JSON XMP span over the 1 MiB cap is a bounded refusal.

        The XMP branch used to json.loads the extracted span before any size
        check, so a label larger than MAX_TC260_VALUE_BYTES parsed fine where
        every bounded entry point refuses it.
        """
        body = b'{"Label":"' + b"x" * MAX_TC260_VALUE_BYTES + b'"}'
        assert len(body) > MAX_TC260_VALUE_BYTES
        assert json.loads(body) == {"Label": "x" * MAX_TC260_VALUE_BYTES}  # parses when unbounded
        big = b"<TC260:AIGC>" + body + b"</TC260:AIGC>"
        assert label_from_metadata(big) is None

    def test_a_label_beyond_the_scan_window_is_not_found(self) -> None:
        """Only the first MAX_TC260_SCAN_WINDOW_BYTES are scanned (documented).

        _scan_label used to latin-1-decode the whole payload (up to ~100 MB)
        on every call; the scan is now a bounded prefix window, so a label
        placed beyond the window is not found, and one inside it still is.
        """
        wrapped = b'{"AIGC":' + _label_json().encode() + b"}"
        assert label_from_metadata(b"\x00" * (MAX_TC260_SCAN_WINDOW_BYTES + 16) + wrapped) is None
        assert label_span(b"\x00" * (MAX_TC260_SCAN_WINDOW_BYTES + 16) + wrapped) is None
        inside = b"\x00" * (MAX_TC260_SCAN_WINDOW_BYTES - len(wrapped) - 8) + wrapped
        found = label_from_metadata(inside)
        assert found is not None and found["Label"] == "1"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF)
    )


def _png_with_text(key: bytes, value: bytes) -> bytes:
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", b"\x00"),
            _png_chunk(b"tEXt", key + b"\x00" + value),
            _png_chunk(b"IEND", b""),
        )
    )


class TestImageSeam:
    """Fixture file in -> inspect findings out (the agreed seam)."""

    def test_exif_embedded_label_yields_field_aware_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "tc260_exif_label.jpg")
        assert report.format == "jpeg"
        assert report.has_ai_metadata
        label_lines = [f for f in report.findings if "TC260 AIGC label" in f]
        assert len(label_lines) == 1
        # Field-aware: label type and the registry-resolved producer.
        assert "Label=1" in label_lines[0]
        assert DOUBAO_USCC in label_lines[0]
        assert "ByteDance" in label_lines[0]
        # Confidence rides the existing report shape.
        confidence = report.to_dict()["findings_confidence"]
        index = report.findings.index(label_lines[0])
        assert confidence[index] == "probable"

    def test_valid_signature_reports_a_verified_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "tc260_exif_label.jpg")
        verified = [f for f in report.findings if "TC260 label signature verified" in f]
        assert len(verified) == 1
        assert "own embedded key" in verified[0]

    def test_xmp_label_without_security_data_reports_no_signature_line(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "tc260_xmp_label.jpg")
        assert report.has_ai_metadata
        assert any("TC260 AIGC label" in f for f in report.findings)
        assert not any("TC260 label signature" in f for f in report.findings)

    def test_failed_signature_reports_an_integrity_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "tc260_exif_label_tampered.jpg")
        assert any("TC260 label signature does not verify" in f for f in report.findings)

    def test_png_text_label_is_detected_field_aware(self) -> None:
        label = json.dumps({"Label": "1", "ContentProducer": DOUBAO_USCC}, separators=(",", ":"))
        has_c2pa, has_ai, findings, _notes = image_meta.inspect_png(
            _png_with_text(b"AIGC", label.encode())
        )
        assert has_ai and not has_c2pa
        label_lines = [f for f in findings if "TC260 AIGC label" in f]
        assert len(label_lines) == 1
        assert "ByteDance" in label_lines[0]

    def test_oversized_png_text_label_is_a_findings_free_refusal(self) -> None:
        oversized = json.dumps({"Label": "1", "pad": "x" * (MAX_TC260_VALUE_BYTES + 16)}).encode()
        assert len(oversized) > MAX_TC260_VALUE_BYTES
        _c2pa, _ai, findings, _notes = image_meta.inspect_png(_png_with_text(b"AIGC", oversized))
        # Bounded refusal: parsed nothing, crashed nothing, flagged the raw
        # marker only through the pre-existing flat hint path.
        assert not any("TC260 AIGC label" in f for f in findings)
        assert not any("TC260 label signature" in f for f in findings)

    def test_cleaning_removes_the_exif_label_carrier(self) -> None:
        data = (FIXTURES / "tc260_exif_label.jpg").read_bytes()
        cleaned, _actions = image_meta.strip_jpeg(data, strip_all_app=False)
        _c, _a, findings, _n = image_meta.inspect_jpeg(cleaned)
        assert not any("TC260" in f for f in findings)

    def test_cleaning_removes_the_png_text_label_carrier(self) -> None:
        label = json.dumps({"Label": "1", "ContentProducer": DOUBAO_USCC}, separators=(",", ":"))
        data = _png_with_text(b"Comment", label.encode())
        cleaned, _actions = image_meta.strip_png(data, strip_all_text=False)
        _c, _a, findings, _n = image_meta.inspect_png(cleaned)
        assert not any("TC260" in f for f in findings)


def _bmff_box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _bmff_fullbox(kind: bytes, version: int, payload: bytes) -> bytes:
    return _bmff_box(kind, bytes([version]) + b"\x00\x00\x00" + payload)


def _heif_with_aigc_item(payload: bytes, major: bytes = b"heic") -> bytes:
    """HEIF/AVIF with an ISO-BMFF AIGC tag: a mime (application/json) iinf
    item named AIGC whose file extent carries the label JSON."""
    ftyp = _bmff_box(b"ftyp", major + b"\x00\x00\x00\x00" + b"mif1" + major)
    item_info = struct.pack(">HH", 1, 0) + b"mime" + b"AIGC\x00" + b"application/json\x00"
    infe = _bmff_fullbox(b"infe", 2, item_info)
    iinf = _bmff_fullbox(b"iinf", 0, struct.pack(">H", 1) + infe)

    def iloc(off: int) -> bytes:
        body = (
            bytes([0x44, 0x00])
            + struct.pack(">H", 1)
            + struct.pack(">HHH", 1, 0, 1)
            + struct.pack(">II", off, len(payload))
        )
        return _bmff_fullbox(b"iloc", 0, body)

    hdlr = _bmff_fullbox(b"hdlr", 0, b"\x00" * 4 + b"pict" + b"\x00" * 12 + b"Picture\x00")
    meta0 = _bmff_fullbox(b"meta", 0, hdlr + iinf + iloc(0))
    meta = _bmff_fullbox(b"meta", 0, hdlr + iinf + iloc(len(ftyp) + len(meta0) + 8))
    return ftyp + meta + _bmff_box(b"mdat", payload)


def _signed_label_json() -> bytes:
    sys.path.insert(0, str(ROOT / "tests"))
    from test_tc260_verify import CHOSEN_K, EMBEDDED_KEY, _label, _signed_label

    return json.dumps(
        _signed_label(EMBEDDED_KEY, CHOSEN_K, _label()), separators=(",", ":")
    ).encode()


class TestHeifSeam:
    def test_isobmff_aigc_tag_yields_the_same_finding_class(self) -> None:
        data = (FIXTURES / "tc260_label.heic").read_bytes()
        has_c2pa, has_ai, findings, _notes, _details = heif_meta.inspect_heif(data)
        assert has_ai and not has_c2pa
        label_lines = [f for f in findings if "TC260 AIGC label" in f]
        assert len(label_lines) == 1
        assert (
            "Label=1" in label_lines[0]
            and DOUBAO_USCC in label_lines[0]
            and "ByteDance" in label_lines[0]
        )
        assert any("TC260 label signature verified" in f for f in findings)

    def test_the_same_class_routes_through_inspect_image(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "tc260_label.heic")
        assert report.format == "heif"
        assert any("TC260 AIGC label" in f and "ByteDance" in f for f in report.findings)

    def test_an_avif_brand_yields_the_same_class(self) -> None:
        data = _heif_with_aigc_item(_signed_label_json(), major=b"avif")
        assert image_meta.detect_format(data) == "avif"
        _c2pa, has_ai, findings, _notes = image_meta.inspect_avif(data)
        assert has_ai
        assert any("TC260 AIGC label" in f and "ByteDance" in f for f in findings)

    def test_cleaning_neutralizes_the_aigc_tag_at_equal_length(self) -> None:
        data = _heif_with_aigc_item(_signed_label_json())
        for strip_all in (True, False):
            cleaned, _actions = heif_meta.neutralize_heif(data, strip_all_metadata=strip_all)
            assert len(cleaned) == len(data)  # offsets never move
            _c, _a, findings, _n, _d = heif_meta.inspect_heif(cleaned)
            assert not any("TC260" in f for f in findings)

    def test_a_non_utf8_aigc_item_is_a_findings_free_refusal(self) -> None:
        data = _heif_with_aigc_item(b'{"Label":"1","ContentProducer":"\xff\xfe\xfd"}')
        has_c2pa, _ai, findings, _notes, _details = heif_meta.inspect_heif(data)
        assert not has_c2pa
        assert not any("TC260" in f for f in findings)
        # Not flagged -> untouched by targeted cleaning; strip-all still zeroes
        # every metadata item by its own contract.
        kept, _actions = heif_meta.neutralize_heif(data, strip_all_metadata=False)
        assert kept == data
        stripped, _actions = heif_meta.neutralize_heif(data, strip_all_metadata=True)
        assert stripped != data

    def test_an_oversized_aigc_item_is_a_findings_free_refusal(self) -> None:
        big = b'{"Label":"' + b"x" * (MAX_TC260_VALUE_BYTES + 1) + b'"}'
        data = _heif_with_aigc_item(big)
        _c, _ai, findings, _n, _d = heif_meta.inspect_heif(data)
        assert not any("TC260" in f for f in findings)
