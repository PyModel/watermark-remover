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

import image_meta
from tc260_label import MAX_TC260_VALUE_BYTES, label_from_metadata, parse_tc260_aigc_json

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
