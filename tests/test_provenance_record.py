"""Provenance record: versioned collection schema, version guard, JSON safety.

The record is the transport format between collection (touches the file once)
and judgment (identify.py, touches nothing). Parity with the verdict lives in
tests/test_identify.py; this file pins the record contract itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from provenance_record import (
    PROVENANCE_RECORD_SCHEMA_VERSION,
    PROVENANCE_RECORD_TYPE,
    UnsupportedRecordVersion,
    collect_provenance_record,
    require_record_version,
)

FIXTURES = ROOT / "tests" / "fixtures" / "identify"
TC260_FIXTURES = ROOT / "tests" / "fixtures" / "tc260"


class TestVersionGuard:
    def test_current_version_is_accepted(self) -> None:
        assert require_record_version(1) == 1

    def test_unknown_version_is_rejected_stably(self) -> None:
        for bad in (0, 2, 99, "1", None, 1.0):
            with pytest.raises(UnsupportedRecordVersion):
                require_record_version(bad)

    def test_guard_error_names_the_contract_and_versions(self) -> None:
        with pytest.raises(UnsupportedRecordVersion, match="provenance record") as excinfo:
            require_record_version(7)
        assert "7" in str(excinfo.value) and "1" in str(excinfo.value)

    def test_collect_rejects_an_unimplemented_schema_up_front(self) -> None:
        with pytest.raises(UnsupportedRecordVersion):
            collect_provenance_record(FIXTURES / "clean_camera_exif.jpg", schema_version=2)


class TestRecordShape:
    def test_record_is_versioned_and_typed(self) -> None:
        record = collect_provenance_record(FIXTURES / "clean_camera_exif.jpg")
        assert record["schema_version"] == PROVENANCE_RECORD_SCHEMA_VERSION == 1
        assert record["record_type"] == PROVENANCE_RECORD_TYPE
        assert record["status"] == "complete"
        assert record["container"] == "jpeg"
        assert record["issues"] == []

    def test_record_is_json_safe(self) -> None:
        for name in sorted(p.name for p in FIXTURES.iterdir()):
            record = collect_provenance_record(FIXTURES / name)
            round_tripped = json.loads(json.dumps(record))
            assert round_tripped == record, name

    def test_clean_camera_exif_carries_no_ai_evidence(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "clean_camera_exif.jpg")["evidence"]
        assert evidence["has_ai_metadata"] is False
        assert evidence["has_c2pa"] is False
        assert evidence["tc260_label"] is None
        assert evidence["tc260_signature"] is None
        assert evidence["samsung_genai_type"] is None
        assert evidence["xai_pair"] is False
        assert evidence["iptc_digital_source"] is None
        assert evidence["hf_job_id"] is None
        assert evidence["generator_products"] == []
        assert evidence["c2pa_hints"] is None

    def test_tc260_label_is_collected_typed_with_signature_state(self) -> None:
        record = collect_provenance_record(TC260_FIXTURES / "tc260_exif_label.jpg")
        evidence = record["evidence"]
        assert evidence["tc260_label"] is not None
        assert evidence["tc260_label"]["ContentProducer"] == "91110102MACQD9K640"
        assert evidence["tc260_signature"] is not None
        assert evidence["tc260_signature"]["status"] == "verified"
        # The findings vocabulary rides along for attribution.
        assert any("TC260 AIGC label" in f for f in evidence["findings"])

    def test_unsigned_tc260_label_reports_signature_none(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "tc260_labeled.jpg")["evidence"]
        assert evidence["tc260_label"] is not None
        assert evidence["tc260_signature"] is None

    def test_samsung_genai_type_is_typed(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "samsung_galaxy_ai.jpg")["evidence"]
        assert evidence["samsung_genai_type"] == 1

    def test_xai_pair_is_typed(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "xai_grok_pair.jpg")["evidence"]
        assert evidence["xai_pair"] is True

    def test_iptc_digital_source_kind_is_typed(self) -> None:
        evidence = collect_provenance_record(
            TC260_FIXTURES.parent / "vendors" / "iptc_generated.jpg"
        )["evidence"]
        assert evidence["iptc_digital_source"] is not None
        assert evidence["iptc_digital_source"]["kind"] == "generated"
        enhanced = collect_provenance_record(
            TC260_FIXTURES.parent / "vendors" / "iptc_enhanced.jpg"
        )["evidence"]
        assert enhanced["iptc_digital_source"]["kind"] == "enhanced"

    def test_hf_job_id_is_typed(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "mixed_signals.jpg")["evidence"]
        assert evidence["hf_job_id"] == "7c9e6679-7425-40de-944b-e07fc1f90ae7"

    def test_generator_products_are_extracted_from_the_finding(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "ai_generator_image.png")["evidence"]
        assert evidence["generator_products"] == ["DALL-E"]

    def test_c2pa_cabx_hints_are_collected_for_png(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "c2pa_cabx_ai_tool.png")["evidence"]
        assert evidence["has_c2pa"] is True
        hints = evidence["c2pa_hints"]
        assert hints is not None
        assert hints["claim_generator"] == "Adobe Firefly"
        assert hints["claim_generator_platforms"] == ["Adobe Firefly"]
        assert hints["cbor"]["status"] == "ok"

    def test_mixed_fixture_carries_every_typed_signal(self) -> None:
        evidence = collect_provenance_record(FIXTURES / "mixed_signals.jpg")["evidence"]
        assert evidence["tc260_label"] is not None
        assert evidence["samsung_genai_type"] == 1
        assert evidence["hf_job_id"] is not None
        assert evidence["has_ai_metadata"] is True


class TestStableRejections:
    def test_missing_file_is_a_status_error_not_an_exception(self) -> None:
        record = collect_provenance_record(FIXTURES / "does_not_exist.jpg")
        assert record["status"] == "error"
        assert record["issues"]
        assert record["evidence"]["findings"] == []

    def test_unsupported_format_is_collected_with_a_note(self) -> None:
        record = collect_provenance_record(FIXTURES / "tc260_labeled.jpg", force_kind="text")
        assert record["container"] == "text"
        assert record["status"] == "complete"
        assert record["evidence"]["has_ai_metadata"] is False

    def test_inspector_failure_is_a_partial_record_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import image_meta

        def _explode(_data: object) -> object:
            raise RuntimeError("malformed container")

        monkeypatch.setattr(image_meta, "inspect_jpeg", _explode)
        record = collect_provenance_record(FIXTURES / "tc260_labeled.jpg")
        assert record["status"] == "partial"
        assert record["issues"] == [{"stage": "image", "code": "collection-failed"}]
        assert record["evidence"]["findings"] == []
