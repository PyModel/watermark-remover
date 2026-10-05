"""Identify: one provenance verdict from collected evidence.

Tri-state honesty is the tier-1 contract: absent signals mean ``unknown``,
never "clean"; a failed label signature is an integrity clash; classifiers
and optional tools are never auto-run; the verdict from a JSON round-trip of
the record equals the verdict from the file (parity contract).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
import server
from identify import (
    PROVENANCE_VERDICT_SCHEMA_VERSION,
    ProvenanceVerdict,
    identify_file,
    identify_from_record,
    unknown_verdict,
)
from provenance_record import (
    UnsupportedRecordVersion,
    collect_provenance_record,
)

FIXTURES = ROOT / "tests" / "fixtures" / "identify"
TC260_FIXTURES = ROOT / "tests" / "fixtures" / "tc260"
VENDOR_FIXTURES = ROOT / "tests" / "fixtures" / "vendors"


def verdict_of(name: str, fixtures: Path = FIXTURES) -> ProvenanceVerdict:
    return identify_file(fixtures / name)


class TestTriStateHonesty:
    def test_no_signals_is_unknown_never_clean(self) -> None:
        verdict = verdict_of("clean_camera_exif.jpg")
        assert verdict.is_ai_generated is None
        assert verdict.platform is None
        assert verdict.ai_source_kind is None
        assert verdict.ai_from_metadata is False
        assert verdict.integrity_clashes == []
        assert verdict.c2pa_validation is None

    def test_the_verdict_never_carries_a_clean_assertion(self) -> None:
        # False is never produced: stripped metadata leaves no local proof of
        # a clean origin, so the judge has no False branch at all.
        verdict = verdict_of("clean_camera_exif.jpg")
        assert verdict.to_dict()["is_ai_generated"] is None

    def test_unknown_verdict_helper_is_all_unknown(self) -> None:
        verdict = unknown_verdict()
        assert verdict.is_ai_generated is None
        assert verdict.ai_from_metadata is False
        assert verdict.to_dict() == {
            "schema_version": PROVENANCE_VERDICT_SCHEMA_VERSION,
            "is_ai_generated": None,
            "platform": None,
            "ai_source_kind": None,
            "ai_from_metadata": False,
            "integrity_clashes": [],
            "c2pa_validation": None,
        }


class TestFixtureMatrix:
    def test_ai_generator_product_asserts_ai(self) -> None:
        verdict = verdict_of("ai_generator_image.png")
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is True
        assert verdict.platform == "DALL-E"
        assert verdict.integrity_clashes == []

    def test_tc260_label_resolves_the_producer_registry(self) -> None:
        verdict = verdict_of("tc260_labeled.jpg")
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is True
        assert verdict.platform == "ByteDance"
        assert verdict.ai_source_kind is None

    def test_samsung_alone_is_ai_at_medium_evidence(self) -> None:
        verdict = verdict_of("samsung_galaxy_ai.jpg")
        assert verdict.is_ai_generated is True
        # The Samsung marker is an editing hint, not generation provenance:
        # it does not claim the ai-from-metadata target set.
        assert verdict.ai_from_metadata is False
        assert verdict.platform == "Samsung Galaxy AI"

    def test_xai_pair_asserts_ai_from_metadata(self) -> None:
        verdict = verdict_of("xai_grok_pair.jpg")
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is True
        assert verdict.platform == "xAI Grok"

    def test_iptc_digital_source_sets_the_source_kind(self) -> None:
        generated = verdict_of("iptc_generated.jpg", VENDOR_FIXTURES)
        assert generated.is_ai_generated is True
        assert generated.ai_source_kind == "generated"
        assert generated.ai_from_metadata is True
        enhanced = verdict_of("iptc_enhanced.jpg", VENDOR_FIXTURES)
        assert enhanced.ai_source_kind == "enhanced"
        assert enhanced.is_ai_generated is True

    def test_hf_job_hint_alone_is_medium_evidence(self) -> None:
        verdict = verdict_of("hf_job_hint.jpg", VENDOR_FIXTURES)
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is False
        assert verdict.platform == "Higgsfield"

    def test_c2pa_cabx_names_the_platform_and_reports_validation(self) -> None:
        verdict = verdict_of("c2pa_cabx_ai_tool.png")
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is True
        assert verdict.platform == "Adobe Firefly"
        validation = verdict.c2pa_validation
        assert validation is not None
        assert validation["source"] == "stdlib-cbor-fallback"
        # The stdlib fallback cannot validate integrity/signature/trust:
        # each dimension reports unknown, never a clean bill.
        assert validation["integrity"] == "unknown"
        assert validation["signature"] == "unknown"
        assert validation["signer_trust"] == "unknown"

    def test_mixed_signals_take_the_strongest_platform_and_stay_ai(self) -> None:
        verdict = verdict_of("mixed_signals.jpg")
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is True
        assert verdict.platform == "ByteDance"
        assert verdict.integrity_clashes == []

    def test_existing_repo_tc260_fixtures_identify(self) -> None:
        verdict = verdict_of("tc260_exif_label.jpg", TC260_FIXTURES)
        assert verdict.is_ai_generated is True
        assert verdict.platform == "ByteDance"
        assert verdict.integrity_clashes == []

    def test_existing_repo_heic_tc260_fixture_identifies(self) -> None:
        verdict = verdict_of("tc260_label.heic", TC260_FIXTURES)
        assert verdict.is_ai_generated is True
        assert verdict.ai_from_metadata is True


class TestIntegrityClash:
    def test_failed_label_signature_reports_the_clash(self) -> None:
        verdict = verdict_of("tc260_signature_clash.jpg")
        assert verdict.is_ai_generated is True
        assert len(verdict.integrity_clashes) == 1
        clash = verdict.integrity_clashes[0]
        assert "does not verify" in clash
        assert "91110102MACQD9K640" in clash
        assert "tampered or transplanted" in clash

    def test_clash_survives_the_record_round_trip(self) -> None:
        record = collect_provenance_record(FIXTURES / "tc260_signature_clash.jpg")
        from_record = identify_from_record(json.loads(json.dumps(record)))
        assert (
            from_record.integrity_clashes
            == identify_file(FIXTURES / "tc260_signature_clash.jpg").integrity_clashes
        )


class TestRecordParity:
    @pytest.mark.parametrize(
        "name",
        [
            "clean_camera_exif.jpg",
            "ai_generator_image.png",
            "tc260_labeled.jpg",
            "tc260_signature_clash.jpg",
            "samsung_galaxy_ai.jpg",
            "xai_grok_pair.jpg",
            "c2pa_cabx_ai_tool.png",
            "mixed_signals.jpg",
        ],
    )
    def test_verdict_from_json_record_equals_verdict_from_file(self, name: str) -> None:
        path = FIXTURES / name
        file_verdict = identify_file(path)
        record = collect_provenance_record(path)
        transported = json.loads(json.dumps(record))
        record_verdict = identify_from_record(transported)
        assert record_verdict.to_dict() == file_verdict.to_dict()

    def test_record_with_an_unknown_version_is_rejected(self) -> None:
        record = collect_provenance_record(FIXTURES / "tc260_labeled.jpg")
        record["schema_version"] = 2
        with pytest.raises(UnsupportedRecordVersion):
            identify_from_record(record)

    def test_record_with_the_wrong_type_is_rejected(self) -> None:
        record = collect_provenance_record(FIXTURES / "tc260_labeled.jpg")
        record["record_type"] = "something_else"
        with pytest.raises(ValueError, match="record_type"):
            identify_from_record(record)

    def test_incomplete_record_is_an_unknown_verdict_not_a_guess(self) -> None:
        record = collect_provenance_record(FIXTURES / "does_not_exist.jpg")
        verdict = identify_from_record(record)
        assert verdict.is_ai_generated is None
        assert verdict.ai_from_metadata is False


class TestClassifiersNeverAutoRun:
    def test_identify_file_never_touches_optional_tools_or_synthid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(*args: object, **kwargs: object) -> object:
            raise AssertionError("classifier auto-run during identify")

        monkeypatch.setattr(image_meta, "run_optional_tools", _explode)
        monkeypatch.setattr(image_meta, "run_synthid_score", _explode)
        monkeypatch.setattr(image_meta, "which", _explode)
        verdict = identify_file(FIXTURES / "mixed_signals.jpg")
        assert verdict.is_ai_generated is True

    def test_identify_module_imports_no_pixel_stack(self) -> None:
        import identify

        source = Path(identify.__file__).read_text(encoding="utf-8")
        for banned in ("import cv2", "import numpy", "from PIL", "import PIL"):
            assert banned not in source
        # And the classifier entry points are not called from it.
        assert "run_synthid_score" not in source
        assert "run_optional_tools" not in source


class TestInspectCliSurface:
    """`inspect_file.py --identify` — the CLI identify surface.

    Without the flag the CLI behaves exactly as today
    (tests/test_inspect_file_partial.py + tests/test_json_exit_code.py pin
    that); these tests pin the flag's own behavior.
    """

    def _run(self, *args: str) -> tuple[int, str, str]:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "inspect_file.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_identify_prints_the_verdict_for_a_clean_file(self) -> None:
        code, out, err = self._run(str(FIXTURES / "clean_camera_exif.jpg"), "--identify")
        assert code == 0, (code, out, err)
        assert "AI generated: unknown" in out
        assert "Platform: unknown" in out
        assert "Integrity clashes: none" in out
        assert "C2PA validation: none" in out

    def test_identify_prints_the_verdict_for_a_tc260_file(self) -> None:
        code, out, _err = self._run(str(FIXTURES / "tc260_labeled.jpg"), "--identify")
        assert code == 0
        assert "AI generated: yes" in out
        assert "Platform: ByteDance" in out
        assert "AI from metadata: yes" in out

    def test_identify_json_carries_the_six_field_verdict(self) -> None:
        code, out, _err = self._run(
            str(FIXTURES / "tc260_signature_clash.jpg"), "--identify", "--json"
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["kind"] == "image"
        verdict = payload["identify"]
        assert verdict["is_ai_generated"] is True
        assert len(verdict["integrity_clashes"]) == 1
        assert set(verdict) == {
            "schema_version",
            "is_ai_generated",
            "platform",
            "ai_source_kind",
            "ai_from_metadata",
            "integrity_clashes",
            "c2pa_validation",
        }

    def test_identify_without_the_flag_has_no_identify_key(self) -> None:
        code, out, _err = self._run(str(FIXTURES / "tc260_labeled.jpg"), "--json")
        assert code == 1  # today's suspicious exit
        assert "identify" not in json.loads(out)

    def test_identify_malformed_image_still_verdicts_unknown(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"\xff\xd8not really a jpeg")
            path = Path(handle.name)
        try:
            code, out, _err = self._run(str(path), "--identify", "--json")
            # A .jpg suffix classifies as image, so the file IS scanned —
            # same as today's inspect; the verdict is unknown, never clean.
            assert code == 0
            payload = json.loads(out)
            assert payload["kind"] == "image"
            assert payload["identify"]["is_ai_generated"] is None
        finally:
            path.unlink(missing_ok=True)

    def test_identify_on_an_unknown_container_reports_unscanned(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as handle:
            handle.write(b"mysterious bytes")
            path = Path(handle.name)
        try:
            code, out, _err = self._run(str(path), "--identify", "--json")
            assert code == 3  # EXIT_PARTIAL: the input was not scanned
            payload = json.loads(out)
            assert payload["unscanned"] is True
        finally:
            path.unlink(missing_ok=True)

    def test_identify_refuses_a_missing_input_like_the_cli(self) -> None:
        code, _out, err = self._run(str(FIXTURES / "does_not_exist.jpg"), "--identify")
        assert code == 2
        assert "not a regular file" in err


@pytest.fixture(scope="module")
def server_conn():
    import http.client
    import threading

    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


class TestServerIdentifySurface:
    """POST /inspect?identify=true — the HTTP identify surface.

    The existing server suite (tests/test_http_server.py) stays unmodified
    and pins today's no-param behavior; these tests pin the param.
    """

    def _post(self, conn, path: str, data: bytes, name: str) -> tuple[int, dict]:
        import base64
        import json as _json

        body = _json.dumps({"file": base64.b64encode(data).decode("ascii"), "name": name})
        conn.request("POST", path, body=body.encode(), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        return resp.status, _json.loads(payload) if payload else {}

    def test_identify_param_returns_the_verdict(self, server_conn) -> None:
        code, payload = self._post(
            server_conn,
            "/inspect?identify=true",
            (FIXTURES / "tc260_signature_clash.jpg").read_bytes(),
            "x.jpg",
        )
        assert code == 200
        # today's response shape intact, verdict added:
        assert payload["ok"] is True and payload["kind"] == "image"
        assert "report" in payload and "suspicious" in payload and "detection_status" in payload
        verdict = payload["identify"]
        assert verdict["is_ai_generated"] is True
        assert verdict["platform"] == "ByteDance"
        assert len(verdict["integrity_clashes"]) == 1
        assert set(verdict) == {
            "schema_version",
            "is_ai_generated",
            "platform",
            "ai_source_kind",
            "ai_from_metadata",
            "integrity_clashes",
            "c2pa_validation",
        }

    def test_identify_accepts_1_and_true_spellings(self, server_conn) -> None:
        data = (FIXTURES / "tc260_labeled.jpg").read_bytes()
        for query in ("?identify=1", "?identify=true", "?identify=TRUE"):
            code, payload = self._post(server_conn, f"/inspect{query}", data, "x.jpg")
            assert code == 200, query
            assert payload["identify"]["platform"] == "ByteDance", query

    def test_without_the_param_the_response_has_no_identify_key(self, server_conn) -> None:
        data = (FIXTURES / "tc260_labeled.jpg").read_bytes()
        for path in ("/inspect", "/inspect?identify=false", "/inspect?identify=0"):
            code, payload = self._post(server_conn, path, data, "x.jpg")
            assert code == 200, path
            assert "identify" not in payload, path

    def test_identify_on_text_is_all_unknown(self, server_conn) -> None:
        code, payload = self._post(server_conn, "/inspect?identify=true", b"plain text", "a.txt")
        assert code == 200
        verdict = payload["identify"]
        assert verdict["is_ai_generated"] is None
        assert verdict["ai_from_metadata"] is False

    def test_openapi_documents_the_identify_param(self, server_conn) -> None:
        code, spec = self._get_spec(server_conn)
        assert code == 200
        op = spec["paths"]["/inspect"]["post"]
        params = {p["name"]: p for p in op.get("parameters", [])}
        assert "identify" in params
        assert params["identify"]["in"] == "query"
        assert params["identify"]["schema"]["type"] == "boolean"
        assert (
            "identify"
            in op["responses"]["200"]["content"]["application/json"]["schema"]["properties"]
        )

    def _get_spec(self, conn):
        import json as _json

        conn.request("GET", "/openapi.json")
        resp = conn.getresponse()
        return resp.status, _json.loads(resp.read())


class TestVerdictShape:
    def test_to_dict_is_json_safe_and_versioned(self) -> None:
        verdict = verdict_of("mixed_signals.jpg")
        payload = json.loads(json.dumps(verdict.to_dict()))
        assert payload["schema_version"] == 1
        assert set(payload) == {
            "schema_version",
            "is_ai_generated",
            "platform",
            "ai_source_kind",
            "ai_from_metadata",
            "integrity_clashes",
            "c2pa_validation",
        }

    def test_findings_only_c2pa_still_reports_validation_unknowns(self) -> None:
        # A JPEG whose APP11 carries C2PA markers but no stdlib caBX scan:
        # validation is reported with unknown dimensions, source = findings.
        record = collect_provenance_record(FIXTURES / "clean_camera_exif.jpg")
        record["evidence"]["has_c2pa"] = True
        record["evidence"]["findings"] = ["JPEG APP11 segment: c2pa, jumb"]
        verdict = identify_from_record(record)
        assert verdict.c2pa_validation is not None
        assert verdict.c2pa_validation["source"] == "findings"
        assert verdict.c2pa_validation["integrity"] == "unknown"

    def test_bare_c2pa_presence_alone_does_not_assert_ai(self) -> None:
        record = collect_provenance_record(FIXTURES / "clean_camera_exif.jpg")
        record["evidence"]["has_c2pa"] = True
        record["evidence"]["findings"] = ["PNG chunk caBX (possible C2PA container)"]
        verdict = identify_from_record(record)
        assert verdict.is_ai_generated is None
        assert verdict.c2pa_validation is not None
