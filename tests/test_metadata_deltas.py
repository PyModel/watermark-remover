"""Vendor metadata deltas: Samsung Galaxy AI trailer, xAI signature pair, IPTC
digital-source semantics, Higgsfield hf-job-id hint — detection, cleaning, and
residual re-inspection. Fixture file in -> inspect findings out.

Upstream vectors: research/remove-ai-watermarks-main metadata.py (Samsung
trailer + EOI-aware strip, xAI signature+UUID pair, IPTC 2025.1 digital-source
values, hf-job-id hint).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
from tc260_label import label_from_metadata

FIXTURES = ROOT / "tests" / "fixtures" / "vendors"
TC260_FIXTURES = ROOT / "tests" / "fixtures" / "tc260"

HF_UUID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"


def _png_with_text(key: bytes, value: bytes) -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF)
        )

    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)),
            chunk(b"IDAT", b"\x00"),
            chunk(b"tEXt", key + b"\x00" + value),
            chunk(b"IEND", b""),
        )
    )


def confidence_of(report_or_scan, findings: list[str]) -> dict[str, str]:
    """finding text -> confidence, riding the existing report shape."""
    if isinstance(report_or_scan, image_meta.ImageInspectReport):
        conns = report_or_scan.to_dict()["findings_confidence"]
    else:
        conns = [image_meta.classify_finding_confidence(f) for f in findings]
    return dict(zip(findings, conns, strict=False))


def vendor_findings(findings: list[str]) -> list[str]:
    """The labeled vendor-delta findings among all findings."""
    keys = ("Samsung Galaxy", "xAI/Grok", "IPTC digital-source", "hf-job-id")
    return [f for f in findings if any(k in f for k in keys)]


class TestSamsungTrailer:
    def test_trailer_yields_a_labeled_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "samsung_galaxy_ai_trailer.jpg")
        assert report.format == "jpeg"
        assert report.has_ai_metadata
        hits = [f for f in report.findings if "Samsung Galaxy" in f]
        assert len(hits) == 1
        assert "genAIType=1" in hits[0]
        assert "PhotoEditor_Re_Edit_Data" in hits[0]
        assert confidence_of(report, report.findings)[hits[0]] == "probable"

    def test_a_plain_jpeg_yields_no_samsung_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "xai_grok_lone_text.jpg")
        assert not any("Samsung Galaxy" in f for f in report.findings)

    def test_cleaning_truncates_the_trailer_and_reinspects_residual_free(self) -> None:
        data = (FIXTURES / "samsung_galaxy_ai_trailer.jpg").read_bytes()
        for strip_all in (True, False):
            cleaned, actions = image_meta.strip_jpeg(data, strip_all_app=strip_all)
            assert (
                cleaned == data[: data.index(b"\xff\xd9") + 2]
            )  # truncated at the EOI, byte-exactly
            drops = [a.to_dict() for a in actions if "post-EOI trailer" in a.text]
            assert drops, "Samsung trailer drop must be reported, not silent"
            assert drops[0]["code"] == "drop_jpeg_segment"
            assert "Samsung Galaxy AI trailer" in drops[0]["text"]
            _c, has_ai, findings, _n = image_meta.inspect_jpeg(cleaned)
            assert not has_ai
            assert not vendor_findings(findings)

    def test_a_jpeg_without_ai_metadata_strips_to_itself(self) -> None:
        data = (FIXTURES / "iptc_bare_algorithmic.jpg").read_bytes()
        cleaned, actions = image_meta.strip_jpeg(data, strip_all_app=False)
        assert cleaned == data  # nothing detected -> nothing dropped
        assert not any("post-EOI trailer" in a.text for a in actions)


class TestXaiSignaturePair:
    def test_pair_yields_a_labeled_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "xai_grok_signature_pair.jpg")
        hits = [f for f in report.findings if "xAI/Grok" in f]
        assert len(hits) == 1
        assert "Signature" in hits[0] and "UUID" in hits[0]
        assert confidence_of(report, report.findings)[hits[0]] == "probable"

    def test_lone_grok_text_is_not_an_xai_pair(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "xai_grok_lone_text.jpg")
        assert not any("xAI/Grok" in f for f in report.findings)

    def test_an_unrelated_uuid_in_another_record_does_not_complete_the_pair(self) -> None:
        sig = b"Signature: " + b"A1b2C3d4" * 8
        # The two values sit in DIFFERENT rdf:Description blocks: no pair.
        data = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xe1" + len(b"").to_bytes(2, "big")
        )
        xmp = (
            b"<x:xmpmeta><rdf:RDF>"
            b"<rdf:Description><dc:description>" + sig + b"</dc:description></rdf:Description>"
            b"<rdf:Description><dc:creator>3f2b8a1c-9d4e-4c7a-b1e2-5a6f7c8d9e0f"
            b"</dc:creator></rdf:Description>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        data = (
            data[:-2]
            + (len(xmp) + 2).to_bytes(2, "big")
            + xmp
            + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x00\xff\xd9"
        )
        _c, _ai, findings, _n = image_meta.inspect_jpeg(data)
        assert not any("xAI/Grok" in f for f in findings)

    def test_cleaning_drops_the_pair_carrier_and_reinspects_residual_free(self) -> None:
        data = (FIXTURES / "xai_grok_signature_pair.jpg").read_bytes()
        for strip_all in (True, False):
            cleaned, actions = image_meta.strip_jpeg(data, strip_all_app=strip_all)
            assert any(a.code.value == "drop_jpeg_segment" for a in actions)
            assert b"Signature: A1b2" not in cleaned
            _c, _ai, findings, _n = image_meta.inspect_jpeg(cleaned)
            assert not vendor_findings(findings)


class TestIptcDigitalSource:
    def test_generated_value_yields_an_ai_generated_finding(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "iptc_generated.jpg")
        hits = [f for f in report.findings if "IPTC digital-source" in f]
        assert len(hits) == 1
        assert "trainedAlgorithmicMedia" in hits[0]
        assert "AI-generated" in hits[0]
        assert confidence_of(report, report.findings)[hits[0]] == "confirmed"
        # The labeled finding replaces the old generic token list for this payload.
        assert not any(f.startswith("JPEG APP1: digitalSourceType") for f in report.findings)

    def test_enhanced_value_is_labeled_composite_not_generated(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "iptc_enhanced.jpg")
        hits = [f for f in report.findings if "IPTC digital-source" in f]
        assert len(hits) == 1
        assert "compositeWithTrainedAlgorithmicMedia" in hits[0]
        assert "AI-composite/enhanced" in hits[0]
        assert "AI-generated" not in hits[0]

    def test_bare_algorithmic_media_yields_nothing(self) -> None:
        # IPTC: purely algorithmic media is NOT AI/ML generation.
        report = image_meta.inspect_image(FIXTURES / "iptc_bare_algorithmic.jpg")
        assert report.findings == []
        assert not report.has_ai_metadata

    def test_cleaning_drops_the_carrier_and_reinspects_residual_free(self) -> None:
        data = (FIXTURES / "iptc_generated.jpg").read_bytes()
        for strip_all in (True, False):
            cleaned, _actions = image_meta.strip_jpeg(data, strip_all_app=strip_all)
            assert b"trainedAlgorithmicMedia" not in cleaned
            _c, has_ai, findings, _n = image_meta.inspect_jpeg(cleaned)
            assert not has_ai
            assert not vendor_findings(findings)


class TestHfJobHint:
    def test_png_text_chunk_yields_a_medium_confidence_hint(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "hf_job_hint.png")
        assert report.format == "png"
        hits = [f for f in report.findings if "hf-job-id" in f]
        assert len(hits) == 1
        assert HF_UUID in hits[0]
        assert confidence_of(report, report.findings)[hits[0]] == "probable"

    def test_jpeg_carrier_yields_the_same_hint(self) -> None:
        report = image_meta.inspect_image(FIXTURES / "hf_job_hint.jpg")
        hits = [f for f in report.findings if "hf-job-id" in f]
        assert len(hits) == 1
        assert HF_UUID in hits[0]

    def test_a_non_uuid_value_after_the_key_is_not_a_hint(self) -> None:
        _c, _ai, findings, _n = image_meta.inspect_png(_png_with_text(b"hf-job-id", b"not-a-uuid"))
        assert not any("hf-job-id" in f for f in findings)

    def test_cleaning_drops_the_hint_and_reinspects_residual_free(self) -> None:
        for name, cleaner in (
            ("hf_job_hint.png", image_meta.strip_png),
            ("hf_job_hint.jpg", image_meta.strip_jpeg),
        ):
            data = (FIXTURES / name).read_bytes()
            for strip_all in (True, False):
                cleaned, _actions = (
                    cleaner(data, strip_all_text=strip_all)
                    if name.endswith(".png")
                    else cleaner(data, strip_all_app=strip_all)
                )
                assert HF_UUID.encode() not in cleaned, name
                _c, _ai, findings, _n = (
                    image_meta.inspect_png(cleaned)
                    if name.endswith(".png")
                    else image_meta.inspect_jpeg(cleaned)
                )
                assert not vendor_findings(findings), name


class TestCorruptedStripFailsClosed:
    def test_truncated_jpeg_still_reports_and_strip_never_claims_clean(self) -> None:
        data = (FIXTURES / "samsung_trailer_truncated.jpg").read_bytes()
        # Detection is resilient: the marker is still reported.
        _c, has_ai, findings, _n = image_meta.inspect_jpeg(data)
        assert has_ai
        assert any("Samsung Galaxy" in f for f in findings)
        # The strip cannot complete -> it fails loudly, never returns "clean" bytes.
        with pytest.raises(ValueError):
            image_meta.strip_jpeg(data, strip_all_app=True)
        with pytest.raises(ValueError):
            image_meta.strip_jpeg(data, strip_all_app=False)

    def test_clean_image_refuses_to_claim_a_clean_output(self, tmp_path) -> None:
        src = tmp_path / "corrupted.jpg"
        src.write_bytes((FIXTURES / "samsung_trailer_truncated.jpg").read_bytes())
        with pytest.raises(ValueError):
            image_meta.clean_image(src, tmp_path / "out.jpg")


class TestCleanImageResidualFree:
    def test_every_vendor_fixture_cleans_residual_free(self, tmp_path) -> None:
        for name in (
            "samsung_galaxy_ai_trailer.jpg",
            "xai_grok_signature_pair.jpg",
            "iptc_generated.jpg",
            "iptc_enhanced.jpg",
            "hf_job_hint.png",
            "hf_job_hint.jpg",
        ):
            src = FIXTURES / name
            dest = tmp_path / ("clean." + name)
            report = image_meta.clean_image(src, dest)
            assert not report["still_has_ai_metadata"], (name, report["post_findings"])
            assert not vendor_findings(report["post_findings"]), name


class TestTc260RasterIntact:
    @staticmethod
    def _scan_bytes(data: bytes) -> bytes:
        """Entropy-coded scan between the SOS header and the first EOI (empty when scan-less)."""
        sos = data.find(b"\xff\xda")
        if sos == -1:
            return b""
        length = int.from_bytes(data[sos + 2 : sos + 4], "big")
        start = sos + 2 + length
        eoi = image_meta._find_jpeg_eoi(data, start)
        assert eoi is not None
        return data[start:eoi]

    @staticmethod
    def _scan_bearing_tc260_jpeg() -> bytes:
        """JPEG with a real entropy-coded scan plus the ticket-04 EXIF label carrier."""
        label = b'{"AIGC":{"Label":"1","ContentProducer":"91110102MACQD9K640"}}'
        app1 = b"Exif\x00\x00" + label
        return b"".join(
            (
                b"\xff\xd8",
                b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1,
                b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00",
                b"\x11\x22\x33\x44\x55\x66\x77\x88",
                b"\xff\xd9",
            )
        )

    def test_scan_bearing_label_fixture_keeps_raster_bytes_identical(self, tmp_path) -> None:
        src = tmp_path / "tc260_scan.jpg"
        data = self._scan_bearing_tc260_jpeg()
        src.write_bytes(data)
        before = hashlib.sha256(self._scan_bytes(data)).hexdigest()
        report = image_meta.clean_image(src, tmp_path / "clean.jpg")
        cleaned = (tmp_path / "clean.jpg").read_bytes()
        assert hashlib.sha256(self._scan_bytes(cleaned)).hexdigest() == before
        assert not any("TC260" in f for f in report["post_findings"])
        assert label_from_metadata(cleaned) is None

    @pytest.mark.parametrize("name", ["tc260_exif_label.jpg", "tc260_xmp_label.jpg"])
    def test_cleaning_drops_the_carrier_without_touching_any_other_byte(
        self, tmp_path, name: str
    ) -> None:
        """The ticket-04 fixtures are scan-less: everything except the dropped
        APP1 carrier must survive byte-for-byte (raster bytes included)."""
        src = TC260_FIXTURES / name
        data = src.read_bytes()
        report = image_meta.clean_image(src, tmp_path / ("clean." + name))
        cleaned = (tmp_path / ("clean." + name)).read_bytes()
        assert not any("TC260" in f for f in report["post_findings"])
        assert label_from_metadata(cleaned) is None
        assert b"TC260:AIGC" not in cleaned
        # The kept remainder (SOI ... EOI without the carrier segment) is identical.
        assert self._non_carrier_remainder(data) == self._non_carrier_remainder(cleaned)

    @staticmethod
    def _non_carrier_remainder(data: bytes) -> bytes:
        """The file minus every APPn segment: header markers, scan, and EOI."""
        assert data.startswith(b"\xff\xd8")
        out = bytearray(b"\xff\xd8")
        i, n = 2, len(data)
        while i + 1 < n:
            if data[i] != 0xFF:
                out.extend(data[i:])
                break
            marker = data[i + 1]
            if marker == 0xD9:
                out.extend(b"\xff\xd9")
                break
            if marker == 0xDA:
                out.extend(data[i:])
                break
            if 0xD0 <= marker <= 0xD7 or marker == 0x01:
                out.extend(data[i : i + 2])
                i += 2
                continue
            seglen = int.from_bytes(data[i + 2 : i + 4], "big")
            if seglen < 2 or i + seglen > n:
                out.extend(data[i:])
                break
            if not 0xE0 <= marker <= 0xEF:
                out.extend(data[i : i + seglen + 2])
            i += seglen + 2
        return bytes(out)
