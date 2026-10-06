"""Container provenance readers: synthetic Matroska, FLV, and AVI markers.

The byte layouts are built in pure Python, adapted from the reference tree's
``tests/test_video.py`` (wiltodelta/remove-ai-watermarks, Apache-2.0). Routing
these through a full encode would turn a layout contract into an integration
test; the readers are stdlib-only and take bytes, so the seam stays here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import container_readers
from tc260_label import MAX_TC260_VALUE_BYTES

DOUBAO_USCC = "91110102MACQD9K640"
PRODUCE_ID = "c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"
LABEL = (
    b'{"Label":"1","ContentProducer":"91110102MACQD9K640",'
    b'"ProduceID":"c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"}'
)


# --- Matroska / WebM -------------------------------------------------------


def _ebml_size(value: int) -> bytes:
    for length in range(1, 9):
        if value < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | value).to_bytes(length, "big")
    raise ValueError("EBML test value is too large")


def _ebml_element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _ebml_size(len(payload)) + payload


def _matroska(value: bytes = LABEL) -> bytes:
    simple_tag = _ebml_element(
        b"\x67\xc8",
        _ebml_element(b"\x45\xa3", b"AIGC") + _ebml_element(b"\x44\x87", value),
    )
    tags = _ebml_element(b"\x12\x54\xc3\x67", _ebml_element(b"\x73\x73", simple_tag))
    segment = _ebml_element(b"\x18\x53\x80\x67", tags)
    return _ebml_element(b"\x1a\x45\xdf\xa3", b"") + segment


class TestMatroskaProvenance:
    def test_a_synthetic_matroska_marker_is_a_finding(self) -> None:
        findings = container_readers.matroska_findings(_matroska())
        assert len(findings) == 1
        assert findings[0]["container"] == "matroska"
        assert findings[0]["label"] == {
            "Label": "1",
            "ContentProducer": DOUBAO_USCC,
            "ProduceID": PRODUCE_ID,
        }


# --- FLV -------------------------------------------------------------------


def _amf0_string(value: bytes) -> bytes:
    return b"\x02" + len(value).to_bytes(2, "big") + value


def _flv(value: bytes = LABEL) -> bytes:
    payload = (
        _amf0_string(b"onMetaData")
        + b"\x08\x00\x00\x00\x02"
        + len(b"AIGC").to_bytes(2, "big")
        + b"AIGC"
        + _amf0_string(value)
        + len(b"duration").to_bytes(2, "big")
        + b"duration"
        + b"\x00"
        + b"\x00\x00\x00\x00\x00\x00\x00\x00"
        + b"\x00\x00\x09"
    )
    tag_header = b"\x12" + len(payload).to_bytes(3, "big") + b"\x00" * 7
    return (
        b"FLV\x01\x05\x00\x00\x00\x09"
        + b"\x00\x00\x00\x00"
        + tag_header
        + payload
        + (11 + len(payload)).to_bytes(4, "big")
    )


class TestFlvProvenance:
    def test_a_synthetic_flv_marker_is_a_finding(self) -> None:
        findings = container_readers.flv_findings(_flv())
        assert len(findings) == 1
        assert findings[0]["container"] == "flv"
        assert findings[0]["label"] == {
            "Label": "1",
            "ContentProducer": DOUBAO_USCC,
            "ProduceID": PRODUCE_ID,
        }


# --- AVI -------------------------------------------------------------------


def _riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return chunk_id + len(payload).to_bytes(4, "little") + payload + pad


def _avi(value: bytes = LABEL) -> bytes:
    info = _riff_chunk(b"AIGC", value) + _riff_chunk(b"INAM", b"standard title\x00")
    body = b"AVI " + _riff_chunk(b"LIST", b"INFO" + info) + _riff_chunk(b"JUNK", b"\x00" * 64)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


class TestAviProvenance:
    def test_a_synthetic_avi_marker_is_a_finding(self) -> None:
        findings = container_readers.avi_findings(_avi())
        assert len(findings) == 1
        assert findings[0]["container"] == "avi"
        assert findings[0]["label"] == {
            "Label": "1",
            "ContentProducer": DOUBAO_USCC,
            "ProduceID": PRODUCE_ID,
        }
