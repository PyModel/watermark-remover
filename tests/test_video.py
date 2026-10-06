"""Behavioral contract for video provenance inspection and verifiable cleaning.

Two agreed seams. The inspect surface is the data seam
(``inspect_file.inspect_asset``) the CLI, the service, and the TUI all read.
The clean surface is the pipeline seam (``clean_pipeline.execute``: a
``CleanRequest`` in, a result and bytes out), crossed the way the CLI does.

The byte layouts are built in pure Python (adapted from the reference tree's
``tests/test_video.py``, wiltodelta/remove-ai-watermarks, Apache-2.0) so an
inspect contract does not become an ffmpeg integration test. The ffmpeg
subprocess is faked at the command runner (``video_encoding.run_command``),
the existing seam; one real-tool test measures the actual bytes when ffmpeg
and ffprobe are on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from conftest import fake_command_result

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

DOUBAO_USCC = "91110102MACQD9K640"
DOUBAO_ID = "c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"
OTHER_USCC = "91330100MA27X2YQ4B"
OTHER_ID = "1f0e9b2c-2f5a-4a3e-9b7c-5d4e6f7a8b90"

LABEL = (
    b'{"Label":"1","ContentProducer":"91110102MACQD9K640",'
    b'"ProduceID":"c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"}'
)
SECOND_LABEL = (
    b'{"Label":"1","ContentProducer":"91330100MA27X2YQ4B",'
    b'"ProduceID":"1f0e9b2c-2f5a-4a3e-9b7c-5d4e6f7a8b90"}'
)

#: The Matroska AIGC location the shipped reader targets, named in a finding.
MATROSKA_LOCATION = "Segment.Tags.Tag.SimpleTag TagName=AIGC"


# --- synthetic containers (pure Python) ------------------------------------


def _ebml_size(value: int) -> bytes:
    for length in range(1, 9):
        if value < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | value).to_bytes(length, "big")
    raise ValueError("EBML test value is too large")


def _ebml_element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _ebml_size(len(payload)) + payload


def matroska(*values: bytes) -> bytes:
    """A minimal Matroska file carrying one ``SimpleTag TagName=AIGC`` per value."""
    simple_tags = b"".join(
        _ebml_element(
            b"\x67\xc8",
            _ebml_element(b"\x45\xa3", b"AIGC") + _ebml_element(b"\x44\x87", value),
        )
        for value in values
    )
    tags = _ebml_element(b"\x12\x54\xc3\x67", _ebml_element(b"\x73\x73", simple_tags))
    segment = _ebml_element(b"\x18\x53\x80\x67", tags)
    return _ebml_element(b"\x1a\x45\xdf\xa3", b"") + segment


def _amf0_string(value: bytes) -> bytes:
    return b"\x02" + len(value).to_bytes(2, "big") + value


def flv(value: bytes = LABEL) -> bytes:
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


def _riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return chunk_id + len(payload).to_bytes(4, "little") + payload + pad


def avi(value: bytes = LABEL) -> bytes:
    info = _riff_chunk(b"AIGC", value) + _riff_chunk(b"INAM", b"standard title\x00")
    body = b"AVI " + _riff_chunk(b"LIST", b"INFO" + info) + _riff_chunk(b"JUNK", b"\x00" * 64)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


BUILDERS = {
    "matroska": lambda: matroska(LABEL),
    "flv": flv,
    "avi": avi,
}


# --- inspection (the data seam) --------------------------------------------


def test_inspecting_a_video_names_each_marker_and_where(tmp_path: Path) -> None:
    import inspect_file

    path = tmp_path / "clip.mkv"
    path.write_bytes(matroska(LABEL, SECOND_LABEL))

    report = inspect_file.inspect_asset(path)

    assert report["kind"] == "video"
    assert report["format"] == "matroska"
    assert report["verdict"] == "markers-found"
    assert report["suspicious"] is True
    assert [(m["container"], m["label"]["ContentProducer"]) for m in report["markers"]] == [
        ("matroska", DOUBAO_USCC),
        ("matroska", OTHER_USCC),
    ]
    assert [m["location"] for m in report["markers"]] == [MATROSKA_LOCATION, MATROSKA_LOCATION]


@pytest.mark.parametrize(("container", "builder"), sorted(BUILDERS.items()))
def test_inspecting_each_container_names_its_marker(
    tmp_path: Path, container: str, builder
) -> None:
    import inspect_file

    path = tmp_path / f"clip.{container}"
    path.write_bytes(builder())

    report = inspect_file.inspect_asset(path)

    assert report["kind"] == "video"
    assert report["format"] == container
    assert [m["container"] for m in report["markers"]] == [container]
    assert report["markers"][0]["label"]["ContentProducer"] == DOUBAO_USCC


def test_a_video_with_no_marker_is_unknown_never_clean(tmp_path: Path) -> None:
    import inspect_file

    path = tmp_path / "bare.mkv"
    path.write_bytes(matroska())

    report = inspect_file.inspect_asset(path)

    assert report["kind"] == "video"
    assert report["markers"] == []
    assert report["verdict"] == "unknown"
    assert report["suspicious"] is False
    # An absent signal is unknown, never a clean bill of health.
    assert "clean" not in json.dumps(report).casefold()
    assert any("unknown" in note.casefold() for note in report["notes"])


def test_inspecting_a_video_never_falls_through_to_the_container_inspector(
    tmp_path: Path,
) -> None:
    import inspect_file

    path = tmp_path / "clip.mkv"
    path.write_bytes(matroska(LABEL))

    report = inspect_file.inspect_asset(path)

    assert report["kind"] == "video"
    assert "unsupported container" not in json.dumps(report)
