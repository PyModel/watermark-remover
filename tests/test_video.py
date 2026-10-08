# Ported from wiltodelta/remove-ai-watermarks tests/test_video.py, Apache-2.0;
# modified: the byte layouts drive this tree's inspect and clean seams, and the
# external tools are faked through the command runner.
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


def read_findings(path: Path) -> tuple:
    """The shipped reader's findings for one file's bytes."""
    import container_readers

    return container_readers.read_container_findings(path.read_bytes())


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


# --- cleaning through the pipeline -----------------------------------------


def _fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the runtime preflight without depending on the host's tools."""
    bindir = tmp_path / "tools"
    bindir.mkdir(exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        tool = bindir / name
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))


def _ffmpeg_writing(output: bytes):
    """A command-runner fake that publishes ``output`` at the staged path."""

    def fake_run(argv, **_kwargs):
        Path(argv[-1]).write_bytes(output)
        return fake_command_result(0)

    return fake_run


def _clean_video(src: Path, *, output: bytes, monkeypatch: pytest.MonkeyPatch):
    import clean_pipeline
    import video_encoding
    from clean_request import CleanRequest

    dest = src.with_name(f"{src.stem}-cleaned{src.suffix}")
    monkeypatch.setattr(video_encoding, "run_command", _ffmpeg_writing(output))
    run = clean_pipeline.execute(CleanRequest(paths=(src,), output=dest))
    return run.payload, dest


def test_cleaning_a_video_names_removed_markers_and_reports_audio_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))

    payload, dest = _clean_video(src, output=matroska(), monkeypatch=monkeypatch)

    assert payload["kind"] == "video"
    assert payload["format"] == "matroska"
    assert [m["label"]["ContentProducer"] for m in payload["markers_detected"]] == [DOUBAO_USCC]
    assert [m["label"]["ContentProducer"] for m in payload["markers_removed"]] == [DOUBAO_USCC]
    assert payload["markers_remaining"] == []
    assert payload["audio"] == "unverified"
    assert payload["audio_verified"] is False
    assert payload["exit_code"] == 0
    assert dest.exists()


def test_a_surviving_marker_is_named_and_fails_the_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))

    # ffmpeg finished but kept the tag: the clean must fail closed.
    payload, _dest = _clean_video(src, output=src.read_bytes(), monkeypatch=monkeypatch)

    assert payload["markers_removed"] == []
    assert [m["label"]["ContentProducer"] for m in payload["markers_remaining"]] == [DOUBAO_USCC]
    assert payload["exit_code"] == 1
    assert payload["audio"] == "unverified"


def test_a_video_with_no_marker_is_cleaned_by_the_video_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "bare.mkv"
    src.write_bytes(matroska())

    payload, dest = _clean_video(src, output=matroska(), monkeypatch=monkeypatch)

    assert payload["kind"] == "video"
    assert payload["markers_detected"] == []
    assert payload["markers_removed"] == []
    assert payload["markers_remaining"] == []
    assert payload["exit_code"] == 0
    assert dest.exists()
    assert "unsupported container" not in json.dumps(payload)


def test_the_default_video_plan_removes_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clean_request import CleanRequest, plan_work, select_request_inputs

    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))
    request = CleanRequest(paths=(src,), output=tmp_path / "out.mkv")

    selection = select_request_inputs(request)
    work = plan_work(selection.items, request, selection.batch)
    _item, _output, plan = work[0]

    assert plan.forced_kind == "video"
    assert plan.video is not None
    assert plan.video.remove_metadata is True
    # The default touches no pixel: no visible work and no image pixel plan.
    assert plan.visible is None
    assert plan.pixel is None


def test_an_explicit_visible_flag_on_a_video_is_carried_not_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clean_request import CleanRequest, plan_work, select_request_inputs

    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))
    request = CleanRequest(
        paths=(src,),
        output=tmp_path / "out.mkv",
        visible_box=(0, 0, 8, 8),
        visible_backend="simple",
    )

    selection = select_request_inputs(request)
    work = plan_work(selection.items, request, selection.batch)
    _item, _output, plan = work[0]

    assert plan.video is not None
    assert plan.video.remove_metadata is True
    assert plan.video.visible is not None
    assert plan.video.visible.box is not None


def test_batch_mode_visible_reaches_a_video_as_mark_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--mode visible`` sets ``--mark auto``; a video must accept it, not refuse."""
    from clean_request import CleanRequest, plan_work, select_request_inputs

    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))
    request = CleanRequest(paths=(src,), output=tmp_path / "out.mkv", mark="auto", mode="visible")

    selection = select_request_inputs(request)
    _item, _output, plan = plan_work(selection.items, request, selection.batch)[0]

    assert plan.video is not None
    assert plan.video.visible is not None
    assert plan.video.visible.mark == "auto"


# --- real tools: the measurement a fake cannot make ------------------------

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
requires_video_tools = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe are not on PATH; their absence is the labeled-default path",
)


def _tool(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, check=False, timeout=120.0)


def _stream_md5(path: Path, stream: str) -> str:
    """One elementary stream's bitstream hash, read with a stream copy."""
    result = _tool(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", stream, "-c", "copy", "-f", "md5", "-"]
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


@requires_video_tools
def test_a_real_video_loses_its_marker_and_keeps_its_frames_and_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one measurement a fake cannot make: real ffmpeg, real bytes."""
    import clean_pipeline
    from clean_request import CleanRequest

    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))
    src = tmp_path / "source.mkv"
    created = _tool(
        [
            _FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-metadata",
            f"AIGC={LABEL.decode()}",
            str(src),
        ]
    )
    assert created.returncode == 0, created.stderr.decode()
    assert read_findings(src)
    video_before = _stream_md5(src, "0:v:0")
    audio_before = _stream_md5(src, "0:a:0")

    dest = tmp_path / "cleaned.mkv"
    run = clean_pipeline.execute(CleanRequest(paths=(src,), output=dest))
    payload = run.payload

    assert payload["kind"] == "video"
    assert payload["exit_code"] == 0
    assert [m["label"]["ContentProducer"] for m in payload["markers_removed"]] == [DOUBAO_USCC]
    assert payload["markers_remaining"] == []
    assert payload["audio"] == "unverified"
    assert dest.exists()
    # The marker is actually gone, read back from the output bytes.
    assert read_findings(dest) == ()
    # No frame changed, and the audio is the original bitstream.
    assert _stream_md5(dest, "0:v:0") == video_before
    assert _stream_md5(dest, "0:a:0") == audio_before


def test_a_remux_that_fails_leaves_the_destination_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive tier 2: the destination appears only after a completed write."""
    import clean_pipeline
    import video_encoding
    from clean_request import CleanRequest

    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))
    dest = tmp_path / "out.mkv"
    dest.write_bytes(b"the previous good output")

    def failed(argv, **_kwargs):
        Path(argv[-1]).write_bytes(b"half a container")  # ffmpeg died mid-write
        return fake_command_result(1, stderr="Invalid argument")

    monkeypatch.setattr(video_encoding, "run_command", failed)
    run = clean_pipeline.execute(CleanRequest(paths=(src,), output=dest))

    assert run.payload["exit_code"] == 1
    assert dest.read_bytes() == b"the previous good output"
    assert not list(tmp_path.glob(".*"))  # the staged sibling was removed


def test_an_in_place_clean_stages_the_write_and_keeps_a_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clean_pipeline
    import video_encoding
    from clean_request import CleanRequest

    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))

    monkeypatch.setattr(video_encoding, "run_command", _ffmpeg_writing(matroska()))
    run = clean_pipeline.execute(CleanRequest(paths=(src,), in_place=True))

    assert run.payload["exit_code"] == 0
    assert run.payload["markers_removed"]
    # The source now holds the stripped bytes; the unstripped original is kept.
    assert read_findings(src) == ()
    assert read_findings(Path(f"{src}.bak")) != ()
