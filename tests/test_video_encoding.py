"""Behavioral contract for the ffprobe/ffmpeg video encode adapter.

The agreed seam is the command runner's existing fake: the adapter calls
``external_command.run_command`` with an argument vector, so the tests drive it
by patching ``video_encoding.run_command`` with faked subprocess results and
assert on the probe profile and the bytes written, never on internal calls.

One integration test (below) calls the same public entry points with the real
ffprobe/ffmpeg when they are on PATH, because "playable" and "the audio
bitstream is byte-identical" can only be measured against real tools. It is
skipped, never silently passed, when they are absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import fake_command_result

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import video_encoding  # noqa: E402

#: One ffprobe video-stream row: a full-range 4:2:2 source whose time base and
#: frame rate both need normalizing, with the raw bit depth the probe reports.
_PROFILE_JSON = json.dumps(
    {
        "streams": [
            {
                "pix_fmt": "yuvj422p",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "time_base": "2/180000",
                "start_pts": 180000,
                "r_frame_rate": "10/1",
                "avg_frame_rate": "10/1",
                "bits_per_raw_sample": "8",
            }
        ]
    }
)


def _recording_runner(result, seen: list) -> object:
    """A ``run_command`` fake that records each argument vector it is handed."""

    def fake_run(argv, **_kwargs):
        seen.append(argv)
        return result

    return fake_run


def test_probing_a_clip_reports_its_pixel_format_and_frame_timing(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")
    seen: list = []
    monkeypatch.setattr(
        video_encoding,
        "run_command",
        _recording_runner(fake_command_result(0, stdout=_PROFILE_JSON), seen),
    )

    profile = video_encoding.probe_video_encode_profile(source)

    assert profile.pixel_format == "yuv422p"
    assert profile.source_pixel_format == "yuvj422p"
    assert profile.frame_rate == "10/1"
    assert profile.time_base == "1/90000"
    assert profile.start_pts == 180000
    assert profile.color_range == "tv"
    assert profile.color_space == "bt709"
    assert profile.color_transfer == "bt709"
    assert profile.color_primaries == "bt709"
    assert profile.component_depth == 8
    assert profile.is_default is False
    assert profile.default_reason is None

    argv = seen[0]
    assert argv[0] == "ffprobe"
    assert str(source) in argv


def test_probing_frame_timing_reads_best_effort_timestamps(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")
    monkeypatch.setattr(
        video_encoding,
        "run_command",
        lambda *_args, **_kwargs: fake_command_result(0, stdout="0.000000\n0.041667\n"),
    )

    assert video_encoding.probe_video_timestamps(source) == (0.0, 0.041667)


def test_unusable_frame_timing_returns_no_timestamps(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")
    for stdout in ("", "N/A\n", "0.0\nnan\n", "0.0\ninf\n"):
        monkeypatch.setattr(
            video_encoding,
            "run_command",
            lambda *_args, _stdout=stdout, **_kwargs: fake_command_result(0, stdout=_stdout),
        )
        assert video_encoding.probe_video_timestamps(source) == ()
