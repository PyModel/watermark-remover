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
import types
from pathlib import Path

import pytest
from conftest import fake_command_result
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import video_encoding

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


def test_probing_a_clip_reports_its_pixel_format_and_frame_timing(
    monkeypatch, tmp_path: Path
) -> None:
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


def test_absent_ffprobe_labels_the_probe_as_a_default(monkeypatch, tmp_path: Path) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(video_encoding, "run_command", missing)

    profile = video_encoding.probe_video_encode_profile(tmp_path / "clip.mp4")

    assert profile.is_default is True
    assert profile.pixel_format == "yuv420p"
    assert profile.default_reason is not None
    assert "ffprobe" in profile.default_reason


def test_a_probe_that_cannot_answer_labels_the_fallback(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")

    def timed_out(*_args, **_kwargs):
        raise video_encoding.ExternalCommandTimeout("ffprobe timed out")

    def failed(*_args, **_kwargs):
        return fake_command_result(1, stderr="Invalid data found when processing input")

    cases = [
        timed_out,
        failed,
        lambda *_a, **_k: fake_command_result(0, stdout="not json at all"),
        lambda *_a, **_k: fake_command_result(0, stdout=""),
        lambda *_a, **_k: fake_command_result(0, stdout='{"streams": []}'),
        lambda *_a, **_k: fake_command_result(0, stdout='{"streams": {}}'),
        lambda *_a, **_k: fake_command_result(0, stdout="[]"),
    ]
    for runner in cases:
        monkeypatch.setattr(video_encoding, "run_command", runner)
        profile = video_encoding.probe_video_encode_profile(source)
        assert profile.is_default is True
        assert profile.pixel_format == "yuv420p"
        assert profile.frame_rate is None


def test_a_truncated_probe_answer_is_not_a_measurement(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")
    truncated = fake_command_result(0, stdout=_PROFILE_JSON)
    truncated.stdout_truncated = True
    monkeypatch.setattr(video_encoding, "run_command", lambda *_a, **_k: truncated)

    profile = video_encoding.probe_video_encode_profile(source)

    assert profile.is_default is True
    assert video_encoding.probe_video_timestamps(source) == ()


def test_component_depth_falls_back_to_the_probes_own_bit_depth(
    monkeypatch, tmp_path: Path
) -> None:
    # ``None`` in sys.modules is the documented way to make a later import
    # raise ImportError, so this pins the PyAV-absent path even on a machine
    # that has av installed.
    monkeypatch.setitem(sys.modules, "av", None)
    source = tmp_path / "clip10.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")
    monkeypatch.setattr(
        video_encoding,
        "run_command",
        lambda *_a, **_k: fake_command_result(
            0,
            stdout=json.dumps(
                {"streams": [{"pix_fmt": "yuv420p10le", "bits_per_raw_sample": "10"}]}
            ),
        ),
    )

    profile = video_encoding.probe_video_encode_profile(source)

    assert profile.pixel_format == "yuv420p"
    assert profile.source_pixel_format == "yuv420p10le"
    assert profile.component_depth == 10


def test_component_depth_consults_pyav_for_that_one_lookup(monkeypatch, tmp_path: Path) -> None:
    looked_up: list[str] = []

    class _Component:
        bits = 12

    class _VideoFormat:
        def __init__(self, name: str) -> None:
            looked_up.append(name)
            self.components = (_Component(), _Component(), _Component())

    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(VideoFormat=_VideoFormat))
    source = tmp_path / "clip10.mp4"
    source.write_bytes(b"not really a video; ffprobe is faked")
    monkeypatch.setattr(
        video_encoding,
        "run_command",
        lambda *_a, **_k: fake_command_result(
            0,
            stdout=json.dumps({"streams": [{"pix_fmt": "yuvj422p", "bits_per_raw_sample": "8"}]}),
        ),
    )

    profile = video_encoding.probe_video_encode_profile(source)

    assert looked_up == ["yuvj422p"]
    assert profile.component_depth == 12


def test_remux_copies_every_stream_and_publishes_atomically(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source-bytes")
    dest = tmp_path / "remuxed.mp4"
    seen: list[tuple] = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        Path(argv[-1]).write_bytes(b"remuxed-bytes")
        return fake_command_result(0)

    monkeypatch.setattr(video_encoding, "run_command", fake_run)

    video_encoding.remux_video(source, dest)

    # The bytes the subprocess wrote reached the destination, and the staged
    # sibling is gone (it was renamed, not left behind).
    assert dest.read_bytes() == b"remuxed-bytes"
    assert not list(tmp_path.glob(".*"))
    argv, kwargs = seen[0]
    assert not isinstance(argv, (str, bytes))
    assert argv[0] == "ffmpeg"
    assert str(source) in argv
    assert argv[argv.index("-map") + 1] == "0"
    assert argv[argv.index("-c") + 1] == "copy"
    assert argv[argv.index("-map_metadata") + 1] == "-1"
    assert argv[argv.index("-map_chapters") + 1] == "-1"
    assert argv[argv.index("-movflags") + 1] == "+faststart"
    assert argv[-1] != str(dest)
    assert Path(argv[-1]).parent == dest.parent
    assert kwargs["timeout"] > 0


def test_remux_can_keep_metadata_when_asked(monkeypatch, tmp_path: Path) -> None:
    seen: list = []

    def fake_run(argv, **_kwargs):
        seen.append(argv)
        return fake_command_result(0)

    monkeypatch.setattr(video_encoding, "run_command", fake_run)

    video_encoding.remux_video(tmp_path / "clip.mp4", tmp_path / "out.mkv", strip_metadata=False)

    assert "-map_metadata" not in seen[0]
    assert "-movflags" not in seen[0]


def test_a_remux_killed_mid_write_leaves_no_destination(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source-bytes")
    dest = tmp_path / "never-published.mp4"

    def killed(argv, **_kwargs):
        Path(argv[-1]).write_bytes(b"half a container")  # ffmpeg died mid-write
        raise video_encoding.ExternalCommandTimeout("killed after partial output")

    monkeypatch.setattr(video_encoding, "run_command", killed)

    with pytest.raises(video_encoding.VideoEncodingError):
        video_encoding.remux_video(source, dest)

    assert not dest.exists()
    assert not list(tmp_path.glob(".*"))


def test_a_failed_remux_never_replaces_an_existing_destination(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source-bytes")
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"the previous good output")

    def failed(argv, **_kwargs):
        Path(argv[-1]).write_bytes(b"half a container")
        return fake_command_result(1, stderr="Invalid argument")

    monkeypatch.setattr(video_encoding, "run_command", failed)

    with pytest.raises(video_encoding.VideoEncodingError, match="Invalid argument"):
        video_encoding.remux_video(source, dest)

    assert dest.read_bytes() == b"the previous good output"
    assert not list(tmp_path.glob(".*"))


def test_an_absent_ffmpeg_fails_labeled_without_writing(monkeypatch, tmp_path: Path) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(video_encoding, "run_command", missing)
    dest = tmp_path / "out.mp4"

    with pytest.raises(video_encoding.VideoEncodingError, match="ffmpeg is unavailable"):
        video_encoding.remux_video(tmp_path / "clip.mp4", dest)

    assert not dest.exists()


def test_a_filename_with_quotes_and_spaces_is_passed_as_one_argument(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / 'clip "quoted" and spaced.mp4'
    source.write_bytes(b"source-bytes")
    dest = tmp_path / 'out "copy" 2026.mp4'
    seen: list = []

    def fake_run(argv, **_kwargs):
        seen.append(argv)
        if argv[0] == "ffprobe":
            return fake_command_result(0, stdout=_PROFILE_JSON)
        Path(argv[-1]).write_bytes(b"remuxed-bytes")
        return fake_command_result(0)

    monkeypatch.setattr(video_encoding, "run_command", fake_run)

    profile = video_encoding.probe_video_encode_profile(source)
    video_encoding.remux_video(source, dest)

    assert profile.pixel_format == "yuv422p"
    assert dest.read_bytes() == b"remuxed-bytes"
    for argv in seen:
        assert not isinstance(argv, (str, bytes))
        assert all(isinstance(argument, str) for argument in argv)
        assert sum(1 for argument in argv if argument == str(source)) == 1


_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
requires_video_tools = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe are not on PATH; their absence is the labeled-default path",
)


def _tool(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, check=False, timeout=120.0)


def _audio_bitstream_md5(path: Path) -> str:
    """The audio elementary bitstream's hash, read with a stream copy."""
    result = _tool(
        [_FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"]
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


@requires_video_tools
def test_a_real_clip_probes_and_remuxes_with_the_audio_bitstream_intact(tmp_path: Path) -> None:
    """The one measurement a fake cannot make: real tools, real bytes."""
    source = tmp_path / "source.mp4"
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
            str(source),
        ]
    )
    assert created.returncode == 0, created.stderr.decode()
    assert source.stat().st_size > 0

    profile = video_encoding.probe_video_encode_profile(source)
    assert profile.is_default is False
    assert profile.pixel_format == "yuv420p"
    assert profile.frame_rate == "10/1"
    assert profile.time_base is not None
    assert profile.start_pts == 0

    timestamps = video_encoding.probe_video_timestamps(source)
    assert len(timestamps) == 10
    assert timestamps[0] == 0.0

    dest = tmp_path / "remuxed.mp4"
    video_encoding.remux_video(source, dest)

    # Playable: ffprobe reads the destination and both streams are present.
    assert dest.exists()
    streams = _tool(
        [
            _FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(dest),
        ]
    )
    assert streams.returncode == 0, streams.stderr.decode()
    kinds = streams.stdout.decode().split()
    assert "video" in kinds
    assert "audio" in kinds

    # The audio bitstream is byte-identical: the packets were copied, not
    # re-encoded.
    assert _audio_bitstream_md5(dest) == _audio_bitstream_md5(source)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
def test_the_video_extra_declares_the_pyav_pin_and_the_python_floor() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    # The core keeps the 3.10 floor; the video extra carries av's own 3.11
    # floor, expressed as the environment marker that scopes the dependency.
    assert project["requires-python"] == ">=3.10"
    requirements = [Requirement(entry) for entry in project["optional-dependencies"]["video"]]
    pyav = next(requirement for requirement in requirements if requirement.name == "av")
    assert str(pyav.specifier) == "<19,>=18"
    assert pyav.marker is not None
    assert pyav.marker.evaluate({"python_version": "3.10"}) is False
    assert pyav.marker.evaluate({"python_version": "3.11"}) is True
    assert pyav.marker.evaluate({"python_version": "3.14"}) is True
