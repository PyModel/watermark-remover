"""Behavioral contract for video asset-kind recognition and runtime refusal.

Two agreed seams. The classifier's own interface (``classify_asset`` /
``classify`` / ``classify_bytes``) and the pipeline seam — a ``CleanRequest``
in, a refusal out — crossed the way the CLI and the demo do.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

#: The Matroska/WebM EBML header — the only magic the ticket names.
EBML_MAGIC = b"\x1a\x45\xdf\xa3"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv")


def _routing_module():
    return importlib.import_module("asset_kind")


@pytest.mark.parametrize("suffix", VIDEO_EXTENSIONS)
def test_video_extension_classifies_as_video(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"clip{suffix}"
    path.write_bytes(b"bytes the extension overrides")

    assert _routing_module().classify_asset(path) == "video"


def test_matroska_magic_with_a_non_video_extension_is_video(tmp_path: Path) -> None:
    path = tmp_path / "recording.payload"
    path.write_bytes(EBML_MAGIC + b"\x00" * 64)

    assert _routing_module().classify_asset(path) == "video"


def test_matroska_magic_without_an_extension_is_video(tmp_path: Path) -> None:
    path = tmp_path / "recording"
    path.write_bytes(EBML_MAGIC + b"\x00" * 64)

    assert _routing_module().classify(path) == "video"


def test_matroska_magic_on_the_bytes_surface_is_video() -> None:
    module = _routing_module()
    payload = EBML_MAGIC + b"\x00" * 64

    assert module.classify_bytes(payload, ".payload") == "video"
    assert module.classify_bytes(payload) == "video"


def test_truncated_matroska_magic_is_unknown(tmp_path: Path) -> None:
    # Three of the four EBML bytes: malformed input stays `unknown`, never a
    # guess and never a crash.
    path = tmp_path / "almost.bin"
    path.write_bytes(b"\x1a\x45\xdf")

    assert _routing_module().classify_asset(path) == "unknown"


def test_video_is_never_text(tmp_path: Path) -> None:
    module = _routing_module()
    for suffix in VIDEO_EXTENSIONS:
        path = tmp_path / f"clip{suffix}"
        path.write_bytes(b"plain ascii text")
        assert module.classify_asset(path) != "text"
    ebml = tmp_path / "recording.payload"
    ebml.write_bytes(EBML_MAGIC + b"\x00" * 64)
    assert module.classify_asset(ebml) != "text"


def test_video_sniff_is_bounded_and_reads_only_the_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _routing_module()
    path = tmp_path / "clip.bin"
    path.write_bytes(EBML_MAGIC + b"\x00" * 64)

    def reject_whole_file_read(_path):
        raise AssertionError("classification must not read the whole file")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    assert module.classify_asset(path) == "video"


# ---------------------------------------------------------------------------
# Runtime refusal at the pipeline seam (ticket 01)
# ---------------------------------------------------------------------------

#: The exact shell-quoted install command the refusal must carry. The extra's
#: brackets are a shell glob, so the hint is copy-pasteable only when quoted.
QUOTED_VIDEO_INSTALL = "pip install 'watermark-remover[video]'"


def _video_file(path: Path) -> Path:
    path.write_bytes(EBML_MAGIC + b"\x00" * 64)
    return path


def _simulate_absent_video_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force ffmpeg/ffprobe and the video extra absent, whatever the host has."""
    empty = tmp_path / "no-tools"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))
    # A ``None`` entry makes the import system raise ImportError for the name,
    # so the probe sees the extra absent even where PyAV is installed.
    monkeypatch.setitem(sys.modules, "av", None)


@pytest.mark.parametrize("name", ["clip.mp4", "recording.payload"])
def test_clean_video_without_runtime_refuses_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    import clean_file

    src = _video_file(tmp_path / name)
    dest = tmp_path / f"cleaned{src.suffix}"
    _simulate_absent_video_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["clean_file.py", str(src), "-o", str(dest)])

    exit_code = clean_file.main()

    err = capsys.readouterr().err
    assert exit_code == 1
    assert QUOTED_VIDEO_INSTALL in err
    assert "pip install watermark-remover[video]" not in err
    for missing in ("ffmpeg", "ffprobe", "av"):
        assert missing in err
    assert not dest.exists()


def test_demo_video_without_runtime_returns_the_same_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from demo import clean_upload

    src = _video_file(tmp_path / "clip.webm")
    _simulate_absent_video_runtime(monkeypatch, tmp_path)

    report, output, _prompt = clean_upload(str(src), False, False, "paraphrase")

    assert report.startswith("**Error cleaning")
    assert QUOTED_VIDEO_INSTALL in report
    assert "pip install watermark-remover[video]" not in report
    assert "ffmpeg" in report
    assert output is None


def _fake_tool(directory: Path, name: str) -> None:
    if sys.platform == "win32":
        # Windows lookup only resolves PATHEXT executables, so the fake
        # must be a .bat; the POSIX spelling does not exist on win32.
        tool = directory / f"{name}.bat"
        tool.write_text("@echo off\r\nexit 0\r\n", encoding="utf-8")
        return
    tool = directory / name
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(0o755)


@pytest.mark.parametrize(
    ("tools", "av_present", "missing"),
    [
        (("ffprobe",), True, "ffmpeg"),
        (("ffmpeg",), True, "ffprobe"),
        (("ffmpeg", "ffprobe"), False, "av"),
    ],
)
def test_clean_video_refuses_when_one_runtime_piece_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tools: tuple[str, ...],
    av_present: bool,
    missing: str,
) -> None:
    # "ffmpeg, ffprobe, or the video extra is absent" means any one of them:
    # each piece alone must refuse, not only the all-absent case.
    import clean_file

    bindir = tmp_path / "tools"
    bindir.mkdir()
    for name in tools:
        _fake_tool(bindir, name)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av") if av_present else None)
    src = _video_file(tmp_path / "clip.mkv")
    dest = tmp_path / "cleaned.mkv"
    monkeypatch.setattr(sys, "argv", ["clean_file.py", str(src), "-o", str(dest)])

    exit_code = clean_file.main()

    err = capsys.readouterr().err
    assert exit_code == 1
    assert f"(missing: {missing})" in err
    assert QUOTED_VIDEO_INSTALL in err
    assert not dest.exists()
