"""Behavioral contract for video asset-kind recognition and runtime refusal.

Two agreed seams. The classifier's own interface (``classify_asset`` /
``classify`` / ``classify_bytes``) and the pipeline seam — a ``CleanRequest``
in, a refusal out — crossed the way the CLI and the demo do.
"""

from __future__ import annotations

import importlib
import sys
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


def test_video_sniff_is_bounded_and_reads_only_the_header(tmp_path: Path, monkeypatch) -> None:
    module = _routing_module()
    path = tmp_path / "clip.bin"
    path.write_bytes(EBML_MAGIC + b"\x00" * 64)

    def reject_whole_file_read(_path):
        raise AssertionError("classification must not read the whole file")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    assert module.classify_asset(path) == "video"
