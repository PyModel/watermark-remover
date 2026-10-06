"""Video provenance inspection and verifiable metadata removal.

The product path for a video asset: read the native container provenance
markers (Matroska/WebM, FLV, AVI) through the already-shipped bounded readers,
and strip container metadata through the already-shipped ffmpeg adapter. The
metadata-report shape -- detected markers, remaining markers, and an
unverified-audio boundary -- follows wiltodelta/remove-ai-watermarks
``video.py`` (Apache-2.0); no code is copied from it.

Three entry points, and no fourth: :func:`video_runtime` is the preflight the
planning step calls, :func:`inspect_video` is read-only, and
:func:`clean_video` is the only write path. No surface calls a video cleaner
directly -- ``clean_asset`` dispatches here for the ``video`` kind.

A note on absence: a video that carries no supported marker is reported as
``unknown``, never as clean. The readers cover three fixed containers; a
signal outside them was not measured, and "not measured" is not "not there".
"""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from common import MAX_INPUT_BYTES, read_bytes_bounded
from container_readers import read_container_findings
from optional_deps import module_available
from video_encoding import remux_video

#: The audio disposition every video result carries. The original stream is
#: bitstream-copied by the remux, but nothing inspected it, so the value is a
#: constant, not a measurement: "unverified".
AUDIO_UNVERIFIED = "unverified"

#: The video runtime a video asset needs before it can be planned: the two
#: executables the video path shells out to, and the importable half of the
#: ``video`` extra (PyAV). Every piece is required, so a missing runtime is a
#: refusal before any write, never a degraded encode.
VIDEO_EXTRA = "video"
VIDEO_RUNTIME_BINARIES = ("ffmpeg", "ffprobe")
VIDEO_RUNTIME_MODULES = ("av",)

#: Where each shipped reader stores the TC260 AIGC label. The readers return
#: the container name; the location is fixed by the container format, so it is
#: named here rather than re-derived per finding.
_CONTAINER_LOCATIONS = {
    "matroska": "Segment.Tags.Tag.SimpleTag TagName=AIGC",
    "flv": "script.onMetaData.AIGC",
    "avi": "LIST/INFO/AIGC",
}

#: The EBML header that opens every Matroska and WebM file.
_MATROSKA_MAGIC = b"\x1a\x45\xdf\xa3"


def video_runtime() -> tuple[str, ...]:
    """The absent video-runtime pieces, executables first, in a stable order."""
    missing = [name for name in VIDEO_RUNTIME_BINARIES if shutil.which(name) is None]
    missing.extend(name for name in VIDEO_RUNTIME_MODULES if not module_available(name))
    return tuple(missing)


def video_runtime_hint(missing: tuple[str, ...] | list[str]) -> str:
    """The labeled refusal for a video asset whose runtime is incomplete.

    The install command is shell-quoted: the extra's brackets are a shell glob,
    and a copied hint that silently splits the package name is worse than none.
    """
    detail = f" (missing: {', '.join(missing)})" if missing else ""
    install = shlex.quote(f"watermark-remover[{VIDEO_EXTRA}]")
    return (
        f"video assets need the video runtime{detail}. "
        f"Install: pip install {install} and a system ffmpeg/ffprobe"
    )


@dataclass(frozen=True, slots=True)
class VideoPlan:
    """The frozen video policy carried on ``CleanPlan``.

    ``remove_metadata`` is the default: a clean with no video flags strips the
    native provenance markers and touches no pixel. A later visible stage joins
    as its own gate rather than widening this flag.
    """

    remove_metadata: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.remove_metadata, bool):
            raise TypeError("remove_metadata must be a bool")


def _container_name(data: bytes) -> str:
    """The container the bytes open, or ``unknown`` for anything else."""
    if data.startswith(_MATROSKA_MAGIC):
        return "matroska"
    if data[:3] == b"FLV":
        return "flv"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "avi"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "isobmff"
    return "unknown"


def _marker(container: str, label: dict[str, str]) -> dict:
    """One JSON-friendly provenance marker: its container, where, and label."""
    return {
        "container": container,
        "location": _CONTAINER_LOCATIONS.get(container, container),
        "label": dict(label),
    }


def read_video_markers(path: Path) -> tuple[dict, ...]:
    """Every supported provenance marker in one video file, in file order."""
    data = read_bytes_bounded(path, MAX_INPUT_BYTES, label="video")
    return tuple(_marker(finding["container"], finding["label"]) for finding in read_container_findings(data))


def marker_key(marker: dict) -> tuple[str, str]:
    """A stable identity for one marker: container plus its canonical label."""
    return marker["container"], repr(sorted(marker["label"].items()))


def inspect_video(path: Path) -> dict:
    """The read-only video report: the markers carried, and where.

    A video with no supported marker reports ``verdict: "unknown"``. It is
    never reported as clean: only three fixed containers are read, so absence
    of a finding is not evidence of absence of a marker.
    """
    data = read_bytes_bounded(path, MAX_INPUT_BYTES, label="video")
    markers = [_marker(finding["container"], finding["label"]) for finding in read_container_findings(data)]
    container = _container_name(data)
    if markers:
        notes = [
            f"{len(markers)} provenance marker(s) read from the {container} container; "
            "a clean removes these"
        ]
    else:
        notes = [
            f"No supported provenance marker was read from this video "
            f"({container} container); absence is unknown, not a statement that the "
            "video carries none"
        ]
    return {
        "format": container,
        "markers": markers,
        "verdict": "markers-found" if markers else "unknown",
        "notes": notes,
    }
