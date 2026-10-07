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

Metadata removal is verifiable and on by default. Visible-mark removal is
opt-in through ``VideoPlan.visible`` and labeled best-effort when a detector
authorized the fill, or verifiable when the operator named the region. Its
engine lives in ``video_visible``, which is imported only when the opt-in is
present so a metadata-only clean never needs OpenCV.

A note on absence: a video that carries no supported marker is reported as
``unknown``, never as clean. The readers cover three fixed containers; a
signal outside them was not measured, and "not measured" is not "not there".
"""

from __future__ import annotations

import shlex
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from common import MAX_INPUT_BYTES, read_bytes_bounded
from container_readers import read_container_findings
from optional_deps import module_available
from video_encoding import DEFAULT_VIDEO_TIMEOUT, remux_video

if TYPE_CHECKING:
    # Annotation-only. The visible engine needs OpenCV, so its plan section is
    # imported at runtime only when the operator opts in -- the metadata-only
    # default must not pull a pixel library into the clean path.
    from video_visible import VisibleVideoPlan

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


def video_runtime_hint(missing: Sequence[str]) -> str:
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
    native provenance markers and touches no pixel. ``visible`` is the opt-in:
    ``None`` means no frame is decoded and no frame can change, and a present
    plan means a visible pass runs under its own gate.
    """

    remove_metadata: bool = True
    visible: VisibleVideoPlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.remove_metadata, bool):
            raise TypeError("remove_metadata must be a bool")
        if self.visible is not None:
            # Imported here, not at module load: the visible engine needs
            # OpenCV, and a metadata-only clean must not require it.
            from video_visible import VisibleVideoPlan as _VisibleVideoPlan

            if not isinstance(self.visible, _VisibleVideoPlan):
                raise TypeError("visible must be a VisibleVideoPlan or None")


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


def _read_video(path: Path) -> tuple[str, tuple[dict, ...]]:
    """Read one video once: its container name and every supported marker."""
    data = read_bytes_bounded(path, MAX_INPUT_BYTES, label="video")
    container = _container_name(data)
    markers = tuple(
        _marker(finding["container"], finding["label"]) for finding in read_container_findings(data)
    )
    return container, markers


def _marker_key(marker: dict) -> tuple[str, str]:
    """A stable identity for one marker: container plus its canonical label."""
    return marker["container"], repr(sorted(marker["label"].items()))


@dataclass(frozen=True, slots=True)
class VideoCleanResult:
    """The verifiable outcome of one metadata strip, plus the audio boundary.

    ``visible`` is the visible pass's own report, or ``None`` when no visible
    pass was requested. It carries the field that matters most here:
    ``verification``, ``best-effort`` for detection-driven removal and
    ``verifiable`` for a region the operator named.
    """

    container: str
    markers_detected: tuple[dict, ...]
    markers_removed: tuple[dict, ...]
    markers_remaining: tuple[dict, ...]
    audio: str = AUDIO_UNVERIFIED
    visible: dict | None = None

    @property
    def residual(self) -> bool:
        """True when a provenance marker survived the strip: the clean fails."""
        return bool(self.markers_remaining)

    def to_dict(self) -> dict:
        return {
            "format": self.container,
            "markers_detected": [dict(marker) for marker in self.markers_detected],
            "markers_removed": [dict(marker) for marker in self.markers_removed],
            "markers_remaining": [dict(marker) for marker in self.markers_remaining],
            "audio": self.audio,
            "audio_verified": False,
            "visible": dict(self.visible) if self.visible is not None else None,
        }


def clean_video(
    source: Path,
    dest: Path,
    plan: VideoPlan,
    *,
    timeout: float | None = None,
) -> VideoCleanResult:
    """Strip a video's provenance markers into ``dest`` and verify the result.

    The source's markers are read before the write and the destination's after,
    so every removed and every surviving marker is named. A survivor is not an
    error: it is a reported residual that fails the clean. With no visible plan
    the remux copies every stream, audio included; with a visible plan the
    frames are re-encoded through the fill backend and the audio is still
    stream-copied. Either way the destination is staged, so a failure leaves
    ``dest`` untouched.
    """
    missing = video_runtime()
    if missing:
        raise ValueError(video_runtime_hint(missing))
    source_container, detected = _read_video(source)
    visible: dict | None = None
    if plan.visible is None:
        remux_kwargs = {} if timeout is None else {"timeout": timeout}
        remux_video(source, dest, strip_metadata=plan.remove_metadata, **remux_kwargs)
    else:
        # Imported here, not at module load: the visible engine needs OpenCV,
        # and a metadata-only clean (the default) must not require it.
        from video_visible import clean_visible_video

        visible = clean_visible_video(
            source,
            dest,
            mark=plan.visible.mark,
            mask_path=plan.visible.mask_path,
            box=plan.visible.box,
            backend=plan.visible.backend,
            remove_metadata=plan.remove_metadata,
            temporal_consistency=plan.visible.temporal_consistency,
            timeout=DEFAULT_VIDEO_TIMEOUT if timeout is None else timeout,
        ).to_dict()
    dest_container, remaining = _read_video(dest)
    remaining_keys = {_marker_key(marker) for marker in remaining}
    removed = tuple(marker for marker in detected if _marker_key(marker) not in remaining_keys)
    return VideoCleanResult(
        container=dest_container if dest_container != "unknown" else source_container,
        markers_detected=detected,
        markers_removed=removed,
        markers_remaining=remaining,
        visible=visible,
    )


def inspect_video(path: Path) -> dict:
    """The read-only video report: the markers carried, and where.

    A video with no supported marker reports ``verdict: "unknown"``. It is
    never reported as clean: only three fixed containers are read, so absence
    of a finding is not evidence of absence of a marker.
    """
    container, markers = _read_video(path)
    listed = [dict(marker) for marker in markers]
    if listed:
        notes = [
            f"{len(listed)} provenance marker(s) read from the {container} container; "
            "a metadata strip removes these"
        ]
    else:
        notes = [
            f"No supported provenance marker was read from this video "
            f"({container} container); absence is unknown, not a statement that the "
            "video carries none"
        ]
    return {
        "format": container,
        "markers": listed,
        "verdict": "markers-found" if listed else "unknown",
        "notes": notes,
    }
