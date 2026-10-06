# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/video_encoding.py,
# Apache-2.0; modified: every subprocess goes through this repo's
# external_command.run_command with a timeout and an argument vector (no
# subprocess.run, no shutil.which), a probe that cannot measure anything
# returns a labeled default instead of a silent one, the destination is staged
# and published with one atomic rename, and the streaming raw-frame encoder
# belongs to the per-frame ticket rather than this one.
"""Source-profile probe and stream-copy remux for the video path.

The video cleaner needs two things from ffmpeg: the source's encode profile
(pixel format, color properties, frame timing) so a re-encode can preserve it,
and a remux that copies the original audio through untouched. Both shell out
through ``external_command.run_command`` with an argument vector and a timeout;
the filename is one argument element, never interpolated into a shell string.

PyAV is used for exactly one lookup -- component bit depth -- and its absence
falls back to ffprobe's own ``bits_per_raw_sample``. The mux itself is ffmpeg,
not PyAV. A probe that cannot measure anything (no ffprobe, a failure, garbage
output, a timeout) returns a profile whose ``default_reason`` labels it as a
default; it never reports an invented measurement as fact.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from external_command import ExternalCommandTimeout, run_command

log = logging.getLogger(__name__)

#: A remux is I/O-bound but bounded: two minutes covers a feature-length
#: stream copy. A probe is a metadata read and gets ten seconds.
DEFAULT_VIDEO_TIMEOUT = 120.0
DEFAULT_PROBE_TIMEOUT = 10.0
#: ffprobe JSON is a few KiB; the bound keeps a misbehaving child from
#: growing the adapter's memory.
DEFAULT_OUTPUT_LIMIT = 1_000_000

_PIXEL_FORMATS = frozenset({"yuv420p", "yuv422p", "yuv444p"})
_PIXEL_FORMAT_ALIASES = {
    "yuvj420p": "yuv420p",
    "yuvj422p": "yuv422p",
    "yuvj444p": "yuv444p",
}
_COLOR_RANGES = frozenset({"tv", "pc"})
_COLOR_SPACES = frozenset({"bt709", "fcc", "bt470bg", "smpte170m", "smpte240m"})
_COLOR_TRANSFERS = frozenset(
    {
        "bt709",
        "gamma22",
        "gamma28",
        "smpte170m",
        "smpte240m",
        "linear",
        "log",
        "log_sqrt",
        "iec61966-2-4",
        "bt1361e",
        "iec61966-2-1",
        "bt2020-10",
        "bt2020-12",
        "smpte2084",
        "smpte428",
        "arib-std-b67",
    }
)
_COLOR_PRIMARIES = frozenset(
    {
        "bt709",
        "bt470m",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "film",
        "bt2020",
        "smpte428",
        "smpte431",
        "smpte432",
        "jedec-p22",
        "ebu3213",
    }
)
#: Containers whose muxer can move the index to the front.
_FASTSTART_SUFFIXES = frozenset({".mp4", ".mov", ".m4v"})


class VideoEncodingError(RuntimeError):
    """A video subprocess failed; the destination was never published."""


@dataclass(frozen=True)
class VideoEncodeProfile:
    """Source video properties that an encode or remux can preserve.

    ``default_reason`` is the label that separates a measured profile from a
    fallback: it is set (and ``is_default`` is true) only when no measurement
    was possible, and it names why. A successfully probed profile leaves it
    None even when a single unsupported field was substituted.
    """

    pixel_format: str = "yuv420p"
    frame_rate: str | None = None
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    time_base: str | None = None
    start_pts: int | None = None
    source_pixel_format: str | None = None
    component_depth: int | None = None
    default_reason: str | None = None

    @property
    def is_default(self) -> bool:
        """True when this profile is a labeled default, not a measurement."""
        return self.default_reason is not None


def _known_value(value: object, allowed: frozenset[str]) -> str | None:
    """Return a supported ffmpeg enum value, otherwise omit it."""
    return value if isinstance(value, str) and value in allowed else None


def _ratio(value: object) -> str | None:
    """Normalize a positive ffprobe rational (time base, frame rate)."""
    if not isinstance(value, str):
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if fraction <= 0:
        return None
    return f"{fraction.numerator}/{fraction.denominator}"


def _pixel_component_depth(pixel_format: object, raw_bits: object) -> int | None:
    """Return the largest component depth reported by ffprobe or PyAV.

    PyAV is optional and serves this one lookup only; when it is absent (or
    rejects the pixel format) the probe's own bit-depth field is the answer.
    """
    depths: list[int] = []
    if isinstance(raw_bits, str) and raw_bits.isdigit():
        depths.append(int(raw_bits))
    if isinstance(pixel_format, str):
        try:
            import av

            depths.extend(component.bits for component in av.VideoFormat(pixel_format).components)
        except (ImportError, ValueError):
            pass
    return max(depths) if depths else None


def _run_ffprobe(
    source: Path,
    arguments: Sequence[str],
    *,
    purpose: str,
    output_format: str,
    timeout: float,
) -> str | None:
    """Run one ffprobe query; return None (never raise) when it cannot answer."""
    command = ["ffprobe", "-v", "error", *arguments, "-of", output_format, str(source)]
    try:
        result = run_command(command, timeout=timeout, output_limit=DEFAULT_OUTPUT_LIMIT)
    except FileNotFoundError:
        log.warning("ffprobe is unavailable; cannot inspect %s of %s", purpose, source)
        return None
    except ExternalCommandTimeout:
        log.warning("ffprobe timed out inspecting %s of %s", purpose, source)
        return None
    log.info(
        "ffprobe %s: command=%s status=%s stderr=%s",
        purpose,
        command,
        result.returncode,
        result.stderr_text,
    )
    if result.returncode != 0:
        log.warning("ffprobe could not inspect %s for %s", purpose, source)
        return None
    return result.stdout_text


def probe_video_encode_profile(
    source: Path,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> VideoEncodeProfile:
    """Read the source properties an encode or remux should preserve."""
    raw_profile = _run_ffprobe(
        source,
        (
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,bits_per_raw_sample,color_range,color_space,color_transfer,"
            "color_primaries,time_base,start_pts,r_frame_rate",
        ),
        purpose="video profile",
        output_format="json",
        timeout=timeout,
    )
    if raw_profile is None:
        return _labeled_default("ffprobe unavailable or failed")
    try:
        payload = json.loads(raw_profile)
        streams = payload.get("streams", [])
        stream = streams[0]
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
        log.warning("ffprobe returned no usable video profile for %s; using defaults", source)
        return _labeled_default("ffprobe returned no usable video stream")

    raw_pixel_format = stream.get("pix_fmt")
    pixel_format = _PIXEL_FORMAT_ALIASES.get(raw_pixel_format, raw_pixel_format)
    if pixel_format not in _PIXEL_FORMATS:
        pixel_format = "yuv420p"
    time_base = _ratio(stream.get("time_base"))
    raw_start_pts = stream.get("start_pts")
    start_pts = raw_start_pts if isinstance(raw_start_pts, int) and time_base is not None else None
    return VideoEncodeProfile(
        pixel_format=pixel_format,
        frame_rate=_ratio(stream.get("r_frame_rate")),
        color_range=_known_value(stream.get("color_range"), _COLOR_RANGES),
        color_space=_known_value(stream.get("color_space"), _COLOR_SPACES),
        color_transfer=_known_value(stream.get("color_transfer"), _COLOR_TRANSFERS),
        color_primaries=_known_value(stream.get("color_primaries"), _COLOR_PRIMARIES),
        time_base=time_base,
        start_pts=start_pts,
        source_pixel_format=raw_pixel_format if isinstance(raw_pixel_format, str) else None,
        component_depth=_pixel_component_depth(raw_pixel_format, stream.get("bits_per_raw_sample")),
    )


def _labeled_default(reason: str) -> VideoEncodeProfile:
    """The defaults a probe falls back to, with the reason attached."""
    return VideoEncodeProfile(default_reason=reason)
