# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/video_visible.py,
# Apache-2.0; modified: the provider table is collapsed to the eight frozen
# policies this tree can justify, provenance relaxation dropped because this
# tree's AIGC readers carry no generator claim to relax it with, the mask and
# fill primitives are this tree's ``region_eraser`` (the numpy-level surface the
# image backends already expose), the raw-frame encoder lives here rather than
# in the encode adapter (ticket 03 shipped the probe and remux only), every
# subprocess is an argument vector, the frame stream is a pipe rather than a
# whole-file read, and a constant-frame-rate source is required rather than
# approximated.
"""Visible AI-watermark removal for video.

The pass has three states, and no fourth: off (no plan section), a detection
run, or an operator-named region. A detection run decodes once to collect one
untrusted candidate per frame, accepts a candidate only when it recurs at the
same place across adjacent frames, and fills the accepted region through the
same fill backend the image path uses. A candidate seen in a single frame is
never a fill: the temporal arbiter is what turns a score into a mask, and an
isolated lookalike fails it. Every mark this build serves -- sora, veo,
seedance, doubao, dola, hailuo, vidu, kling -- runs that one arbiter, gate and
fill; only its detector and its policy row differ.

Detection-driven removal is **best-effort** and says so in the report. An
operator-named mask or box is **verifiable**: the region that changed is
exactly the region that was named, and no detection stands between the request
and the pixels.

The frame stream never holds the whole video: OpenCV decodes one frame at a
time and the cleaned frames cross a pipe into one ffmpeg encode. Audio is
stream-copied from the source by a second ffmpeg pass, and the destination is
staged so a failure leaves it untouched.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
import region_eraser
from external_command import ExternalCommandTimeout, run_command
from image_io import imread
from morphomod import read_pgm
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from video_encoding import (
    DEFAULT_OUTPUT_LIMIT,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_VIDEO_TIMEOUT,
    VideoEncodeProfile,
    atomic_video_output,
    probe_video_encode_profile,
    remux_video,
)
from video_temporal import stabilize_filled_frame

log = logging.getLogger(__name__)

#: A localization region in source-frame pixels: ``(x, y, width, height)``.
Region = tuple[int, int, int, int]

#: The synthetic sora silhouette is 180x64 at its native size. It is built from
#: geometric primitives and Pillow's bundled font; no source frame, and no
#: provider asset, contributes a pixel.
_SORA_TEMPLATE_SIZE = (180, 64)
_SORA_RELATIVE_HEIGHTS = (0.065, 0.075, 0.085, 0.095, 0.105)

#: Veo's current four-point diamond is matched at its measured corner layout;
#: the legacy ``Veo`` text arm is a bounded corner search. Both silhouettes are
#: primitive/Pillow renders.
_VEO_STRICT_WEAK_CONFIDENCE = 0.50
_VEO_STRONG_CONFIDENCE = 0.55
_VEO_REFERENCE_SHORT_SIDE = 720
_VEO_DIAMOND_PROFILES = ((56, 92, 92), (48, 72, 72), (44, 29, 40))
_MIN_VEO_STABLE_FRAMES = 12

#: Seedance's boxed ``AI`` label is a fixed-position mark; its floors and
#: stability length are the reference's measured ones.
_SEEDANCE_WEAK_CONFIDENCE = 0.38
_SEEDANCE_STRONG_CONFIDENCE = 0.43
_MIN_FIXED_MARK_STABLE_FRAMES = 12

#: Doubao's bottom-right "豆包AI生成" run is the same synthetic outline the image
#: engine matches; measured floors from the reference (0.30 already accepted a
#: negative, so the weak bar stays at 0.35).
_DOUBAO_WEAK_CONFIDENCE = 0.35
_DOUBAO_STRONG_CONFIDENCE = 0.55

#: Dola's compact text label, matched from the frozen Hershey duplex render at
#: a flat sweep of relative heights.
_DOLA_STRICT_WEAK_CONFIDENCE = 0.50
_DOLA_STRONG_CONFIDENCE = 0.52
_DOLA_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(22, 41))

#: Hailuo's composite MINIMAX/hailuo AI label is white; the reference measured
#: 0.02 of bright low-saturation pixels separating the label from a wood-grain
#: lookalike that outscored it.
_HAILUO_WEAK_CONFIDENCE = 0.30
_HAILUO_STRONG_CONFIDENCE = 0.34
_HAILUO_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(28, 56, 3))
_HAILUO_MIN_WHITE_FRACTION = 0.02

#: Vidu (ShengShu) "Vidu AI" logo plus wordmark, bottom-right. Measured floors
#: from the reference separate the one real export (0.72+) from a 0.41 corpus max.
_VIDU_WEAK_CONFIDENCE = 0.52
_VIDU_STRONG_CONFIDENCE = 0.58
_VIDU_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(34, 61, 3))

#: Kling's wordmark core. Its per-frame score does not separate texture from the
#: label (an OpenArt export with no Kling mark scored 0.81 against 0.63 for a
#: real overlay), so the run must stay anchored far tighter than the shared 0.80
#: and the label must reach the bottom-right frame edge.
_KLING_WEAK_CONFIDENCE = 0.20
_KLING_STRONG_CONFIDENCE = 0.24
_KLING_MIN_WHITE_FRACTION = 0.02
_KLING_RELATIVE_HEIGHTS = tuple(value / 1000 for value in range(24, 49, 3))

#: Every detector runs on a frame whose short side is at most this many pixels,
#: so one measurement holds at every input resolution.
_NORMALIZED_SHORT_SIDE = 480

#: The temporal arbiter's sora numbers, measured upstream and kept as data.
#: ``strong_floor`` is the only entry bar this tree can honour: the AIGC
#: readers carry no generator claim, so no metadata run relaxes it.
_SORA_WEAK_CONFIDENCE = 0.60
_SORA_STRONG_CONFIDENCE = 0.65
_SORA_TRANSITION_CONFIDENCE = 0.45
_SORA_MIN_STABLE_FRAMES = 5
_SORA_PADDING_FRACTION = 0.28
_STABLE_IOU = 0.55
_MAX_STABLE_GAP = 2

_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})

#: Renderer-synthetic silhouettes ship beside this module. No captured pixel
#: enters them: each is regenerated by an in-tree renderer (see
#: ``assets/PROVENANCE.md`` and ``assets/video/PROVENANCE.md``).
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

#: One libx264 encode is the visible pass's quality floor: high enough that a
#: filled region is not itself an artifact, low enough to stay an ordinary
#: re-encode. It is not exposed on the plan because nothing measured a reason
#: to move it yet.
_VISIBLE_CRF = 14

#: The visible pass re-encodes at one constant frame rate. ``constant-rate`` is
#: a measurement of a uniform timeline; a timeline that could not be read says
#: so, and a non-uniform one is refused rather than resampled.
VISIBLE_TIMING_CONSTANT = "constant-rate"
VISIBLE_TIMING_UNMEASURED = "unmeasured-timeline"
VISIBLE_TIMING_NOT_RUN = "not-run"

_FFMPEG_STDERR_LIMIT = 8_000
_FFMPEG_STDERR_TRUNCATION = b"\n... [truncated]\n"

#: The video marks this build can localize, in the order the reference lists
#: them. Sora ships first because it is the mark the engine was proved end to
#: end on; the other seven run the same pass with their own detector and policy.
VISIBLE_VIDEO_MARKS = ("sora", "veo", "seedance", "doubao", "dola", "hailuo", "vidu", "kling")
#: ``--mark auto`` on a video: scan every shipped mark and fill the one the
#: temporal arbiter accepts on the most frames.
AUTO_VIDEO_MARK = "auto"

#: Fill backends, in the vocabulary the image path uses. ``cv2`` is the
#: classical floor (always available); ``migan`` and ``lama`` are the same
#: learned backends ``region_eraser.erase`` resolves for images.
VISIBLE_FILL_BACKENDS = ("cv2", "migan", "lama")


@dataclass(frozen=True, slots=True)
class VisibleVideoPlan:
    """The explicit opt-in for a visible pass on one video.

    Constructing this at all is the opt-in: a ``VideoPlan`` without it does no
    pixel work and never even decodes a frame. Exactly one localization source
    is configured -- a detection-driven ``mark``, or a region the operator
    named as a ``mask_path`` or a ``box``. The named-region sources are
    reported as verifiable; the detection-driven source is reported as
    best-effort, and its fill additionally requires temporal evidence.
    """

    mark: str | None = None
    mask_path: Path | None = None
    box: tuple[int, int, int, int] | None = None
    backend: str = "cv2"
    #: Motion-align a safely covered prior fill after the per-frame fill. It
    #: only ever writes inside the current removal mask, so it cannot widen the
    #: named region of an explicit mask or box.
    temporal_consistency: bool = True

    def __post_init__(self) -> None:
        sources = sum(source is not None for source in (self.mark, self.mask_path, self.box))
        if sources != 1:
            raise ValueError("exactly one visible video localization source is required")
        if self.mark is not None and self.mark not in (*VISIBLE_VIDEO_MARKS, AUTO_VIDEO_MARK):
            raise ValueError(
                f"unknown visible video mark {self.mark!r}; this build ships "
                f"{', '.join(VISIBLE_VIDEO_MARKS)}"
            )
        if self.box is not None:
            if len(self.box) != 4 or any(
                isinstance(value, bool) or not isinstance(value, int) for value in self.box
            ):
                raise TypeError("visible video box must be four integers: x,y,w,h")
            x, y, width, height = self.box
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(
                    "visible video box must have a non-negative origin and positive size"
                )
        if self.backend not in VISIBLE_FILL_BACKENDS:
            raise ValueError(
                f"unknown visible video backend {self.backend!r}; expected one of "
                f"{', '.join(VISIBLE_FILL_BACKENDS)}"
            )
        if not isinstance(self.temporal_consistency, bool):
            raise TypeError("temporal_consistency must be a bool")


@dataclass(frozen=True, slots=True)
class FrameLocalization:
    """The best untrusted sora candidate found in one decoded frame."""

    frame_index: int
    confidence: float
    region: Region | None


@dataclass(frozen=True, slots=True)
class VideoScan:
    """Decoded video geometry plus one localization candidate per frame."""

    width: int
    height: int
    fps: float
    detections: tuple[FrameLocalization, ...]


@dataclass(frozen=True, slots=True)
class VideoVisiblePolicy:
    """One provider's temporal-arbiter tuning plus the fill geometry it needs.

    ``cover_after_confirmation`` stays False for sora: the mark moves, so a
    confirmed position is never carried onto a frame that showed nothing.
    """

    weak_floor: float
    strong_floor: float
    transition_floor: float
    min_stable_frames: int
    cover_after_confirmation: bool
    padding_fraction: float
    #: Optional tighter anchor: every candidate in a run must stay within this IoU
    #: of the run's first candidate, not only of its neighbour. Kling sets it, on
    #: a measured reason -- its per-frame score does not separate texture from the
    #: label, but the overlay's box is identical on every frame while texture
    #: matches drift a few pixels.
    anchor_iou: float | None = None


SORA_POLICY = VideoVisiblePolicy(
    weak_floor=_SORA_WEAK_CONFIDENCE,
    strong_floor=_SORA_STRONG_CONFIDENCE,
    transition_floor=_SORA_TRANSITION_CONFIDENCE,
    min_stable_frames=_SORA_MIN_STABLE_FRAMES,
    cover_after_confirmation=False,
    padding_fraction=_SORA_PADDING_FRACTION,
)

VEO_POLICY = VideoVisiblePolicy(
    weak_floor=_VEO_STRICT_WEAK_CONFIDENCE,
    strong_floor=_VEO_STRONG_CONFIDENCE,
    transition_floor=0.35,
    min_stable_frames=_MIN_VEO_STABLE_FRAMES,
    cover_after_confirmation=True,
    padding_fraction=0.18,
)

SEEDANCE_POLICY = VideoVisiblePolicy(
    weak_floor=_SEEDANCE_WEAK_CONFIDENCE,
    strong_floor=_SEEDANCE_STRONG_CONFIDENCE,
    transition_floor=0.30,
    min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
    cover_after_confirmation=True,
    padding_fraction=0.0,
    anchor_iou=0.80,
)

DOUBAO_POLICY = VideoVisiblePolicy(
    weak_floor=_DOUBAO_WEAK_CONFIDENCE,
    strong_floor=_DOUBAO_STRONG_CONFIDENCE,
    transition_floor=0.30,
    min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
    cover_after_confirmation=True,
    padding_fraction=0.0,
    anchor_iou=0.80,
)

DOLA_POLICY = VideoVisiblePolicy(
    weak_floor=_DOLA_STRICT_WEAK_CONFIDENCE,
    strong_floor=_DOLA_STRONG_CONFIDENCE,
    transition_floor=0.40,
    min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
    cover_after_confirmation=True,
    padding_fraction=0.20,
    anchor_iou=0.80,
)

HAILUO_POLICY = VideoVisiblePolicy(
    weak_floor=_HAILUO_WEAK_CONFIDENCE,
    strong_floor=_HAILUO_STRONG_CONFIDENCE,
    transition_floor=0.28,
    min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
    cover_after_confirmation=True,
    padding_fraction=0.12,
    anchor_iou=0.80,
)

VIDU_POLICY = VideoVisiblePolicy(
    weak_floor=_VIDU_WEAK_CONFIDENCE,
    strong_floor=_VIDU_STRONG_CONFIDENCE,
    transition_floor=0.45,
    min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
    cover_after_confirmation=True,
    padding_fraction=0.12,
    anchor_iou=0.80,
)

KLING_POLICY = VideoVisiblePolicy(
    weak_floor=_KLING_WEAK_CONFIDENCE,
    strong_floor=_KLING_STRONG_CONFIDENCE,
    transition_floor=0.30,
    min_stable_frames=_MIN_FIXED_MARK_STABLE_FRAMES,
    cover_after_confirmation=True,
    anchor_iou=0.95,
    padding_fraction=0.12,
)

#: One row per served mark. The arbiter, gate and fill are shared; a row is the
#: only per-mark tuning. Provenance relaxation is dropped from every row: this
#: tree's AIGC readers carry no generator claim that could relax a floor, so
#: ``strong_floor`` stays a hard requirement of every run.
VISIBLE_MARK_POLICIES: dict[str, VideoVisiblePolicy] = {
    "sora": SORA_POLICY,
    "veo": VEO_POLICY,
    "seedance": SEEDANCE_POLICY,
    "doubao": DOUBAO_POLICY,
    "dola": DOLA_POLICY,
    "hailuo": HAILUO_POLICY,
    "vidu": VIDU_POLICY,
    "kling": KLING_POLICY,
}


@dataclass(frozen=True, slots=True)
class VisibleRemoval:
    """What one visible pass did, and how much its word is worth.

    ``verification`` is the load-bearing field: ``"best-effort"`` means a
    probabilistic detector authorized the fill, ``"verifiable"`` means the
    operator named the region and the pixels that changed are the ones named.
    """

    source: str
    mark: str | None
    status: str
    verification: str
    backend: str
    total_frames: int
    detected_frames: int
    removed_frames: int
    timing: str
    note: str

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "mark": self.mark,
            "status": self.status,
            "verification": self.verification,
            "backend": self.backend,
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "removed_frames": self.removed_frames,
            "timing": self.timing,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class _PreparedFrame:
    """The grayscale representations every detected frame shares."""

    gray: NDArray[Any]
    normalized_gray: NDArray[Any]
    normalized_scale: float


# --- the synthetic sora silhouette -----------------------------------------


def _scalable_default_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load Pillow's bundled scalable font, with a Pillow 10.0 fallback."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # ``size=`` was added after Pillow 10.0, which is still inside this
        # package's supported dependency range.
        return ImageFont.load_default()


@lru_cache(maxsize=1)
def _sora_templates() -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the synthetic full-wordmark and mascot-only silhouettes.

    The cloud-like mascot is assembled from primitive shapes and the word is
    rendered with Pillow's bundled font, so the template is reproducible from
    this source alone and carries no provider pixels.
    """
    width, height = _SORA_TEMPLATE_SIZE
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((4, 8, 58, 57), radius=22, fill=255)
    draw.ellipse((0, 18, 26, 50), fill=255)
    draw.ellipse((38, 16, 64, 51), fill=255)
    draw.ellipse((16, 18, 29, 43), fill=0)
    draw.ellipse((35, 18, 48, 43), fill=0)

    font = _scalable_default_font(50)
    if isinstance(font, ImageFont.FreeTypeFont):
        draw.text((67, 0), "Sora", font=font, fill=255, stroke_width=1, stroke_fill=255)
    else:
        text_box = font.getbbox("Sora")
        text = Image.new("L", (max(1, text_box[2]), max(1, text_box[3])), 0)
        ImageDraw.Draw(text).text((0, 0), "Sora", font=font, fill=255)
        text = text.resize((102, 50), Image.Resampling.NEAREST)
        canvas.paste(text, (67, 0), text)

    full = np.asarray(canvas, dtype=np.uint8)
    return full, full[:, :64]


@lru_cache(maxsize=1)
def _template_sources() -> dict[str, NDArray[Any]]:
    """Return every immutable synthetic template by detector-local key."""
    sora_word, sora_icon = _sora_templates()
    veo_diamond, veo_text = _veo_templates()
    sources = {
        "sora-word": sora_word,
        "sora-icon": sora_icon,
        "veo-diamond": veo_diamond,
        "veo-text": veo_text,
        "seedance": _seedance_template(),
        "doubao": _asset_template("doubao_outline_alpha.png"),
        "dola": _asset_template("video/video_dola_hershey_duplex.png"),
        "hailuo": _hailuo_template(),
        "vidu": _vidu_template(),
        "kling-logo": _kling_logo_template(),
    }
    sources.update(
        {f"kling-{index}": template for index, template in enumerate(_kling_templates())}
    )
    return sources


@cache
def _asset_template(relative_path: str) -> NDArray[Any]:
    """Load a renderer-synthetic silhouette from ``assets/``.

    The image engines' alpha maps are the same synthetic silhouettes the video
    detectors match; a video mark that shares the glyph reuses the asset rather
    than re-rendering it. No source frame contributes pixels.
    """
    template = imread(str(_ASSETS_DIR / relative_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError(f"missing or undecodable video template asset: {relative_path}")
    return template


@lru_cache(maxsize=512)
def _resized_template_feature(
    template_key: str, width: int, height: int, kernel_size: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Resize one template and cache its invariant top-hat representation."""
    template = cv2.resize(
        _template_sources()[template_key],
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return template, _top_hat(template, kernel_size=kernel_size)


def _top_hat(gray: NDArray[Any], *, kernel_size: int = 7) -> NDArray[Any]:
    """Suppress the low-frequency background a translucent overlay sits on."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def _normalized_gray(image_bgr: NDArray[Any]) -> tuple[NDArray[Any], float]:
    """Return the detection-resolution gray frame and the scale back to source."""
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    short_side = min(height, width)
    if short_side <= 0:
        return gray, 1.0
    scale = min(1.0, _NORMALIZED_SHORT_SIDE / short_side)
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def _prepare_frame(image_bgr: NDArray[Any]) -> _PreparedFrame:
    """Compute the shared grayscale representations for one decoded frame."""
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    normalized_gray, normalized_scale = _normalized_gray(gray)
    return _PreparedFrame(gray, normalized_gray, normalized_scale)


def _expanded_region(
    location: tuple[int, int],
    template_width: int,
    template_height: int,
    *,
    icon_only: bool,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Region:
    """Map a normalized-resolution match back to a source-frame region."""
    x = round(location[0] / scale)
    y = round(location[1] / scale)
    height = max(1, round(template_height / scale))
    width = max(1, round(template_width / scale))
    if icon_only:
        width = round(height * _SORA_TEMPLATE_SIZE[0] / _SORA_TEMPLATE_SIZE[1])
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    return x, y, max(1, width), max(1, height)


def detect_sora_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest sora-wordmark match in one frame.

    The candidate is intentionally untrusted, and so is its score: a mask is
    built only for a run the temporal arbiter confirms across frames.
    """
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    prepared = prepared or _prepare_frame(image_bgr)
    gray, scale = prepared.normalized_gray, prepared.normalized_scale
    normalized_height, normalized_width = gray.shape[:2]
    feature = _top_hat(gray)
    best_confidence = 0.0
    best_region: Region | None = None

    for icon_only, template_key in ((False, "sora-word"), (True, "sora-icon")):
        for relative_height in _SORA_RELATIVE_HEIGHTS:
            base_width, base_height = _SORA_TEMPLATE_SIZE
            if icon_only:
                base_width = 64
            template_height = max(
                16, round(min(normalized_height, normalized_width) * relative_height)
            )
            template_width = max(1, round(base_width * template_height / base_height))
            if template_height >= normalized_height or template_width >= normalized_width:
                continue
            template, template_feature = _resized_template_feature(
                template_key, template_width, template_height, 7
            )
            scores = cv2.matchTemplate(feature, template_feature, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(scores)
            if confidence <= best_confidence:
                continue
            if icon_only:
                # Edge-only correlation also accepts diamonds and scene texture.
                # Confirm the full mascot silhouette, eyes included.
                x, y = location
                candidate = gray[y : y + template_height, x : x + template_width]
                shape_confidence = float(
                    cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)[0, 0]
                )
                if shape_confidence < 0.50:
                    continue
            best_confidence = float(confidence)
            best_region = _expanded_region(
                location,
                template_width,
                template_height,
                icon_only=icon_only,
                scale=scale,
                frame_width=frame_width,
                frame_height=frame_height,
            )

    return FrameLocalization(frame_index, best_confidence, best_region)


# --- the other synthetic silhouettes ---------------------------------------


def _crop_nonzero(image: NDArray[Any]) -> NDArray[Any]:
    """Crop a synthetic template to its nonzero footprint."""
    ys, xs = np.where(image > 0)
    if len(xs) == 0:
        return image
    return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


@lru_cache(maxsize=1)
def _veo_templates() -> tuple[NDArray[Any], NDArray[Any]]:
    """Return synthetic current-diamond and legacy-text Veo silhouettes."""
    size = 256
    diamond_canvas = Image.new("L", (size, size), 0)
    diamond_points = (
        (0.50, 0.02),
        (0.60, 0.39),
        (0.98, 0.50),
        (0.60, 0.61),
        (0.50, 0.98),
        (0.40, 0.61),
        (0.02, 0.50),
        (0.40, 0.39),
    )
    ImageDraw.Draw(diamond_canvas).polygon(
        [(round(x * size), round(y * size)) for x, y in diamond_points],
        fill=255,
    )

    text_canvas = Image.new("L", (140, 60), 0)
    text_draw = ImageDraw.Draw(text_canvas)
    text_draw.text((2, 0), "Veo", font=_scalable_default_font(48), fill=255)
    text = _crop_nonzero(np.asarray(text_canvas, dtype=np.uint8))
    return np.asarray(diamond_canvas, dtype=np.uint8), text


@lru_cache(maxsize=1)
def _seedance_template() -> NDArray[Any]:
    """Return a synthetic boxed-AI silhouette for Seedance exports."""
    canvas = Image.new("L", (160, 120), 0)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((8, 8, 142, 105), radius=28, outline=255, width=7)
    draw.text(
        (35, 17),
        "AI",
        font=_scalable_default_font(70),
        fill=255,
        stroke_width=1,
        stroke_fill=255,
    )
    draw.rounded_rectangle((142, 94, 157, 109), radius=3, outline=255, width=2)
    draw.text((145, 94), "AI", font=_scalable_default_font(8), fill=255)
    return np.asarray(canvas, dtype=np.uint8)


@lru_cache(maxsize=1)
def _hailuo_template() -> NDArray[Any]:
    """Return a synthetic MINIMAX/Hailuo composite-label silhouette."""
    canvas = Image.new("L", (680, 112), 0)
    draw = ImageDraw.Draw(canvas)

    # Five symmetric waveform strokes approximate the provider-independent
    # geometry at the left edge without copying pixels from an export.
    waveform_heights = (42, 70, 96, 70, 42)
    center_y = 56
    for index, height in enumerate(waveform_heights):
        x = 10 + index * 13
        draw.rounded_rectangle(
            (x, center_y - height // 2, x + 5, center_y + height // 2),
            radius=2,
            fill=255,
        )

    font = _scalable_default_font(58)
    draw.text((84, 18), "MINIMAX", font=font, fill=255, stroke_width=1, stroke_fill=255)
    draw.rectangle((342, 18, 347, 92), fill=255)

    # The Hailuo symbol is a ring with a small offset highlight.
    draw.ellipse((370, 20, 446, 96), outline=255, width=12)
    draw.ellipse((397, 38, 434, 76), fill=255)
    draw.ellipse((397, 29, 420, 52), fill=0)
    draw.text((456, 18), "hailuo AI", font=font, fill=255, stroke_width=1, stroke_fill=255)
    return np.asarray(canvas, dtype=np.uint8)


def _vidu_capsule(height: int, width: int, angle: float, stroke: int) -> NDArray[Any]:
    canvas = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (100 - width // 2, 100 - height // 2, 100 + width // 2, 100 + height // 2),
        radius=width // 2,
        outline=255,
        width=stroke,
    )
    rotation = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
    return _crop_nonzero(cv2.warpAffine(np.asarray(canvas, dtype=np.uint8), rotation, (200, 200)))


@lru_cache(maxsize=1)
def _vidu_template() -> NDArray[Any]:
    """Return a synthetic Vidu logo plus "Vidu AI" wordmark silhouette.

    The logo is two outlined capsules leaning into a V; the text is
    font-rendered. Proportions follow the measured 1080p mark (232 x 50 px:
    logo 65 x 50, text 155 x 35 centered 12 px after it). No pixels come from
    an export.
    """
    canvas = np.zeros((100, 464), dtype=np.uint8)
    logo = np.zeros((100, 130), dtype=np.uint8)
    left = _vidu_capsule(96, 40, 24, 11)
    left = cv2.resize(left, (left.shape[1] * 100 // left.shape[0], 100))
    logo[:, : left.shape[1]] = np.maximum(logo[:, : left.shape[1]], left)
    right = _vidu_capsule(96, 40, -24, 11)
    right = cv2.resize(right, (right.shape[1] * 100 // right.shape[0], 100))
    offset = 130 - right.shape[1]
    logo[:, offset:] = np.maximum(logo[:, offset:], right)
    canvas[:, :130] = logo
    text = Image.new("L", (600, 140), 0)
    ImageDraw.Draw(text).text(
        (4, 10),
        "Vidu AI",
        font=_scalable_default_font(96),
        fill=255,
        stroke_width=1,
        stroke_fill=255,
    )
    word = cv2.resize(
        _crop_nonzero(np.asarray(text, dtype=np.uint8)),
        (310, 70),
        interpolation=cv2.INTER_AREA,
    )
    canvas[15:85, 154:464] = np.maximum(canvas[15:85, 154:464], word)
    return canvas


@lru_cache(maxsize=1)
def _kling_templates() -> tuple[NDArray[Any], ...]:
    """Return synthetic font and capitalization variants for the Kling wordmark core."""
    templates: list[NDArray[Any]] = []
    for text, asset_stem in (("KLING AI", "kling_ai"), ("KlingAI", "klingai")):
        canvas = Image.new("L", (430, 104), 0)
        ImageDraw.Draw(canvas).text(
            (2, 12),
            text,
            font=_scalable_default_font(64),
            fill=255,
            stroke_width=1,
            stroke_fill=255,
        )
        templates.append(_crop_nonzero(np.asarray(canvas, dtype=np.uint8)))
        # Frozen OpenCV 4 Hershey renders; see assets/video/PROVENANCE.md for why.
        templates.extend(
            _asset_template(f"video/video_{asset_stem}_hershey_{font}.png")
            for font in ("simplex", "duplex")
        )
    return tuple(templates)


@lru_cache(maxsize=1)
def _kling_logo_template() -> NDArray[Any]:
    """Return a synthetic ring approximation of the Kling swirl."""
    template = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(template, (50, 50), 34, 255, 14, cv2.LINE_AA)
    return _crop_nonzero(template)


# --- fixed-mark detection primitives ---------------------------------------


def _match_template(
    gray: NDArray[Any],
    template: NDArray[Any],
    *,
    region: Region,
    kernel_size: int,
    template_feature: NDArray[Any] | None = None,
) -> tuple[float, Region | None]:
    """Match one synthetic silhouette inside a bounded frame region."""
    x, y, width, height = region
    roi = gray[y : y + height, x : x + width]
    template_height, template_width = template.shape[:2]
    if roi.size == 0 or template_height >= roi.shape[0] or template_width >= roi.shape[1]:
        return 0.0, None
    feature = _top_hat(roi, kernel_size=kernel_size)
    if template_feature is None:
        template_feature = _top_hat(template, kernel_size=kernel_size)
    scores = cv2.matchTemplate(feature, template_feature, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(scores)
    return float(confidence), (
        x + location[0],
        y + location[1],
        template_width,
        template_height,
    )


def _restore_region(
    region: Region | None,
    *,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Region | None:
    """Map a localization from normalized pixels back to the source frame."""
    if region is None:
        return None
    x, y, width, height = region
    source_x = round(x / scale)
    source_y = round(y / scale)
    source_width = min(frame_width - source_x, max(1, round(width / scale)))
    source_height = min(frame_height - source_y, max(1, round(height / scale)))
    return source_x, source_y, source_width, source_height


def _bounded_region(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    frame_width: int,
    frame_height: int,
) -> Region:
    """Clip an expanded region to the frame without changing its anchor."""
    bounded_x = max(0, x)
    bounded_y = max(0, y)
    return (
        bounded_x,
        bounded_y,
        min(frame_width - bounded_x, width + x - bounded_x),
        min(frame_height - bounded_y, height + y - bounded_y),
    )


def _detect_fixed_mark(
    image_bgr: NDArray[Any],
    template_key: str,
    *,
    relative_heights: tuple[float, ...],
    search_origin: tuple[float, float],
    kernel_fraction: float,
    prefer_larger_within: float = 0.0,
    normalized: tuple[NDArray[Any], float] | None = None,
    frame_index: int,
) -> FrameLocalization:
    """Match one fixed synthetic mark inside a normalized-frame search region."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray, scale = normalized if normalized is not None else _normalized_gray(image_bgr)
    normalized_height, normalized_width = gray.shape[:2]
    short_side = min(normalized_height, normalized_width)
    search_x = round(normalized_width * search_origin[0])
    search_y = round(normalized_height * search_origin[1])
    search_region = (
        search_x,
        search_y,
        normalized_width - search_x,
        normalized_height - search_y,
    )
    matches: list[tuple[float, Region]] = []
    base_template = _template_sources()[template_key]
    for relative_height in relative_heights:
        template_height = max(6, round(short_side * relative_height))
        template_width = max(
            1, round(base_template.shape[1] * template_height / base_template.shape[0])
        )
        kernel_size = max(3, round(template_height * kernel_fraction) | 1)
        resized, template_feature = _resized_template_feature(
            template_key,
            template_width,
            template_height,
            kernel_size,
        )
        confidence, candidate = _match_template(
            gray,
            resized,
            region=search_region,
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if candidate is not None and confidence > 0:
            matches.append((confidence, candidate))
    if not matches:
        return FrameLocalization(frame_index, 0.0, None)
    best_confidence, best_region = max(matches, key=lambda match: match[0])
    if prefer_larger_within > 0:
        eligible = [
            (confidence, candidate)
            for confidence, candidate in matches
            if confidence >= best_confidence - prefer_larger_within
        ]
        best_confidence, best_region = max(
            eligible,
            key=lambda match: (match[1][2] * match[1][3], match[0]),
        )

    return FrameLocalization(
        frame_index,
        best_confidence,
        _restore_region(
            best_region,
            scale=scale,
            frame_width=frame_width,
            frame_height=frame_height,
        ),
    )


def detect_seedance_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Seedance boxed-AI candidate."""
    return _detect_fixed_mark(
        image_bgr,
        "seedance",
        relative_heights=(0.065, 0.075, 0.085, 0.095, 0.105),
        search_origin=(0.68, 0.72),
        kernel_fraction=0.12,
        normalized=None
        if prepared is None
        else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def detect_doubao_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Doubao "豆包AI生成" text candidate.

    The ByteDance Doubao mark registered for images also appears burned into the
    bottom-right of Doubao video exports; this detector reuses the image
    engine's renderer-synthetic outline alpha as the template.
    """
    return _detect_fixed_mark(
        image_bgr,
        "doubao",
        relative_heights=tuple(value / 1000 for value in range(32, 61, 4)),
        search_origin=(0.60, 0.76),
        kernel_fraction=0.14,
        prefer_larger_within=0.04,
        normalized=None
        if prepared is None
        else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def detect_dola_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Dola AI text candidate."""
    return _detect_fixed_mark(
        image_bgr,
        "dola",
        relative_heights=_DOLA_RELATIVE_HEIGHTS,
        search_origin=(0.65, 0.85),
        kernel_fraction=0.50,
        normalized=None
        if prepared is None
        else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def _white_fraction(roi: NDArray[Any]) -> float:
    """Share of bright, low-saturation pixels, the color of the white video labels."""
    if roi.size == 0:
        return 0.0
    if roi.ndim == 2:
        return float(np.mean(roi >= 180))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 180)))


def detect_hailuo_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed MINIMAX/Hailuo composite-label candidate."""
    detection = _detect_fixed_mark(
        image_bgr,
        "hailuo",
        relative_heights=_HAILUO_RELATIVE_HEIGHTS,
        search_origin=(0.28, 0.76),
        kernel_fraction=0.18,
        normalized=None
        if prepared is None
        else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )
    if detection.region is None:
        return detection
    frame_width = image_bgr.shape[1]
    x, y, width, height = detection.region
    if _white_fraction(image_bgr[y : y + height, x : x + width]) < _HAILUO_MIN_WHITE_FRACTION:
        return FrameLocalization(frame_index, 0.0, None)
    horizontal_padding = round(height * 1.25)
    region = _bounded_region(
        x - horizontal_padding,
        y,
        width + horizontal_padding * 2,
        height,
        frame_width=frame_width,
        frame_height=image_bgr.shape[0],
    )
    return FrameLocalization(frame_index, detection.confidence, region)


def detect_vidu_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest fixed Vidu logo-plus-wordmark candidate."""
    return _detect_fixed_mark(
        image_bgr,
        "vidu",
        relative_heights=_VIDU_RELATIVE_HEIGHTS,
        search_origin=(0.70, 0.84),
        kernel_fraction=0.18,
        normalized=None
        if prepared is None
        else (prepared.normalized_gray, prepared.normalized_scale),
        frame_index=frame_index,
    )


def _reaches_kling_edge(region: Region, frame_width: int, frame_height: int) -> bool:
    """Whether a Kling candidate box reaches the bottom-right frame edge."""
    x, y, width, height = region
    return x + width >= frame_width * 0.96 and y + height >= frame_height * 0.94


def detect_kling_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the fixed Kling wordmark core and include its version suffix."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)
    frame_height, frame_width = image_bgr.shape[:2]
    prepared = prepared or _prepare_frame(image_bgr)
    normalized = prepared.normalized_gray, prepared.normalized_scale
    expanded: list[FrameLocalization] = []
    for template_index, _template in enumerate(_kling_templates()):
        detection = _detect_fixed_mark(
            image_bgr,
            f"kling-{template_index}",
            relative_heights=_KLING_RELATIVE_HEIGHTS,
            search_origin=(0.64, 0.84),
            kernel_fraction=0.18,
            prefer_larger_within=0.06,
            normalized=normalized,
            frame_index=frame_index,
        )
        if detection.region is None:
            continue
        x, y, width, height = detection.region
        left_padding = round(height * 1.8)
        right_padding = round(height * 4.0)
        vertical_padding = round(height * 0.4)
        region = _bounded_region(
            x - left_padding,
            y - vertical_padding,
            width + left_padding + right_padding,
            height + vertical_padding * 2,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        expanded.append(FrameLocalization(frame_index, detection.confidence, region))
    if not expanded:
        return FrameLocalization(frame_index, 0.0, None)
    edge_candidates = [
        candidate
        for candidate in expanded
        if candidate.region is not None
        and _reaches_kling_edge(candidate.region, frame_width, frame_height)
    ]
    font_candidate = (
        None
        if not edge_candidates
        else max(edge_candidates, key=lambda candidate: candidate.confidence)
    )

    logo = _detect_fixed_mark(
        image_bgr,
        "kling-logo",
        relative_heights=tuple(value / 1000 for value in range(20, 61, 3)),
        search_origin=(0.62, 0.90),
        kernel_fraction=0.18,
        prefer_larger_within=0.03,
        normalized=normalized,
        frame_index=frame_index,
    )
    logo_candidate: FrameLocalization | None = None
    logo_x: int | None = None
    if logo.region is not None and logo.confidence >= 0.44:
        logo_x, logo_y, _, logo_height = logo.region
        logo_region = _bounded_region(
            logo_x - round(logo_height * 0.2),
            logo_y - round(logo_height * 0.25),
            round(logo_height * 7.8),
            round(logo_height * 1.5),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        # The swirl arm must reach the same frame edge as the text arm: a swirl
        # match on wood grain 11% short of the right edge covered every frame of
        # a Higgsfield Cinema Studio clip (2026-09-25).
        if _reaches_kling_edge(logo_region, frame_width, frame_height):
            logo_candidate = FrameLocalization(frame_index, logo.confidence, logo_region)

    best = font_candidate or logo_candidate
    if (
        font_candidate is not None
        and font_candidate.region is not None
        and logo_candidate is not None
        and logo_x is not None
    ):
        font_x, _, font_width, _ = font_candidate.region
        if font_x - round(font_width * 0.08) <= logo_x <= font_x + round(font_width * 0.50):
            best = logo_candidate
    if best is None or best.region is None:
        return FrameLocalization(frame_index, 0.0, None)
    x, y, width, height = best.region
    if _white_fraction(image_bgr[y : y + height, x : x + width]) < _KLING_MIN_WHITE_FRACTION:
        return FrameLocalization(frame_index, 0.0, None)
    return best


def detect_veo_frame(
    image_bgr: NDArray[Any],
    *,
    frame_index: int = 0,
    prepared: _PreparedFrame | None = None,
) -> FrameLocalization:
    """Locate the strongest current-diamond or legacy-text Veo candidate."""
    if image_bgr.size == 0:
        return FrameLocalization(frame_index, 0.0, None)

    frame_height, frame_width = image_bgr.shape[:2]
    gray = (prepared or _prepare_frame(image_bgr)).gray
    short_scale = min(frame_height, frame_width) / _VEO_REFERENCE_SHORT_SIDE
    _, text_base = _veo_templates()
    best_confidence = 0.0
    best_region: Region | None = None

    diamond_sizes: set[int] = set()
    for base_size, right_margin, bottom_margin in _VEO_DIAMOND_PROFILES:
        diamond_size = max(16, round(base_size * short_scale))
        diamond_sizes.add(diamond_size)
        kernel_size = max(3, round(7 * short_scale) | 1)
        template, template_feature = _resized_template_feature(
            "veo-diamond", diamond_size, diamond_size, kernel_size
        )
        expected_x = round(frame_width - (right_margin + base_size) * short_scale)
        expected_y = round(frame_height - (bottom_margin + base_size) * short_scale)
        search_padding = max(6, round(diamond_size * 0.25))
        search_x = max(0, expected_x - search_padding)
        search_y = max(0, expected_y - search_padding)
        search_width = min(frame_width - search_x, diamond_size + search_padding * 2)
        search_height = min(frame_height - search_y, diamond_size + search_padding * 2)
        confidence, candidate = _match_template(
            gray,
            template,
            region=(search_x, search_y, search_width, search_height),
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    # Provider layouts have moved before. A bounded corner search is a safety
    # net for a relocated diamond, but it is admitted only at a much stronger
    # per-frame score than the known-profile path. Without this gate, recurring
    # bright scene details in clean API exports can become stable false matches.
    corner_x = round(frame_width * 0.65)
    corner_y = round(frame_height * 0.65)
    corner_region = (corner_x, corner_y, frame_width - corner_x, frame_height - corner_y)
    for diamond_size in diamond_sizes:
        kernel_size = max(3, round(7 * short_scale) | 1)
        template, template_feature = _resized_template_feature(
            "veo-diamond", diamond_size, diamond_size, kernel_size
        )
        confidence, candidate = _match_template(
            gray,
            template,
            region=corner_region,
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if confidence >= 0.70 and confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    text_region_width = min(frame_width, max(32, round(180 * short_scale)))
    text_region_height = min(frame_height, max(24, round(120 * short_scale)))
    text_region = (
        frame_width - text_region_width,
        frame_height - text_region_height,
        text_region_width,
        text_region_height,
    )
    text_heights = sorted({max(5, round(height * short_scale)) for height in range(8, 22)})
    for text_height in text_heights:
        text_width = max(1, round(text_base.shape[1] * text_height / text_base.shape[0]))
        kernel_size = max(3, round(3 * short_scale) | 1)
        template, template_feature = _resized_template_feature(
            "veo-text", text_width, text_height, kernel_size
        )
        confidence, candidate = _match_template(
            gray,
            template,
            region=text_region,
            kernel_size=kernel_size,
            template_feature=template_feature,
        )
        if confidence > best_confidence:
            best_confidence = confidence
            best_region = candidate

    return FrameLocalization(frame_index, best_confidence, best_region)


# --- the temporal arbiter ---------------------------------------------------


def _region_intersection(left: Region, right: Region) -> int:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0, min(ly + lh, ry + rh) - max(ly, ry))


def _region_iou(left: Region, right: Region) -> float:
    intersection = _region_intersection(left, right)
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def stabilize_localizations(
    detections: tuple[FrameLocalization, ...],
    *,
    policy: VideoVisiblePolicy = SORA_POLICY,
) -> list[Region | None]:
    """Accept only spatially and temporally recurring candidates.

    A run is consecutive frames whose candidates clear ``weak_floor``, stay
    within ``_STABLE_IOU`` of their neighbour, and are no more than
    ``_MAX_STABLE_GAP + 1`` frames apart. A run shorter than
    ``min_stable_frames`` never fills, and a run whose best score stays below
    ``strong_floor`` never fills either: one confident frame is a lookalike
    until the frames around it agree.
    """
    accepted: list[Region | None] = [None] * len(detections)
    runs: list[list[int]] = []
    current: list[int] = []

    for position, detection in enumerate(detections):
        if detection.region is None or detection.confidence < policy.weak_floor:
            continue
        if current:
            previous = detections[current[-1]]
            anchor = detections[current[0]]
            frame_gap = detection.frame_index - previous.frame_index
            if (
                previous.region is None
                or anchor.region is None
                or frame_gap > _MAX_STABLE_GAP + 1
                or _region_iou(previous.region, detection.region) < _STABLE_IOU
                or (
                    policy.anchor_iou is not None
                    and _region_iou(anchor.region, detection.region) < policy.anchor_iou
                )
            ):
                runs.append(current)
                current = []
        current.append(position)
    if current:
        runs.append(current)

    for run in runs:
        strong = max(detections[position].confidence for position in run) >= policy.strong_floor
        if len(run) < policy.min_stable_frames or not strong:
            continue
        for position in run:
            accepted[position] = detections[position].region
        for left_position, right_position in pairwise(run):
            if right_position - left_position <= 1:
                continue
            left = detections[left_position]
            right = detections[right_position]
            if (
                left.region is None
                or right.region is None
                or _region_iou(left.region, right.region) < _STABLE_IOU
            ):
                continue
            for missing_position in range(left_position + 1, right_position):
                distance_left = missing_position - left_position
                distance_right = right_position - missing_position
                accepted[missing_position] = (
                    left.region if distance_left <= distance_right else right.region
                )

    if policy.cover_after_confirmation:
        confirmed_positions = [
            position for position, region in enumerate(accepted) if region is not None
        ]
        carry_position = confirmed_positions[0] if confirmed_positions else None
        last_supported = confirmed_positions[-1] if confirmed_positions else -1
        for position, region in enumerate(accepted):
            if region is not None:
                carry_position = position
                continue
            raw = detections[position]
            if (
                raw.region is not None
                and raw.confidence >= policy.transition_floor
                and any(
                    accepted[confirmed] is not None
                    and _region_iou(raw.region, accepted[confirmed]) >= _STABLE_IOU
                    for confirmed in confirmed_positions
                )
            ):
                accepted[position] = raw.region
                last_supported = max(last_supported, position)
                continue
            if carry_position is None or position > last_supported + policy.min_stable_frames:
                continue
            accepted[position] = accepted[carry_position]

    return accepted


# --- decode -----------------------------------------------------------------


def _open_capture(source: Path) -> tuple[Any, int, int, float]:
    """Open a source for frame-at-a-time decoding, or refuse with a label."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not decode video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"video has no decodable stream geometry or frame rate: {source}")
    return capture, width, height, fps


def _scan_with_detector(source: Path, detector: Callable[..., FrameLocalization]) -> VideoScan:
    """Decode the source once and collect one untrusted candidate per frame."""
    capture, width, height, fps = _open_capture(source)
    detections: list[FrameLocalization] = []
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError(f"video changes frame dimensions at frame {index}: {source}")
            detections.append(detector(frame, frame_index=index, prepared=_prepare_frame(frame)))
            index += 1
    finally:
        capture.release()
    if not detections:
        raise ValueError(f"video contains no decodable frames: {source}")
    return VideoScan(width, height, fps, tuple(detections))


#: One detector per served mark. The arbiter, the opt-in gate and the fill are
#: shared; a detector is the only per-mark code, and it only localizes.
_MARK_DETECTORS: dict[str, Callable[..., FrameLocalization]] = {
    "sora": detect_sora_frame,
    "veo": detect_veo_frame,
    "seedance": detect_seedance_frame,
    "doubao": detect_doubao_frame,
    "dola": detect_dola_frame,
    "hailuo": detect_hailuo_frame,
    "vidu": detect_vidu_frame,
    "kling": detect_kling_frame,
}


def _mark_detector(mark: str) -> Callable[..., FrameLocalization]:
    """The detector for one mark, or a labeled refusal for a mark we do not serve."""
    detector = _MARK_DETECTORS.get(mark)
    if detector is None:
        raise ValueError(
            f"unknown visible video mark {mark!r}; this build ships "
            f"{', '.join(VISIBLE_VIDEO_MARKS)}"
        )
    return detector


def scan_sora(source: Path) -> VideoScan:
    """Decode the source once and collect one untrusted sora candidate per frame."""
    return _scan_with_detector(source, detect_sora_frame)


def scan_mark(source: Path, mark: str) -> VideoScan:
    """Decode the source once and collect one untrusted candidate per frame for ``mark``."""
    return _scan_with_detector(source, _mark_detector(mark))


# --- masks and fills --------------------------------------------------------


def mask_for_region(
    frame_shape: tuple[int, ...],
    region: Region,
    *,
    padding_fraction: float = _SORA_PADDING_FRACTION,
) -> NDArray[Any]:
    """Build the removal mask for one accepted region.

    A padded rectangle with no dilation is exactly ``region_eraser.boxes_to_mask``,
    the same primitive the image fill uses: a glyph-shaped mask leaves a
    translucent rim outside the approximate synthetic silhouette, and the
    classical fill then pulls that rim back into the hole.
    """
    height, width = frame_shape[:2]
    x, y, region_width, region_height = region
    padding = max(4, round(region_height * padding_fraction))
    return region_eraser.boxes_to_mask(
        (height, width),
        [(x - padding, y - padding, region_width + 2 * padding, region_height + 2 * padding)],
        dilate=0,
    )


def fill_frame(frame_bgr: NDArray[Any], mask: NDArray[Any], *, backend: str) -> NDArray[Any]:
    """Fill one frame's mask through the backend the image path already uses."""
    return region_eraser.erase(frame_bgr, mask=mask, backend=backend)


# --- the encode pipe --------------------------------------------------------


@dataclass
class _RawEncoder:
    """A running ffmpeg process fed BGR frames on standard input."""

    process: subprocess.Popen[bytes]
    stdin: BinaryIO
    stderr_buffer: BinaryIO

    def poll(self) -> int | None:
        return self.process.poll()

    def collect_stderr(self) -> str:
        """Return bounded ffmpeg diagnostics and release the temporary file."""
        if self.stderr_buffer.closed:
            raise RuntimeError("ffmpeg diagnostics have already been collected")
        try:
            return _read_bounded_stderr(self.stderr_buffer)
        finally:
            self.stderr_buffer.close()

    def discard_stderr(self) -> None:
        """Release the diagnostic buffer after an abort."""
        if not self.stderr_buffer.closed:
            self.stderr_buffer.close()


def _read_bounded_stderr(buffer: BinaryIO) -> str:
    """Read bounded head-and-tail diagnostics from a seekable binary stream."""
    buffer.seek(0, os.SEEK_END)
    size = buffer.tell()
    if size <= _FFMPEG_STDERR_LIMIT:
        buffer.seek(0)
        raw_stderr = buffer.read()
    else:
        payload_limit = _FFMPEG_STDERR_LIMIT - len(_FFMPEG_STDERR_TRUNCATION)
        head_size = payload_limit // 2
        tail_size = payload_limit - head_size
        buffer.seek(0)
        head = buffer.read(head_size)
        buffer.seek(-tail_size, os.SEEK_END)
        tail = buffer.read(tail_size)
        raw_stderr = head + _FFMPEG_STDERR_TRUNCATION + tail
    return raw_stderr.decode("utf-8", errors="replace")


def _ffmpeg_path() -> str:
    """The ffmpeg executable, or a labeled refusal; never a guessed name."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("ffmpeg is required for the visible video pass and was not found")
    return ffmpeg


def _profile_args(profile: VideoEncodeProfile) -> list[str]:
    """Build the output options that carry the source's own colour properties."""
    args = ["-pix_fmt", profile.pixel_format]
    for option, value in (
        ("-color_range", profile.color_range),
        ("-colorspace", profile.color_space),
        ("-color_trc", profile.color_transfer),
        ("-color_primaries", profile.color_primaries),
    ):
        if value is not None:
            args.extend([option, value])
    if profile.time_base is not None:
        args.extend(["-enc_time_base:v", profile.time_base])
    return args


def _raw_video_command(
    output: Path,
    *,
    width: int,
    height: int,
    fps: str,
    profile: VideoEncodeProfile,
    itsoffset: float | None,
) -> list[str]:
    """Build the ffmpeg command that reads BGR frames from standard input.

    The frame rate is a measurement (the probe's ``r_frame_rate``, or the
    decoder's own value): a visible pass re-encodes at a constant rate, so a
    source with non-uniform intervals is refused rather than resampled.
    """
    command = [_ffmpeg_path(), "-y", "-loglevel", "error"]
    if itsoffset is not None:
        command.extend(["-itsoffset", f"{itsoffset:.6f}"])
    command.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            fps,
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(_VISIBLE_CRF),
            *_profile_args(profile),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            str(output),
        ]
    )
    return command


def _start_raw_encoder(command: list[str]) -> _RawEncoder:
    """Start ffmpeg and validate its raw-frame pipe.

    Diagnostics go to a temporary file rather than a pipe: an encode that fills
    a stderr pipe nobody drains would deadlock, and a bounded head-and-tail read
    is what makes the failure message useful.
    """
    log.info("Starting ffmpeg visible encode: command=%s", command)
    # The encoder owns this buffer's lifetime, not a ``with`` block: it must
    # outlive ``_start_raw_encoder`` and is released by ``collect_stderr`` or
    # ``discard_stderr``. Same contract as the shipped frame-timing probe.
    stderr_buffer = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_buffer,
        )
    except Exception:
        stderr_buffer.close()
        raise
    if process.stdin is None:
        process.kill()
        process.wait()
        stderr_buffer.close()
        raise RuntimeError("could not open the ffmpeg input pipe")
    return _RawEncoder(process, process.stdin, stderr_buffer)


def _stopped_encoder_error(
    encoder: _RawEncoder, output: Path, error: OSError, *, stderr: str | None = None
) -> ValueError:
    """Label an encoder that stopped reading frames, keeping ffmpeg's own words.

    Writing to a dead ffmpeg raises ``BrokenPipeError``, and the frame pipe is
    the one place in this module that crosses a raw subprocess boundary, so it
    must answer with the encode's diagnostics rather than ``[Errno 32] Broken
    pipe``. ffmpeg has usually already exited and left its reason on stderr;
    *stderr* is passed in when a caller has already collected it.
    """
    if stderr is None and not encoder.stderr_buffer.closed:
        if encoder.poll() is None:
            # ffmpeg closed the pipe but has not exited yet; stop it so its
            # diagnostics can be read rather than discarded as an abort.
            encoder.process.kill()
            encoder.process.wait()
        stderr = encoder.collect_stderr()
    detail = (stderr or "").strip() or str(error)
    return ValueError(f"ffmpeg stopped reading frames for {output.name}: {detail[:500]}")


def _finish_raw_encoder(encoder: _RawEncoder, output: Path, *, timeout: float) -> None:
    """Close the frame stream and raise when ffmpeg rejects the encode."""
    close_error: OSError | None = None
    try:
        encoder.stdin.close()
    except OSError as error:
        # A dead encoder breaks the close; the wait and the stderr read below
        # still have to happen before the reason can be reported.
        close_error = error
    try:
        return_code = encoder.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        encoder.process.kill()
        encoder.process.wait()
        stderr = encoder.collect_stderr()
        raise ValueError(
            f"ffmpeg timed out encoding {output.name}: {stderr.strip()[:500]}"
        ) from error
    stderr = encoder.collect_stderr()
    log.info("ffmpeg visible encode finished: status=%s stderr=%s", return_code, stderr)
    if close_error is not None:
        raise _stopped_encoder_error(encoder, output, close_error, stderr=stderr) from close_error
    if return_code != 0:
        raise ValueError(f"ffmpeg failed to encode {output.name}: {stderr.strip()[:500]}")


def _abort_raw_encoder(encoder: _RawEncoder) -> None:
    """Stop an incomplete ffmpeg encode and release its diagnostic buffer."""
    if encoder.poll() is None:
        encoder.process.kill()
    encoder.process.wait()
    with suppress(OSError):
        # The pipe is already broken. Closing is best-effort cleanup: without
        # it the buffered writer raises this same error again from its
        # finalizer, where nothing can catch it.
        encoder.stdin.close()
    encoder.discard_stderr()


def _mux_with_source_audio(
    encoded: Path,
    source: Path,
    output: Path,
    *,
    strip_metadata: bool,
    copy_input_timestamps: bool,
    timeout: float,
) -> None:
    """Copy the encoded video and the source's own audio into ``output``.

    ``-c copy`` is what keeps the audio byte-identical: no decode, no re-encode,
    the original packets are remuxed.
    """
    command = [
        _ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        *(["-copyts"] if copy_input_timestamps else []),
        "-i",
        str(encoded),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c",
        "copy",
        "-map_metadata",
        "-1" if strip_metadata else "1",
        "-map_chapters",
        "-1" if strip_metadata else "1",
    ]
    if copy_input_timestamps:
        command.extend(["-avoid_negative_ts", "disabled"])
    if output.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(output))
    log.info("ffmpeg visible mux: command=%s", command)
    try:
        result = run_command(command, timeout=timeout, output_limit=100_000)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg is required to mux the visible pass output") from error
    except ExternalCommandTimeout as error:
        raise ValueError(f"ffmpeg timed out muxing {output.name}") from error
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed to mux {output.name}: {result.stderr_text.strip()[:500]}")


def _measured_frame_times(source: Path, *, timeout: float) -> tuple[float, ...] | None:
    """The source's frame display times, or ``None`` when they cannot be read.

    ``video_encoding.probe_video_timestamps`` cannot answer for a source whose
    first frame carries SEI side data: ffprobe's CSV writer appends the
    side-data warning to the first value, so the whole read is discarded as
    unparseable (``clip.mkv`` from this ticket's own fixture is one). This is a
    JSON read of the same field, so a measurable timeline stays measurable. An
    unreadable one returns ``None`` and is never reported as a constant one.
    """
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(source),
            ],
            timeout=timeout,
            output_limit=DEFAULT_OUTPUT_LIMIT,
        )
    except (FileNotFoundError, ExternalCommandTimeout):
        return None
    if result.returncode != 0 or result.stdout_truncated:
        return None
    try:
        frames = json.loads(result.stdout_text)["frames"]
        times = tuple(float(frame["best_effort_timestamp_time"]) for frame in frames)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(times) < 3 or not all(math.isfinite(time) for time in times):
        return None
    return times


def _timing_label(source: Path, *, timeout: float) -> str:
    """Label the source's frame timeline, refusing a non-uniform one.

    A constant-rate encode of a variable-rate source would move frames against
    the bitstream-copied audio, so the pass refuses instead of approximating.
    """
    times = _measured_frame_times(source, timeout=timeout)
    if times is None:
        return VISIBLE_TIMING_UNMEASURED
    intervals = [right - left for left, right in pairwise(times)]
    if (
        intervals
        and intervals[0] > 0
        and all(abs(interval - intervals[0]) < 1e-6 for interval in intervals)
    ):
        return VISIBLE_TIMING_CONSTANT
    raise ValueError(
        "visible video removal re-encodes at a constant frame rate; this source has "
        "non-uniform frame timestamps, so a fill would desynchronize the copied audio"
    )


def _require_sdr_8bit(profile: VideoEncodeProfile) -> None:
    """Refuse a source the 8-bit BGR pipe cannot carry back unchanged."""
    if (profile.component_depth or 0) > 8 or profile.color_transfer in _HDR_TRANSFERS:
        source_format = profile.source_pixel_format or "unknown pixel format"
        raise ValueError(
            "visible video removal supports SDR 8-bit input only; refusing to "
            f"silently reduce {source_format} / "
            f"{profile.color_transfer or 'unknown transfer'} to 8-bit SDR"
        )


def _itsoffset_seconds(profile: VideoEncodeProfile) -> float | None:
    """The source's non-zero start offset in seconds, when it is measurable."""
    if profile.start_pts in (None, 0) or profile.time_base is None:
        return None
    offset = profile.start_pts * float(Fraction(profile.time_base))
    return offset if offset > 0 else None


@contextmanager
def _staged_frames_video(dest: Path) -> Iterator[Path]:
    """Yield the encoded-video temporary path, removed whatever happens."""
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{dest.name}-video-", suffix=".mkv", dir=dest.parent
    )
    os.close(handle)
    encoded = Path(temp_name)
    try:
        yield encoded
    finally:
        encoded.unlink(missing_ok=True)


# --- the pass ---------------------------------------------------------------


def _explicit_mask_provider(
    mask_path: Path,
) -> Callable[[NDArray[Any], int], NDArray[Any] | None]:
    """A per-frame mask provider for an operator-named PGM mask."""
    mask = read_pgm(mask_path)
    expected = (mask.height, mask.width)
    flat = np.frombuffer(bytes(mask.data), dtype=np.uint8)

    def provider(frame_bgr: NDArray[Any], _index: int) -> NDArray[Any] | None:
        shape = frame_bgr.shape[:2]
        if shape != expected:
            raise ValueError(
                f"mask {mask_path} is {mask.width}x{mask.height} but the video frame is "
                f"{shape[1]}x{shape[0]}"
            )
        return flat.reshape(expected)

    return provider


def _box_mask_provider(
    box: tuple[int, int, int, int],
) -> Callable[[NDArray[Any], int], NDArray[Any] | None]:
    """A per-frame mask provider for an operator-named box."""

    def provider(frame_bgr: NDArray[Any], _index: int) -> NDArray[Any] | None:
        return region_eraser.boxes_to_mask(frame_bgr.shape[:2], [box], dilate=0)

    return provider


def encode_clean_video(
    source: Path,
    dest: Path,
    *,
    mask_provider: Callable[[NDArray[Any], int], NDArray[Any] | None],
    backend: str,
    remove_metadata: bool,
    temporal_consistency: bool,
    timeout: float,
    expected_frames: int | None = None,
) -> tuple[int, str]:
    """Decode, fill every masked frame, and publish the result atomically.

    Returns the number of frames whose pixels changed and the measured timing
    label. The destination appears only after a completed encode and mux; any
    failure removes the staged files and leaves the destination exactly as it
    was.
    """
    if backend not in VISIBLE_FILL_BACKENDS:
        raise ValueError(
            f"unknown visible fill backend {backend!r}; expected one of "
            f"{', '.join(VISIBLE_FILL_BACKENDS)}"
        )
    profile = probe_video_encode_profile(source, timeout=min(timeout, DEFAULT_PROBE_TIMEOUT))
    _require_sdr_8bit(profile)
    timing = _timing_label(source, timeout=min(timeout, DEFAULT_PROBE_TIMEOUT))
    itsoffset = _itsoffset_seconds(profile)

    capture, width, height, capture_fps = _open_capture(source)
    fps = profile.frame_rate or f"{capture_fps:.12g}"
    removed_frames = 0
    frame_index = 0
    encoder: _RawEncoder | None = None
    try:
        with atomic_video_output(dest) as temporary, _staged_frames_video(dest) as encoded:
            encoder = _start_raw_encoder(
                _raw_video_command(
                    encoded,
                    width=width,
                    height=height,
                    fps=fps,
                    profile=profile,
                    itsoffset=itsoffset,
                )
            )
            previous_source: NDArray[Any] | None = None
            previous_cleaned: NDArray[Any] | None = None
            previous_mask: NDArray[Any] | None = None
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (height, width):
                        raise ValueError(
                            f"video changes frame dimensions at frame {frame_index}: {source}"
                        )
                    source_frame = frame
                    mask = mask_provider(frame, frame_index)
                    if mask is not None:
                        cleaned = fill_frame(frame, mask, backend=backend)
                        if (
                            temporal_consistency
                            and previous_source is not None
                            and previous_cleaned is not None
                            and previous_mask is not None
                        ):
                            cleaned = stabilize_filled_frame(
                                previous_source,
                                previous_cleaned,
                                previous_mask,
                                source_frame,
                                cleaned,
                                mask,
                                copy=False,
                            )
                        frame = cleaned
                        removed_frames += 1
                        previous_source = source_frame
                        previous_cleaned = frame
                        previous_mask = mask
                    try:
                        encoder.stdin.write(frame.tobytes())
                    except OSError as error:
                        raise _stopped_encoder_error(encoder, encoded, error) from error
                    frame_index += 1
            except BaseException:
                _abort_raw_encoder(encoder)
                encoder = None
                raise
            _finish_raw_encoder(encoder, encoded, timeout=timeout)
            encoder = None
            if expected_frames is not None and frame_index != expected_frames:
                raise ValueError(
                    f"video ended after {frame_index} frames; the scan measured "
                    f"{expected_frames} — refusing to fill a partial clip"
                )
            _mux_with_source_audio(
                encoded,
                source,
                temporary,
                strip_metadata=remove_metadata,
                copy_input_timestamps=itsoffset is not None,
                timeout=timeout,
            )
    finally:
        capture.release()
        if encoder is not None:
            _abort_raw_encoder(encoder)
    return removed_frames, timing


def _accepted_regions(source: Path, mark: str) -> tuple[VideoScan, list[Any], int]:
    """Scan *source* for *mark* and keep the localizations the arbiter accepts."""
    scan = scan_mark(source, mark)
    regions = stabilize_localizations(scan.detections, policy=VISIBLE_MARK_POLICIES[mark])
    return scan, regions, sum(region is not None for region in regions)


def _best_auto_mark(source: Path) -> tuple[str, VideoScan, list[Any], int]:
    """Scan every shipped mark; keep the one accepted on the most frames.

    Several detectors can accept the same overlay, so a tie on accepted frames
    goes to the higher mean confidence on those frames. An all-zero result
    keeps the first mark, so a clip with no mark reports ``no-mark`` and fills
    nothing.
    """
    # ponytail: decodes the clip once per mark (8 passes); share one decode
    # across detectors if auto scans on long clips get slow.
    best: tuple[str, VideoScan, list[Any], int] | None = None
    best_rank: tuple[int, float] = (-1, 0.0)
    for key in VISIBLE_VIDEO_MARKS:
        scan, regions, detected = _accepted_regions(source, key)
        accepted = [
            detection.confidence
            for detection, region in zip(scan.detections, regions, strict=False)
            if region is not None
        ]
        rank = (detected, sum(accepted) / len(accepted) if accepted else 0.0)
        if rank > best_rank:
            best, best_rank = (key, scan, regions, detected), rank
    assert best is not None
    return best


def clean_visible_video(
    source: Path,
    dest: Path,
    *,
    mark: str | None = None,
    mask_path: Path | None = None,
    box: tuple[int, int, int, int] | None = None,
    backend: str = "cv2",
    remove_metadata: bool = True,
    temporal_consistency: bool = True,
    timeout: float = DEFAULT_VIDEO_TIMEOUT,
) -> VisibleRemoval:
    """Run one visible pass: detection-driven, or an operator-named region.

    A detection pass with no accepted candidate writes the metadata-only
    remux: no frame is filled without evidence. An explicit mask or box fills
    the named region on every frame and reports the removal as verifiable.
    """
    explicit = mask_path is not None or box is not None
    if explicit and mark is not None:
        raise ValueError("exactly one localization source may be configured")
    if not explicit and mark is None:
        raise ValueError("a visible video pass needs a mark, a mask, or a box")

    if explicit:
        if mask_path is not None:
            source_label = f"mask:{mask_path}"
            mask_provider = _explicit_mask_provider(mask_path)
        else:
            if box is None:
                raise ValueError("a box pass needs a box")
            source_label = f"box:{','.join(map(str, box))}"
            mask_provider = _box_mask_provider(box)
        removed_frames, timing = encode_clean_video(
            source,
            dest,
            mask_provider=mask_provider,
            backend=backend,
            remove_metadata=remove_metadata,
            temporal_consistency=temporal_consistency,
            timeout=timeout,
        )
        return VisibleRemoval(
            source=source_label,
            mark=None,
            status="cleaned",
            verification="verifiable",
            backend=backend,
            total_frames=removed_frames,
            detected_frames=removed_frames,
            removed_frames=removed_frames,
            timing=timing,
            note=(
                "the operator named the region; the pixels that changed are exactly the "
                "named region, so this removal is verifiable rather than best-effort"
            ),
        )

    if mark == AUTO_VIDEO_MARK:
        mark, scan, regions, detected_frames = _best_auto_mark(source)
    else:
        scan, regions, detected_frames = _accepted_regions(source, mark)
    policy = VISIBLE_MARK_POLICIES[mark]
    if detected_frames == 0:
        # A detection pass with no accepted candidate changes no pixel: the
        # metadata strip still produces the destination, as a stream copy.
        remux_video(source, dest, strip_metadata=remove_metadata, timeout=timeout)
        return VisibleRemoval(
            source=f"detection:{mark}",
            mark=mark,
            status="no-mark",
            verification="best-effort",
            backend=backend,
            total_frames=len(scan.detections),
            detected_frames=0,
            removed_frames=0,
            timing=VISIBLE_TIMING_NOT_RUN,
            note=(
                "no candidate recurred across adjacent frames, so no frame was filled; "
                "detection-driven removal is best-effort and this run found nothing to remove"
            ),
        )

    def provider(frame_bgr: NDArray[Any], index: int) -> NDArray[Any] | None:
        if index >= len(regions):
            raise ValueError(
                f"video has more frames than the {len(regions)}-frame scan; refusing to "
                "fill a clip whose length changed between the scan and the encode"
            )
        region = regions[index]
        if region is None:
            return None
        return mask_for_region(frame_bgr.shape, region, padding_fraction=policy.padding_fraction)

    removed_frames, timing = encode_clean_video(
        source,
        dest,
        mask_provider=provider,
        backend=backend,
        remove_metadata=remove_metadata,
        temporal_consistency=temporal_consistency,
        timeout=timeout,
        expected_frames=len(scan.detections),
    )
    return VisibleRemoval(
        source=f"detection:{mark}",
        mark=mark,
        status="cleaned" if removed_frames else "no-mark",
        verification="best-effort",
        backend=backend,
        total_frames=len(scan.detections),
        detected_frames=detected_frames,
        removed_frames=removed_frames,
        timing=timing,
        note=(
            "the temporal arbiter accepted a mark that recurs at the same place across "
            "adjacent frames; a detection-driven fill is best-effort, not a guarantee"
        ),
    )
