# Ported from wiltodelta/remove-ai-watermarks tests/test_video_visible.py,
# Apache-2.0; adapted: this tree's synthetic container builders and its
# ``clean_asset`` execution seam replace the upstream helper imports, the clip
# is generated at the pipeline input rather than loaded from a fixture corpus,
# and the assertions are written against this tree's report fields.
"""Behavioral contract for the opt-in visible-mark pass on a video.

One seam, the pipeline: a request in (the frozen ``VideoPlan`` the request
builder composes and the source path), frames and a result out. The tests
cross it the way the executor does -- ``clean_asset``, the execution step
``clean_pipeline.execute`` calls -- and assert on the bytes at the
destination and the words in the report, never on internal calls. The
synthetic container builders and the faked command runner are this tree's
existing seams (``tests/test_video.py``); the clip tests use real ffmpeg,
ffprobe and OpenCV because a fill is a pixel measurement a fake cannot make,
and skip cleanly when those are absent.

Scope: the sora mark only. The other seven marks are ticket 06.
"""

from __future__ import annotations

import importlib.util
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

LABEL = (
    b'{"Label":"1","ContentProducer":"91110102MACQD9K640",'
    b'"ProduceID":"c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"}'
)

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_HAS_CV2 = importlib.util.find_spec("cv2") is not None
requires_visible_runtime = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None or not _HAS_CV2,
    reason="the visible video pass needs ffmpeg, ffprobe and OpenCV",
)


# --- synthetic container bytes (pure Python, this tree's builder) -----------


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


def _stub_video_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the video extra's import without a PyAV install.

    ``probe_video_encode_profile`` asks PyAV for component bit depth and falls
    back to ffprobe's own field when PyAV rejects the pixel format. The stub
    keeps that contract by rejecting, so the pixel tests exercise the fallback
    rather than crashing on a bare module object.
    """
    extra = types.ModuleType("av")

    def _reject(_pixel_format):
        raise ValueError("the PyAV stub resolves no pixel formats")

    extra.VideoFormat = _reject
    monkeypatch.setitem(sys.modules, "av", extra)


# --- the clip the pixel tests measure --------------------------------------


def _sora_frames(
    *,
    width: int = 160,
    height: int = 120,
    count: int = 12,
    marked_frames: range | tuple[int, ...] = (),
    mark_origin: tuple[int, int] = (8, 8),
) -> list:
    """Synthetic BGR frames carrying the sora wordmark on selected frames.

    The mark is the engine's own synthetic silhouette, resized exactly the way
    the detector resizes its template, so a detection here is a real detection
    and not a lookalike. The background is flat and dark: a classical fill over
    a padded box can restore it exactly, which is what makes the post-fill
    measurement meaningful.
    """
    import cv2
    import numpy as np
    import video_visible

    word, _icon = video_visible._sora_templates()
    template_height = max(16, round(min(height, width) * 0.085))
    template_width = max(1, round(word.shape[1] * template_height / word.shape[0]))
    mark = cv2.resize(word, (template_width, template_height), interpolation=cv2.INTER_AREA)
    x, y = mark_origin
    frames = []
    for index in range(count):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # A slow-moving band: motion the detector must tolerate, well away from
        # the mark's fixed corner.
        band = 8 + (index * 3) % (width // 2)
        frame[height - 24 : height - 12, band : band + 10] = 120
        if index in tuple(marked_frames):
            region = frame[y : y + template_height, x : x + template_width]
            region[mark > 127] = 255
        frames.append(frame)
    return frames


def _encode_clip(path: Path, frames: list, *, fps: int = 10) -> None:
    """Write BGR frames into a clip with ffmpeg, near-losslessly.

    The clip also carries an AIGC provenance tag, so a ``clean_video`` caller
    proves on the same run that the visible pass still strips and verifies the
    container markers rather than only changing pixels.
    """
    height, width = frames[0].shape[:2]
    raw = b"".join(frame.tobytes() for frame in frames)
    result = subprocess.run(
        [
            str(_FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-metadata",
            f"AIGC={LABEL.decode()}",
            str(path),
        ],
        input=raw,
        capture_output=True,
        check=False,
        timeout=120.0,
    )
    assert result.returncode == 0, result.stderr.decode()


def _sora_clip(tmp_path: Path, name: str = "clip.mkv", **kwargs) -> Path:
    """A real video carrying the sora mark on the frames the caller names."""
    marked = kwargs.pop("marked_frames", range(12))
    frames = _sora_frames(count=12, marked_frames=marked, **kwargs)
    path = tmp_path / name
    _encode_clip(path, frames)
    return path


def _stream_md5(path: Path, stream: str) -> str:
    """One elementary stream's bitstream hash, read with a stream copy."""
    result = subprocess.run(
        [
            str(_FFMPEG),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            stream,
            "-c",
            "copy",
            "-f",
            "md5",
            "-",
        ],
        capture_output=True,
        check=False,
        timeout=120.0,
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


def _decoded_frames(path: Path) -> list:
    """Every frame of a clip as a BGR array, decoded by the engine's decoder."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert frames, path
    return frames


def _clean_video_asset(src: Path, dest: Path, video_plan):
    import clean_asset
    from clean_asset import CleanPlan

    return clean_asset.clean_asset(src, dest, CleanPlan(forced_kind="video", video=video_plan))


def _scan(path: Path):
    import video_visible

    return video_visible.scan_sora(path)


def _accepted_frames(path: Path) -> int:
    """How many frames the temporal arbiter would authorize a fill for."""
    import video_visible

    regions = video_visible.stabilize_localizations(_scan(path).detections)
    return sum(region is not None for region in regions)


def _assert_change_confined_to(
    source: Path,
    dest: Path,
    box: tuple[int, int, int, int],
    *,
    tolerance: int = 16,
    changed: int = 40,
    minimum_fraction: float = 0.05,
) -> None:
    """Assert the region that changed is the named box, within codec noise.

    Outside the box the two clips differ only by a lossy re-encode of the same
    pixels, measured at ``tolerance`` levels; inside it the fill replaced the
    mark, measured as a floor on the changed-pixel fraction.
    """
    import numpy as np

    x, y, width, height = box
    before_frames = _decoded_frames(source)
    after_frames = _decoded_frames(dest)
    assert len(before_frames) == len(after_frames)
    for before, after in zip(before_frames, after_frames, strict=True):
        diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2)
        inside = diff[y : y + height, x : x + width]
        outside = diff.copy()
        outside[y : y + height, x : x + width] = 0
        assert int(outside.max()) <= tolerance
        assert float(np.count_nonzero(inside > changed)) / inside.size >= minimum_fraction


# --- criterion 1: without the opt-in flag, no frame changes ----------------


def test_the_default_video_plan_carries_no_visible_opt_in() -> None:
    """The gate is the plan shape: no flag, no visible section at all."""
    from video import VideoPlan

    assert VideoPlan().visible is None


def test_a_video_clean_without_the_opt_in_never_transcodes_a_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The faked-seam control: the run issues a stream copy, never an encode."""
    import video_encoding
    from video import VideoPlan

    _fake_runtime(tmp_path, monkeypatch)
    src = tmp_path / "clip.mkv"
    src.write_bytes(matroska(LABEL))
    dest = tmp_path / "clean.mkv"

    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        Path(argv[-1]).write_bytes(matroska())
        return fake_command_result(0)

    monkeypatch.setattr(video_encoding, "run_command", fake_run)
    result = _clean_video_asset(src, dest, VideoPlan())

    assert result.to_dict()["visible"] is None
    assert calls, "the metadata strip must still write the destination"
    assert all("-c:v" not in argv for argv in calls)
    assert any(argv[argv.index("-c") + 1] == "copy" for argv in calls if "-c" in argv)


@requires_visible_runtime
def test_a_real_sora_clip_without_the_opt_in_keeps_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real measurement: the marked frames survive a no-opt-in clean."""
    from video import VideoPlan

    _stub_video_extra(monkeypatch)
    src = _sora_clip(tmp_path)
    dest = tmp_path / "clean.mkv"
    frames_before = _stream_md5(src, "0:v:0")

    result = _clean_video_asset(src, dest, VideoPlan())
    payload = result.to_dict()

    assert payload["kind"] == "video"
    assert payload["visible"] is None
    assert dest.exists()
    assert _stream_md5(dest, "0:v:0") == frames_before


# --- criterion 2: the opt-in, a stable mark, and a one-frame flicker -------


def test_the_opt_in_vocabulary_is_closed_to_marks_this_build_cannot_serve() -> None:
    """The other seven marks are ticket 06: configuring one is refused, not ignored."""
    import pytest as _pytest

    _pytest.importorskip("cv2")

    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    for unsupported in ("veo", "seedance", "kling"):
        with _pytest.raises(ValueError, match="mark"):
            VisibleVideoPlan(mark=unsupported)
    with _pytest.raises(ValueError, match="exactly one"):
        VisibleVideoPlan(mark="sora", box=(0, 0, 1, 1))
    with _pytest.raises(ValueError, match="exactly one"):
        VisibleVideoPlan()
    with _pytest.raises(ValueError, match="backend"):
        VisibleVideoPlan(mark="sora", backend="magic")
    assert VideoPlan(visible=VisibleVideoPlan(mark="sora")).visible is not None


@requires_visible_runtime
def test_a_stable_sora_mark_is_removed_and_reported_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence-backed removal: a recurring mark loses it, and says best-effort."""
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _sora_clip(tmp_path)
    dest = tmp_path / "clean.mkv"
    # The fixture really carries the mark before the pass runs.
    assert _accepted_frames(src) == 12

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mark="sora"))
    ).to_dict()
    visible = payload["visible"]

    assert visible is not None
    assert visible["source"] == "detection:sora"
    assert visible["mark"] == "sora"
    assert visible["status"] == "cleaned"
    assert visible["verification"] == "best-effort"
    assert visible["backend"] == "cv2"
    assert visible["detected_frames"] == 12
    assert visible["removed_frames"] == 12
    # The same run still strips and verifies the container's provenance marker.
    assert payload["markers_removed"] != []
    assert payload["markers_remaining"] == []
    assert payload["audio"] == "unverified"
    assert payload["audio_verified"] is False
    # The mark is gone: the arbiter finds nothing left to authorize.
    assert _accepted_frames(dest) == 0
    # The frames were re-encoded, so a stream copy cannot explain the result.
    assert _stream_md5(dest, "0:v:0") != _stream_md5(src, "0:v:0")


@requires_visible_runtime
def test_a_mark_in_only_one_frame_is_never_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-frame flicker is not evidence: no candidate run, no fill."""
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _sora_clip(tmp_path, marked_frames=(5,))
    dest = tmp_path / "clean.mkv"
    frames_before = _stream_md5(src, "0:v:0")

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mark="sora"))
    ).to_dict()
    visible = payload["visible"]

    assert visible["status"] == "no-mark"
    assert visible["verification"] == "best-effort"
    assert visible["detected_frames"] == 0
    assert visible["removed_frames"] == 0
    assert visible["timing"] == "not-run"
    assert dest.exists()
    # No frame changed at all, and the flicker is still readable in the output.
    assert _stream_md5(dest, "0:v:0") == frames_before
    assert max(detection.confidence for detection in _scan(dest).detections) > 0.9


# --- criterion 3: an explicit region is verifiable ------------------------


@requires_visible_runtime
def test_an_explicit_box_removes_the_named_region_and_is_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No detector stands between the operator's box and the pixels."""
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _sora_clip(tmp_path)
    dest = tmp_path / "clean.mkv"
    box = (4, 4, 60, 30)

    payload = _clean_video_asset(src, dest, VideoPlan(visible=VisibleVideoPlan(box=box))).to_dict()
    visible = payload["visible"]

    assert visible["source"] == "box:4,4,60,30"
    assert visible["mark"] is None
    assert visible["status"] == "cleaned"
    assert visible["verification"] == "verifiable"
    assert visible["removed_frames"] == visible["total_frames"] == 12
    # The named mark was inside the box, and the fill removed it.
    assert _accepted_frames(dest) == 0
    _assert_change_confined_to(src, dest, box)


@requires_visible_runtime
def test_an_explicit_mask_removes_the_named_region_and_is_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PGM mask is the same contract as a box: exactly the named region."""
    import morphomod
    from region_eraser import boxes_to_mask
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _sora_clip(tmp_path)
    dest = tmp_path / "clean.mkv"
    box = (4, 4, 60, 30)
    mask = boxes_to_mask((120, 160), [box], dilate=0)
    mask_path = tmp_path / "region.pgm"
    morphomod.write_pgm(morphomod.Mask(160, 120, bytearray(mask.tobytes())), mask_path)

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mask_path=mask_path))
    ).to_dict()
    visible = payload["visible"]

    assert visible["source"] == f"mask:{mask_path}"
    assert visible["mark"] is None
    assert visible["verification"] == "verifiable"
    assert visible["removed_frames"] == visible["total_frames"] == 12
    assert _accepted_frames(dest) == 0
    _assert_change_confined_to(src, dest, box)


# --- the encoder boundary: a crash is labeled, never a bare pipe error -----


@requires_visible_runtime
def test_a_dead_encoder_is_reported_as_a_labeled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An encoder that dies mid-encode fails the clean with ffmpeg's own reason.

    The frame pipe crosses a raw subprocess boundary: writing to a dead encoder
    raises ``BrokenPipeError``, and the error the operator sees must be the
    encode's diagnostic, not ``[Errno 32] Broken pipe``. No partial output may
    survive the failure either.
    """
    import video_visible
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _sora_clip(tmp_path)
    dest = tmp_path / "clean.mkv"

    dead = tmp_path / "ffmpeg"
    dead.write_text("#!/bin/sh\necho 'boom: encoder died' >&2\nexit 1\n", encoding="utf-8")
    dead.chmod(0o755)
    monkeypatch.setattr(video_visible, "_ffmpeg_path", lambda: str(dead))

    with pytest.raises(ValueError, match=r"stopped reading frames.*encoder died"):
        _clean_video_asset(src, dest, VideoPlan(visible=VisibleVideoPlan(box=(4, 4, 60, 30))))
    assert not dest.exists()
