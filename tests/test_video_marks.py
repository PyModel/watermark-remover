# Ported from wiltodelta/remove-ai-watermarks tests/test_video_visible.py,
# Apache-2.0; modified: this file covers the seven marks ticket 06 adds to the
# visible pass (veo, seedance, doubao, dola, hailuo, vidu, kling) through the
# same pipeline seam ticket 05 established, reusing that file's synthetic clip
# builders, faked runtime and report assertions instead of its sora fixture.
"""Behavioral contract for the seven remaining visible-video marks.

One seam, the pipeline: a request in (the frozen ``VideoPlan`` the request
builder composes and the source path), frames and a result out. Every test
crosses it the way the executor does -- ``clean_asset``, the execution step
``clean_pipeline.execute`` calls -- and asserts on the bytes at the destination
and the words in the report, never on internal calls. Each fixture draws the
mark from the detector's own synthetic template, exactly as ticket 05's sora
fixture does, so a detection is a real detection and not a lookalike. The clip
tests use real ffmpeg, ffprobe and OpenCV because a fill is a pixel measurement
a fake cannot make, and skip cleanly when those are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_video_visible import (
    _clean_video_asset,
    _encode_clip,
    _stream_md5,
    _stub_video_extra,
    requires_visible_runtime,
)

#: The marks this file covers. Ticket 05's sora pass is pinned in
#: ``test_video_visible.py``; these seven run through the same engine.
_MARKS = ("veo", "seedance", "doubao", "dola", "hailuo", "vidu", "kling")

#: Where each mark is drawn and how tall, chosen so the mark sits inside the
#: detector's own search region and at a size the detector's height sweep hits.
#: ``(template key, template height, origin x, origin y)`` on the 160x120 clip.
_PLACEMENT: dict[str, tuple[str, int, int, int]] = {
    "veo": ("veo-diamond", 16, 142, 102),
    # seedance: template height 10 = round(120 * 0.085), a swept height, drawn
    # inside the detector's bottom-right search region.
    "seedance": ("seedance", 10, 120, 95),
    # doubao: template height 6 = max(6, round(120 * 0.048)) is in the swept
    # range; the mark is the tree's renderer-synthetic outline alpha.
    "doubao": ("doubao", 6, 110, 100),
    # dola: all swept heights collapse to 6 at this clip size; the mark is the
    # frozen Hershey duplex render the detector matches.
    "dola": ("dola", 6, 115, 106),
    # hailuo: the composite MINIMAX/hailuo label is wide, so it starts well left
    # of the corner but inside the detector's bottom-left search region.
    "hailuo": ("hailuo", 6, 60, 100),
    # vidu: logo plus wordmark, bottom-right, inside the (0.70, 0.84) search.
    "vidu": ("vidu", 6, 120, 106),
    # kling: the Pillow-rendered "KlingAI" wordmark variant. The wordmark's own
    # expanded box must reach the frame edge, so it is drawn flush toward the
    # corner. The "KLING AI" variant is avoided because its matching ring
    # lookalike contracts the region onto the glyph's right half and leaves the
    # left half behind after the fill; this wider variant's box still covers it.
    "kling": ("kling-3", 6, 112, 108),
}


def _mark_frames(
    mark: str,
    *,
    width: int = 160,
    height: int = 120,
    count: int = 12,
    marked_frames: range | tuple[int, ...] = (),
) -> list:
    """Synthetic BGR frames carrying one mark on the selected frames.

    The mark is the detector's own synthetic template, resized the way the
    detector resizes it, so a detection here is a real detection. The
    background is flat and dark: a classical fill over a padded box can restore
    it, which is what makes the post-fill measurement meaningful.
    """
    import cv2
    import numpy as np
    import video_visible

    template_key, template_height, x, y = _PLACEMENT[mark]
    base = video_visible._template_sources()[template_key]
    template_width = max(1, round(base.shape[1] * template_height / base.shape[0]))
    mark_image = cv2.resize(base, (template_width, template_height), interpolation=cv2.INTER_AREA)
    frames = []
    for index in range(count):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # A slow-moving band: motion the detector must tolerate, well away from
        # the mark's fixed corner.
        band = 8 + (index * 3) % (width // 2)
        frame[height - 24 : height - 12, band : band + 10] = 120
        if index in tuple(marked_frames):
            region = frame[y : y + template_height, x : x + template_width]
            np.maximum(region, mark_image[:, :, None], out=region)
        frames.append(frame)
    return frames


def _mark_clip(
    tmp_path: Path, mark: str, *, marked_frames: range | tuple[int, ...] = range(12)
) -> Path:
    """A real video carrying one mark on the frames the caller names."""
    frames = _mark_frames(mark, marked_frames=marked_frames)
    path = tmp_path / f"{mark}.mkv"
    _encode_clip(path, frames)
    return path


def _accepted_frames(path: Path, mark: str) -> int:
    """How many frames the temporal arbiter authors a fill for, for one mark."""
    import video_visible

    scan = video_visible.scan_mark(path, mark)
    regions = video_visible.stabilize_localizations(
        scan.detections, policy=video_visible.VISIBLE_MARK_POLICIES[mark]
    )
    return sum(region is not None for region in regions)


# --- criterion 1: each mark is removed from a clip carrying it stably ------


@requires_visible_runtime
@pytest.mark.parametrize("mark", _MARKS)
def test_a_stable_mark_is_removed_and_reported_best_effort(
    mark: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence-backed removal: a recurring mark loses it, and says best-effort."""
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _mark_clip(tmp_path, mark)
    dest = tmp_path / "clean.mkv"
    # The fixture really carries the mark before the pass runs.
    assert _accepted_frames(src, mark) == 12

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mark=mark))
    ).to_dict()
    visible = payload["visible"]

    assert visible is not None
    assert visible["source"] == f"detection:{mark}"
    assert visible["mark"] == mark
    assert visible["status"] == "cleaned"
    assert visible["verification"] == "best-effort"
    assert visible["backend"] == "cv2"
    assert visible["detected_frames"] == 12
    assert visible["removed_frames"] == 12
    # The same run still strips and verifies the container's provenance marker.
    assert payload["markers_removed"] != []
    assert payload["markers_remaining"] == []
    # The mark is gone: the arbiter finds nothing left to authorize.
    assert _accepted_frames(dest, mark) == 0
    # The frames were re-encoded, so a stream copy cannot explain the result.
    assert _stream_md5(dest, "0:v:0") != _stream_md5(src, "0:v:0")


# --- criterion 2: a one-frame flicker from this set is never filled --------


@requires_visible_runtime
@pytest.mark.parametrize("mark", _MARKS)
def test_a_mark_in_only_one_frame_is_never_filled(
    mark: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-frame flicker is not evidence: no candidate run, no fill."""
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _mark_clip(tmp_path, mark, marked_frames=(5,))
    dest = tmp_path / "clean.mkv"
    frames_before = _stream_md5(src, "0:v:0")

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mark=mark))
    ).to_dict()
    visible = payload["visible"]

    assert visible is not None
    assert visible["mark"] == mark
    assert visible["status"] == "no-mark"
    assert visible["verification"] == "best-effort"
    assert visible["detected_frames"] == 0
    assert visible["removed_frames"] == 0
    assert visible["timing"] == "not-run"
    assert dest.exists()
    # No frame changed at all, so the isolated flicker survives untouched.
    assert _stream_md5(dest, "0:v:0") == frames_before


# --- --mark auto: every shipped mark is scanned, the accepted one is filled --


@requires_visible_runtime
@pytest.mark.parametrize("mark", _MARKS)
def test_auto_finds_the_mark_the_clip_carries(
    mark: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--mark auto`` (the batch ``--mode visible`` default) works on video."""
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _mark_clip(tmp_path, mark)
    dest = tmp_path / "clean.mkv"

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mark="auto"))
    ).to_dict()
    visible = payload["visible"]

    assert visible["mark"] == mark
    assert visible["status"] == "cleaned"
    assert visible["verification"] == "best-effort"
    assert _accepted_frames(dest, mark) == 0


@requires_visible_runtime
def test_auto_on_a_clip_with_no_mark_fills_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video import VideoPlan
    from video_visible import VisibleVideoPlan

    _stub_video_extra(monkeypatch)
    src = _mark_clip(tmp_path, "vidu", marked_frames=())
    dest = tmp_path / "clean.mkv"
    frames_before = _stream_md5(src, "0:v:0")

    payload = _clean_video_asset(
        src, dest, VideoPlan(visible=VisibleVideoPlan(mark="auto"))
    ).to_dict()

    assert payload["visible"]["status"] == "no-mark"
    assert payload["visible"]["mark"] == "auto"
    assert payload["visible"]["source"] == "detection:auto"
    assert payload["visible"]["removed_frames"] == 0
    assert _stream_md5(dest, "0:v:0") == frames_before
