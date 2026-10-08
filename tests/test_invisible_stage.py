"""Contracts for the invisible-watermark stage on the image path.

The stage is the seam: a request in, a plan, and bytes out. It runs only when a
local decoder finds an invisible target, it is labeled best-effort, and it
writes into the image path's staging directory so a failure never reaches the
destination. When the engine is unavailable the stage is skipped, the output is
byte-identical to a run without it, and the command exits 3.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_asset as clean_asset_module
import clean_file
import clean_pipeline
import invisible_watermark
from clean_asset import CleanPlan, ImagePixelPlan, InvisiblePlan
from clean_asset import clean_asset as run_clean_asset
from clean_request import CleanRequest, build_clean_plan


def _png(path: Path, size: tuple[int, int] = (32, 32)) -> Path:
    Image.new("RGB", size, (120, 140, 160)).save(path, format="PNG")
    return path


def _engine_command(tmp_path: Path, body: str, name: str = "engine.py") -> str:
    """A shell-free command template that runs *body* as the fake engine.

    ``{input}``/``{output}``/``{profile}``/``{strength}``/``{seed}`` are filled by
    the stage; quoting here keeps a path with spaces a single argv element.
    """
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} "
        "{input} {output} {profile} {strength} {seed}"
    )


def _recording_engine(tmp_path: Path, record: Path) -> str:
    return _engine_command(
        tmp_path,
        "import shutil, sys\n"
        "src, dst, profile, strength, seed = sys.argv[1:6]\n"
        "shutil.copyfile(src, dst)\n"
        f"open({str(record)!r}, 'w', encoding='utf-8').write("
        '"|".join((profile, strength, seed)))\n',
    )


@pytest.fixture
def target_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decoder that reports a known open scheme for every file."""
    monkeypatch.setattr(
        invisible_watermark,
        "detect_invisible_watermark",
        lambda path, **kwargs: "Stable Diffusion XL",
    )


@pytest.fixture
def engine_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fail-closed capability probe to report the engine missing."""
    monkeypatch.setattr(
        clean_asset_module,
        "pixel_backend_capability",
        lambda name: {
            "requested": name,
            "supported": True,
            "available": False,
            "executed": False,
            "reason": "pixel backend is unavailable",
        },
    )


def test_the_stage_runs_only_when_the_file_carries_a_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine present, no target: the engine is never invoked."""
    src = _png(tmp_path / "input.png")
    dest = tmp_path / "output.png"
    record = tmp_path / "record.txt"
    monkeypatch.setattr(
        invisible_watermark, "detect_invisible_watermark", lambda path, **kwargs: None
    )

    result = run_clean_asset(
        src,
        dest,
        CleanPlan(
            forced_kind="image",
            pixel=ImagePixelPlan(),
            invisible=InvisiblePlan(command=_recording_engine(tmp_path, record)),
        ),
    )

    assert dest.is_file()
    assert not record.exists(), "the engine ran with no invisible target present"
    # Nothing to remove: no stage report at all, so the common image clean keeps
    # its byte-identical JSON shape.
    assert "invisible" not in result.to_dict()
    assert result.backend_unavailable is False


def test_a_completed_stage_is_labeled_best_effort(tmp_path: Path, target_present: None) -> None:
    src = _png(tmp_path / "input.png")
    dest = tmp_path / "output.png"
    record = tmp_path / "record.txt"

    result = run_clean_asset(
        src,
        dest,
        CleanPlan(
            forced_kind="image",
            pixel=ImagePixelPlan(),
            invisible=InvisiblePlan(
                profile="qwen-zimage",
                command=_recording_engine(tmp_path, record),
            ),
        ),
    )

    report = result.to_dict()["invisible"]
    assert report["status"] == "completed"
    assert report["best_effort"] is True
    assert report["profile"] == "qwen-zimage"
    assert dest.is_file()
    assert record.read_text(encoding="utf-8").split("|")[0] == "qwen-zimage"


def test_profile_selection_changes_the_profile_the_stage_uses(
    tmp_path: Path, target_present: None
) -> None:
    profiles = {}
    for profile in ("qwen-zimage", "sdxl-zimage"):
        workspace = tmp_path / profile
        workspace.mkdir()
        src = _png(workspace / "input.png")
        dest = workspace / "output.png"
        record = workspace / "record.txt"
        result = run_clean_asset(
            src,
            dest,
            CleanPlan(
                forced_kind="image",
                pixel=ImagePixelPlan(),
                invisible=InvisiblePlan(
                    profile=profile,
                    command=_recording_engine(workspace, record),
                ),
            ),
        )
        profiles[profile] = (result.to_dict()["invisible"], record.read_text(encoding="utf-8"))

    qwen_report, qwen_record = profiles["qwen-zimage"]
    sdxl_report, sdxl_record = profiles["sdxl-zimage"]
    assert qwen_report["profile"] == "qwen-zimage"
    assert sdxl_report["profile"] == "sdxl-zimage"
    assert qwen_record.split("|")[0] == "qwen-zimage"
    assert sdxl_record.split("|")[0] == "sdxl-zimage"
    # The selected profile also moves its calibrated strength, not only a label.
    assert qwen_record.split("|")[1] != sdxl_record.split("|")[1]


def test_a_failing_engine_leaves_the_destination_untouched(
    tmp_path: Path, target_present: None
) -> None:
    src = _png(tmp_path / "input.png")
    dest = tmp_path / "output.png"
    dest.write_bytes(b"ORIGINAL-DESTINATION")
    failing = _engine_command(
        tmp_path,
        "import sys\n"
        "with open(sys.argv[2], 'wb') as fh:\n"
        "    fh.write(b'partial-regeneration')\n"
        "raise SystemExit(1)\n",
    )

    with pytest.raises(RuntimeError, match="invisible-watermark engine failed"):
        run_clean_asset(
            src,
            dest,
            CleanPlan(
                forced_kind="image",
                pixel=ImagePixelPlan(),
                invisible=InvisiblePlan(command=failing),
            ),
        )

    assert dest.read_bytes() == b"ORIGINAL-DESTINATION"


def test_an_absent_engine_skips_the_stage_and_preserves_the_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_present: None, engine_absent: None
) -> None:
    """The control run has no stage at all; the stage run must match it byte for byte."""
    control_src = _png(tmp_path / "control.png")
    control_dest = tmp_path / "control.out.png"
    control = run_clean_asset(
        control_src,
        control_dest,
        CleanPlan(forced_kind="image", pixel=ImagePixelPlan()),
    )

    src = _png(tmp_path / "input.png")
    dest = tmp_path / "output.png"
    staged = run_clean_asset(
        src,
        dest,
        CleanPlan(
            forced_kind="image",
            pixel=ImagePixelPlan(),
            invisible=InvisiblePlan(),
        ),
    )

    assert dest.read_bytes() == control_dest.read_bytes()
    report = staged.to_dict()["invisible"]
    assert report["status"] == "skipped"
    assert report["reason"] == "engine-unavailable"
    assert report["best_effort"] is True
    assert staged.backend_unavailable is True
    assert control.backend_unavailable is False


def test_an_absent_engine_exits_three_through_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_present: None, engine_absent: None
) -> None:
    src = _png(tmp_path / "input.png")
    request = CleanRequest(paths=(src,), json=True)
    run = clean_pipeline.execute(request)

    payload = run.payload
    assert payload["exit_code"] == 3
    assert payload["invisible"]["reason"] == "engine-unavailable"


def test_an_absent_engine_exits_three_from_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_present: None, engine_absent: None
) -> None:
    src = _png(tmp_path / "input.png")
    dest = tmp_path / "output.png"
    monkeypatch.setattr(
        sys, "argv", ["clean_file.py", str(src), "-o", str(dest), "--json", "--quiet"]
    )

    assert clean_file.main() == 3
    assert dest.is_file()


def test_a_video_request_carries_the_visible_opt_in_into_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request no longer refuses visible cleaning for video."""
    monkeypatch.setattr("clean_request.missing_video_runtime", lambda: ())
    from video_visible import VisibleVideoPlan

    dest = tmp_path / "clip.cleaned.mkv"
    mask = tmp_path / "region.pgm"
    mask.write_bytes(b"P5\n1 1\n255\n\x00")

    for request, expected in (
        (CleanRequest(paths=(tmp_path / "a.mkv",), visible_box=(4, 4, 8, 8)), "box"),
        (CleanRequest(paths=(tmp_path / "b.mkv",), visible_mask=mask), "mask_path"),
        (CleanRequest(paths=(tmp_path / "c.mkv",), mark="dola"), "mark"),
    ):
        plan = build_clean_plan(request, dest, "video")
        assert isinstance(plan.video.visible, VisibleVideoPlan)
        assert getattr(plan.video.visible, expected) is not None
        # Metadata-only default is preserved when no visible flag is present.
        assert build_clean_plan(CleanRequest(), dest, "video").video.visible is None
