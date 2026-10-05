"""CLI regressions for dry-run, timeout, and image-only degradation policy."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import morphomod
import pytest
from morphomod import (
    Raster,
    VisiblePlan,
    box_mask,
    decode_png,
    encode_png,
    remove_visible,
    write_pgm,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
CLEAN_FILE = SCRIPTS / "clean_file.py"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLEAN_FILE), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_dry_run_does_not_execute_commands_or_create_outputs(tmp_path) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(3, 3, 3, bytearray([10, 20, 30] * 9))))
    marker = tmp_path / "detector-ran"
    detector = tmp_path / "detector.py"
    detector.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('called')\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "new-output-dir"
    destination = output_dir / "cleaned.png"
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(detector))} {{input}} {{mask}}"

    result = _run(
        str(source),
        "-o",
        str(destination),
        "--detect-command",
        command,
        "--dry-run",
        "--timeout",
        "12.5",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry-run"
    assert payload["timeout"] == 12.5
    assert not marker.exists()
    assert not destination.exists()
    assert not Path(payload["mask"]).exists()
    assert not output_dir.exists()
    assert not source.with_suffix(".png.bak").exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_visible_plan_rejects_invalid_timeout(timeout) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        VisiblePlan(box=(0, 0, 1, 1), timeout=timeout)


def test_detect_command_receives_input_mask_prompt(tmp_path) -> None:
    # F5: external --detect-command must receive {input}, {mask}, and {prompt}
    # exactly as documented, and the detector's mask output must feed the
    # inpaint path so a real output is produced.
    source = tmp_path / "input.png"
    # 9x9 image so dilation of a single center pixel never covers the whole frame.
    source.write_bytes(encode_png(Raster(9, 9, 3, bytearray([10, 20, 30] * 81))))
    log = tmp_path / "detector.log"
    detector = tmp_path / "detector.py"
    detector.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(SCRIPTS)!r})",
                "import morphomod",
                "from morphomod import box_mask, write_pgm",
                "input_arg, mask_arg, prompt_arg = sys.argv[1], sys.argv[2], sys.argv[3]",
                f"Path({str(log)!r}).write_text(",
                "    f'input={input_arg}\\nmask={mask_arg}\\nprompt={prompt_arg}',",
                "    encoding='utf-8',",
                ")",
                "write_pgm(box_mask(9, 9, (4, 4, 1, 1)), Path(mask_arg))",
            ]
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "output.png"
    prompt = "remove the watermark box please"
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(detector))} {{input}} {{mask}} {{prompt}}"
    )

    result = _run(
        str(source),
        "-o",
        str(dest),
        "--detect-command",
        command,
        "--visible-prompt",
        prompt,
        "--visible-backend",
        "simple",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    assert dest.is_file()
    log_text = log.read_text(encoding="utf-8")
    assert f"input={source}" in log_text
    assert "mask=" in log_text
    assert f"prompt={prompt}" in log_text


def test_frictionless_default_report_retains_transparency(tmp_path) -> None:
    # F6: "no transparency on what was removed" is satisfied by leaving no
    # artifacts/audit trail — but the report itself must still state exactly
    # what was removed (deterministic counts, best-effort labels where relevant).
    source = tmp_path / "input.txt"
    source.write_text("hello \u200bworld", encoding="utf-8")
    dest = tmp_path / "output.txt"
    result = _run(str(source), "-o", str(dest), "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    # Deterministic cleaners report exact removed/replaced counts (honesty
    # contract); best-effort layers are labeled as such elsewhere.
    assert report["stats"]["removed_count"] > 0
    assert "removed_count" in report["stats"] and "replaced_count" in report["stats"]
    # Frictionless: no audit trail, no artifacts on disk by default.
    assert not (tmp_path / "wm-audit.json").exists()


def test_detector_receives_plan_timeout(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(3, 3, 3, bytearray([10, 20, 30] * 9))))
    destination = tmp_path / "output.png"
    observed: list[float] = []

    def fake_run(template: str, *, timeout: float, **values: str) -> None:
        observed.append(timeout)
        write_pgm(morphomod.box_mask(3, 3, (1, 1, 1, 1)), Path(values["mask"]))

    monkeypatch.setattr(morphomod, "_run_template", fake_run)
    remove_visible(
        source,
        destination,
        VisiblePlan(
            detect_command="detector {input} {mask}",
            backend="simple",
            dilation_radius=0,
            timeout=7.25,
        ),
    )

    assert observed == [7.25]


def test_external_inpainter_receives_plan_timeout(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(3, 3, 3, bytearray([10, 20, 30] * 9))))
    mask = tmp_path / "input-mask.pgm"
    write_pgm(morphomod.box_mask(3, 3, (1, 1, 1, 1)), mask)
    destination = tmp_path / "output.png"
    observed: list[float] = []

    def fake_run(template: str, *, timeout: float, **values: str) -> None:
        observed.append(timeout)
        Path(values["output"]).write_bytes(Path(values["input"]).read_bytes())

    monkeypatch.setattr(morphomod, "_run_template", fake_run)
    remove_visible(
        source,
        destination,
        VisiblePlan(
            mask_path=mask,
            backend="external",
            command="inpainter {input} {mask} {output}",
            dilation_radius=0,
            timeout=8.5,
        ),
    )

    assert observed == [8.5]


@pytest.mark.parametrize("flag", [("--degrade", "blur"), ("--morpho", "grid")])
def test_non_image_degradation_is_rejected_without_output(tmp_path, flag) -> None:
    source = tmp_path / "input.txt"
    source.write_text("hello", encoding="utf-8")
    destination = tmp_path / "output.txt"

    result = _run(str(source), "-o", str(destination), *flag)

    assert result.returncode == 1
    assert "only valid for image assets" in result.stderr
    assert not destination.exists()


def _text_source(tmp_path: Path, name: str = "input.txt") -> Path:
    src = tmp_path / name
    src.write_text("hello \u200bworld", encoding="utf-8")
    return src


def test_quiet_suppresses_success_output_but_not_errors(tmp_path: Path) -> None:
    # F3: silent batch op — success/progress output is suppressed, failures
    # still surface so nothing is silently dropped.
    src = _text_source(tmp_path)
    dest = tmp_path / "output.txt"
    result = _run(str(src), "-o", str(dest), "--quiet")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert dest.is_file()

    # Errors must surface even under --quiet.
    missing = tmp_path / "missing.txt"
    result = _run(str(missing), "-o", str(tmp_path / "out.txt"), "--quiet")
    assert result.returncode != 0
    assert "not a" in result.stderr or "error" in result.stderr


def test_audit_writes_report_only_when_requested(tmp_path: Path) -> None:
    # F3: no audit trail by default (frictionless); --audit writes a JSON report.
    src = _text_source(tmp_path)
    dest = tmp_path / "output.txt"
    result = _run(str(src), "-o", str(dest))
    assert result.returncode == 0
    assert not (tmp_path / "wm-audit.json").exists()

    dest2 = tmp_path / "output2.txt"
    result = _run(str(src), "-o", str(dest2), "--audit")
    assert result.returncode == 0, result.stderr
    audit_path = tmp_path / "wm-audit.json"
    assert audit_path.is_file()
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    assert report["output"] == str(dest2)
    assert report["stats"]["removed_count"] > 0


def test_audit_custom_path(tmp_path: Path) -> None:
    src = _text_source(tmp_path)
    dest = tmp_path / "output.txt"
    audit = tmp_path / "custom" / "report.json"
    result = _run(str(src), "-o", str(dest), "--audit", str(audit))
    assert result.returncode == 0, result.stderr
    assert audit.is_file()
    assert json.loads(audit.read_text(encoding="utf-8"))["output"] == str(dest)


# ---------------------------------------------------------------------------
# cv2 fill backends at the wm CLI seam (ticket 02)
# ---------------------------------------------------------------------------


def _write_stub_opencv(directory: Path) -> None:
    """Ship importable stub cv2/numpy modules for the wm subprocess.

    Same trick as test_stability_server.py's stub cv2: the child resolves
    them through PYTHONPATH, so the test exercises the real import path —
    'OpenCV importable' — without the wheel in the pooled venv.  The stub
    inpaint fills each marked pixel from its nearest unmasked row neighbour.
    """
    (directory / "numpy.py").write_text(
        "\n".join(
            [
                "uint8 = 'uint8'",
                "",
                "class _Array:",
                "    def __init__(self, data, shape):",
                "        self._data = bytes(data)",
                "        self.shape = tuple(shape)",
                "",
                "    def reshape(self, *shape):",
                "        return _Array(self._data, tuple(shape))",
                "",
                "    def tobytes(self):",
                "        return self._data",
                "",
                "def frombuffer(data, dtype=None):",
                "    return _Array(data, (len(data),))",
            ]
        ),
        encoding="utf-8",
    )
    (directory / "cv2.py").write_text(
        "\n".join(
            [
                "INPAINT_TELEA = 1",
                "INPAINT_NS = 2",
                "",
                "def inpaint(src, mask, radius, flags):",
                "    width = src.shape[1]",
                "    channels = src.shape[2]",
                "    data = bytearray(src._data)",
                "    marks = mask._data",
                "    for index, value in enumerate(marks):",
                "        if not value:",
                "            continue",
                "        row_start = (index // width) * width",
                "        replacement = bytes(channels)",
                "        for neighbour in (*range(index - 1, row_start - 1, -1),",
                "                          *range(index + 1, row_start + width)):",
                "            if not marks[neighbour]:",
                "                replacement = data[neighbour * channels:(neighbour + 1) * channels]",
                "                break",
                "        data[index * channels:(index + 1) * channels] = replacement",
                "    return type(src)(bytes(data), src.shape)",
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("backend", ["cv2-telea", "cv2-ns"])
def test_clean_file_cv2_backend_end_to_end_with_stub_opencv(tmp_path: Path, backend: str) -> None:
    stubs = tmp_path / "opencv-stubs"
    stubs.mkdir()
    _write_stub_opencv(stubs)
    width = height = 16
    background = bytearray([9, 8, 7] * width * height)
    marked = bytearray(background)
    index = (7 * width + 7) * 3  # single white pixel at (7, 7)
    marked[index : index + 3] = b"\xff\xff\xff"
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(width, height, 3, marked)))
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(width, height, (7, 7, 1, 1)), mask)
    dest = tmp_path / "cleaned.png"
    env = {**os.environ, "PYTHONPATH": str(stubs)}

    result = subprocess.run(
        [
            sys.executable,
            str(CLEAN_FILE),
            str(src),
            "-o",
            str(dest),
            "--visible-mask",
            str(mask),
            "--dilate",
            "0",
            "--visible-backend",
            backend,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["visible"]["status"] == "completed"
    assert report["visible"]["backend"] == backend
    # The filled output verifies: every pixel outside the effective mask is
    # preserved, and the marked pixel was filled from the background.
    assert report["verification"]["outside_mask_preserved"] is True
    assert decode_png(dest.read_bytes()).data == background


def _write_broken_opencv(directory: Path) -> None:
    """Shadow cv2 with a module that raises on import for the wm subprocess.

    Inverse of _write_stub_opencv: instead of faking a working OpenCV, the
    stub makes ``import cv2`` fail inside the child, so the availability
    boundary ``optional_deps._import_safe("cv2")`` (importlib.import_module)
    observes OpenCV as absent on any machine — the fail-soft path is
    exercised deterministically instead of assuming environment absence.
    """
    (directory / "cv2.py").write_text(
        "raise ModuleNotFoundError('OpenCV is not installed (blocked by test)')\n",
        encoding="utf-8",
    )


def test_clean_file_cv2_backend_fails_soft_without_opencv(tmp_path: Path) -> None:
    # Block the cv2 import at the same boundary backend resolution probes
    # (optional_deps -> importlib.import_module("cv2")) rather than trusting
    # the host venv to lack OpenCV: with the stub first on PYTHONPATH the
    # child sees optional_deps.cv2 as None, no cv2 backend registers, and
    # _resolve_fill_backend takes the labeled-unavailable path everywhere.
    stubs = tmp_path / "blocked-opencv"
    stubs.mkdir()
    _write_broken_opencv(stubs)
    env = {**os.environ, "PYTHONPATH": str(stubs)}

    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(16, 16, 3, bytearray([9, 8, 7] * 256))))
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(16, 16, (7, 7, 1, 1)), mask)
    dest = tmp_path / "cleaned.png"

    result = _run(
        str(src),
        "-o",
        str(dest),
        "--visible-mask",
        str(mask),
        "--dilate",
        "0",
        "--visible-backend",
        "cv2-telea",
        env=env,
    )

    # The flag parses (choices widened); the failure is the labeled
    # unavailability on the normal runtime error path — exit code unchanged
    # (FAILED -> 1), install hint in stderr, no output written.
    assert result.returncode == 1
    assert "invalid choice" not in result.stderr
    assert "watermark-remover[visible]" in result.stderr
    assert not dest.exists()

    # Default backends still run unchanged in the same OpenCV-less environment.
    simple_dest = tmp_path / "simple.png"
    ok = _run(
        str(src),
        "-o",
        str(simple_dest),
        "--visible-mask",
        str(mask),
        "--dilate",
        "0",
        "--visible-backend",
        "simple",
        "--json",
        env=env,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["visible"]["status"] == "completed"


# ---------------------------------------------------------------------------
# Learned fill tiers at the plan + CLI seam (ticket 03)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["migan", "lama"])
def test_visible_plan_accepts_learned_backend_names(backend: str) -> None:
    plan = VisiblePlan(
        mask_path=Path("mask.pgm"),
        backend=backend,
        mask_output=Path("out.mask.pgm"),
    )
    assert plan.backend == backend

    with pytest.raises(ValueError, match="unknown visible backend"):
        VisiblePlan(mask_path=Path("mask.pgm"), backend="bogus")


@pytest.mark.parametrize("backend", ["migan", "lama"])
def test_remove_visible_learned_backend_fails_soft_labeled(tmp_path, backend: str) -> None:
    # No learned backend is registered in this build (the migan/lama ONNX
    # backends land with ticket 04), so resolution must take the labeled-
    # unavailable path deterministically -- registration is code-driven, not
    # environment-driven.
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(16, 16, 3, bytearray([9, 8, 7] * 256))))
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(16, 16, (7, 7, 1, 1)), mask)
    dest = tmp_path / "cleaned.png"

    with pytest.raises(ValueError, match=rf"watermark-remover\[{backend}\]"):
        remove_visible(
            src,
            dest,
            VisiblePlan(
                mask_path=mask,
                backend=backend,
                dilation_radius=0,
                mask_output=tmp_path / "out.mask.pgm",
            ),
        )
    assert not dest.exists()


@pytest.mark.parametrize("backend", ["migan", "lama"])
def test_clean_file_learned_backend_choice_parses_and_fails_soft(
    tmp_path: Path, backend: str
) -> None:
    # The flag parses (choices widened for the learned tiers); the failure is
    # the labeled unavailability on the normal runtime error path -- exit code
    # unchanged (FAILED -> 1), install hint in stderr, no output written.
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(16, 16, 3, bytearray([9, 8, 7] * 256))))
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(16, 16, (7, 7, 1, 1)), mask)
    dest = tmp_path / "cleaned.png"

    result = _run(
        str(src),
        "-o",
        str(dest),
        "--visible-mask",
        str(mask),
        "--dilate",
        "0",
        "--visible-backend",
        backend,
    )

    assert result.returncode == 1
    assert "invalid choice" not in result.stderr
    assert f"watermark-remover[{backend}]" in result.stderr
    assert not dest.exists()


# ---------------------------------------------------------------------------
# Registry auto-mark flags at the wm CLI seam (ticket 09)
# ---------------------------------------------------------------------------

#: sha256 of the pinned no-flag run over :func:`_pin_fixture` + an explicit mask
#: with the texture backend, computed on the pre-change tree (ticket 09 base
#: 96b9061). The pinned contract: absent --mark/--sensitivity, the CLI's output
#: bytes are exactly what base produced.
_PIN_SHA256 = "993992de5829684f4d20d4d65f646b9bf46f16cedec27545c4f007a21682cd71"


def _pin_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """The deterministic 64x64 input + mask the pin was computed against."""
    src = tmp_path / "pin-input.png"
    bg = bytearray([9, 8, 7] * 64 * 64)
    for y in (28, 29, 30, 31):
        for x in (28, 29, 30, 31):
            i = (y * 64 + x) * 3
            bg[i : i + 3] = b"\xfa\xfb\xfc"
    src.write_bytes(encode_png(Raster(64, 64, 3, bg)))
    mask = tmp_path / "pin.mask.pgm"
    write_pgm(box_mask(64, 64, (26, 26, 8, 8)), mask)
    return src, mask


def test_no_flag_invocation_is_byte_for_byte_pinned(tmp_path: Path) -> None:
    """Without --mark/--sensitivity the CLI behaves byte-for-byte like base.

    The run uses the flags base had (--visible-mask + --visible-backend texture)
    and nothing from ticket 09; the output must hash to the base-computed pin.
    """
    src, mask = _pin_fixture(tmp_path)
    dest = tmp_path / "pin-actual.png"

    result = _run(
        str(src),
        "-o",
        str(dest),
        "--visible-mask",
        str(mask),
        "--visible-backend",
        "texture",
    )

    assert result.returncode == 0, result.stderr
    assert dest.is_file()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == _PIN_SHA256


def _planted_fixture(path: Path, *, marked: bool = True) -> None:
    """A bottom-right Doubao-class mark composed at the engine's measured geometry.

    Same forward model as the engine vectors: ``out = a*fg + (1-a)*bg`` from the
    shipped renderer-synthetic silhouette.
    """
    import cv2
    import numpy as np

    asset = cv2.imread(str(SCRIPTS / "assets" / "doubao_outline_alpha.png"), cv2.IMREAD_GRAYSCALE)
    assert asset is not None
    at = asset.astype(np.float32) / 255.0
    w, h, base = 1024, 1365, 1024
    gw = int(0.1636 * base)
    gh = int(gw * 77 / 346)
    ax = w - int(0.015 * base) - gw
    ay = h - int(0.015 * base) - gh
    img = np.full((h, w, 3), 100, np.uint8)
    if marked:
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
        a3 = amap[:, :, None]
        img = (a3 * 255 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path.write_bytes(encode_png(Raster(w, h, 3, bytearray(rgb.tobytes()))))


def test_mark_auto_detects_removes_and_validates(tmp_path: Path) -> None:
    """``--mark auto`` on a planted fixture: detected, removed, validated, reported
    best-effort, exit 0."""
    pytest.importorskip("cv2")
    src = tmp_path / "marked.png"
    _planted_fixture(src)
    dest = tmp_path / "cleaned.png"

    result = _run(str(src), "-o", str(dest), "--mark", "auto", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 0
    visible = payload["visible"]
    assert visible["status"] == "completed"
    assert visible["mark_status"] == "cleaned"
    assert visible["mark"] == "auto"
    assert visible["sensitivity"] == "auto"
    assert [m["key"] for m in visible["marks"]] == ["doubao"]
    removal = visible["marks"][0]
    assert removal["status"] == "cleaned"
    assert removal["confidence_before"] > removal["confidence_after"]
    assert removal["note"].startswith("best-effort")
    codes = [a["code"] for a in visible["action_details"]]
    assert "mask_source" in codes and "inpaint" in codes
    assert dest.is_file()
    # the output actually differs from the input (the mark was filled)
    assert dest.read_bytes() != src.read_bytes()


def test_mark_auto_on_clean_image_reports_no_mark_exit_zero(tmp_path: Path) -> None:
    """Nothing found under --mark auto: status no_mark, exit 0, output written as
    an unchanged copy (no new exit codes)."""
    pytest.importorskip("cv2")
    src = tmp_path / "clean.png"
    _planted_fixture(src, marked=False)
    dest = tmp_path / "cleaned.png"

    result = _run(str(src), "-o", str(dest), "--mark", "auto", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 0
    visible = payload["visible"]
    assert visible["mark_status"] == "no_mark"
    assert visible["marks"] == []
    assert [a["code"] for a in visible["action_details"]][-1] == "nothing_removed"
    assert decode_png(dest.read_bytes()).data == decode_png(src.read_bytes()).data


def test_sensitivity_flag_parses_and_rides_the_report(tmp_path: Path) -> None:
    """``--sensitivity strict`` is accepted with --mark and reported verbatim.

    The strict behavioral gates (never relaxes; relaxation needs same-product
    evidence) are pinned at the registry seam in test_watermark_registry.py;
    this pins the flag surface.
    """
    pytest.importorskip("cv2")
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(64, 64, 3, bytearray([9, 8, 7] * 64 * 64))))
    dest = tmp_path / "cleaned.png"

    result = _run(str(src), "-o", str(dest), "--mark", "auto", "--sensitivity", "strict", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["visible"]["sensitivity"] == "strict"
    assert payload["visible"]["mark_status"] == "no_mark"


def test_mark_and_explicit_sources_are_mutually_exclusive(tmp_path: Path) -> None:
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(8, 8, 3, bytearray([9, 8, 7] * 64))))
    dest = tmp_path / "cleaned.png"
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(8, 8, (3, 3, 1, 1)), mask)

    result = _run(str(src), "-o", str(dest), "--mark", "auto", "--visible-mask", str(mask))

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
    assert not dest.exists()


def test_unknown_mark_is_a_usage_error(tmp_path: Path) -> None:
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(8, 8, 3, bytearray([9, 8, 7] * 64))))
    result = _run(str(src), "-o", tmp_path / "cleaned.png", "--mark", "bogus")
    assert result.returncode == 2


def test_visible_plan_rejects_unknown_mark() -> None:
    with pytest.raises(ValueError, match="unknown mark"):
        VisiblePlan(mark="bogus")
    with pytest.raises(ValueError, match="unknown sensitivity"):
        VisiblePlan(mark="auto", sensitivity="aggressive")
    with pytest.raises(ValueError, match="exactly one localization source"):
        VisiblePlan(box=(0, 0, 1, 1), mark="auto")
    plan = VisiblePlan(mark="auto", sensitivity="strict")
    assert plan.mark == "auto" and plan.sensitivity == "strict"


# ---------------------------------------------------------------------------
# Stage 8 fix pass 2: honest mark-path reporting (findings F3 + F7)
# ---------------------------------------------------------------------------


def test_mark_auto_reports_the_actual_fill_backend_not_the_planned_one(tmp_path: Path) -> None:
    """F3: a mark plan resolves its fill through the registry, so the report's
    ``backend`` field must name the backend that actually filled, not the
    ``--visible-backend`` value (whose texture/external choices the registry
    pass never uses)."""
    pytest.importorskip("cv2")
    src = tmp_path / "marked.png"
    _planted_fixture(src)
    dest = tmp_path / "cleaned.png"

    result = _run(str(src), "-o", str(dest), "--mark", "auto", "--json")

    assert result.returncode == 0, result.stderr
    visible = json.loads(result.stdout)["visible"]
    assert visible["marks"], "fixture must fire a mark for this pin"
    assert visible["backend"] == visible["marks"][0]["backend"]


def test_mark_auto_with_external_backend_reports_the_registry_fill_and_warns(tmp_path: Path) -> None:
    """F3: `--visible-backend external` cannot drive a registry mark fill (the
    external command template has no place in the localize->fill->validate
    pass), so the report must name the registry backend that actually ran and
    carry a labeled warning that external was not applied."""
    pytest.importorskip("cv2")
    src = tmp_path / "marked.png"
    _planted_fixture(src)
    dest = tmp_path / "cleaned.png"

    result = _run(
        str(src),
        "-o",
        str(dest),
        "--mark",
        "auto",
        "--visible-backend",
        "external",
        "--inpaint-command",
        'inpainter --image "{input}" --mask "{mask}" --output "{output}"',
        "--json",
    )

    assert result.returncode == 0, result.stderr
    visible = json.loads(result.stdout)["visible"]
    assert visible["marks"], "fixture must fire a mark for this pin"
    assert visible["backend"] == visible["marks"][0]["backend"]
    assert visible["backend"] != "external"
    warnings = [a for a in visible["action_details"] if a["effect"] == "warning"]
    assert any("external" in a["text"] and "registry" in a["text"] for a in warnings), (
        "expected a labeled warning that the external backend was not applied"
    )


def test_mark_auto_reports_dilation_as_engine_controlled(tmp_path: Path) -> None:
    """F7: the registry engines own their dilation, so the mark report must not
    imply a user `--dilate` radius applied: it states 0 user-applied pixels and
    labels the dilation engine-controlled."""
    pytest.importorskip("cv2")
    src = tmp_path / "clean.png"
    _planted_fixture(src, marked=False)
    dest = tmp_path / "cleaned.png"

    result = _run(str(src), "-o", str(dest), "--mark", "auto", "--dilate", "5", "--json")

    assert result.returncode == 0, result.stderr
    visible = json.loads(result.stdout)["visible"]
    assert visible["dilation_radius"] == 0
    assert visible["dilation"] == "engine-controlled (0 user-applied)"
