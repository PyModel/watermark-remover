"""CLI regressions for dry-run, timeout, and image-only degradation policy."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import morphomod
import pytest
from morphomod import Raster, VisiblePlan, encode_png, remove_visible, write_pgm

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
CLEAN_FILE = SCRIPTS / "clean_file.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLEAN_FILE), *args],
        capture_output=True,
        text=True,
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
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(detector))} {{input}} {{mask}} {{prompt}}"

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
