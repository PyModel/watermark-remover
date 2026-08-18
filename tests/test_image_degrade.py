"""End-to-end tests for the --degrade/--morpho pipeline through clean_asset."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_asset as clean_asset_module
from clean_asset import CleanPlan, ImageDegradePlan, clean_asset
from morphomod import Raster, decode_png, encode_png


def _make_png(path: Path, size: int = 16, channels: int = 3) -> Path:
    data = bytearray()
    for i in range(size * size):
        pixel = [7 * i % 256, 11 * i % 256, 13 * i % 256]
        if channels == 4:
            pixel.append(255)
        data += bytes(pixel)
    path.write_bytes(encode_png(Raster(size, size, channels, data)))
    return path


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_file.py"), *argv],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("strategy", ["freq-dct", "blur", "median", "jpeg", "rotate", "two-stage"])
def test_degrade_strategies_end_to_end(tmp_path: Path, strategy: str) -> None:
    source = _make_png(tmp_path / "in.png")
    dest = tmp_path / "out.png"
    plan = CleanPlan(
        degrade=ImageDegradePlan(strategy=strategy, strength=0.4, seed=3),
    )
    result = clean_asset(source, dest, plan)
    assert result.residual is False
    payload = result.to_dict()
    assert payload["degrade"]["strategy"] == strategy
    raster = decode_png(dest.read_bytes())
    assert (raster.width, raster.height) == (16, 16)


@pytest.mark.parametrize("strategy", ["grid", "diagonal", "noise", "quantize"])
def test_morpho_strategies_end_to_end(tmp_path: Path, strategy: str) -> None:
    source = _make_png(tmp_path / "in.png", channels=4)
    dest = tmp_path / "out.png"
    plan = CleanPlan(degrade=ImageDegradePlan(strategy=strategy, seed=3))
    result = clean_asset(source, dest, plan)
    assert result.residual is False
    assert result.to_dict()["degrade"]["strategy"] == strategy
    raster = decode_png(dest.read_bytes())
    assert raster.channels == 4
    # Alpha must be untouched for RGBA input.
    assert raster.data[3::4] == bytes([255] * (16 * 16))


def test_degrade_preserves_output_file_mode(tmp_path: Path) -> None:
    source = _make_png(tmp_path / "in.png")
    dest = tmp_path / "out.png"
    clean_asset(
        source,
        dest,
        CleanPlan(degrade=ImageDegradePlan(strategy="blur")),
    )
    dest.chmod(0o640)
    clean_asset(
        source,
        dest,
        CleanPlan(degrade=ImageDegradePlan(strategy="median")),
    )
    if os.name == "posix":
        assert dest.stat().st_mode & 0o777 == 0o640


def test_degrade_rejects_non_png_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_png(tmp_path / "in.png")
    dest = tmp_path / "out.png"
    clean_asset(source, dest, CleanPlan(strip_all_metadata=False))
    monkeypatch.setattr(clean_asset_module, "detect_format", lambda data: "jpeg")
    with pytest.raises(ValueError, match="requires PNG output"):
        clean_asset(
            source,
            dest,
            CleanPlan(degrade=ImageDegradePlan(strategy="blur")),
        )


def test_image_degrade_plan_validates_strength() -> None:
    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        ImageDegradePlan(strength=1.5)
    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        ImageDegradePlan(strength="strong")


def test_clean_file_degrade_cli_exits_zero_without_residual(tmp_path: Path) -> None:
    source = _make_png(tmp_path / "in.png")
    dest = tmp_path / "out.png"
    result = _run_cli(
        str(source),
        "-o",
        str(dest),
        "--degrade",
        "freq-dct",
        "--degrade-strength",
        "0.4",
        "--degrade-seed",
        "3",
    )
    assert result.returncode == 0, result.stderr
    assert dest.is_file()
    assert "residual" not in result.stderr
    decode_png(dest.read_bytes())


def test_clean_file_morpho_cli_exits_zero(tmp_path: Path) -> None:
    source = _make_png(tmp_path / "in.png", channels=4)
    dest = tmp_path / "out.png"
    result = _run_cli(str(source), "-o", str(dest), "--morpho", "grid", "--degrade-seed", "3")
    assert result.returncode == 0, result.stderr
    assert dest.is_file()
    assert "residual" not in result.stderr
