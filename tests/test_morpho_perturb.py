"""Tests for morphological perturbation strategies and their CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from morpho_perturb import (
    combined_morpho,
    morpho_perturb,
)
from morphomod import Raster, decode_png, encode_png


def _raw(size: int = 8, channels: int = 4) -> bytes:
    pixel = [60, 120, 180]
    if channels == 4:
        pixel.append(255)
    return bytes(pixel * (size * size))


@pytest.mark.parametrize("strategy", ["grid", "diagonal", "noise", "quantize"])
def test_alpha_channel_never_modified(strategy: str) -> None:
    kwargs = {
        "grid": {"spacing": 2, "opacity": 0.5, "seed": 1},
        "diagonal": {"spacing": 2, "opacity": 0.5, "seed": 1},
        "noise": {"sigma": 10.0, "seed": 1},
        "quantize": {"levels": 8},
    }[strategy]
    result = morpho_perturb(_raw(), 8, 8, 4, strategy=strategy, **kwargs)
    assert result.data[3::4] == bytes([255] * (8 * 8))


def test_wrong_raw_length_rejected() -> None:
    with pytest.raises(ValueError, match="raw length"):
        morpho_perturb(b"\x00\x01\x02", 4, 4, 3, strategy="grid")


def test_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        morpho_perturb(bytes(12), 0, 4, 3, strategy="grid")


def test_invalid_channels_rejected() -> None:
    with pytest.raises(ValueError, match="channels"):
        morpho_perturb(bytes(8 * 8 * 2), 8, 8, 2, strategy="grid")


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="unknown morphological strategy"):
        morpho_perturb(_raw(), 8, 8, 4, strategy="lava")


def test_unexpected_keyword_rejected() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        morpho_perturb(_raw(), 8, 8, 4, strategy="noise", opacity=0.1)


@pytest.mark.parametrize(
    ("strategy", "kwargs", "message"),
    [
        ("grid", {"spacing": 1}, "spacing"),
        ("grid", {"opacity": 1.5}, "opacity"),
        ("noise", {"sigma": -1.0}, "sigma"),
        ("quantize", {"levels": 1}, "levels"),
    ],
)
def test_parameter_validation(strategy: str, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        morpho_perturb(_raw(), 8, 8, 4, strategy=strategy, **kwargs)


def test_combined_morpho_deterministic_with_seed() -> None:
    first = combined_morpho(_raw(), 8, 8, 4, strategies=("grid", "noise"), seed=42)
    second = combined_morpho(_raw(), 8, 8, 4, strategies=("grid", "noise"), seed=42)
    assert bytes(first.data) == bytes(second.data)


def test_combined_morpho_seed_stays_deterministic_across_quantize() -> None:
    strategies = ("grid", "quantize", "noise")
    first = combined_morpho(_raw(), 8, 8, 4, strategies=strategies, seed=7)
    second = combined_morpho(_raw(), 8, 8, 4, strategies=strategies, seed=7)
    assert bytes(first.data) == bytes(second.data)


def test_combined_morpho_strategy_string() -> None:
    result = combined_morpho(_raw(), 8, 8, 4, strategies=("grid", "noise"), seed=1)
    assert result.strategy == "grid+noise"


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "morpho_perturb.py"), *argv],
        capture_output=True,
        text=True,
    )


def _make_png(path: Path) -> Path:
    data = bytearray()
    for _ in range(16 * 16):
        data += bytes([200, 100, 50, 255])
    path.write_bytes(encode_png(Raster(16, 16, 4, data)))
    return path


def test_cli_writes_output_and_json_report(tmp_path: Path) -> None:
    source = _make_png(tmp_path / "in.png")
    dest = tmp_path / "out.png"
    result = _run_cli(str(source), "-o", str(dest), "--strategy", "grid", "--seed", "3", "--json")
    assert result.returncode == 0, result.stderr
    assert dest.is_file()
    raster = decode_png(dest.read_bytes())
    assert raster.channels == 4
    assert raster.data[3::4] == bytes([255] * (16 * 16))
    assert '"strategy": "grid"' in result.stdout


def test_cli_error_path_reports_and_exits_one(tmp_path: Path) -> None:
    result = _run_cli(str(tmp_path / "missing.png"))
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "error:" in result.stderr
