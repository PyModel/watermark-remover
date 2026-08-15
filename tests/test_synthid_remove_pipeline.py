"""End-to-end regression tests: SynthID removal wired into the pipeline (F1).

These prove the removal is not just a standalone library — it is reachable
from ``clean_image``, ``clean_asset``/``CleanPlan``, and both CLIs, and it
observably destroys a seeded SynthID-class signal.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/remove-ai-marks/scripts"
sys.path.insert(0, str(SCRIPTS))

from clean_asset import CleanPlan
from image_meta import clean_image
from morphomod import Raster, decode_png, encode_png
from synthid_remove import detect_synthid_pattern, embed_synthid_pattern


def _flat_rgb(width: int, height: int) -> bytearray:
    data = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            base = 120 + ((x * 7 + y * 3) % 40)
            data[idx] = base
            data[idx + 1] = base - 10
            data[idx + 2] = base - 20
    return data


def _embedded_png(tmp_path: Path, name: str = "in.png") -> Path:
    src = tmp_path / name
    raw = bytes(_flat_rgb(64, 48))
    embedded = embed_synthid_pattern(raw, 64, 48, 3, seed=42, strength=0.25)
    src.write_bytes(encode_png(Raster(64, 48, 3, embedded)))
    return src


def _detect(path: Path) -> float:
    raster = decode_png(path.read_bytes())
    det = detect_synthid_pattern(
        bytes(raster.data), raster.width, raster.height, raster.channels, seed=42
    )
    return det.confidence


class TestCleanImage:
    def test_remove_synthid_destroys_signal(self, tmp_path: Path):
        src = _embedded_png(tmp_path)
        assert _detect(src) >= 0.2
        dest = tmp_path / "out.png"
        result = clean_image(src, dest, remove_synthid=True, synthid_strength=0.6)
        assert result["synthid_removal"] is not None
        assert result["synthid_removal"]["strategy"] == "synthid-band-dct"
        assert _detect(dest) < 0.1

    def test_metadata_strip_alone_keeps_signal(self, tmp_path: Path):
        # Without removal, the metadata cleaner must leave the spectral carrier
        # intact — proving removal adds real value, not a no-op.
        src = _embedded_png(tmp_path)
        dest = tmp_path / "out.png"
        clean_image(src, dest, remove_synthid=False)
        assert _detect(dest) >= 0.2

    def test_remove_synthid_rejects_non_png(self, tmp_path: Path):
        # Minimal JPEG: SOI, APP0 JFIF, SOS stub, EOI.
        app0 = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        app0_seg = b"\xff\xe0" + struct.pack(">H", len(app0) + 2) + app0
        sos_payload = b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
        sos = b"\xff\xda" + struct.pack(">H", len(sos_payload) + 2) + sos_payload
        jpeg = b"\xff\xd8" + app0_seg + sos + b"\x00\x00" + b"\xff\xd9"
        src = tmp_path / "in.jpg"
        src.write_bytes(jpeg)
        dest = tmp_path / "out.png"
        with pytest.raises(ValueError, match="PNG"):
            clean_image(src, dest, remove_synthid=True)

    def test_remove_synthid_rejects_bad_strength(self, tmp_path: Path):
        src = _embedded_png(tmp_path)
        dest = tmp_path / "out.png"
        with pytest.raises(ValueError, match="strength"):
            clean_image(src, dest, remove_synthid=True, synthid_strength=1.5)


class TestCleanAsset:
    def test_clean_plan_remove_synthid_destroys_signal(self, tmp_path: Path):
        from clean_asset import clean_asset

        src = _embedded_png(tmp_path)
        assert _detect(src) >= 0.2
        dest = tmp_path / "out.png"
        result = clean_asset(
            src,
            dest,
            CleanPlan(forced_kind="image", remove_synthid=True, synthid_strength=0.6),
        )
        assert result.kind == "image"
        assert _detect(dest) < 0.1

    def test_clean_plan_validates_strength(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CleanPlan(forced_kind="image", remove_synthid=True, synthid_strength=2.0)


class TestCli:
    def test_clean_file_cli_remove_synthid_e2e(self, tmp_path: Path):
        src = _embedded_png(tmp_path)
        dest = tmp_path / "out.png"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "clean_file.py"),
                str(src),
                "-o",
                str(dest),
                "--remove-synthid",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert dest.is_file()
        assert _detect(dest) < 0.1

    def test_clean_image_cli_remove_synthid_e2e(self, tmp_path: Path):
        src = _embedded_png(tmp_path)
        dest = tmp_path / "out.png"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "clean_image.py"),
                str(src),
                "-o",
                str(dest),
                "--remove-synthid",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert dest.is_file()
        assert _detect(dest) < 0.1
