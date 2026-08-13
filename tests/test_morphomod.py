"""Tests at the MorphoMod module interface and pure mask/raster seams."""

from __future__ import annotations

import json
import shlex
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from morphomod import (
    MAX_PIXELS,
    Mask,
    Raster,
    box_mask,
    composite,
    decode_png,
    dilate,
    encode_png,
    fill_holes,
    read_pgm,
    remove_visible,
    simple_inpaint,
    write_pgm,
)


def test_dilation_is_bounded_not_cascading():
    data = bytearray(7 * 7)
    data[3 * 7 + 3] = 255
    mask = Mask(7, 7, data)
    d1 = dilate(mask, 1)
    d2 = dilate(mask, 2)
    assert d1.marked == 9
    assert d2.marked == 25
    assert d1.data[0] == 0  # no accidental flood-fill to image edge


def test_dilation_clips_at_border():
    mask = box_mask(5, 5, (0, 0, 1, 1))
    assert dilate(mask, 1).marked == 4


def test_fill_holes():
    data = bytearray(5 * 5)
    for x, y in (
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ):
        data[y * 5 + x] = 255
    filled = fill_holes(Mask(5, 5, data))
    assert filled.marked == 9
    assert filled.data[2 * 5 + 2] == 255
    assert filled.data[0] == 0


def test_pgm_roundtrip_even_if_first_pixel_is_whitespace_byte(tmp_path: Path):
    # Writer emits binary 0/255; also verify parser doesn't consume binary bytes.
    mask = box_mask(4, 3, (1, 1, 2, 1))
    path = tmp_path / "mask.pgm"
    write_pgm(mask, path)
    assert read_pgm(path) == mask


def test_png_roundtrip_rgb_and_rgba():
    for channels in (1, 3, 4):
        pixels = bytearray((i * 17) % 256 for i in range(3 * 2 * channels))
        raster = Raster(3, 2, channels, pixels)
        decoded = decode_png(encode_png(raster))
        assert decoded == raster


def test_png_decoder_rejects_oversized_dimensions_before_decompression():
    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", MAX_PIXELS + 1, 1, 8, 2, 0, 0, 0)
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )
    try:
        decode_png(raw)
    except ValueError as error:
        assert "safety limit" in str(error)
    else:
        raise AssertionError("expected safety-limit rejection")


def test_simple_inpaint_and_composite_restore_outside_mask():
    # Blue image with a red 3x3 watermark; mask covers that region.
    pixels = bytearray()
    for y in range(5):
        for x in range(5):
            pixels.extend((255, 0, 0) if 1 <= x <= 3 and 1 <= y <= 3 else (0, 0, 255))
    original = Raster(5, 5, 3, pixels)
    mask = box_mask(5, 5, (1, 1, 3, 3))
    filled = simple_inpaint(original, mask)
    restored = composite(original, filled, mask)
    center = (2 * 5 + 2) * 3
    assert restored.data[center : center + 3] == bytearray((0, 0, 255))
    assert restored.data[0:3] == original.data[0:3]  # outside restored exactly


def test_remove_visible_simple_pipeline(tmp_path: Path):
    pixels = bytearray([20, 40, 60] * 25)
    # Distinct center mark
    pixels[(2 * 5 + 2) * 3 : (2 * 5 + 2) * 3 + 3] = b"\xff\xff\xff"
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(5, 5, 3, pixels)))
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(5, 5, (2, 2, 1, 1)), mask)
    dest = tmp_path / "cleaned.png"
    report = remove_visible(
        src,
        dest,
        mask_path=mask,
        backend="simple",
        dilation_radius=1,
    )
    assert report["status"] == "completed"
    assert report["initial_mask_pixels"] == 1
    assert report["refined_mask_pixels"] == 9
    assert dest.is_file()
    out = decode_png(dest.read_bytes())
    center = (2 * 5 + 2) * 3
    assert out.data[center : center + 3] == bytearray((20, 40, 60))
    assert Path(report["mask"]).is_file()


def test_remove_visible_plan_does_not_claim_removal(tmp_path: Path):
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(2, 2, 3, bytearray([0, 0, 0] * 4))))
    report = remove_visible(src, None)
    assert report["status"] == "plan-only"
    assert report["output"] is None
    assert "No blind segmenter" in report["note"]


def test_external_adapter_and_restore(tmp_path: Path):
    src = tmp_path / "input.png"
    original = encode_png(Raster(3, 3, 3, bytearray([10, 20, 30] * 9)))
    src.write_bytes(original)
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(3, 3, (1, 1, 1, 1)), mask)
    copier = tmp_path / "copy.py"
    copier.write_text(
        "import shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[3])\n",
        encoding="utf-8",
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(copier))} "
        '"{input}" "{mask}" "{output}"'
    )
    dest = tmp_path / "external.png"
    report = remove_visible(
        src,
        dest,
        mask_path=mask,
        backend="external",
        command=command,
        dilation_radius=0,
    )
    assert report["status"] == "completed"
    assert decode_png(dest.read_bytes()) == decode_png(original)


def test_clean_file_visible_pipeline(tmp_path: Path):
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(3, 3, 3, bytearray([9, 8, 7] * 9))))
    dest = tmp_path / "cleaned.png"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(src),
            "-o",
            str(dest),
            "--visible-box",
            "1,1,1,1",
            "--dilate",
            "0",
            "--visible-backend",
            "simple",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["visible"]["status"] == "completed"
    assert dest.is_file()


def test_clean_file_refuses_dilation_without_mask_source(tmp_path: Path):
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(3, 3, 3, bytearray([9, 8, 7] * 9))))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_file.py"), str(src), "--dilate", "3"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert "requires --visible-mask" in r.stderr


def test_morphomod_cli(tmp_path: Path):
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(3, 3, 3, bytearray([9, 8, 7] * 9))))
    dest = tmp_path / "out.png"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "morphomod.py"),
            str(src),
            "-o",
            str(dest),
            "--box",
            "1,1,1,1",
            "--dilation",
            "0",
            "--backend",
            "simple",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["status"] == "completed"
    assert dest.is_file()
