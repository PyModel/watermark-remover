"""Tests at the MorphoMod module interface and pure mask/raster seams."""

from __future__ import annotations

import json
import shlex
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"

import morphomod
from morphomod import (
    MAX_ENCODED_BYTES,
    MAX_PIXELS,
    Mask,
    Raster,
    VisiblePlan,
    box_mask,
    composite,
    decode_png,
    dilate,
    encode_png,
    fill_holes,
    read_pgm,
    remove_visible,
    simple_inpaint,
    texture_patch_inpaint,
    write_pgm,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "unknown"}, "unknown visible backend"),
        ({"backend": "texture"}, "localization source"),
        (
            {"mask_path": Path("mask.pgm"), "box": (0, 0, 1, 1)},
            "exactly one localization source",
        ),
        ({"box": (0, 0, 1, 1), "backend": "external"}, "command required"),
        (
            {
                "box": (0, 0, 1, 1),
                "backend": "texture",
                "command": "tool {input}",
            },
            "only valid for external",
        ),
        ({"box": (0, 0, 1, 1), "dilation_radius": -1}, "dilation"),
    ],
)
def test_visible_plan_rejects_invalid_mode_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VisiblePlan(**kwargs)


def test_visible_backend_requires_destination_before_writing_mask(tmp_path: Path):
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(2, 2, 3, bytearray([0, 0, 0] * 4))))
    mask_output = tmp_path / "mask.pgm"
    plan = VisiblePlan(
        box=(0, 0, 1, 1),
        backend="texture",
        mask_output=mask_output,
    )

    with pytest.raises(ValueError, match="output required"):
        remove_visible(source, None, plan)

    assert not mask_output.exists()


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


def test_decode_png_ignores_trailing_data_after_iend():
    raster = Raster(2, 1, 3, bytearray([10, 20, 30, 40, 50, 60]))
    encoded = encode_png(raster)

    assert decode_png(encoded + b"HIDDEN-PAYLOAD") == decode_png(encoded)


def test_remove_visible_rejects_oversized_encoded_input(tmp_path: Path, monkeypatch):
    src = tmp_path / "oversized.png"
    src.write_bytes(b"x" * 32)
    monkeypatch.setattr("morphomod.MAX_ENCODED_BYTES", 16)
    try:
        remove_visible(src, None, VisiblePlan())
    except ValueError as error:
        assert "encoded file exceeds safety limit" in str(error)
    else:
        raise AssertionError("expected encoded-size rejection")
    assert MAX_ENCODED_BYTES > 16


def test_mask_bounds_large_sparse_mask_does_not_build_marked_lists():
    # Regression: bounds calculation must remain one-pass/constant auxiliary memory.
    mask = box_mask(1000, 1000, (999, 999, 1, 1))
    result = dilate(mask, 0)
    assert result.marked == 1


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


def test_remove_visible_rejects_output_aliases_before_writing(tmp_path: Path):
    src = tmp_path / "input.png"
    original = encode_png(Raster(3, 3, 3, bytearray([1, 2, 3] * 9)))
    src.write_bytes(original)
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(3, 3, (1, 1, 1, 1)), mask)

    for destination, mask_output in ((src, None), (tmp_path / "out.png", src), (mask, None)):
        try:
            remove_visible(
                src,
                destination,
                VisiblePlan(
                    mask_path=mask,
                    mask_output=mask_output,
                    backend="simple",
                    dilation_radius=0,
                ),
            )
        except ValueError as error:
            assert "alias" in str(error)
        else:
            raise AssertionError("expected output alias rejection")
    assert src.read_bytes() == original


def test_texture_backend_fully_replaces_small_mark_after_default_dilation(tmp_path: Path):
    width = height = 32
    pixels = bytearray([30, 90, 30] * width * height)
    mark_x = mark_y = 24
    index = (mark_y * width + mark_x) * 3
    pixels[index : index + 3] = b"\xff\xff\xff"
    src = tmp_path / "small-mark.png"
    src.write_bytes(encode_png(Raster(width, height, 3, pixels)))
    dest = tmp_path / "small-mark.cleaned.png"

    remove_visible(
        src,
        dest,
        VisiblePlan(box=(mark_x, mark_y, 1, 1), backend="texture"),
    )

    cleaned = decode_png(dest.read_bytes())
    assert cleaned.data[index : index + 3] == bytearray((30, 90, 30))


def test_texture_patch_inpaint_preserves_outside_and_replaces_texture():
    width = height = 48
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            value = 70 + ((x * 11 + y * 7 + (x * y) % 13) % 55)
            pixels.extend((value // 2, value, value // 2, 255))
    raster = Raster(width, height, 4, pixels)
    mask = box_mask(width, height, (30, 30, 8, 8))
    for y in range(30, 38):
        for x in range(30, 38):
            index = (y * width + x) * 4
            raster.data[index : index + 4] = b"\xff\xff\xff\xff"

    result, match = texture_patch_inpaint(raster, mask, feather=2)

    assert match.width == 8 and match.height == 8
    assert result.data[: 30 * width * 4] == raster.data[: 30 * width * 4]
    center = (34 * width + 34) * 4
    assert result.data[center : center + 3] != b"\xff\xff\xff"
    assert result.data[center + 3] == 255


def test_texture_large_mask_raises_actionable_error():
    # Mask spans most of the image: no same-size source patch can fit in any
    # margin, so the search region clamps below the mask size and every
    # candidate would "overlap".  The feasibility check must fail fast with
    # guidance instead of scanning and raising a dead-end error.
    width, height = 48, 32
    raster = Raster(width, height, 3, bytearray([100, 110, 120] * width * height))
    mask = box_mask(width, height, (10, 10, 40, 24))
    with pytest.raises(ValueError, match="cannot be placed non-overlapping"):
        morphomod._find_texture_match(raster, mask, 0)
    with pytest.raises(ValueError, match="--backend simple or external"):
        morphomod._find_texture_match(raster, mask, 0)


def test_texture_match_boundary_feasibility():
    width, height = 48, 32
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            value = 60 + ((x * 11 + y * 7) % 80)
            pixels.extend((value, value, value))
    raster = Raster(width, height, 3, pixels)
    # 22-wide mask at (0,0): right margin 22+2+22=46 <= 48 → feasible.
    match = morphomod._find_texture_match(raster, box_mask(width, height, (0, 0, 22, 20)), 0)
    assert match.width == 22 and match.height == 20
    # 24-wide mask at (0,0): right margin 24+2+24=50 > 48 → infeasible.
    with pytest.raises(ValueError, match="cannot be placed non-overlapping"):
        morphomod._find_texture_match(raster, box_mask(width, height, (0, 0, 24, 20)), 0)


def test_texture_match_reaches_margin_beyond_radius_window():
    # gap = feather = 254 pushes the only valid right-margin placement to
    # x = 40 + 254 = 294, beyond the old radius window (max(256, 6*40)=256)
    # which raised "no non-overlapping texture patch candidate found".
    width, height = 334, 100
    raster = Raster(width, height, 3, bytearray([100] * width * height * 3))
    mask = box_mask(width, height, (0, 0, 40, 40))
    match = morphomod._find_texture_match(raster, mask, 254)
    assert match.width == 40 and match.height == 40
    assert match.x == 294
    assert match.y == 0


def test_remove_visible_texture_pipeline(tmp_path: Path):
    width = height = 48
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            value = 60 + ((x * 13 + y * 5) % 80)
            pixels.extend((value // 2, value, value // 2))
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(width, height, 3, pixels)))
    dest = tmp_path / "cleaned.png"
    report = remove_visible(
        src,
        dest,
        VisiblePlan(
            box=(30, 30, 8, 8),
            backend="texture",
            dilation_radius=2,
        ),
    )
    assert report["status"] == "completed"
    assert any("texture-patch" in action for action in report["actions"])
    assert dest.is_file()


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
        VisiblePlan(
            mask_path=mask,
            backend="simple",
            dilation_radius=1,
        ),
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
    report = remove_visible(src, None, VisiblePlan())
    assert report["status"] == "plan-only"
    assert report["output"] is None
    assert "No blind segmenter" in report["note"]


def test_external_command_failure_caps_diagnostics(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(morphomod, "MAX_COMMAND_DIAGNOSTIC_BYTES", 32)
    noisy = tmp_path / "noisy.py"
    noisy.write_text(
        "import sys\nsys.stdout.write('prefix-' + 'x' * 128 + '-tail')\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    try:
        morphomod._run_template(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(noisy))}",
            timeout=5,
        )
    except RuntimeError as error:
        message = str(error)
        assert "failed (7)" in message
        assert message.endswith("-tail")
        assert "prefix-" not in message
        assert len(message) < 100
    else:
        raise AssertionError("expected external command failure")


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
        VisiblePlan(
            mask_path=mask,
            backend="external",
            command=command,
            dilation_radius=0,
        ),
    )
    assert report["status"] == "completed"
    assert decode_png(dest.read_bytes()) == decode_png(original)


def test_external_adapter_rejects_oversized_png_output(tmp_path: Path, monkeypatch):
    src = tmp_path / "input.png"
    original = encode_png(Raster(3, 3, 3, bytearray([10, 20, 30] * 9)))
    src.write_bytes(original)
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(3, 3, (1, 1, 1, 1)), mask)
    limit = max(len(original), mask.stat().st_size) + 16
    monkeypatch.setattr("morphomod.MAX_ENCODED_BYTES", limit)

    writer = tmp_path / "oversized.py"
    writer.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"Path(sys.argv[3]).write_bytes(b'x' * {limit + 1})\n",
        encoding="utf-8",
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(writer))} "
        '"{input}" "{mask}" "{output}"'
    )
    dest = tmp_path / "external.png"

    try:
        remove_visible(
            src,
            dest,
            VisiblePlan(
                mask_path=mask,
                backend="external",
                command=command,
                dilation_radius=0,
            ),
        )
    except ValueError as error:
        assert "encoded file exceeds safety limit" in str(error)
    else:
        raise AssertionError("expected oversized external-output rejection")
    assert not dest.exists()


def test_clean_file_in_place_batch_preflights_generated_masks(tmp_path: Path):
    first = tmp_path / "first.png"
    first.write_bytes(encode_png(Raster(3, 3, 3, bytearray([9, 8, 7] * 9))))
    colliding = tmp_path / "first.mask.pgm"
    colliding.write_bytes(b"do not overwrite")
    original = colliding.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(first),
            str(colliding),
            "--in-place",
            "--detect-command",
            "detector {input} {mask}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "mask output aliases an input" in result.stderr
    assert colliding.read_bytes() == original
    assert not first.with_suffix(".png.bak").exists()


def test_clean_file_rejects_output_aliasing_visible_mask(tmp_path: Path):
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(3, 3, 3, bytearray([9, 8, 7] * 9))))
    mask = tmp_path / "mask.pgm"
    write_pgm(box_mask(3, 3, (1, 1, 1, 1)), mask)
    original_mask = mask.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(src),
            "-o",
            str(mask),
            "--visible-mask",
            str(mask),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert mask.read_bytes() == original_mask


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
        check=False,
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
        check=False,
    )
    assert r.returncode == 1
    assert "requires a localization source" in r.stderr


@pytest.mark.parametrize(
    "options",
    [
        ("--inpaint-command", "tool {input} {mask} {output}"),
        ("--visible-backend", "external"),
    ],
)
def test_clean_file_rejects_visible_options_without_localization_before_writing(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(3, 3, 3, bytearray([9, 8, 7] * 9))))
    destination = tmp_path / "output.png"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(source),
            "-o",
            str(destination),
            *options,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requires a localization source" in result.stderr
    assert not destination.exists()


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
        check=False,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["status"] == "completed"
    assert dest.is_file()


def test_clean_file_frictionless_no_mask_artifact_by_default(tmp_path: Path):
    # F2: automated/frictionless — visible cleaning publishes no .mask.pgm
    # unless --keep-artifacts is requested.
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(16, 16, 3, bytearray([9, 8, 7] * 256))))
    dest = tmp_path / "cleaned.png"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(src),
            "-o",
            str(dest),
            "--visible-box",
            "4,4,4,4",
            "--dilate",
            "0",
            "--visible-backend",
            "simple",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["visible"]["status"] == "completed"
    assert dest.is_file()
    # Frictionless: no published mask artifact next to the output.
    assert not dest.with_name("cleaned.mask.pgm").exists()
    assert report["visible"]["mask"] is None

    # Same run with --keep-artifacts publishes the mask.
    dest2 = tmp_path / "kept.png"
    r2 = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(src),
            "-o",
            str(dest2),
            "--visible-box",
            "4,4,4,4",
            "--dilate",
            "0",
            "--visible-backend",
            "simple",
            "--keep-artifacts",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0, r2.stderr
    assert dest2.with_name("kept.mask.pgm").is_file()
    assert json.loads(r2.stdout)["visible"]["mask"] is not None


def test_remove_visible_respects_publish_mask_direct(tmp_path: Path):
    from morphomod import VisiblePlan, remove_visible

    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(16, 16, 3, bytearray([9, 8, 7] * 256))))
    dest = tmp_path / "out.png"
    plan = VisiblePlan(
        box=(4, 4, 4, 4),
        backend="simple",
        dilation_radius=0,
        mask_output=tmp_path / "out.mask.pgm",
        publish_mask=False,
    )
    report = remove_visible(src, dest, plan)
    assert report["status"] == "completed"
    assert dest.is_file()
    assert not (tmp_path / "out.mask.pgm").exists()
    assert report["mask"] is None
