"""Tests for JPEG input support in morphomod."""

from __future__ import annotations

import io
import shlex
import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image")
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import morphomod
import optional_deps
from morphomod import (
    Raster,
    VisiblePlan,
    box_mask,
    decode_jpeg,
    decode_png,
    decode_to_raster,
    encode_png,
    remove_visible,
    write_pgm,
)


def _make_sample_jpeg_bytes(
    mode: str = "RGB",
    size: tuple[int, int] = (4, 4),
    exif: dict | None = None,
    icc_profile: bytes | None = None,
) -> bytes:
    if mode == "RGB":
        data = bytes(
            [
                (x * 50 + y * 20) % 240 + 10
                for y in range(size[1])
                for x in range(size[0])
                for _ in range(3)
            ]
        )
        im = Image.frombytes("RGB", size, data)
    elif mode == "L":
        data = bytes([(x * 50 + y * 20) % 240 + 10 for y in range(size[1]) for x in range(size[0])])
        im = Image.frombytes("L", size, data)
    elif mode == "CMYK":
        data = bytes(
            [
                (x * 40 + y * 15) % 200 + 20
                for y in range(size[1])
                for x in range(size[0])
                for _ in range(4)
            ]
        )
        im = Image.frombytes("CMYK", size, data)
    else:
        raise ValueError(f"unsupported test mode: {mode}")

    buf = io.BytesIO()
    save_kwargs: dict = {"format": "JPEG", "quality": 95}
    if exif:
        exif_obj = im.getexif()
        for k, v in exif.items():
            exif_obj[k] = v
        save_kwargs["exif"] = exif_obj
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    im.save(buf, **save_kwargs)
    return buf.getvalue()


def test_decode_jpeg_roundtrip_rgb_and_l():
    rgb_bytes = _make_sample_jpeg_bytes("RGB", (16, 16))
    raster_rgb = decode_jpeg(rgb_bytes)
    assert raster_rgb.width == 16
    assert raster_rgb.height == 16
    assert raster_rgb.channels == 3
    pil_rgb = Image.open(io.BytesIO(rgb_bytes))
    pil_bytes = pil_rgb.tobytes()
    for b1, b2 in zip(raster_rgb.data, pil_bytes, strict=True):
        assert abs(b1 - b2) == 0

    l_bytes = _make_sample_jpeg_bytes("L", (8, 8))
    raster_l = decode_jpeg(l_bytes)
    assert raster_l.width == 8
    assert raster_l.height == 8
    assert raster_l.channels == 1


def test_decode_jpeg_exif_orientation():
    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (30, 20), exif={0x0112: 6})
    raster = decode_jpeg(jpeg_bytes)
    assert (raster.width, raster.height) == (20, 30)
    assert raster.channels == 3


def test_decode_jpeg_cmyk_converts_to_rgb_and_is_deterministic():
    cmyk_bytes_no_icc = _make_sample_jpeg_bytes("CMYK", (8, 8))
    raster_no_icc_1 = decode_jpeg(cmyk_bytes_no_icc)
    raster_no_icc_2 = decode_jpeg(cmyk_bytes_no_icc)
    assert raster_no_icc_1.channels == 3
    assert raster_no_icc_1.width == 8
    assert raster_no_icc_1.height == 8
    assert raster_no_icc_1.data == raster_no_icc_2.data

    srgb_prof = ImageCms.createProfile("sRGB")
    icc_bytes = ImageCms.ImageCmsProfile(srgb_prof).tobytes()
    cmyk_bytes_icc = _make_sample_jpeg_bytes("CMYK", (8, 8), icc_profile=icc_bytes)
    raster_icc = decode_jpeg(cmyk_bytes_icc)
    assert raster_icc.channels == 3
    assert raster_icc.width == 8
    assert raster_icc.height == 8


def test_png_roundtrip_preservation():
    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (8, 8))
    raster = decode_jpeg(jpeg_bytes)
    mask = box_mask(8, 8, (2, 2, 4, 4))

    inpainted = morphomod.simple_inpaint(raster, mask)
    restored = morphomod.composite(raster, inpainted, mask)
    encoded = encode_png(restored)
    decoded = decode_png(encoded)

    assert (decoded.width, decoded.height, decoded.channels) == (8, 8, 3)
    for i in range(8 * 8):
        if not mask.data[i]:
            orig = bytes(raster.data[i * 3 : (i + 1) * 3])
            after_roundtrip = bytes(decoded.data[i * 3 : (i + 1) * 3])
            assert after_roundtrip == orig


def test_rgba_alpha_preserved_through_texture_backend(tmp_path: Path):
    data = bytearray()
    for y in range(16):
        for x in range(16):
            data.extend([x * 10, y * 10, 100, (x + y) * 8])
    orig_raster = Raster(16, 16, 4, data)
    src = tmp_path / "rgba_input.png"
    src.write_bytes(encode_png(orig_raster))

    mask = box_mask(16, 16, (4, 4, 4, 4))
    mask_path = tmp_path / "mask.pgm"
    write_pgm(mask, mask_path)

    dest = tmp_path / "rgba_output.png"
    report = remove_visible(
        src,
        dest,
        VisiblePlan(
            mask_path=mask_path,
            backend="texture",
            dilation_radius=0,
        ),
    )
    assert report["status"] == "completed"
    out_raster = decode_png(dest.read_bytes())
    assert out_raster.channels == 4
    for i in range(16 * 16):
        if not mask.data[i]:
            assert out_raster.data[i * 4 : (i + 1) * 4] == orig_raster.data[i * 4 : (i + 1) * 4]


def test_backend_output_shape_mismatch_raises(tmp_path: Path):
    src = tmp_path / "input.png"
    src.write_bytes(encode_png(Raster(4, 4, 3, bytearray([10, 20, 30] * 16))))
    mask_path = tmp_path / "mask.pgm"
    write_pgm(box_mask(4, 4, (1, 1, 1, 1)), mask_path)

    inpaint_script = tmp_path / "wrong_size.py"
    inpaint_script.write_text(
        "import sys\n"
        "from PIL import Image\n"
        "im = Image.new('RGB', (8, 8), (255, 255, 255))\n"
        "im.save(sys.argv[3], format='PNG')\n",
        encoding="utf-8",
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(inpaint_script))} "
        '"{input}" "{mask}" "{output}"'
    )
    dest = tmp_path / "output.png"
    with pytest.raises(ValueError, match=r"output shape .* does not match"):
        remove_visible(
            src,
            dest,
            VisiblePlan(
                mask_path=mask_path,
                backend="external",
                command=command,
                dilation_radius=0,
            ),
        )


def test_texture_backend_end_to_end_on_jpeg(tmp_path: Path):
    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (20, 20))
    src = tmp_path / "input.jpg"
    src.write_bytes(jpeg_bytes)
    dest = tmp_path / "output.png"

    orig_raster = decode_jpeg(jpeg_bytes)

    report = remove_visible(
        src,
        dest,
        VisiblePlan(
            box=(4, 4, 4, 4),
            backend="texture",
            dilation_radius=0,
        ),
    )
    assert report["status"] == "completed"
    assert dest.is_file()
    out_raster = decode_png(dest.read_bytes())
    assert (out_raster.width, out_raster.height, out_raster.channels) == (20, 20, 3)

    eff_mask = box_mask(20, 20, (4, 4, 4, 4))
    for i in range(20 * 20):
        if not eff_mask.data[i]:
            assert out_raster.data[i * 3 : (i + 1) * 3] == orig_raster.data[i * 3 : (i + 1) * 3]


def test_non_png_destination_rejected_for_jpeg(tmp_path: Path):
    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (10, 10))
    src = tmp_path / "input.jpg"
    src.write_bytes(jpeg_bytes)
    dest = tmp_path / "output.jpg"

    with pytest.raises(ValueError, match="PNG destination"):
        remove_visible(
            src,
            dest,
            VisiblePlan(
                box=(1, 1, 2, 2),
                backend="texture",
            ),
        )


def test_missing_pillow_raises_actionable_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(optional_deps, "PIL", None)
    with pytest.raises(ValueError) as exc_info:
        decode_jpeg(b"\xff\xd8\xff\xd9")
    msg = str(exc_info.value)
    assert "watermark-remover[visible]" in msg or "[visible]" in msg
    assert "ImportError" not in msg


def test_decode_to_raster_format_dispatch():
    png_bytes = encode_png(Raster(2, 2, 1, bytearray([0, 100, 200, 255])))
    r_png = decode_to_raster(png_bytes, "png")
    assert (r_png.width, r_png.height, r_png.channels) == (2, 2, 1)

    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (4, 4))
    r_jpeg = decode_to_raster(jpeg_bytes, "jpeg")
    assert (r_jpeg.width, r_jpeg.height, r_jpeg.channels) == (4, 4, 3)

    with pytest.raises(ValueError, match="unsupported format"):
        decode_to_raster(b"data", "gif")


def test_decode_jpeg_rejects_decompression_bomb(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(morphomod, "MAX_PIXELS", 50)
    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="safety limit"):
            decode_jpeg(jpeg_bytes)


def test_external_backend_composites_for_jpeg(tmp_path: Path) -> None:
    jpeg_bytes = _make_sample_jpeg_bytes("RGB", (4, 4))
    src = tmp_path / "input.jpg"
    src.write_bytes(jpeg_bytes)

    pil_orig = Image.open(io.BytesIO(jpeg_bytes))
    orig_bytes = pil_orig.tobytes()

    mask_path = tmp_path / "mask.pgm"
    mask = box_mask(4, 4, (1, 1, 1, 1))
    write_pgm(mask, mask_path)

    inpaint_script = tmp_path / "overwrite_all.py"
    inpaint_script.write_text(
        "import sys\n"
        "from PIL import Image\n"
        "im = Image.new('RGB', (4, 4), (255, 255, 255))\n"
        "im.save(sys.argv[3], format='PNG')\n",
        encoding="utf-8",
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(inpaint_script))} "
        '"{input}" "{mask}" "{output}"'
    )

    dest = tmp_path / "output.png"
    report = remove_visible(
        src,
        dest,
        VisiblePlan(
            mask_path=mask_path,
            backend="external",
            command=command,
            dilation_radius=0,
        ),
    )

    assert report["status"] == "completed"
    assert dest.exists()

    result_raster = decode_png(dest.read_bytes())
    assert (result_raster.width, result_raster.height, result_raster.channels) == (4, 4, 3)

    for i in range(4 * 4):
        if not mask.data[i]:
            orig_pixel = orig_bytes[i * 3 : (i + 1) * 3]
            res_pixel = bytes(result_raster.data[i * 3 : (i + 1) * 3])
            assert res_pixel == orig_pixel, (
                f"Pixel {i} outside mask was not restored! Got {res_pixel!r}, expected {orig_pixel!r}"
            )
