#!/usr/bin/env python3
"""MorphoMod-inspired visible watermark removal.

Pipeline: mask → hole-fill/refine → morphological dilation (d=3 default) →
inpaint → restore/composite.

The stdlib core provides:
  - correct non-cascading binary dilation (O(width*height), sliding window)
  - PGM/PNG mask I/O
  - PNG 8-bit gray/RGB/RGBA decode + encode (non-interlaced)
  - a simple nearest-boundary inpaint fallback
  - external detector/inpainter adapters for SAM/LaMa/MI-GAN/diffusion tools

No U-Net or LaMa weights are bundled. For production quality, pass
--detect-command and/or --command. Paper-reported gains are not claimed as this
implementation's results; inspect the generated mask and output yourself.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import eprint
from image_meta import detect_format

DEFAULT_DILATION_RADIUS = 3
MAX_PIXELS = 40_000_000  # bounds decompression/allocation; covers 8K UHD
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if width * height > MAX_PIXELS:
        raise ValueError(f"image exceeds safety limit of {MAX_PIXELS:,} pixels")


@dataclass
class Mask:
    width: int
    height: int
    data: bytearray  # row-major; 0=keep, 255=remove

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        if len(self.data) != self.width * self.height:
            raise ValueError("mask data length does not match dimensions")

    @property
    def marked(self) -> int:
        return sum(v != 0 for v in self.data)


@dataclass
class Raster:
    width: int
    height: int
    channels: int  # 1, 3, or 4
    data: bytearray

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        if self.channels not in (1, 3, 4):
            raise ValueError("supported channels: 1, 3, 4")
        if len(self.data) != self.width * self.height * self.channels:
            raise ValueError("raster data length does not match dimensions")


# ---------------------------------------------------------------------------
# Mask operations
# ---------------------------------------------------------------------------


def box_mask(width: int, height: int, box: tuple[int, int, int, int]) -> Mask:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError("box width/height must be positive")
    _validate_dimensions(width, height)
    data = bytearray(width * height)
    for yy in range(max(0, y), min(height, y + h)):
        start = yy * width + max(0, x)
        end = yy * width + min(width, x + w)
        data[start:end] = b"\xff" * max(0, end - start)
    return Mask(width, height, data)


def dilate(mask: Mask, radius: int = DEFAULT_DILATION_RADIUS) -> Mask:
    """Square-kernel binary dilation, computed from the original mask only.

    Two sliding-window max passes avoid the cascading/flood-fill bug common in
    naive in-place implementations and run in O(width*height).
    """
    if radius < 0:
        raise ValueError("dilation radius must be >= 0")
    if radius == 0:
        return Mask(mask.width, mask.height, bytearray(mask.data))
    w, h = mask.width, mask.height
    tmp = bytearray(w * h)
    out = bytearray(w * h)

    # horizontal pass
    for y in range(h):
        row = y * w
        count = sum(mask.data[row + x] != 0 for x in range(min(w, radius + 1)))
        for x in range(w):
            tmp[row + x] = 255 if count else 0
            remove = x - radius
            add = x + radius + 1
            if remove >= 0:
                count -= mask.data[row + remove] != 0
            if add < w:
                count += mask.data[row + add] != 0

    # vertical pass
    for x in range(w):
        count = sum(tmp[y * w + x] != 0 for y in range(min(h, radius + 1)))
        for y in range(h):
            out[y * w + x] = 255 if count else 0
            remove = y - radius
            add = y + radius + 1
            if remove >= 0:
                count -= tmp[remove * w + x] != 0
            if add < h:
                count += tmp[add * w + x] != 0
    return Mask(w, h, out)


def fill_holes(mask: Mask) -> Mask:
    """Fill zero-valued regions not connected to the image border."""
    w, h = mask.width, mask.height
    seen = bytearray(w * h)
    q: deque[int] = deque()

    def seed(x: int, y: int) -> None:
        i = y * w + x
        if not mask.data[i] and not seen[i]:
            seen[i] = 1
            q.append(i)

    for x in range(w):
        seed(x, 0)
        seed(x, h - 1)
    for y in range(h):
        seed(0, y)
        seed(w - 1, y)
    while q:
        i = q.popleft()
        x, y = i % w, i // w
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                ni = ny * w + nx
                if not mask.data[ni] and not seen[ni]:
                    seen[ni] = 1
                    q.append(ni)
    out = bytearray(mask.data)
    for i, value in enumerate(out):
        if not value and not seen[i]:
            out[i] = 255
    return Mask(w, h, out)


def refine_mask(mask: Mask, radius: int = DEFAULT_DILATION_RADIUS) -> Mask:
    return dilate(fill_holes(mask), radius)


# ---------------------------------------------------------------------------
# PGM + PNG I/O
# ---------------------------------------------------------------------------


def read_pgm(path: Path) -> Mask:
    raw = path.read_bytes()
    if not raw.startswith(b"P5"):
        raise ValueError("only binary PGM (P5) masks are supported")
    pos = 2
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while pos < len(raw) and raw[pos] in b" \t\r\n":
            pos += 1
        if pos < len(raw) and raw[pos] == ord("#"):
            pos = raw.find(b"\n", pos)
            if pos < 0:
                raise ValueError("truncated PGM comment")
            continue
        end = pos
        while end < len(raw) and raw[end] not in b" \t\r\n":
            end += 1
        tokens.append(raw[pos:end])
        pos = end
    width, height, maxval = map(int, tokens)
    _validate_dimensions(width, height)
    if maxval != 255:
        raise ValueError("PGM max value must be 255")
    # PGM requires one whitespace delimiter after maxval. Consume exactly one
    # delimiter (or CRLF), because the first binary pixel may itself equal a
    # whitespace byte.
    if raw[pos : pos + 2] == b"\r\n":
        pos += 2
    elif pos < len(raw) and raw[pos] in b" \t\r\n":
        pos += 1
    else:
        raise ValueError("missing PGM pixel delimiter")
    pixels = raw[pos : pos + width * height]
    if len(pixels) != width * height:
        raise ValueError("truncated PGM pixels")
    return Mask(width, height, bytearray(255 if p >= 128 else 0 for p in pixels))


def write_pgm(mask: Mask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{mask.width} {mask.height}\n255\n".encode() + bytes(mask.data))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def decode_png(data: bytes) -> Raster:
    if not data.startswith(PNG_SIG):
        raise ValueError("not PNG")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if pos + 12 + length > len(data):
            raise ValueError("truncated PNG chunk")
        stored_crc = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])[0]
        actual_crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError(f"PNG CRC mismatch in {kind!r}")
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
        pos += 12 + length
    channels = {0: 1, 2: 3, 6: 4}.get(color_type)
    if bit_depth != 8 or channels is None or interlace != 0:
        raise ValueError("PNG must be non-interlaced 8-bit gray/RGB/RGBA")
    _validate_dimensions(width, height)
    stride = width * channels
    expected = height * (stride + 1)
    inflater = zlib.decompressobj()
    raw = inflater.decompress(bytes(idat), expected + 1)
    if len(raw) > expected or not inflater.eof or inflater.unused_data:
        raise ValueError("PNG compressed stream exceeds expected size or is malformed")
    if len(raw) != expected:
        raise ValueError(f"unexpected PNG scan size {len(raw)} != {expected}")
    out = bytearray(width * height * channels)
    prev = bytearray(stride)
    p = 0
    for y in range(height):
        filt = raw[p]
        row = bytearray(raw[p + 1 : p + 1 + stride])
        p += stride + 1
        for i in range(stride):
            left = row[i - channels] if i >= channels else 0
            up = prev[i]
            upper_left = prev[i - channels] if i >= channels else 0
            if filt == 1:
                row[i] = (row[i] + left) & 255
            elif filt == 2:
                row[i] = (row[i] + up) & 255
            elif filt == 3:
                row[i] = (row[i] + ((left + up) // 2)) & 255
            elif filt == 4:
                row[i] = (row[i] + _paeth(left, up, upper_left)) & 255
            elif filt != 0:
                raise ValueError(f"unsupported PNG filter {filt}")
        out[y * stride : (y + 1) * stride] = row
        prev = row
    return Raster(width, height, channels, out)


def encode_png(raster: Raster) -> bytes:
    color_type = {1: 0, 3: 2, 4: 6}[raster.channels]
    ihdr = struct.pack(">IIBBBBB", raster.width, raster.height, 8, color_type, 0, 0, 0)
    stride = raster.width * raster.channels
    rows = bytearray()
    for y in range(raster.height):
        rows.append(0)  # filter None
        rows.extend(raster.data[y * stride : (y + 1) * stride])
    return (
        PNG_SIG
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _png_chunk(b"IEND", b"")
    )


def load_mask(path: Path) -> Mask:
    if path.suffix.lower() == ".pgm":
        return read_pgm(path)
    raster = decode_png(path.read_bytes())
    data = bytearray(raster.width * raster.height)
    for i in range(raster.width * raster.height):
        start = i * raster.channels
        values = raster.data[start : start + min(raster.channels, 3)]
        data[i] = 255 if max(values) >= 128 else 0
    return Mask(raster.width, raster.height, data)


# ---------------------------------------------------------------------------
# Inpainting and restore
# ---------------------------------------------------------------------------


def simple_inpaint(raster: Raster, mask: Mask) -> Raster:
    """Nearest-boundary wavefront fill. Useful fallback, not LaMa-quality."""
    if (raster.width, raster.height) != (mask.width, mask.height):
        raise ValueError("mask/image dimensions differ")
    if mask.marked == raster.width * raster.height:
        raise ValueError("cannot inpaint a mask covering the entire image")
    w, h, ch = raster.width, raster.height, raster.channels
    data = bytearray(raster.data)
    resolved = bytearray(0 if mask.data[i] else 1 for i in range(w * h))
    queued = bytearray(w * h)
    q: deque[int] = deque()

    def neighbors(i: int):
        x, y = i % w, i // w
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if nx != x or ny != y:
                    yield ny * w + nx

    for i in range(w * h):
        if mask.data[i] and any(resolved[n] for n in neighbors(i)):
            queued[i] = 1
            q.append(i)
    while q:
        i = q.popleft()
        if resolved[i]:
            continue
        ns = [n for n in neighbors(i) if resolved[n]]
        if not ns:
            continue
        for c in range(ch):
            data[i * ch + c] = sum(data[n * ch + c] for n in ns) // len(ns)
        resolved[i] = 1
        for n in neighbors(i):
            if mask.data[n] and not resolved[n] and not queued[n]:
                queued[n] = 1
                q.append(n)
    if any(mask.data[i] and not resolved[i] for i in range(w * h)):
        raise RuntimeError("inpaint could not resolve all masked pixels")
    return Raster(w, h, ch, data)


def composite(original: Raster, inpainted: Raster, mask: Mask) -> Raster:
    if (
        original.width != inpainted.width
        or original.height != inpainted.height
        or original.channels != inpainted.channels
        or (mask.width, mask.height) != (original.width, original.height)
    ):
        raise ValueError("original/inpainted/mask dimensions differ")
    out = bytearray(original.data)
    ch = original.channels
    for i, marked in enumerate(mask.data):
        if marked:
            out[i * ch : (i + 1) * ch] = inpainted.data[i * ch : (i + 1) * ch]
    return Raster(original.width, original.height, ch, out)


def _run_template(template: str, **values: str) -> None:
    command = template.format(**values)
    proc = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(
            f"external command failed ({proc.returncode}): {(proc.stderr or proc.stdout)[:1000]}"
        )


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    i = 2
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            break
        length = struct.unpack(">H", data[i : i + 2])[0]
        if marker in range(0xC0, 0xC4) and i + 7 <= len(data):
            return struct.unpack(">H", data[i + 5 : i + 7])[0], struct.unpack(
                ">H", data[i + 3 : i + 5]
            )[0]
        i += length
    return None


def remove_visible(
    path: Path,
    dest: Path | None,
    *,
    mask_path: Path | None = None,
    box: tuple[int, int, int, int] | None = None,
    detect_command: str | None = None,
    backend: str = "print-plan",
    command: str | None = None,
    dilation_radius: int = DEFAULT_DILATION_RADIUS,
    mask_output: Path | None = None,
    prompt: str = "Remove watermark, fill with background",
) -> dict[str, Any]:
    data = path.read_bytes()
    fmt = detect_format(data)
    raster = decode_png(data) if fmt == "png" else None
    dims = (raster.width, raster.height) if raster else _jpeg_dimensions(data)
    actions: list[str] = []

    if mask_path:
        initial = load_mask(mask_path)
        source = f"mask:{mask_path}"
    elif box:
        if not dims:
            raise ValueError("cannot derive dimensions for --box; provide --mask")
        initial = box_mask(*dims, box)
        source = f"box:{','.join(map(str, box))}"
    elif detect_command:
        with tempfile.TemporaryDirectory(prefix="wm-mask-") as td:
            detected = Path(td) / "detected.pgm"
            _run_template(
                detect_command,
                input=str(path),
                mask=str(detected),
                prompt=prompt,
            )
            if not detected.is_file():
                raise RuntimeError("detector command did not create {mask}")
            initial = load_mask(detected)
        source = "external-detector"
    else:
        return {
            "status": "plan-only",
            "input": str(path),
            "output": None,
            "format": fmt,
            "backend": backend,
            "actions": [
                "supply --mask, --box, or --detect-command",
                "then refine/fill holes, dilate d=3, inpaint, restore original outside mask",
            ],
            "note": "No blind segmenter is bundled; no image bytes were changed.",
        }

    if dims and (initial.width, initial.height) != dims:
        raise ValueError(f"mask dimensions {(initial.width, initial.height)} != image {dims}")
    refined = refine_mask(initial, dilation_radius)
    if mask_output is None:
        base = dest or path.with_name(f"{path.stem}.visible.cleaned{path.suffix}")
        mask_output = base.with_name(f"{base.stem}.mask.pgm")
    write_pgm(refined, mask_output)
    actions.extend(
        [
            f"mask source: {source}",
            f"fill holes + dilate radius={dilation_radius}: {initial.marked}->{refined.marked} pixels",
            f"wrote refined mask: {mask_output}",
        ]
    )

    if backend == "print-plan":
        status = "mask-ready"
        output = None
        actions.append("no inpainting run (print-plan backend)")
    elif backend == "simple":
        if raster is None:
            raise ValueError("simple backend supports PNG only; use --backend external")
        if dest is None:
            raise ValueError("-o/--output required for an inpainting backend")
        filled = simple_inpaint(raster, refined)
        restored = composite(raster, filled, refined)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(encode_png(restored))
        status, output = "completed", str(dest)
        actions.append("nearest-boundary inpaint + restore (stdlib fallback)")
    elif backend == "external":
        if not command:
            raise ValueError("--command required for external backend")
        if dest is None:
            raise ValueError("-o/--output required for external backend")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="wm-inpaint-") as td:
            external_out = Path(td) / ("inpainted.png" if fmt == "png" else path.name)
            _run_template(
                command,
                input=str(path),
                mask=str(mask_output),
                output=str(external_out),
                prompt=prompt,
            )
            if not external_out.is_file():
                raise RuntimeError("inpaint command did not create {output}")
            if raster is not None:
                inpainted = decode_png(external_out.read_bytes())
                dest.write_bytes(encode_png(composite(raster, inpainted, refined)))
                actions.append("external inpaint + stdlib restore outside mask")
            else:
                shutil.copyfile(external_out, dest)
                actions.append("external backend output copied (backend owns JPEG compositing)")
        status, output = "completed", str(dest)
    else:
        raise ValueError(f"unknown backend: {backend}")

    return {
        "status": status,
        "input": str(path),
        "output": output,
        "format": fmt,
        "backend": backend,
        "mask": str(mask_output),
        "initial_mask_pixels": initial.marked,
        "refined_mask_pixels": refined.marked,
        "dilation_radius": dilation_radius,
        "actions": actions,
        "note": (
            "MorphoMod-inspired pipeline; CVPR paper metrics are not this run's metrics. "
            "Inspect output fidelity and residual marks manually."
        ),
    }


def _parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(v) for v in value.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError("box must be x,y,w,h") from e
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x,y,w,h")
    return parts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("-o", "--output", type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--mask", type=Path, help="Binary PGM or PNG mask (white=remove)")
    source.add_argument("--box", type=_parse_box, help="Manual x,y,w,h mask")
    source.add_argument(
        "--detect-command",
        help="External detector template; placeholders: {input} {mask} {prompt}",
    )
    p.add_argument("--dilation", type=int, default=DEFAULT_DILATION_RADIUS)
    p.add_argument("--backend", choices=("print-plan", "simple", "external"), default="print-plan")
    p.add_argument(
        "--command",
        help="External inpainter template; placeholders: {input} {mask} {output} {prompt}",
    )
    p.add_argument("--mask-output", type=Path)
    p.add_argument("--prompt", default="Remove watermark, fill with background")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    try:
        report = remove_visible(
            args.path,
            args.output,
            mask_path=args.mask,
            box=args.box,
            detect_command=args.detect_command,
            backend=args.backend,
            command=args.command,
            dilation_radius=args.dilation,
            mask_output=args.mask_output,
            prompt=args.prompt,
        )
    except Exception as e:
        eprint(f"error: {e}")
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"status={report['status']} output={report.get('output')}")
        for action in report["actions"]:
            print(f"  - {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
