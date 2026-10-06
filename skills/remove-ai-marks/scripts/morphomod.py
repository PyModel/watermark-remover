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
import io
import json
import math
import struct
import sys
import tempfile
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import external_command
import pipeline_actions as act
from common import (
    atomic_write_bytes,
    eprint,
    paths_alias,
    read_bytes_bounded,
    validate_output_path,
)
from domain_types import InpaintRequest
from image_meta import detect_format
from pipeline_actions import Action, action_sentences, report_actions
from png_chunks import iter_png_chunks

DEFAULT_DILATION_RADIUS = 3

#: Every classify output repeats this. The honesty contract: absence of the
#: classifier, of weights, or of any signal is unknown — never a clean bill.
CLASSIFY_CAVEAT = (
    "Classification is not provenance: a label is pixel-level evidence, and any"
    " result -- including an unavailable classifier -- "
    "never proves a file is clean."
)
MAX_PIXELS = 40_000_000  # bounds decompression/allocation; covers 8K UHD
MAX_ENCODED_BYTES = 256 * 1024 * 1024
MAX_TEXTURE_PATCH_PIXELS = 1_048_576
MAX_TEXTURE_CANDIDATES = 20_000
MAX_COMMAND_DIAGNOSTIC_BYTES = 64 * 1024
EXTERNAL_COMMAND_TIMEOUT_SECONDS = 1800
PNG_SIG = b"\x89PNG\r\n\x1a\n"

VISIBLE_BACKENDS = (
    "print-plan",
    "texture",
    "simple",
    "external",
    "cv2-telea",
    "cv2-ns",
    "migan",
    "lama",
)
VISIBLE_CLEAN_BACKENDS = ("texture", "simple", "external", "cv2-telea", "cv2-ns", "migan", "lama")
#: Fill backends resolved through the inpaint backend registry: the OpenCV pair
#: plus the learned ONNX tiers (labeled-unavailable until the migan/lama extras
#: register real backends).
REGISTRY_FILL_BACKENDS = ("cv2-telea", "cv2-ns", "migan", "lama")


@dataclass(frozen=True, slots=True)
class VisiblePlan:
    mask_path: Path | None = None
    box: tuple[int, int, int, int] | None = None
    detect_command: str | None = None
    #: Registry-driven localization: "auto" scans every registered mark, or a
    #: single registry key (see watermark_registry.mark_keys). Runs the shared
    #: localize -> fill -> validate pass; None keeps the explicit-mask behavior
    #: byte-for-byte.
    mark: str | None = None
    #: Trust policy for the registry pass: "auto" relaxes a mark's gate only on
    #: same-product evidence (TC260 producer metadata, or a strict sibling mark);
    #: "strict" never relaxes.
    sensitivity: str = "auto"
    backend: str = "print-plan"
    command: str | None = None
    dilation_radius: int = DEFAULT_DILATION_RADIUS
    mask_output: Path | None = None
    prompt: str = "Remove watermark, fill with background"
    #: False = frictionless mode: do not publish a .mask.pgm next to output.
    #: The effective mask is still computed and used internally (and written
    #: for the external backend, which needs a real file as its input).
    publish_mask: bool = True

    timeout: float = EXTERNAL_COMMAND_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.backend not in VISIBLE_BACKENDS:
            raise ValueError(f"unknown visible backend: {self.backend}")
        if self.sensitivity not in ("auto", "strict"):
            raise ValueError(
                f"unknown sensitivity {self.sensitivity!r}; expected 'auto' or 'strict'"
            )
        if self.mark is not None and self.mark != "auto":
            from watermark_registry import mark_keys

            if self.mark not in mark_keys():
                raise ValueError(
                    f"unknown mark {self.mark!r}; expected 'auto' or one of {mark_keys()}"
                )
        sources = sum(
            source is not None
            for source in (self.mask_path, self.box, self.detect_command, self.mark)
        )
        if sources > 1:
            raise ValueError("exactly one localization source may be configured")
        if sources == 0 and self.backend != "print-plan":
            raise ValueError("an inpainting backend requires a localization source")
        if self.backend == "external" and sources and not self.command:
            raise ValueError("command required for external backend")
        if self.command and self.backend != "external":
            raise ValueError("command is only valid for external backend")
        if (
            not isinstance(self.dilation_radius, int)
            or isinstance(self.dilation_radius, bool)
            or self.dilation_radius < 0
        ):
            raise ValueError("dilation radius must be a non-negative integer")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        if not isinstance(self.publish_mask, bool):
            raise TypeError("publish_mask must be a bool")


def _read_bounded(path: Path, limit: int | None = None) -> bytes:
    effective_limit = MAX_ENCODED_BYTES if limit is None else limit
    return read_bytes_bounded(path, effective_limit, label="encoded file")


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


@dataclass(frozen=True)
class TextureMatch:
    x: int
    y: int
    width: int
    height: int
    score: float


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


def filter_components(mask: Mask, *, min_size: int = 10, max_components: int = 50) -> Mask:
    """Remove small disconnected components from a mask.

    Keeps only components whose pixel count >= ``min_size``, up to
    ``max_components`` total components.  Use this after mask acquisition
    to drop noise and false positives from detectors.

    Does NOT modify the original mask.
    """
    w, h = mask.width, mask.height
    data = bytearray(mask.data)
    visited = bytearray(w * h)
    component_sizes: list[int] = []
    component_ids: list[list[int]] = []

    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if data[idx] == 0 or visited[idx]:
                continue
            # BFS to find connected component
            component: list[int] = []
            q: deque[int] = deque([idx])
            visited[idx] = 1
            while q:
                ci = q.popleft()
                component.append(ci)
                cx, cy = ci % w, ci // w
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        ni = ny * w + nx
                        if data[ni] != 0 and not visited[ni]:
                            visited[ni] = 1
                            q.append(ni)
            component_sizes.append(len(component))
            component_ids.append(component)

    # Keep large components, discard small ones
    kept = 0
    for comp_idx, size in enumerate(component_sizes):
        if size >= min_size and kept < max_components:
            kept += 1
        else:
            for ci in component_ids[comp_idx]:
                data[ci] = 0

    return Mask(w, h, data)


def closing(mask: Mask, radius: int = 1) -> Mask:
    """Morphological closing: dilate then erode to fill narrow gaps.

    Operates on marked pixels (255 = mark).  ``radius=1`` fills single-pixel
    gaps and bridges narrow breaks.
    """
    if radius < 1:
        return Mask(mask.width, mask.height, bytearray(mask.data))
    # Dilate first, then erode using the mask's real rectangular dimensions.
    dilated = dilate(mask, radius)
    eroded = _erode_mask(dilated.data, mask.width, mask.height, radius)
    return Mask(mask.width, mask.height, eroded)


def _erode_mask(data: bytearray, width: int, height: int, radius: int) -> bytearray:
    """Erode a binary mask (255=foreground, 0=background) by radius."""
    out = bytearray(data)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if data[idx] == 0:
                continue
            # Check all pixels in the radius window
            keep = True
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and data[ny * width + nx] == 0:
                        keep = False
                        break
                if not keep:
                    break
            if not keep:
                out[idx] = 0
    return out


def feather_blend(
    original: Raster,
    inpainted: Raster,
    mask: Mask,
    *,
    feather_radius: int = 3,
) -> Raster:
    """Blend inpainted pixels into the original with a smooth feather region.

    Unlike texture_patch_inpaint's built-in feather, this operates on
    any inpainted raster by computing a depth-based blend at the mask
    boundary.  The mask must already be the effective mask (post-dilation).

    Returns a new Raster; original and inpainted are not modified.
    """
    if (
        original.width != inpainted.width
        or original.height != inpainted.height
        or original.channels != inpainted.channels
    ):
        raise ValueError("original/inpainted dimensions or channel counts differ")
    if (original.width, original.height) != (mask.width, mask.height):
        raise ValueError("mask dimensions must match raster dimensions")
    if feather_radius <= 0:
        return composite(original, inpainted, mask)

    depths = _mask_depths(mask)
    out = bytearray(original.data)
    ch = original.channels

    for i, marked in enumerate(mask.data):
        if not marked:
            continue
        blend = min(1.0, max(0.0, depths[i] / max(1, feather_radius)))
        blend = blend * blend * (3.0 - 2.0 * blend)  # smoothstep
        for c in range(ch):
            out[i * ch + c] = round(
                original.data[i * ch + c] * (1.0 - blend) + inpainted.data[i * ch + c] * blend
            )
    return Raster(original.width, original.height, ch, out)


# ---------------------------------------------------------------------------
# PGM + PNG I/O
# ---------------------------------------------------------------------------


def read_pgm(path: Path) -> Mask:
    raw = _read_bounded(path)
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
    atomic_write_bytes(
        path,
        f"P5\n{mask.width} {mask.height}\n255\n".encode() + bytes(mask.data),
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def decode_png(data: bytes) -> Raster:
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    for chunk in iter_png_chunks(data, allow_trailing_data=True):
        if chunk.kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk.payload
            )
        elif chunk.kind == b"IDAT":
            idat.extend(chunk.payload)
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


def decode_jpeg(data: bytes) -> Raster:
    """Decode JPEG image bytes into a canonical Raster.

    Contract:
    A Raster has no orientation concept and no external-profile dependency.
    channels == 1 means L (grayscale), 3 means sRGB RGB, and 4 means sRGB RGBA
    with straight alpha. CMYK never enters Raster.
    """
    import optional_deps

    if optional_deps.PIL is None:
        raise ValueError(
            optional_deps.BackendAvailability(
                available=False,
                extra="visible",
                reason="Pillow is required for JPEG input decoding",
            ).hint
        )
    from PIL import Image, ImageCms, ImageOps

    saved_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        try:
            raw_im = Image.open(io.BytesIO(data))
        except Exception as exc:
            raise ValueError(f"malformed JPEG image: {exc}") from exc

        with raw_im:
            _validate_dimensions(raw_im.width, raw_im.height)
            try:
                im = ImageOps.exif_transpose(raw_im)
            except Exception:
                im = raw_im
            if im is None:
                im = raw_im
            _validate_dimensions(im.width, im.height)

            icc_data = im.info.get("icc_profile")
            if icc_data:
                try:
                    input_profile = ImageCms.getOpenProfile(io.BytesIO(icc_data))
                    srgb_profile = ImageCms.createProfile("sRGB")
                    im = ImageCms.profileToProfile(
                        im,
                        inputProfile=input_profile,
                        outputProfile=srgb_profile,
                        outputMode="RGB",
                    )
                except Exception:
                    im = im.convert("RGB")
            elif im.mode not in ("L", "RGB", "RGBA"):
                im = im.convert("RGB")

            channels = len(im.getbands())
            if channels not in (1, 3, 4):
                im = im.convert("RGB")
                channels = 3

            _validate_dimensions(im.width, im.height)
            try:
                pixel_data = im.tobytes()
            except Exception as exc:
                raise ValueError(f"failed to decode JPEG pixels: {exc}") from exc
            return Raster(im.width, im.height, channels, bytearray(pixel_data))
    finally:
        Image.MAX_IMAGE_PIXELS = saved_max


def decode_to_raster(data: bytes, fmt: str) -> Raster:
    """Decode image bytes of a supported format into a canonical Raster."""
    if fmt == "png":
        return decode_png(data)
    if fmt == "jpeg":
        return decode_jpeg(data)
    raise ValueError(f"unsupported format for raster decode: {fmt}")


def load_mask(path: Path) -> Mask:
    if path.suffix.lower() == ".pgm":
        return read_pgm(path)
    raster = decode_png(_read_bounded(path))
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


def _mask_bounds(mask: Mask) -> tuple[int, int, int, int]:
    min_x = mask.width
    min_y = mask.height
    max_x = -1
    max_y = -1
    for index, value in enumerate(mask.data):
        if not value:
            continue
        x, y = index % mask.width, index // mask.width
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if max_x < 0:
        raise ValueError("mask contains no marked pixels")
    return min_x, min_y, max_x + 1, max_y + 1


def _texture_edge_score(
    raster: Raster,
    bounds: tuple[int, int, int, int],
    source_x: int,
    source_y: int,
) -> float:
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    channels = min(raster.channels, 3)
    perimeter = 2 * (width + height)
    sample_step = max(1, perimeter // 1024)
    squared_error = 0
    samples = 0

    def add_error(source_index: int, target_index: int) -> None:
        nonlocal squared_error, samples
        for channel in range(channels):
            difference = raster.data[source_index + channel] - raster.data[target_index + channel]
            squared_error += difference * difference
            samples += 1

    if y0 > 0:
        for offset in range(0, width, sample_step):
            add_error(
                (source_y * raster.width + source_x + offset) * raster.channels,
                ((y0 - 1) * raster.width + x0 + offset) * raster.channels,
            )
    if y1 < raster.height:
        for offset in range(0, width, sample_step):
            add_error(
                ((source_y + height - 1) * raster.width + source_x + offset) * raster.channels,
                (y1 * raster.width + x0 + offset) * raster.channels,
            )
    if x0 > 0:
        for offset in range(0, height, sample_step):
            add_error(
                ((source_y + offset) * raster.width + source_x) * raster.channels,
                ((y0 + offset) * raster.width + x0 - 1) * raster.channels,
            )
    if x1 < raster.width:
        for offset in range(0, height, sample_step):
            add_error(
                ((source_y + offset) * raster.width + source_x + width - 1) * raster.channels,
                ((y0 + offset) * raster.width + x1) * raster.channels,
            )
    if not samples:
        raise ValueError("texture patch has no surrounding pixels to match")
    return squared_error / samples


def _candidate_axis(start: int, stop: int, step: int) -> list[int]:
    values = list(range(start, stop + 1, step))
    if values and values[-1] != stop:
        values.append(stop)
    return values


def _find_texture_match(raster: Raster, mask: Mask, feather: int) -> TextureMatch:
    bounds = _mask_bounds(mask)
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    if width * height > MAX_TEXTURE_PATCH_PIXELS:
        raise ValueError(
            f"texture patch exceeds safety limit of {MAX_TEXTURE_PATCH_PIXELS:,} pixels"
        )

    gap = max(2, feather)
    max_sx = raster.width - width
    max_sy = raster.height - height
    # A same-size source patch must sit entirely in one of the four margins
    # around the mask; their union is exactly the feasible set, so sampling
    # the strips can never miss a valid placement (a radius window around
    # the mask can, when the margin is farther away than the radius).
    strips = [
        (0, x0 - width - gap, 0, max_sy),
        (x1 + gap, max_sx, 0, max_sy),
        (0, max_sx, 0, y0 - height - gap),
        (0, max_sx, y1 + gap, max_sy),
    ]
    strips = [s for s in strips if s[0] <= s[1] and s[2] <= s[3]]
    if not strips:
        # When the mask is too large, no non-overlapping placement exists
        # and scanning the search region would only reject every candidate.
        raise ValueError(
            f"texture patch {width}x{height} at ({x0},{y0}) cannot be placed "
            f"non-overlapping inside {raster.width}x{raster.height} image; "
            "mask is too large for the texture backend (use --backend simple or external)"
        )

    # Two-tier search: prefer a local window around the mask to keep the
    # candidate count bounded and avoid distant semantic mismatches on large
    # images; fall back to the full feasible strips when the margin is pushed
    # outside the local radius (e.g. wide feather).
    radius = max(256, 6 * max(width, height))
    win_x0 = max(0, x0 - radius)
    win_x1 = min(max_sx, x0 + radius)
    win_y0 = max(0, y0 - radius)
    win_y1 = min(max_sy, y0 + radius)
    local_strips = [
        (max(left, win_x0), min(right, win_x1), max(top, win_y0), min(bottom, win_y1))
        for left, right, top, bottom in strips
    ]
    local_strips = [s for s in local_strips if s[0] <= s[1] and s[2] <= s[3]]
    search_strips = local_strips if local_strips else strips

    step = max(1, min(width, height) // 16)

    def axis(lo: int, hi: int) -> list[int]:
        return _candidate_axis(lo, hi, step)

    estimated = sum(len(axis(a, b)) * len(axis(c, d)) for a, b, c, d in search_strips)
    if estimated > MAX_TEXTURE_CANDIDATES:
        step *= math.ceil(math.sqrt(estimated / MAX_TEXTURE_CANDIDATES))

    best: tuple[float, int, int, int] | None = None
    seen: set[tuple[int, int]] = set()
    for left, right, top, bottom in search_strips:
        for source_y in axis(top, bottom):
            for source_x in axis(left, right):
                if (source_x, source_y) in seen:
                    continue
                seen.add((source_x, source_y))
                score = _texture_edge_score(raster, bounds, source_x, source_y)
                candidate = (
                    score,
                    abs(source_x - x0) + abs(source_y - y0),
                    source_x,
                    source_y,
                )
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise ValueError("no non-overlapping texture patch candidate found")
    score, _distance, source_x, source_y = best
    return TextureMatch(source_x, source_y, width, height, score)


def _mask_depths(mask: Mask) -> bytearray:
    depths = bytearray(mask.width * mask.height)
    queue: deque[int] = deque()
    for index, marked in enumerate(mask.data):
        if not marked:
            continue
        x, y = index % mask.width, index // mask.width
        if any(
            nx < 0
            or nx >= mask.width
            or ny < 0
            or ny >= mask.height
            or not mask.data[ny * mask.width + nx]
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        ):
            depths[index] = 1
            queue.append(index)
    while queue:
        index = queue.popleft()
        depth = depths[index]
        x, y = index % mask.width, index // mask.width
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < mask.width and 0 <= ny < mask.height:
                neighbor = ny * mask.width + nx
                if mask.data[neighbor] and not depths[neighbor]:
                    depths[neighbor] = min(255, depth + 1)
                    queue.append(neighbor)
    return depths


def texture_patch_inpaint(
    raster: Raster,
    mask: Mask,
    *,
    feather: int = 0,
) -> tuple[Raster, TextureMatch]:
    """Fill a small marked region from the best matching nearby texture patch.

    The source patch is selected by edge error against known pixels surrounding
    the mask. Optional distance-based feathering can hide seams; the default fully replaces
    the refined mask while preserving every pixel outside it exactly.
    """
    if (raster.width, raster.height) != (mask.width, mask.height):
        raise ValueError("mask/image dimensions differ")
    if feather < 0 or feather > 254:
        raise ValueError("texture feather must be in [0, 254]")
    if mask.marked == raster.width * raster.height:
        raise ValueError("cannot inpaint a mask covering the entire image")

    match = _find_texture_match(raster, mask, feather)
    x0, y0, _x1, _y1 = _mask_bounds(mask)
    depths = _mask_depths(mask)
    output = bytearray(raster.data)
    for index, marked in enumerate(mask.data):
        if not marked:
            continue
        x, y = index % raster.width, index // raster.width
        source_x = match.x + x - x0
        source_y = match.y + y - y0
        source_index = (source_y * raster.width + source_x) * raster.channels
        dest_index = index * raster.channels
        if feather == 0:
            blend = 1.0
        else:
            normalized = min(1.0, max(0.0, (depths[index] - 1) / feather))
            blend = normalized * normalized * (3.0 - 2.0 * normalized)
        for channel in range(raster.channels):
            original = raster.data[dest_index + channel]
            texture = raster.data[source_index + channel]
            output[dest_index + channel] = round(original * (1.0 - blend) + texture * blend)
    return Raster(raster.width, raster.height, raster.channels, output), match


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


def _run_template(template: str, *, timeout: float, **values: str) -> None:
    result = external_command.run_command(
        external_command.command_from_template(template, **values),
        timeout=timeout,
        output_limit=MAX_COMMAND_DIAGNOSTIC_BYTES,
    )
    if result.returncode != 0:
        diagnostics = result.stderr_text or result.stdout_text
        raise RuntimeError(f"external command failed ({result.returncode}): {diagnostics}")


def _validate_visible_paths(
    path: Path,
    dest: Path | None,
    mask_path: Path | None,
    mask_output: Path | None,
) -> None:
    inputs = [path, *([mask_path] if mask_path else [])]
    outputs = [candidate for candidate in (dest, mask_output) if candidate is not None]
    for output in outputs:
        validate_output_path(path, output)
        if output.is_symlink():
            raise ValueError(f"output path is a symlink: {output}")
        if any(paths_alias(output, source) for source in inputs):
            raise ValueError(f"output aliases an input: {output}")
    if len(outputs) == 2 and paths_alias(outputs[0], outputs[1]):
        raise ValueError("image output and mask output alias each other")


def _resolve_fill_backend(name: str) -> Any:
    """Resolve a fill backend by name; fail soft when unavailable.

    Delegates to the inpaint backend registry's ``resolve_backend`` -- the single
    resolution point: ``auto`` walks the lama > migan > cv2 > pure preference
    order skipping unavailable tiers, and a named selection that is unregistered
    or unavailable raises the labeled ValueError with the canonical install hint
    (e.g. ``watermark-remover[migan]``). The exit contract stays with the caller.
    """
    import inpaint_backends

    return inpaint_backends.resolve_backend(name)


def _registry_fill_backend(backend: str) -> str:
    """Map a visible-plan backend to the registry fill vocabulary.

    ``migan``/``lama`` pass through; the cv2 pair maps to the registry's ``cv2``;
    the pure texture/simple backends mean "no learned model demanded", so the
    registry resolves its own quality order (lama > migan > cv2, cv2 as floor)."""
    if backend in ("migan", "lama"):
        return backend
    if backend in REGISTRY_FILL_BACKENDS:  # cv2-telea / cv2-ns
        return "cv2"
    return "auto"


def _remove_visible_marks(
    path: Path,
    dest: Path,
    plan: VisiblePlan,
    *,
    data: bytes,
    fmt: str,
    raster: Raster,
    mask_details: dict[str, Mask] | None,
) -> dict[str, Any]:
    """The ``--mark`` path: registry localize -> fill -> validate on the decoded frame.

    Always writes ``dest`` (an unchanged copy when nothing was detected -- an
    idempotent clean) and returns the standard visible report with the mark
    results riding along under ``mark_status`` / ``marks``. The aggregate status
    stays ``completed`` (the pipeline produced an output); ``mark_status`` is the
    honest ``no_mark`` / ``cleaned`` / ``partial`` / ``unvalidated`` verdict and
    every auto-detected removal is labeled best-effort in the action text.
    """
    # Pixel-path imports, lazily: the stdlib core never loads them; only a caller
    # that passed --mark opted into cv2/numpy through the registry.
    import cv2
    import numpy as np
    import tc260_label
    import tc260_verify
    import watermark_registry as reg

    image = (
        np.frombuffer(bytes(raster.data), dtype=np.uint8)
        .reshape(raster.height, raster.width, raster.channels)
        .copy()
    )
    if raster.channels == 1:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif raster.channels == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Same-product evidence for the `auto` sensitivity: the file's own metadata
    # (non-TC260 signals declared on the registry rows, e.g. Samsung genAIType
    # or a Microsoft platform token in generator fields) plus a TC260 label
    # whose ContentProducer maps (via tc260_verify) to a registered mark's
    # product. Unmapped producers and undeclared signals never relax a gate.
    provenance = reg.metadata_evidence(data)
    label = tc260_label.label_from_metadata(data)
    if label:
        producer = tc260_verify.producer_for_code(label.get("ContentProducer", ""))
        if producer is not None:
            provenance = provenance | frozenset({producer.key})

    fill_backend = _registry_fill_backend(plan.backend)
    if plan.mark == "auto":
        report = reg.remove_auto_marks_detailed(
            bgr,
            sensitivity=plan.sensitivity,  # type: ignore[arg-type]
            provenance=provenance,
            backend=fill_backend,  # type: ignore[arg-type]
        )
    else:
        report = reg.remove_mark_detailed(
            bgr,
            plan.mark,  # type: ignore[arg-type]
            sensitivity=plan.sensitivity,  # type: ignore[arg-type]
            provenance=provenance,
            backend=fill_backend,  # type: ignore[arg-type]
        )
    out_bgr = report.image

    # The union of the masks actually filled: clean_asset's outside-mask
    # verification runs against exactly the pixels that changed, never a bbox proxy.
    union = np.zeros(out_bgr.shape[:2], np.uint8)
    for removal in report.marks:
        if removal.mask is not None:
            union |= removal.mask
    if mask_details is not None:
        union_mask = Mask(out_bgr.shape[1], out_bgr.shape[0], bytearray(union.tobytes()))
        mask_details.update(original=union_mask, effective=union_mask)

    actions: list[Action] = [act.mask_source(f"registry:{plan.mark}")]
    # The registry, not --visible-backend, owns the fill: report the backend
    # that actually ran (None when nothing was filled) and warn loudly when the
    # planned backend cannot apply here (the external command template has no
    # place in the localize -> fill -> validate pass).
    resolved_report_backend = report.marks[0].backend if report.marks else None
    if plan.backend == "external":
        actions.append(
            Action(
                act.ActionCode.VISIBLE_PLAN,
                act.Effect.WARNING,
                "external backend not applicable to registry marks: the fill ran "
                f"through the registry ({resolved_report_backend}); --command was "
                "not invoked",
                {"planned_backend": "external", "used_backend": resolved_report_backend},
            )
        )
    if report.marks:
        resolved = report.marks[0].backend
        labels = ", ".join(m.label for m in report.marks)
        actions.append(
            Action(
                act.ActionCode.INPAINT,
                act.Effect.CHANGE,
                f"registry {resolved} fill (best-effort): removed "
                f"{len(report.marks)} mark(s): {labels}",
                {"backend": resolved, "marks": [m.key for m in report.marks]},
            )
        )
    else:
        actions.append(
            act.nothing_removed(fmt, f"no registered mark detected (--mark {plan.mark})")
        )

    # Encode back into the source raster's channel layout (alpha preserved: the
    # original plane is re-attached untouched).
    if raster.channels == 1:
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
        payload = out
    elif raster.channels == 4:
        rgba = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGBA)
        rgba[..., 3] = image[..., 3]
        payload = rgba
    else:
        payload = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    atomic_write_bytes(
        dest,
        encode_png(
            Raster(
                raster.width,
                raster.height,
                raster.channels,
                bytearray(np.ascontiguousarray(payload).tobytes()),
            )
        ),
    )

    provisional = reg.provisional_marks()
    return {
        "status": "completed",
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        # The backend that actually filled (registry vocabulary), not the
        # planned --visible-backend; plan.backend stays only when nothing was
        # filled and no fill resolution ran.
        "backend": resolved_report_backend if resolved_report_backend is not None else plan.backend,
        "mask": None,  # frictionless: the effective mask lives in mask_details only
        "mark": plan.mark,
        "mark_status": report.status,  # no_mark / cleaned / partial / unvalidated
        "sensitivity": plan.sensitivity,
        "provenance": sorted(provenance),
        "marks": [
            {
                "key": m.key,
                "label": m.label,
                "location": m.location,
                "region": list(m.region) if m.region else None,
                "mask_bbox": list(m.mask_bbox),
                "backend": m.backend,
                "confidence_before": round(m.confidence_before, 4),
                "confidence_after": (
                    round(m.confidence_after, 4) if m.confidence_after is not None else None
                ),
                "status": m.status,
                "residual_region": list(m.residual_region) if m.residual_region else None,
                "provisional": m.key in provisional,
                "note": "best-effort detection and removal; validated by the same detector",
            }
            for m in report.marks
        ],
        "initial_mask_pixels": int((union > 0).sum()),
        "refined_mask_pixels": int((union > 0).sum()),
        "dilation_radius": 0,
        # The registered engines own their dilation; --dilate is not applied on
        # this path, so the field states that instead of implying it was.
        "dilation": "engine-controlled (0 user-applied)",
        **report_actions(actions),
        "note": (
            "Registry localize -> fill -> validate pass. Auto-detected removal is "
            "best-effort: the validation step is the same detector re-run, not an "
            "independent oracle."
        ),
    }


def remove_visible(
    path: Path,
    dest: Path | None,
    plan: VisiblePlan,
    *,
    mask_details: dict[str, Mask] | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, VisiblePlan):
        raise TypeError("plan must be a VisiblePlan")
    _validate_visible_paths(path, dest, plan.mask_path, plan.mask_output)
    data = _read_bounded(path)
    fmt = detect_format(data)
    actions: list[Action] = []

    has_source = any(
        source is not None for source in (plan.mask_path, plan.box, plan.detect_command, plan.mark)
    )
    if not has_source:
        return {
            "status": "plan-only",
            "input": str(path),
            "output": None,
            "format": fmt,
            "backend": plan.backend,
            **report_actions([act.visible_needs_source(), act.visible_plan()]),
            "note": "No blind segmenter is bundled; no image bytes were changed.",
        }
    raster = decode_to_raster(data, fmt)
    dims = (raster.width, raster.height)
    if plan.backend != "print-plan" and dest is None:
        raise ValueError("output required for an inpainting backend")
    if (
        plan.backend != "print-plan"
        and fmt != "png"
        and dest is not None
        and dest.suffix.lower() != ".png"
    ):
        raise ValueError(
            f"{fmt} input must be written to a PNG destination (.png); "
            "outside-mask guarantee cannot survive a JPEG re-encode"
        )

    if plan.mark is not None:
        # A labeled error, never an assert: the mark pass always writes dest, so
        # a dest-less call (e.g. a mask-only probe) is a caller contract breach.
        if dest is None:
            raise ValueError("output required for a mark pass")
        assert raster is not None
        return _remove_visible_marks(
            path,
            dest,
            plan,
            data=data,
            fmt=fmt,
            raster=raster,
            mask_details=mask_details,
        )

    if plan.mask_path is not None:
        initial = load_mask(plan.mask_path)
        source = f"mask:{plan.mask_path}"
    elif plan.box is not None:
        if not dims:
            raise ValueError("cannot derive dimensions for --box; provide --mask")
        initial = box_mask(*dims, plan.box)
        source = f"box:{','.join(map(str, plan.box))}"
    else:
        assert plan.detect_command is not None
        with tempfile.TemporaryDirectory(prefix="wm-mask-") as temp_dir:
            detected = Path(temp_dir) / "detected.pgm"
            _run_template(
                plan.detect_command,
                input=str(path),
                mask=str(detected),
                prompt=plan.prompt,
                timeout=plan.timeout,
            )
            if not detected.is_file():
                raise RuntimeError("detector command did not create {mask}")
            initial = load_mask(detected)
        source = "external-detector"

    if dims and (initial.width, initial.height) != dims:
        raise ValueError(f"mask dimensions {(initial.width, initial.height)} != image {dims}")
    refined = refine_mask(initial, plan.dilation_radius)
    if mask_details is not None:
        mask_details.update(original=initial, effective=refined)
    # The external backend needs a real mask file as its input, so it always
    # publishes one.  Internal backends (texture/simple) may suppress the
    # artifact entirely in frictionless mode (publish_mask=False).
    publish = plan.publish_mask or plan.backend == "external"
    mask_output = plan.mask_output
    if publish:
        if mask_output is None:
            base = dest or path.with_name(f"{path.stem}.visible.cleaned{path.suffix}")
            mask_output = base.with_name(f"{base.stem}.mask.pgm")
            _validate_visible_paths(path, dest, plan.mask_path, mask_output)
        write_pgm(refined, mask_output)
        mask_label = str(mask_output)
    else:
        mask_label = None  # frictionless: effective mask kept in memory only
    actions.extend(
        [
            act.mask_source(source),
            act.refine_mask(plan.dilation_radius, initial.marked, refined.marked),
            act.effective_mask(initial.marked, refined.marked, mask_label),
        ]
    )

    if plan.backend == "print-plan":
        status = "mask-ready"
        output = None
        actions.append(act.inpaint_skipped())
    elif plan.backend == "texture":
        assert raster is not None
        assert dest is not None
        restored, match = texture_patch_inpaint(raster, refined, feather=0)
        atomic_write_bytes(dest, encode_png(restored))
        status, output = "completed", str(dest)
        actions.append(
            act.inpaint_texture(match.x, match.y, match.width, match.height, match.score)
        )
    elif plan.backend == "simple":
        assert raster is not None
        assert dest is not None
        filled = simple_inpaint(raster, refined)
        restored = composite(raster, filled, refined)
        atomic_write_bytes(dest, encode_png(restored))
        status, output = "completed", str(dest)
        actions.append(act.inpaint_simple())
    elif plan.backend in REGISTRY_FILL_BACKENDS:
        assert raster is not None
        assert dest is not None
        backend = _resolve_fill_backend(plan.backend)
        result = backend.inpaint(InpaintRequest(source=raster, mask=refined))
        restored = composite(raster, result.inpainted, refined)
        atomic_write_bytes(dest, encode_png(restored))
        status, output = "completed", str(dest)
        actions.append(
            Action(
                act.ActionCode.INPAINT,
                act.Effect.CHANGE,
                f"{plan.backend} inpaint + stdlib restore outside mask",
                {"backend": plan.backend},
            )
        )
    else:
        assert plan.backend == "external"
        assert plan.command is not None
        assert dest is not None
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="wm-inpaint-") as temp_dir:
            temp_in = Path(temp_dir) / "input.png"
            atomic_write_bytes(temp_in, encode_png(raster))
            external_out = Path(temp_dir) / "inpainted.png"
            _run_template(
                plan.command,
                input=str(temp_in),
                mask=str(mask_output),
                output=str(external_out),
                prompt=plan.prompt,
                timeout=plan.timeout,
            )
            if not external_out.is_file():
                raise RuntimeError("inpaint command did not create {output}")
            inpainted = decode_png(_read_bounded(external_out))
            if (inpainted.width, inpainted.height, inpainted.channels) != (
                raster.width,
                raster.height,
                raster.channels,
            ):
                raise ValueError(
                    f"external backend output shape {(inpainted.width, inpainted.height, inpainted.channels)} "
                    f"does not match source {(raster.width, raster.height, raster.channels)}"
                )
            atomic_write_bytes(dest, encode_png(composite(raster, inpainted, refined)))
            actions.append(act.inpaint_external())
        status, output = "completed", str(dest)

    return {
        "status": status,
        "input": str(path),
        "output": output,
        "format": fmt,
        "backend": plan.backend,
        "mask": mask_label,
        "initial_mask_pixels": initial.marked,
        "refined_mask_pixels": refined.marked,
        "dilation_radius": plan.dilation_radius,
        **report_actions(actions),
        "note": (
            "MorphoMod-inspired pipeline; CVPR paper metrics are not this run's metrics. Inspect output fidelity and residual marks manually."
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


def _classify_main(argv: list[str]) -> int:
    """The explicit ``classify`` verb: metadata-free photo classification.

    Opt-in by construction — nothing on the inspect/clean/detect/server paths
    reaches this handler (pinned by tests/test_photo_classify.py's source
    scan). Unavailability is labeled and carries the caveat; it is never a
    silent pass and never a clean verdict.
    """
    parser = _classify_parser()
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {
        "verb": "classify",
        "status": "error",
        "result": None,
        "reason": None,
        "caveat": CLASSIFY_CAVEAT,
    }

    def emit(
        status: str, code: int, *, result: dict[str, Any] | None = None, reason: str | None = None
    ) -> int:
        payload["status"] = status
        payload["result"] = result
        payload["reason"] = reason
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif status == "ok":
            assert result is not None
            print("pixel classification (not provenance)")
            for key in ("label", "domain", "detector", "provider"):
                print(f"  {key}: {result[key]}")
            print(f"caveat: {CLASSIFY_CAVEAT}")
        else:
            print(f"unavailable: {reason}")
            print(f"caveat: {CLASSIFY_CAVEAT}")
        return code

    if not args.path.is_file():
        message = f"input file not found: {args.path}"
        if args.json:
            payload["status"] = "usage-error"
            payload["reason"] = message
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            eprint(f"error: {message}")
        return 2

    try:
        import classify as classify_mod
    except ImportError as error:  # pragma: no cover - ships in the same tree
        return emit(
            "unavailable",
            1,
            reason=(
                f"classifier module unavailable: {error}."
                " Install: pip install 'watermark-remover[classify]'"
            ),
        )

    try:
        if not classify_mod.is_available():
            return emit(
                "unavailable",
                1,
                reason=(
                    "the pixel-classification dependencies are not installed."
                    " Install: pip install 'watermark-remover[classify]'"
                ),
            )
        result = classify_mod.classify_pixels(
            args.path, device=args.device, backend=args.backend
        ).to_dict()
    except RuntimeError as error:
        # Missing weights / gate artifact / failed pinned download: the gated
        # head did not run, and that fact is the headline, not a label.
        return emit("unavailable", 1, reason=str(error))
    except (OSError, ValueError) as error:
        return emit("error", 1, reason=str(error))
    except ImportError as error:
        return emit("unavailable", 1, reason=f"classifier module unavailable: {error}")
    return emit("ok", 0, result=result)


def _classify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="morphomod.py classify",
        description=(
            "Classify a photograph as AI or camera-like from pixels alone."
            " This is not provenance and it never runs cleanup."
        ),
    )
    parser.add_argument("path", type=Path, help="Image file to classify")
    parser.add_argument("--device", default=None, help="torch device: auto (default), cpu, or cuda")
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = sys.argv[1:] if argv is None else argv
    if args_list and args_list[0] == "classify":
        # The explicit E1 verb: the only sanctioned classify call site.
        return _classify_main(args_list[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mask", type=Path, help="Binary PGM or PNG mask (white=remove)")
    source.add_argument("--box", type=_parse_box, help="Manual x,y,w,h mask")
    source.add_argument(
        "--detect-command",
        help="External detector template; placeholders: {input} {mask} {prompt}",
    )
    parser.add_argument("--dilation", type=int, default=DEFAULT_DILATION_RADIUS)
    parser.add_argument(
        "--backend",
        choices=VISIBLE_BACKENDS,
        default="print-plan",
    )
    parser.add_argument(
        "--command",
        help="External inpainter template; placeholders: {input} {mask} {output} {prompt}",
    )
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--prompt", default="Remove watermark, fill with background")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(args_list)

    try:
        plan = VisiblePlan(
            mask_path=args.mask,
            box=args.box,
            detect_command=args.detect_command,
            backend=args.backend,
            command=args.command,
            dilation_radius=args.dilation,
            mask_output=args.mask_output,
            prompt=args.prompt,
        )
        report = remove_visible(args.path, args.output, plan)
    except Exception as error:
        eprint(f"error: {error}")
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"status={report['status']} output={report.get('output')}")
        for action in action_sentences(report.get("action_details", ())):
            print(f"  - {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
