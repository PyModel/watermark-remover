"""Strict structural iteration for in-memory PNG data."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class PNGChunk:
    kind: bytes
    payload: memoryview
    raw: memoryview


def iter_png_chunks(data: bytes, *, allow_trailing_data: bool = False) -> Iterator[PNGChunk]:
    """Yield CRC-validated chunks and require one complete terminal IEND.

    Per-chunk bounds and CRC checks run as chunks are produced. The IHDR, IDAT,
    and terminal IEND requirements are enforced only when the caller consumes
    the whole iterator; a caller that stops early gets no whole-file guarantee.
    When allow_trailing_data is true, data after IEND is discarded rather than
    preserved.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not PNG")

    view = memoryview(data)
    position = len(PNG_SIGNATURE)
    saw_ihdr = False
    saw_iend = False
    saw_idat = False
    idat_interrupted = False
    while position < len(view):
        if len(view) - position < 12:
            raise ValueError("truncated PNG chunk header or payload")
        length = struct.unpack_from(">I", view, position)[0]
        payload_start = position + 8
        payload_end = payload_start + length
        chunk_end = payload_end + 4
        kind = bytes(view[position + 4 : position + 8])
        if chunk_end > len(view):
            raise ValueError(f"truncated PNG chunk {kind!r}")

        payload = view[payload_start:payload_end]
        stored_crc = struct.unpack_from(">I", view, payload_end)[0]
        actual_crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError(f"PNG CRC mismatch in {kind!r}")
        if not saw_ihdr and kind != b"IHDR":
            raise ValueError("PNG IHDR must be the first chunk")
        if kind == b"IHDR":
            if saw_ihdr:
                raise ValueError("PNG has multiple IHDR chunks")
            if length != 13:
                raise ValueError("PNG IHDR payload must be 13 bytes")
            saw_ihdr = True
        elif kind == b"IDAT":
            if idat_interrupted:
                raise ValueError("PNG IDAT chunks must be consecutive")
            saw_idat = True
        elif saw_idat:
            idat_interrupted = True
        if kind == b"IEND":
            if length != 0:
                raise ValueError("PNG IEND chunk must be empty")
            if not allow_trailing_data and chunk_end != len(view):
                raise ValueError("PNG has trailing bytes after IEND")
            saw_iend = True

        yield PNGChunk(kind, payload, view[position:chunk_end])
        position = chunk_end
        if saw_iend:
            break

    if not saw_ihdr:
        raise ValueError("PNG has no IHDR chunk")
    if not saw_iend:
        raise ValueError("PNG has no complete IEND chunk")
    if not saw_idat:
        raise ValueError("PNG has no IDAT chunk")
