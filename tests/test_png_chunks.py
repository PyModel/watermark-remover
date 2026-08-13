"""Contracts for the shared, strict PNG chunk iterator."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from png_chunks import PNG_SIGNATURE, iter_png_chunks


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _ihdr() -> bytes:
    return struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)


def _png() -> bytes:
    return PNG_SIGNATURE + _chunk(b"IHDR", _ihdr()) + _chunk(b"IEND", b"")


def test_iterator_yields_validated_memory_views() -> None:
    data = _png()

    chunks = list(iter_png_chunks(data))

    assert [chunk.kind for chunk in chunks] == [b"IHDR", b"IEND"]
    assert bytes(chunks[0].payload) == _ihdr()
    assert bytes(chunks[0].raw) == _chunk(b"IHDR", _ihdr())
    assert isinstance(chunks[0].payload, memoryview)


def test_iterator_rejects_crc_mismatch() -> None:
    data = bytearray(_png())
    data[-1] ^= 0xFF

    with pytest.raises(ValueError, match="CRC mismatch"):
        list(iter_png_chunks(bytes(data)))


def test_iterator_rejects_truncated_chunk() -> None:
    data = PNG_SIGNATURE + struct.pack(">I", 20) + b"tEXt" + b"short"

    with pytest.raises(ValueError, match="truncated PNG chunk"):
        list(iter_png_chunks(data))


def test_iterator_requires_iend() -> None:
    data = PNG_SIGNATURE + _chunk(b"IHDR", _ihdr())

    with pytest.raises(ValueError, match="IEND"):
        list(iter_png_chunks(data))


def test_iterator_rejects_trailing_bytes() -> None:
    with pytest.raises(ValueError, match="trailing bytes"):
        list(iter_png_chunks(_png() + b"unexpected"))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            PNG_SIGNATURE + _chunk(b"IDAT", b"pixels") + _chunk(b"IEND", b""),
            "IHDR must be the first",
        ),
        (
            PNG_SIGNATURE
            + _chunk(b"IHDR", _ihdr())
            + _chunk(b"IHDR", _ihdr())
            + _chunk(b"IEND", b""),
            "multiple IHDR",
        ),
        (
            PNG_SIGNATURE + _chunk(b"IHDR", b"short") + _chunk(b"IEND", b""),
            "IHDR payload must be 13 bytes",
        ),
    ],
)
def test_iterator_enforces_ihdr_invariants(data: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(iter_png_chunks(data))
