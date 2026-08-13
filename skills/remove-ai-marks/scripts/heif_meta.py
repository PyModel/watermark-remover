"""Inspect/clean HEIF/AVIF (HEIC) provenance metadata — stdlib ISO-BMFF parsing.

Cleaning is **in-place neutralization**: C2PA/JUMBF boxes are re-typed to `free`
and their payloads zeroed, and AI-marker byte runs inside Exif/XMP item extents
are overwritten at equal length. Offsets never move, so the file stays valid and
camera/editor EXIF outside the matched tokens is preserved (mirrors the
remove-ai-watermarks v0.12 in-place approach).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import atomic_write_bytes
from image_meta import AI_META_HINTS, C2PA_MARKERS, _contains_any

HEIF_BRANDS = {
    b"heic",
    b"heix",
    b"hevc",
    b"heis",
    b"heim",
    b"hevm",
    b"hevs",
    b"mif1",
    b"msf1",
}
AVIF_BRANDS = {b"avif", b"avis"}

C2PA_BOX_TYPES = (b"jumb", b"JUMB", b"c2pa", b"C2PA", b"cabx", b"caBX")
C2PA_BMFF_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")
XMP_MIME = b"application/rdf+xml"


@dataclass
class Box:
    type: bytes
    header_start: int
    payload_start: int
    size: int  # total, header included

    @property
    def end(self) -> int:
        return self.header_start + self.size


def _iter_boxes(data: bytes, start: int, end: int) -> Iterator[Box]:
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos : pos + 4], "big")
        btype = data[pos + 4 : pos + 8]
        header = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            break  # corrupt / truncated
        yield Box(btype, pos, pos + header, size)
        pos += size


def _ftyp_brands(data: bytes) -> set[bytes]:
    for box in _iter_boxes(data, 0, min(len(data), 4096)):
        if box.type != b"ftyp":
            continue
        payload = data[box.payload_start : box.end]
        brands = {payload[:4]}
        for i in range(8, len(payload) - 3, 4):
            brands.add(payload[i : i + 4])
        return brands
    return set()


def detect_heif(data: bytes) -> str:
    """'avif' | 'heif' | 'unknown'."""
    brands = _ftyp_brands(data)
    if brands & AVIF_BRANDS:
        return "avif"
    if brands & HEIF_BRANDS:
        return "heif"
    return "unknown"


def _read_uint(data: bytes, pos: int, size: int, end: int, label: str) -> tuple[int, int]:
    if size < 0 or size > 8:
        raise ValueError(f"invalid {label} field width: {size}")
    if pos < 0 or end > len(data) or pos + size > end:
        raise ValueError(f"truncated {label} field")
    return (int.from_bytes(data[pos : pos + size], "big") if size else 0, pos + size)


def _parse_iinf(data: bytes, start: int, end: int) -> dict[int, tuple[bytes, str, str]]:
    """item_id -> (item_type, name, content_type). Supports infe v0-v3."""
    if start < 0 or end > len(data) or start + 4 > end:
        raise ValueError("truncated iinf full-box header")
    items: dict[int, tuple[bytes, str, str]] = {}
    version = data[start]
    pos = start + 4
    count, pos = _read_uint(data, pos, 2 if version == 0 else 4, end, "iinf count")
    if count > 100_000:
        raise ValueError("iinf entry count exceeds safety limit")
    seen = 0
    for box in _iter_boxes(data, pos, end):
        if box.type != b"infe":
            continue
        seen += 1
        p = box.payload_start
        if p + 4 > box.end:
            raise ValueError("truncated infe full-box header")
        item_version = data[p]
        q = p + 4
        if item_version == 2:
            item_id, q = _read_uint(data, q, 2, box.end, "infe item id")
        elif item_version == 3:
            item_id, q = _read_uint(data, q, 4, box.end, "infe item id")
        elif item_version in (0, 1):
            item_id, q = _read_uint(data, q, 2, box.end, "infe item id")
        else:
            raise ValueError(f"unsupported infe version: {item_version}")
        _, q = _read_uint(data, q, 2, box.end, "infe protection index")
        if item_version >= 2:
            if q + 4 > box.end:
                raise ValueError("truncated infe item type")
            item_type = data[q : q + 4]
            q += 4
        else:
            item_type = b""
        nul = data.find(b"\x00", q, box.end)
        if nul < 0:
            raise ValueError("unterminated infe item name")
        name = data[q:nul].decode("utf-8", errors="replace")
        content_type = ""
        if item_version >= 2 and item_type == b"mime":
            content_start = nul + 1
            content_end = data.find(b"\x00", content_start, box.end)
            if content_end < 0:
                raise ValueError("unterminated infe MIME content type")
            content_type = data[content_start:content_end].decode("ascii", errors="replace")
        items[item_id] = (item_type, name, content_type)
    if seen < count:
        raise ValueError(f"truncated iinf entries ({seen} < {count})")
    return items


def _parse_iloc(data: bytes, start: int, end: int) -> dict[int, tuple[int, list[tuple[int, int]]]]:
    """item_id -> (construction_method, [(absolute_offset, length), ...])."""
    if start < 0 or end > len(data) or start + 6 > end:
        raise ValueError("truncated iloc header")
    version = data[start]
    if version not in (0, 1, 2):
        raise ValueError(f"unsupported iloc version: {version}")
    b0, b1 = data[start + 4], data[start + 5]
    off_size, len_size = b0 >> 4, b0 & 0xF
    base_size = b1 >> 4
    index_size = (b1 & 0xF) if version in (1, 2) else 0
    for label, size in (
        ("offset", off_size),
        ("length", len_size),
        ("base offset", base_size),
        ("index", index_size),
    ):
        if size > 8:
            raise ValueError(f"invalid iloc {label} width: {size}")
    pos = start + 6
    count, pos = _read_uint(data, pos, 2 if version < 2 else 4, end, "iloc item count")
    if count > 100_000:
        raise ValueError("iloc item count exceeds safety limit")
    out: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    for _ in range(count):
        item_id, pos = _read_uint(data, pos, 2 if version < 2 else 4, end, "iloc item id")
        method = 0
        if version in (1, 2):
            raw_method, pos = _read_uint(data, pos, 2, end, "iloc construction method")
            method = raw_method & 0xF
        data_reference_index, pos = _read_uint(data, pos, 2, end, "iloc data reference index")
        if data_reference_index != 0:
            method = -1
        base, pos = _read_uint(data, pos, base_size, end, "iloc base offset")
        extent_count, pos = _read_uint(data, pos, 2, end, "iloc extent count")
        if extent_count > 100_000:
            raise ValueError("iloc extent count exceeds safety limit")
        extents: list[tuple[int, int]] = []
        for _ in range(extent_count):
            if version in (1, 2) and index_size:
                _, pos = _read_uint(data, pos, index_size, end, "iloc extent index")
            offset, pos = _read_uint(data, pos, off_size, end, "iloc extent offset")
            length, pos = _read_uint(data, pos, len_size, end, "iloc extent length")
            absolute = base + offset
            if method == 0 and (absolute > len(data) or length > len(data) - absolute):
                raise ValueError("iloc extent points outside file")
            extents.append((absolute, length))
        out[item_id] = (method, extents)
    return out


def _find_meta(data: bytes) -> Box | None:
    for box in _iter_boxes(data, 0, len(data)):
        if box.type == b"meta":
            return box
    return None


def _provenance_items(
    data: bytes,
) -> tuple[dict[int, tuple[bytes, str, str]], dict[int, tuple[int, list[tuple[int, int]]]]]:
    """(iinf items, iloc extents) from the meta box; empty when absent/corrupt."""
    meta = _find_meta(data)
    if meta is None:
        return {}, {}
    items: dict[int, tuple[bytes, str, str]] = {}
    extents: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    for child in _iter_boxes(data, meta.payload_start + 4, meta.end):
        if child.type == b"iinf":
            items = _parse_iinf(data, child.payload_start, child.end)
        elif child.type == b"iloc":
            extents = _parse_iloc(data, child.payload_start, child.end)
    return items, extents


def _is_c2pa_box(data: bytes, box: Box) -> bool:
    return box.type in C2PA_BOX_TYPES or (
        box.type == b"uuid"
        and box.payload_start + len(C2PA_BMFF_UUID) <= box.end
        and data[box.payload_start : box.payload_start + len(C2PA_BMFF_UUID)] == C2PA_BMFF_UUID
    )


def _c2pa_boxes(data: bytes) -> list[Box]:
    """JUMBF/C2PA boxes at top level or inside meta."""
    found: list[Box] = []
    for box in _iter_boxes(data, 0, len(data)):
        if _is_c2pa_box(data, box):
            found.append(box)
        if box.type == b"meta":
            for child in _iter_boxes(data, box.payload_start + 4, box.end):
                if _is_c2pa_box(data, child):
                    found.append(child)
    return found


def _item_extent_bytes(
    data: bytes,
    items: dict[int, tuple[bytes, str, str]],
    extents: dict[int, tuple[int, list[tuple[int, int]]]],
    wanted_types: tuple[bytes, ...],
) -> list[tuple[int, int, bytes, str]]:
    """(offset, length, item_type, content_type) for editable file extents."""
    out: list[tuple[int, int, bytes, str]] = []
    for item_id, (itype, _name, content_type) in items.items():
        if itype not in wanted_types:
            continue
        loc = extents.get(item_id)
        if not loc:
            continue
        method, exts = loc
        if method != 0:
            continue  # idat-relative extents not supported for editing
        for off, ln in exts:
            if 0 <= off < len(data) and off + ln <= len(data):
                out.append((off, ln, itype))
    return out


def _is_xmp_item(itype: bytes, payload: bytes) -> bool:
    return XMP_MIME in payload[:200] or payload.lstrip()[:1] == b"<"


def _neutralize_runs(buf: bytearray, start: int, length: int, pad: int) -> list[str]:
    """Overwrite AI/C2PA marker byte runs in-place at equal length."""
    seg = bytes(buf[start : start + length])
    lower = seg.lower()
    hits: list[str] = []
    for needle in AI_META_HINTS + C2PA_MARKERS:
        nl = needle.lower()
        i = lower.find(nl)
        while i != -1:
            hits.append(needle.decode("ascii", errors="replace"))
            for k in range(len(nl)):
                buf[start + i + k] = pad
            i = lower.find(nl, i + len(nl))
    return sorted(set(hits))


def inspect_heif(data: bytes) -> tuple[bool, bool, list[str], dict[str, Any]]:
    fmt = detect_heif(data)
    if fmt == "unknown":
        return False, False, ["not a HEIF/AVIF file"], {}
    findings: list[str] = []
    brands = sorted(b.decode("ascii", "replace") for b in _ftyp_brands(data) if b.strip())
    findings.append(f"brands: {', '.join(brands[:6])}")

    has_c2pa = False
    has_ai = False

    for box in _c2pa_boxes(data):
        has_c2pa = True
        findings.append(
            f"C2PA/JUMBF box '{box.type.decode('latin-1')}' at offset {box.header_start}"
        )

    try:
        items, extents = _provenance_items(data)
    except ValueError as error:
        findings.append(f"malformed HEIF metadata table: {error}")
        return (
            has_c2pa,
            True,
            findings,
            {
                "format": fmt,
                "brands": brands,
                "malformed": True,
            },
        )
    for off, ln, itype in _item_extent_bytes(data, items, extents, (b"Exif", b"mime")):
        payload = data[off : off + ln]
        hits = _contains_any(payload, AI_META_HINTS + C2PA_MARKERS)
        label = "Exif item" if itype == b"Exif" else "XMP/mime item"
        if hits:
            has_ai = True
            if any(h.lower() in ("c2pa", "contentcredentials", "jumb", "aigc") for h in hits):
                has_c2pa = True
            findings.append(f"{label} @ {off}: {', '.join(hits[:8])}")
        else:
            findings.append(f"{label} present ({ln} bytes, no AI markers)")
    if not items:
        findings.append("no iinf item table (or unsupported version)")

    if not has_c2pa:
        whole = _contains_any(data, C2PA_MARKERS)
        if whole:
            has_c2pa = True
            findings.append(f"byte-scan C2PA markers: {', '.join(whole[:6])}")
    return has_c2pa, has_ai or has_c2pa, findings, {"format": fmt, "brands": brands}


def neutralize_heif(data: bytes, *, strip_all_metadata: bool = True) -> tuple[bytes, list[str]]:
    """In-place neutralization on raw bytes. Returns (cleaned, actions)."""
    if detect_heif(data) == "unknown":
        raise ValueError("not a HEIF/AVIF file")
    actions: list[str] = []
    buf = bytearray(data)

    # 1. Neutralize C2PA/JUMBF boxes: retype to 'free', zero payload.
    for box in _c2pa_boxes(bytes(buf)):
        name = box.type.decode("latin-1")
        buf[box.header_start + 4 : box.header_start + 8] = b"free"
        for i in range(box.payload_start, box.end):
            buf[i] = 0
        actions.append(f"neutralized '{name}' box -> free (zeroed {box.size - 8} bytes)")

    # 2. Exif / XMP item extents.
    items, extents = _provenance_items(bytes(buf))
    item_exts = _item_extent_bytes(bytes(buf), items, extents, (b"Exif", b"mime"))
    for off, ln, itype in item_exts:
        payload = bytes(buf[off : off + ln])
        label = "Exif item" if itype == b"Exif" else "XMP/mime item"
        # HEIF `mime` items are generic. Only XML/XMP payloads are metadata;
        # never zero an arbitrary MIME item merely because its item_type is mime.
        if itype == b"mime" and not _is_xmp_item(itype, payload):
            continue
        if strip_all_metadata:
            pad = 0x20 if _is_xmp_item(itype, payload) else 0x00
            for i in range(off, off + ln):
                buf[i] = pad
            actions.append(f"zeroed entire {label} payload ({ln} bytes, offsets preserved)")
        else:
            pad = 0x20 if _is_xmp_item(itype, payload) else 0x00
            hits = _neutralize_runs(buf, off, ln, pad)
            if hits:
                actions.append(f"neutralized AI tokens in {label}: {', '.join(hits[:8])}")

    if not actions:
        actions.append("no HEIF/AVIF provenance metadata found")
    return bytes(buf), actions


def clean_heif(
    path: Path,
    dest: Path,
    *,
    strip_all_metadata: bool = True,
) -> dict[str, Any]:
    data = path.read_bytes()
    fmt = detect_heif(data)
    cleaned, actions = neutralize_heif(data, strip_all_metadata=strip_all_metadata)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(cleaned)

    has_c2pa, has_ai, post_findings, _ = inspect_heif(cleaned)
    return {
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        "actions": actions,
        "bytes_in": len(data),
        "bytes_out": dest.stat().st_size,
        "still_has_c2pa": has_c2pa,
        "still_has_ai_metadata": has_ai,
        "post_findings": post_findings,
    }
