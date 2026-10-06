# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/_internal/ebml.py,
# _internal/flv.py, and _internal/riff.py, Apache-2.0; modified: the readers take
# bytes instead of a path (no file I/O), each returns parsed TC260 AIGC labels as
# findings rather than raw payloads, and the three readers plus a magic-byte
# dispatch live in one module instead of one module per container. The bounded
# value size and the JSON parse come from this repo's tc260_label module.
"""Bounded provenance readers for the Matroska, FLV, and AVI containers.

TC260-PG-20257A (China's AI-content labeling standard) stores the same AIGC
label JSON natively in each container: a Matroska ``Segment.Tags.Tag.SimpleTag``
with ``TagName=AIGC``, an AVI ``LIST/INFO/AIGC`` chunk, and an FLV
``script.onMetaData.AIGC`` AMF0 string. Each reader self-gates on its container
magic bytes, walks the container structure while seeking over media payloads,
and validates every candidate value through the already-shipped bounded TC260
parser. Malformed, truncated, or oversized input returns no finding and raises
nothing.
"""

from __future__ import annotations

from typing import TypedDict

from tc260_label import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json


class ContainerFinding(TypedDict):
    """One native-container provenance finding: the container and its label."""

    container: str
    label: dict[str, str]


def _label_finding(value: bytes, container: str) -> ContainerFinding | None:
    """Validate one bounded candidate value into a finding, else None."""
    if not value or len(value) > MAX_TC260_VALUE_BYTES:
        return None
    label = parse_tc260_aigc_json(value)
    if label is None:
        return None
    return {"container": container, "label": label}


# --- Matroska / WebM (EBML) ------------------------------------------------

_EBML_MAGIC = b"\x1aE\xdf\xa3"
_SEGMENT_ID = 0x18538067
_TAGS_ID = 0x1254C367
_TAG_ID = 0x7373
_SIMPLE_TAG_ID = 0x67C8
_TAG_NAME_ID = 0x45A3
_TAG_STRING_ID = 0x4487
_MAX_TAG_NAME_BYTES = 256


def _vint_length(first: int, *, maximum: int) -> int | None:
    """Return an EBML variable-integer length from its first byte."""
    mask = 0x80
    for length in range(1, maximum + 1):
        if first & mask:
            return length
        mask >>= 1
    return None


def _element_header(data: bytes, pos: int, limit: int) -> tuple[int, int, int] | None:
    """Return ``(element_id, payload_start, element_end)`` inside ``limit``."""
    if pos < 0 or pos >= limit:
        return None
    id_length = _vint_length(data[pos], maximum=4)
    if id_length is None or pos + id_length >= limit:
        return None
    element_id = int.from_bytes(data[pos : pos + id_length], "big")

    size_first = pos + id_length
    size_length = _vint_length(data[size_first], maximum=8)
    if size_length is None:
        return None
    size_end = size_first + size_length
    if size_end > limit:
        return None
    marker = 1 << (8 - size_length)
    size_value = int.from_bytes(
        bytes([data[size_first] & (marker - 1)]) + data[size_first + 1 : size_end], "big"
    )
    payload_start = size_end
    unknown_size = size_value == (1 << (7 * size_length)) - 1
    element_end = limit if unknown_size else payload_start + size_value
    if element_end > limit:
        return None
    return element_id, payload_start, element_end


def _iter_elements(data: bytes, start: int, end: int) -> tuple[tuple[int, int, int], ...]:
    """Every valid direct child of one bounded EBML region, in file order."""
    elements: list[tuple[int, int, int]] = []
    pos = start
    while pos < end:
        header = _element_header(data, pos, end)
        if header is None:
            break
        elements.append(header)
        if header[2] <= pos:
            break
        pos = header[2]
    return tuple(elements)


def _bounded_value(data: bytes, start: int, end: int, maximum: int) -> bytes | None:
    size = end - start
    if size < 0 or size > maximum:
        return None
    value = data[start:end]
    return value if len(value) == size else None


def _simple_tag_findings(data: bytes, start: int, end: int) -> tuple[ContainerFinding, ...]:
    name: bytes | None = None
    values: list[bytes] = []
    for element_id, payload_start, element_end in _iter_elements(data, start, end):
        if element_id == _TAG_NAME_ID:
            name = _bounded_value(data, payload_start, element_end, _MAX_TAG_NAME_BYTES)
        elif element_id == _TAG_STRING_ID:
            value = _bounded_value(data, payload_start, element_end, MAX_TC260_VALUE_BYTES)
            if value is not None:
                values.append(value)
    if name != b"AIGC":
        return ()
    findings: list[ContainerFinding] = []
    for value in values:
        finding = _label_finding(value, "matroska")
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


def matroska_findings(data: bytes) -> tuple[ContainerFinding, ...]:
    """Read validated TC260 AIGC labels from Matroska/WebM ``SimpleTag`` entries."""
    if not data.startswith(_EBML_MAGIC):
        return ()
    found: list[ContainerFinding] = []
    for element_id, payload_start, element_end in _iter_elements(data, 0, len(data)):
        if element_id != _SEGMENT_ID:
            continue
        for child_id, child_start, child_end in _iter_elements(data, payload_start, element_end):
            if child_id != _TAGS_ID:
                continue
            for tag_id, tag_start, tag_end in _iter_elements(data, child_start, child_end):
                if tag_id != _TAG_ID:
                    continue
                for simple_id, simple_start, simple_end in _iter_elements(data, tag_start, tag_end):
                    if simple_id == _SIMPLE_TAG_ID:
                        found.extend(_simple_tag_findings(data, simple_start, simple_end))
    return tuple(found)
