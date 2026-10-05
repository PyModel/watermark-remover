# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/metadata.py (TC260 label constants, parse_tc260_aigc_json, aigc_label_from_metadata), Apache-2.0; modified: extracted into this repo's stdlib script package as a standalone module; audio placements (RIFF/ID3/Vorbis) dropped with the deferred audio/video phase
"""Bounded field-aware parsing of China's TC260 AIGC labels.

China's mandatory AI-content labeling (TC260, the national cybersecurity
standards committee) embeds a label JSON object in file metadata. AI
generators serving China write an XMP block in the TC260 namespace —
``<TC260:AIGC>{"Label":"1",...}`` (Doubao/ByteDance) — and the same standard
is mandatory for Jimeng, Kling, Qwen, Ernie and the rest of the China-AIGC
ecosystem. The label also appears as a raw-JSON ``{"AIGC":{...}}`` wrapper in
JPEG EXIF (UserComment), as a bare ``AIGC{...}`` blob near a JPEG APP segment,
and as a PNG ``tEXt`` chunk keyed ``AIGC`` carrying the raw JSON.

Every entry point is bounded: payloads over ``MAX_TC260_VALUE_BYTES`` are
refused before any JSON work, and malformed or non-UTF-8 input is a stable
rejection (None), never an exception.
"""

from __future__ import annotations

import html
import json
import re
from typing import cast

# Container-agnostic byte markers (XMP is text): a quick containment check,
# not evidence by itself.
AIGC_MARKERS: tuple[bytes, ...] = (
    b"tc260.org.cn/ns/AIGC",
    b"TC260:AIGC",
)

# TC260 AIGC-label JSON fields (the standard's labeling object). Doubao writes
# the producer-side schema as a PNG ``tEXt`` chunk keyed ``AIGC`` (raw JSON,
# not XMP), so a JSON object carrying at least one of these is accepted as a
# valid TC260 label even when the namespaced XMP element is absent.
TC260_AIGC_FIELDS: frozenset[str] = frozenset(
    {
        # Producer-side schema (Doubao and most China-served generators).
        "Label",
        "ContentProducer",
        "ProduceID",
        "ContentPropagator",
        "PropagateID",
        "ReservedCode1",
        "ReservedCode2",
        # Service-provider schema (Tencent Cloud's AIGC variant): the same
        # ``{"AIGC":{...}}`` wrapper but keyed ``ServiceProvider`` /
        # ``ServiceUser`` (+ generic ``Time`` / ``ContentId``, not gated on),
        # embedded in EXIF ``ImageDescription``.
        "ServiceProvider",
        "ServiceUser",
    }
)
_TC260_CANONICAL_FIELDS: dict[str, str] = {field.casefold(): field for field in TC260_AIGC_FIELDS}
MAX_TC260_VALUE_BYTES = 1024 * 1024

_XMP_AIGC = re.compile(rb'<TC260:AIGC>(.*?)</TC260:AIGC>|TC260:AIGC\s*=\s*"(.*?)"', re.DOTALL)


def _tc260_fields(parsed: dict[object, object]) -> dict[str, str]:
    """Keep structured reserved fields as JSON rather than Python repr strings."""
    return {
        _TC260_CANONICAL_FIELDS.get(str(key).casefold(), str(key)): (
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if isinstance(item, (dict, list))
            else str(item)
        )
        for key, item in parsed.items()
    }


def parse_tc260_aigc_json(value: bytes) -> dict[str, str] | None:
    """Parse a bounded JSON object carrying at least one normative TC260 field.

    Oversized (> 1 MiB) or non-UTF-8 payloads and non-object JSON are stable
    rejections: None, never an exception.
    """
    if len(value) > MAX_TC260_VALUE_BYTES:
        return None
    try:
        parsed = json.loads(value.rstrip(b"\x00 ").decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    fields = _tc260_fields(cast("dict[object, object]", parsed))
    return fields if TC260_AIGC_FIELDS & fields.keys() else None


def parse_tc260_aigc_text(text: str) -> dict[str, str] | None:
    """``parse_tc260_aigc_json`` for an already-decoded metadata string.

    Callers bound the source bytes before decoding; this re-encodes to UTF-8
    (ASCII metadata such as label codes round-trips losslessly).
    """
    return parse_tc260_aigc_json(text.encode("utf-8"))


def _parse_wrapped(text: str, *, require_tc260_field: bool) -> dict[str, str] | None:
    if require_tc260_field:
        return parse_tc260_aigc_text(text)
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _tc260_fields(cast("dict[object, object]", parsed))


def _scan_label(data: bytes) -> tuple[dict[str, str], tuple[int, int]] | None:
    """``(parsed label, byte span of the label JSON)`` from the XMP or generic
    wrapped forms, or None. The span is exclusive at the end and maps 1:1 onto
    ``data``'s bytes, so callers can overwrite exactly the label in place."""
    for match in _XMP_AIGC.finditer(data):
        group = 1 if match.group(1) is not None else 2
        decoded = html.unescape(match.group(group).decode("utf-8", "replace"))
        result = _parse_wrapped(decoded, require_tc260_field=False)
        if result:
            return result, (match.start(group), match.end(group))

    text = data.decode("latin-1")
    for needle in ('"AIGC"', "AIGC{"):
        start = text.find(needle)
        if start == -1:
            continue
        brace = text.find("{", start)
        if brace == -1:
            continue
        try:
            _, end = json.JSONDecoder().raw_decode(text, brace)
        except ValueError:
            continue
        result = _parse_wrapped(text[brace:end], require_tc260_field=True)
        if result:
            # latin-1 decodes 1:1, so character offsets are byte offsets.
            return result, (brace, end)
    return None


def label_from_metadata(data: bytes, candidates: tuple[str, ...] = ()) -> dict[str, str] | None:
    """Parse a China TC260 AI-labeling block from already collected metadata.

    ``candidates`` are decoded metadata strings (PNG text-chunk values and the
    like); each must parse as a JSON object carrying at least one TC260 field.
    The byte scan over ``data`` recognizes the unambiguous XMP forms (any JSON
    object is accepted there) and the generic wrapped ``{"AIGC":{...}}`` and
    bare ``AIGC{...}`` blobs, which must carry a TC260 field to count.
    """
    for candidate in candidates:
        result = parse_tc260_aigc_text(candidate)
        if result:
            return result
    found = _scan_label(data)
    return found[0] if found else None


def label_span(data: bytes) -> tuple[int, int] | None:
    """The ``(start, end)`` byte span of a label JSON inside ``data``, or None.

    Companion to :func:`label_from_metadata` for in-place neutralization: a
    stripper can overwrite exactly this run at equal length and keep every
    other byte (and file offset) intact.
    """
    found = _scan_label(data)
    return found[1] if found else None


def aigc_label_marker_in(data: bytes) -> bool:
    """True when ``data`` carries an unambiguous TC260 AIGC byte marker."""
    return any(marker in data for marker in AIGC_MARKERS)
