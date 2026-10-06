# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/_internal/tc260_signature.py and _internal/tc260_producers.py, Apache-2.0; modified: the producer registry is merged into this module (the port keeps one TC260 module pair: tc260_label.py + tc260_verify.py), imports flattened to the stdlib script package, and the finding text is built carrier-agnostically for this repo's inspect findings
"""Verify TC260 ``SecurityData`` label signatures (TC260-PG-202511A).

The guide lets a label producer record a ``SecurityData`` JSON object in the
label's ``ReservedCode1``: SM3withSM2 signatures (``PubSD`` entries of type
``DS``), the public key (``PubKey``), and content bindings (``Bindings``).
Only the label signature is checked. MiniMax spells its type ``LabelMataData``
where the guide says ``Md``; both are accepted.

The key travels inside the file it signs, so a signature proves only integrity
unless the key matches a pinned signer in ``TC260_SIGNING_KEYS``. Verification
only: this module never signs or generates keys.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from sm2 import Point, lift_x, verify

Status = Literal["verified", "failed", "absent", "unsupported"]

SM3_WITH_SM2 = "1.2.156.10197.1.501"


# ---------------------------------------------------------------------------
# Registered producer identities (ported registry data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tc260Producer:
    """One producer's measured codes and independently established identity.

    ``image_mark`` / ``video_mark`` route a producer code to this repo's
    visible-mark vocabulary; the video consumers land in a later phase.
    """

    key: str
    codes: tuple[str, ...]
    vendor: str
    image_mark: str | None = None
    video_mark: str | None = None
    video_platform: str | None = None
    signer: str | None = None
    signing_key_x: int | None = None
    image_platform: str | None = None


TC260_PRODUCERS: tuple[Tc260Producer, ...] = (
    # YOYO JPEG exports; USCC independently verified in Honor's privacy statement.
    Tc260Producer("honor", ("91440300MA5G49LC9K",), "Honor", image_platform="Honor"),
    Tc260Producer(
        "doubao",
        ("91110102MACQD9K640", "doubao"),
        "ByteDance",
        "doubao",
        "doubao",
        "ByteDance Doubao",
    ),
    Tc260Producer("jimeng", ("9144030008867405X2",), "ByteDance", "jimeng"),
    Tc260Producer(
        "qwen", ("91440101MA9Y9T4H7A",), "Alibaba", "qwen", video_platform="Alibaba Cloud Qwen"
    ),
    # Tongyi Yunqi (Hangzhou), owned by Alibaba Cloud and Tongyi Lab, signs
    # both Wan and HappyHorse exports, not just one model.
    Tc260Producer(
        "wan",
        ("91330106MA2CFLDG4R",),
        "Alibaba",
        "wan",
        video_platform="Alibaba Tongyi (Wan, HappyHorse)",
    ),
    # Kling 3.0 Turbo served by Higgsfield (2026-09-24) writes the bare "kling".
    Tc260Producer(
        "kling", ("91110108335469089C", "kling"), "Kuaishou", "kling", "kling", "Kuaishou Kling AI"
    ),
    Tc260Producer("yuanbao", ("91440300708461136T",), "Tencent", "yuanbao"),
    Tc260Producer("runninghub", ("91340100MAEB4N8H76", "RunningHub"), "RunningHub", "runninghub"),
    Tc260Producer("baidu", ("91110000802100433B",), "Baidu", "baidu"),
    Tc260Producer("liblib", ("91110105MACJ6K1C8A",), "LiblibAI", "liblib"),
    # ShengShu's USCC, verified against the Beijing municipal list of 2025-03-03.
    Tc260Producer(
        "vidu",
        ("91110108MACC4D63XF",),
        "ShengShu",
        video_mark="vidu",
        video_platform="ShengShu Vidu",
    ),
    # Even-y SM2 point for the x coordinate carried by measured MiniMax exports
    # (Hailuo 2.3, H3 and H3 Max, 2026-09-29; upstream research sweep).
    Tc260Producer(
        "minimax",
        ("MiniMax",),
        "MiniMax",
        video_mark="hailuo",
        video_platform="MiniMax Hailuo AI",
        signer="MiniMax",
        signing_key_x=0xA0B3B0B6A0C9B0C89CAB328342AF4E8221EC5B40799CBE835AB4251F7B47E4FD,
    ),
)


def uscc_of(code: str) -> str:
    """Extract the USCC from ``001`` + type digit + USCC + product suffix."""
    return code[4:22] if len(code) >= 22 and code[:3] == "001" else code


_PRODUCERS_BY_CODE = {code.casefold(): row for row in TC260_PRODUCERS for code in row.codes}


def tc260_producer_in(producer: str, codes: Iterable[str]) -> bool:
    """Match a whole producer identity, USCC-normalized and casefolded."""
    code = uscc_of(producer.strip()).casefold()
    return bool(code) and any(code == candidate.casefold() for candidate in codes)


def producer_for_code(code: str) -> Tc260Producer | None:
    """Resolve a producer field; partial names never establish an identity."""
    return _PRODUCERS_BY_CODE.get(uscc_of(code.strip()).casefold())


def producer_for_signer(signer: str) -> Tc260Producer | None:
    """Resolve the identity returned by successful pinned-key verification."""
    return next((row for row in TC260_PRODUCERS if row.signer == signer), None)


def producer_codes_for_mark(mark: str) -> tuple[str, ...]:
    """Measured producer identities for an image or video mark, if registered."""
    row = next((row for row in TC260_PRODUCERS if mark in (row.image_mark, row.video_mark)), None)
    return row.codes if row else ()


# ---------------------------------------------------------------------------
# SecurityData label-signature verification
# ---------------------------------------------------------------------------


def _pinned(x: int) -> Point:
    point = lift_x(x, odd=False)
    if point is None:
        raise ValueError(f"pinned TC260 key x={x:#x} is not on the SM2 curve")
    return point


# Derived from the same producer rows used by metadata and visible-mark routing.
TC260_SIGNING_KEYS: dict[str, Point] = {
    row.signer: _pinned(row.signing_key_x)  # type: ignore[dict-item]
    for row in TC260_PRODUCERS
    if row.signer is not None and row.signing_key_x is not None
}

_LABEL_TBS_TYPES = frozenset({"Md", "LabelMataData"})
# The producer's signed fields, in GB 45438 appendix E order ("Label" to value3).
_PRODUCER_FIELDS = ("Label", "ContentProducer", "ProduceID")


@dataclass(frozen=True)
class Tc260Signature:
    """Outcome of checking one label's ``SecurityData`` label signature."""

    label: Status
    signer: str | None

    def describe(self) -> str:
        if self.label == "verified":
            if self.signer:
                return f"TC260 label signature verified (pinned {self.signer} key)"
            return "TC260 label signature verified with its own embedded key (no pinned signer)"
        if self.label == "failed":
            return "TC260 label signature does not verify"
        if self.label == "unsupported":
            return (
                "TC260 SecurityData signature algorithm or signed-content selection "
                "is not supported; integrity unverified"
            )
        return "TC260 SecurityData without a label signature"


def _candidate_keys(key_hex: str) -> tuple[list[Point], str | None]:
    """The points a ``KeyValue`` can denote, and the pinned signer when it matches one."""
    try:
        raw = bytes.fromhex(key_hex)
    except ValueError:
        return [], None
    if len(raw) == 65 and raw[0] == 4:
        raw = raw[1:]
    if len(raw) == 33 and raw[0] == 0:
        raw = raw[1:]  # MiniMax pads a bare x coordinate with 00
    if len(raw) == 64:
        points = [(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))]
    elif len(raw) == 33 and raw[0] in (2, 3):
        point = lift_x(int.from_bytes(raw[1:], "big"), odd=raw[0] == 3)
        points = [point] if point else []
    elif len(raw) == 32:
        x = int.from_bytes(raw, "big")  # a bare x coordinate: either y may be meant
        points = [p for p in (lift_x(x, odd=False), lift_x(x, odd=True)) if p]
    else:
        return [], None
    for signer, pinned in TC260_SIGNING_KEYS.items():
        if any(p[0] == pinned[0] for p in points):
            return [pinned], signer
    return points, None


def label_message(label: Mapping[str, str]) -> bytes:
    """The signed producer-field serialization (GB 45438 appendix A.1):
    ``"Label":"1","ContentProducer":"...","ProduceID":"..."``."""
    return ",".join(
        f"{json.dumps(field)}:{json.dumps(label.get(field, ''), ensure_ascii=False)}"
        for field in _PRODUCER_FIELDS
    ).encode()


def _object(value: object) -> dict[str, Any]:
    """``value`` as a JSON object, or empty when it is anything else."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict[str, Any]]:
    """The JSON objects in a JSON array, or none when ``value`` is not an array."""
    items = cast("list[object]", value) if isinstance(value, list) else []
    return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]


# Entries past this combined size skip the bounded cache: a label payload
# carrying megabytes of key/signature hex must not pin them as cache keys.
_MAX_CACHED_VERIFY_BYTES = 64 * 1024


def _verify_cached(point: Point, message: bytes, signature: bytes) -> bool:
    """``_verified`` with a per-entry size bound on the cache.

    ``inspect`` and ``strip`` re-read the same label, so small (key, message,
    signature) triples are cached; anything oversized verifies directly and
    stays out of the cache entirely.
    """
    if len(message) + len(signature) > _MAX_CACHED_VERIFY_BYTES:
        return verify(point, message, signature)
    return _verified(point, message, signature)


@functools.lru_cache(maxsize=256)
def _verified(point: Point, message: bytes, signature: bytes) -> bool:
    # inspect and strip re-read the same label; the pure-Python curve math
    # runs once per (key, message, signature) triple.
    return verify(point, message, signature)


def check_tc260_signature(label: Mapping[str, str]) -> Tc260Signature | None:
    """Check the producer's ``SecurityData`` in ``ReservedCode1``.

    Returns None when the label carries no ``SecurityData`` at all — absence of
    a signature mechanism is not a verdict. Malformed keys or signature
    encodings fail closed as ``failed`` rather than raising.
    """
    try:
        parsed: object = json.loads(label.get("ReservedCode1", ""))
    except ValueError:
        return None
    security = _object(_object(parsed).get("SecurityData"))
    if security.get("Type") != "TC260PG":
        return None
    public = _objects(security.get("PubSD", security.get("PubSd")))
    keys = {
        str(entry.get("KeyID", 0)): _candidate_keys(str(entry.get("KeyValue", "")))
        for entry in reversed(public)  # reversed: the first entry per KeyID wins
        if entry.get("Type") == "PubKey" and entry.get("AlgID") == SM3_WITH_SM2
    }
    message = label_message(label)
    results: list[bool] = []
    unsupported = False
    signer: str | None = None
    for entry in public:
        if entry.get("Type") != "DS":
            continue
        if (
            entry.get("AlgID") != SM3_WITH_SM2
            or _object(entry.get("TBSData")).get("Type") not in _LABEL_TBS_TYPES
        ):
            unsupported = True
            continue
        points, key_signer = keys.get(str(entry.get("KeyID", 0)), ([], None))
        try:
            signature = bytes.fromhex(str(entry.get("Signature", "")))
        except ValueError:
            signature = b""
        verified = any(_verify_cached(point, message, signature) for point in points)
        results.append(verified)
        if verified:
            signer = signer or key_signer
    if results:
        return Tc260Signature(label="verified" if any(results) else "failed", signer=signer)
    return Tc260Signature(label="unsupported" if unsupported else "absent", signer=None)


def tc260_label_findings(label: Mapping[str, str], *, carrier: str) -> list[str]:
    """Field-aware finding lines for one parsed TC260 AIGC label.

    The first line names the label type and the content producer resolved
    against the registry; a second line reports the ``SecurityData`` signature
    verdict when the label carries one. ``carrier`` anchors each line to the
    metadata placement it was found in, so the lines slot into the existing
    inspect findings list shape (no new report format).
    """
    producer = label.get("ContentProducer", "")
    row = producer_for_code(producer)
    producer_text = f"{producer} ({row.vendor})" if row else f"{producer} (not in TC260 registry)"
    fields = [f"Label={label.get('Label', '')}", f"ContentProducer={producer_text}"]
    produce_id = label.get("ProduceID", "")
    if produce_id:
        fields.append(f"ProduceID={produce_id}")
    lines = [f"{carrier}: TC260 AIGC label ({', '.join(fields)})"]
    signature = check_tc260_signature(label)
    if signature is not None:
        lines.append(f"{carrier}: {signature.describe()}")
    return lines
