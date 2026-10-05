# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/_internal/constants.py, Apache-2.0; modified: carve-out — the C2PA issuer/claim-generator/AI-vendor registries only; the CC BY 4.0 soft-binding algorithm snapshot (upstream `_generated_c2pa_soft_bindings.py`) and its derived label map are deliberately NOT ported (deferred-for-scope), so soft-binding lookups resolve to a labeled-unknown; adds a bounded stdlib CBOR mini-parser and PNG caBX scanner with stable length/depth/budget rejections.
"""C2PA constants registry and bounded CBOR/caBX scanning (stdlib only).

Carve-out: this module carries the issuer / claim-generator / AI-vendor
constants shared by C2PA consumers. It intentionally does NOT carry the
official soft-binding algorithm registry (CC BY 4.0 snapshot, deferred for
scope); soft-binding lookups degrade to :data:`SOFT_BINDING_UNKNOWN_LABEL`.

The CBOR mini-parser is bounded by design (untrusted input, defensive tier 1):

- payload size cap, nesting depth cap, item budget, per-string and aggregate
  text budgets (see ``MAX_CBOR_*`` constants);
- an explicit stack instead of recursion, so deep payloads cannot recurse;
- every violation produces a stable ``("rejected", reason)`` outcome — never
  an exception escaping to the caller.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
C2PA_CHUNK_TYPE = b"caBX"
C2PA_SIGNATURES = tuple(
    token.encode()
    for token in (
        "c2pa",
        "C2PA",
        "jumb",
        "jumd",
        "JUMBF",
        "cbor",
        "contentcreds",
        "digid",
        "assertions",
        "manifest",
    )
)

# Bounded-scan caps (defensive tier 1). Real Content Credentials manifests are
# tens to hundreds of KiB; the caps sit well above that and far below anything
# that could exhaust memory or time.
MAX_CBOR_PAYLOAD_BYTES = 1 << 20  # 1 MiB
MAX_CABX_CHUNK_BYTES = 8 << 20  # 8 MiB caBX chunk payload cap
MAX_CBOR_DEPTH = 32
MAX_CBOR_ITEMS = 4096
MAX_CBOR_TEXT_BYTES = 1 << 16  # per-string decode cap
MAX_CBOR_TOTAL_TEXT_BYTES = 1 << 18  # aggregate decoded-text cap

# Stable rejection reasons surfaced in scan results.
_CBOR_REASONS = {
    "too_large": "payload exceeds the CBOR scan cap",
    "truncated": "truncated CBOR item",
    "depth": "CBOR depth cap exceeded",
    "items": "CBOR item budget exceeded",
    "indefinite": "unsupported indefinite-length CBOR item",
    "text_budget": "CBOR text budget exceeded",
}


@dataclass(frozen=True, slots=True)
class CborScanResult:
    """Outcome of one bounded CBOR scan (never raises on malformed input)."""

    status: str  # "ok" | "rejected"
    reason: str | None
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class C2paAiVendor:
    """One C2PA identity and its AI-product or neutral-signer attribution."""

    issuer: bytes
    org: str
    platform: str | None
    needle: str | None
    synthid: bool = False
    asserts_ai: bool = False
    synthid_requires_watermark_action: bool = False
    signer_platform: str | None = None
    raw_signer_platforms: tuple[tuple[bytes, str], ...] = ()


def _vendor(
    issuer: bytes | str,
    org: str,
    platform: str | None,
    needle: str | None,
    *,
    synthid: bool = False,
    asserts_ai: bool = False,
    synthid_requires_watermark_action: bool = False,
    signer_platform: str | None = None,
    raw_signer_platforms: tuple[tuple[bytes, str], ...] = (),
) -> C2paAiVendor:
    token = issuer.encode() if isinstance(issuer, str) else issuer
    return C2paAiVendor(
        token,
        org,
        platform,
        needle,
        synthid=synthid,
        asserts_ai=asserts_ai,
        synthid_requires_watermark_action=synthid_requires_watermark_action,
        signer_platform=signer_platform,
        raw_signer_platforms=raw_signer_platforms,
    )


# Order is product priority when a manifest mentions more than one organization.
C2PA_AI_VENDORS: tuple[C2paAiVendor, ...] = (
    _vendor(b"Microsoft", "Microsoft", "Microsoft (Copilot / Designer)", "Microsoft"),
    _vendor(b"Adobe", "Adobe", "Adobe Firefly", "Adobe"),
    _vendor(
        b"OpenAI",
        "OpenAI",
        "OpenAI (ChatGPT / GPT Image / DALL·E / Sora)",
        "OpenAI",
        synthid=True,
        synthid_requires_watermark_action=True,
    ),
    _vendor(b"Google", "Google LLC", "Google (Gemini / Imagen)", "Google", synthid=True),
    _vendor(
        b"Stability AI", "Stability AI", "Stability AI (Stable Image / DreamStudio)", "Stability AI"
    ),
    _vendor(
        b"Black Forest Labs", "Black Forest Labs", "Black Forest Labs (FLUX)", "Black Forest Labs"
    ),
    _vendor(
        b"volcengine",
        "ByteDance (Volcano Engine)",
        "ByteDance Volcano Engine",
        "Volcano Engine",
    ),
    _vendor(
        "北京火山引擎科技有限公司",
        "ByteDance (Volcano Engine)",
        "ByteDance Volcano Engine",
        "Volcano Engine",
    ),
    _vendor(b"Byteplus", "BytePlus (ByteDance)", "BytePlus (ByteDance)", "BytePlus"),
    _vendor(
        b"Dreamina",
        "ByteDance (Dreamina)",
        "ByteDance Dreamina",
        "Dreamina",
        asserts_ai=True,
    ),
    # Amazon Bedrock signs Nova Canvas images and Nova Reel video as "Amazon Web
    # Services, Inc." with claim generator "Amazon Bedrock" (aws-samples outputs,
    # checked 2026-09-26). Listed before Canva: the Nova Canvas software agent
    # contains the Canva token, so a byte scan finds both.
    _vendor(b"Amazon Web Services", "Amazon", "Amazon Bedrock (Nova)", "Amazon"),
    _vendor(b"Canva", "Canva", "Canva (Magic Media)", "Canva"),
    _vendor(b"Eleven Labs", "ElevenLabs", "ElevenLabs", "ElevenLabs"),
    # fal signs as "fal - Features & Labels Inc." (common name "fal.ai"); "fal-ai" is
    # its claim-generator and software-agent prefix, which only the byte fallback reads.
    _vendor(b"Features & Labels", "fal.ai", "fal.ai", "fal.ai", asserts_ai=True),
    _vendor(b"fal-ai", "fal.ai", "fal.ai", "fal.ai", asserts_ai=True),
    _vendor(b"Bria", "Bria Artificial Intelligence", "Bria AI", "Bria", asserts_ai=True),
    # Ideogram signs its downloads' Content Credentials with "Ideogram, Inc"; the
    # issuer token is the org prefix (same substring-match class as "Bria" in
    # "Bria Artificial Intelligence"). Found as an unmapped signer on 4 corpus
    # uploads 2026-08-08 that identify reported as unknown-signer C2PA.
    _vendor(b"Ideogram", "Ideogram", "Ideogram", "Ideogram", asserts_ai=True),
    _vendor(b"xAI Grok Imagine", "xAI Grok Imagine", "xAI Grok Imagine", "xAI", asserts_ai=True),
    _vendor(b"Producer.ai", "Producer.ai", "Producer.ai", "Producer.ai", asserts_ai=True),
    # This signature names only SPRING's legal entity, not a product. Preserve
    # that signer identity without guessing which product produced a file.
    _vendor(b"SPRING (SG) PTE. LTD.", "SPRING (SG) PTE. LTD.", "SPRING (SG) PTE. LTD.", "SPRING"),
    # These companies also sign ordinary edits and publications. Register the
    # signer identity for provenance display, but provide no AI platform and do
    # not let the identity assert an AI verdict.
    _vendor(
        b"TikTok Inc.",
        "TikTok",
        None,
        None,
        signer_platform="TikTok (C2PA signer)",
    ),
    _vendor(
        b"Bytedance Pte",
        "ByteDance",
        None,
        None,
        signer_platform="ByteDance (C2PA signer)",
        raw_signer_platforms=(
            (b"Bytedance Pte. Ltd.", "ByteDance (C2PA signer)"),
            (b"CapCut/", "CapCut (C2PA signer)"),
            (b"capcut c2pa-rs", "CapCut (C2PA signer)"),
        ),
    ),
    _vendor(
        b"Anthropic Claude Content Signing",
        "Anthropic Claude",
        None,
        None,
        signer_platform="Anthropic Claude (C2PA signer)",
    ),
    _vendor(
        b"Samsung Galaxy",
        "Samsung Galaxy",
        None,
        None,
        signer_platform="Samsung Galaxy (C2PA)",
    ),
    _vendor(
        b"com.asus.gallery",
        "ASUS Gallery",
        None,
        None,
        signer_platform="ASUS Gallery (C2PA signer)",
    ),
    # Certificate common name on Google Photos edits (Android and iOS). The
    # manifest also names Google LLC, so an AI edit establishes Google SynthID
    # and its platform label lives in identify; an edited Gemini or OpenAI
    # generation keeps its generator.
    _vendor(b"Google Photos", "Google Photos", None, None),
    # Runway signs only its own models (Gen-4 image, Gen-4.5 video, measured
    # 2026-09-24: issuer "RUNWAY AI, INC.", software agent "Runway Image/Video
    # Generation"); a third-party model run inside Runway keeps its vendor's own
    # manifest. The C2PA conformance list spells the organization "Runway AI, Inc".
    _vendor(b"RUNWAY AI, INC.", "Runway", "Runway", "Runway"),
    _vendor(b"Runway AI, Inc", "Runway", "Runway", "Runway"),
    _vendor(b"Truepic", "Truepic", None, None),
)

C2PA_ISSUERS = {vendor.issuer: vendor.org for vendor in C2PA_AI_VENDORS}
# Google signers that re-encode media rather than generate it, so their Google
# LLC identity alone says nothing about SynthID. YouTube re-signs every upload;
# a manifest from one of these signers establishes SynthID only when the chain
# records a SynthID action. Google Photos is deliberately absent: its manifests
# record no SynthID action.
SYNTHID_EDIT_SIGNERS: tuple[bytes, ...] = (b"YouTube",)
C2PA_IDENTITY_AI_ORGS = frozenset(vendor.org for vendor in C2PA_AI_VENDORS if vendor.asserts_ai)
C2PA_SIGNER_PLATFORM_BY_ORG = {
    vendor.org: vendor.signer_platform
    for vendor in C2PA_AI_VENDORS
    if vendor.signer_platform is not None
}
C2PA_SIGNER_PLATFORMS = tuple(
    pair
    for vendor in C2PA_AI_VENDORS
    if vendor.signer_platform is not None
    for pair in (vendor.raw_signer_platforms or ((vendor.issuer, vendor.signer_platform),))
)

# Product-specific claim generators can sign through a different upstream issuer.
C2PA_CLAIM_GENERATOR_PLATFORMS: tuple[tuple[str, str], ...] = (
    ("grok imagine", "xAI Grok Imagine"),
    ("suno", "Suno"),
    ("adobe_firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("dreamina", "ByteDance Dreamina"),
    ("higgsfield ai", "Higgsfield AI"),
    ("recraft.ai", "Recraft"),
    ("topaz labs image api", "Topaz Labs"),
    ("tiktok ad creative toolbox", "TikTok Ad Creative Toolbox"),
    ("fastvid", "FastVid"),
    ("capcut", "CapCut (C2PA signer)"),
)

C2PA_AI_TOOLS = {
    token.encode(): label
    for token, label in (
        ("GPT-4o", "GPT-4o"),
        ("ChatGPT", "ChatGPT"),
        ("Sora", "Sora"),
        ("DALL-E", "DALL·E"),
        ("DALL", "DALL·E"),
        ("Imagen", "Imagen"),
        ("Firefly", "Firefly"),
        ("Dreamina", "Dreamina"),
    )
}

_C2PA_ACTION_NAMES = (
    "created",
    "converted",
    "edited",
    "filtered",
    "cropped",
    "resized",
    "opened",
    "placed",
    "watermarked.unbound",
)
C2PA_ACTIONS = {f"c2pa.{action}".encode(): action for action in _C2PA_ACTION_NAMES}

# ---------------------------------------------------------------------------
# Soft-binding carve-out: the official algorithm registry (CC BY 4.0 snapshot)
# is NOT ported. Every lookup resolves to a stable labeled-unknown.
# ---------------------------------------------------------------------------

SOFT_BINDING_REGISTRY_STATE = "not-ported"
SOFT_BINDING_UNKNOWN_LABEL = "unknown soft-binding algorithm (registry not ported)"


def soft_binding_label(algorithm: str | bytes) -> str:
    """Return the stable labeled-unknown for any soft-binding algorithm.

    The CC BY 4.0 soft-binding snapshot is deferred-for-scope, so this module
    cannot name algorithms. It must never guess: the label carries no algorithm
    identity, only the fact that the registry is unavailable.
    """
    return SOFT_BINDING_UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Bounded CBOR mini-parser (RFC 8949 subset, definite lengths only)
# ---------------------------------------------------------------------------


def _cbor_argument(
    payload: bytes, cursor: int, additional: int
) -> tuple[int | None, int, str | None]:
    """Decode one CBOR head argument. Returns (value, next_cursor, reject_reason)."""
    if additional < 24:
        return additional, cursor, None
    width = {24: 1, 25: 2, 26: 4, 27: 8}.get(additional)
    if width is None:  # 28-31: indefinite lengths / break
        return None, cursor, _CBOR_REASONS["indefinite"]
    end = cursor + width
    if end > len(payload):
        return None, cursor, _CBOR_REASONS["truncated"]
    return int.from_bytes(payload[cursor:end], "big"), end, None


def decode_cbor_texts(
    payload: bytes,
    *,
    max_depth: int = MAX_CBOR_DEPTH,
    max_items: int = MAX_CBOR_ITEMS,
    max_payload_bytes: int = MAX_CBOR_PAYLOAD_BYTES,
) -> CborScanResult:
    """Decode definite-length CBOR text strings from ``payload`` (bounded).

    Walks one CBOR sequence with an explicit stack — no recursion. Any length,
    depth, or budget violation stops the scan with a stable rejection reason.
    Malformed UTF-8 inside a text string decodes with replacements (stable, and
    still bounded); it is not a structural violation.
    """
    if len(payload) > max_payload_bytes:
        return CborScanResult("rejected", _CBOR_REASONS["too_large"], ())
    stack: list[int] = []  # remaining child slots per open container
    texts: list[str] = []
    decoded_bytes = 0
    items = 0
    cursor = 0
    size = len(payload)
    while True:
        while stack and stack[-1] == 0:
            stack.pop()
        if not stack and cursor >= size:
            break
        if cursor >= size:
            return CborScanResult("rejected", _CBOR_REASONS["truncated"], ())
        if stack and stack[-1] < 0:  # defensive; cannot happen with the accounting below
            return CborScanResult("rejected", _CBOR_REASONS["items"], ())
        head = payload[cursor]
        cursor += 1
        argument, cursor, reject = _cbor_argument(payload, cursor, head & 0x1F)
        if reject is not None:
            return CborScanResult("rejected", reject, ())
        assert argument is not None  # narrow for type checkers; reject covers None
        items += 1
        if items > max_items:
            return CborScanResult("rejected", _CBOR_REASONS["items"], ())
        major = head >> 5
        if major in (0, 1, 7):
            pass  # integers, negative integers, simple values: no content bytes
        elif major == 2:  # byte string: skipped, never decoded
            cursor += argument
            if cursor > size:
                return CborScanResult("rejected", _CBOR_REASONS["truncated"], ())
        elif major == 3:  # text string
            if cursor + argument > size:
                return CborScanResult("rejected", _CBOR_REASONS["truncated"], ())
            if (
                argument <= MAX_CBOR_TEXT_BYTES
                and decoded_bytes + argument <= MAX_CBOR_TOTAL_TEXT_BYTES
            ):
                texts.append(payload[cursor : cursor + argument].decode("utf-8", errors="replace"))
                decoded_bytes += argument
            elif (
                argument > MAX_CBOR_TEXT_BYTES
                or decoded_bytes + argument > MAX_CBOR_TOTAL_TEXT_BYTES
            ):
                return CborScanResult("rejected", _CBOR_REASONS["text_budget"], ())
            cursor += argument
        else:  # major 4 (array) / major 5 (map)
            if len(stack) + 1 > max_depth:
                return CborScanResult("rejected", _CBOR_REASONS["depth"], ())
            if stack:
                stack[-1] -= 1
            stack.append(argument if major == 4 else argument * 2)
            continue
        if stack:
            stack[-1] -= 1
    return CborScanResult("ok", None, tuple(texts))


def cbor_text_after(payload: bytes, key: bytes) -> str | None:
    """Decode a definite-length CBOR text value immediately following ``key``.

    Ported from the upstream PNG fallback (bounded targeted extraction for
    JUMBF-wrapped claims where a full CBOR walk cannot start at offset 0).
    Capped at :data:`MAX_CBOR_TEXT_BYTES` per string.
    """
    key_end = payload.find(key)
    if key_end < 0:
        return None
    cursor = key_end + len(key)
    if cursor >= len(payload):
        return None
    initial = payload[cursor]
    if 0x60 <= initial <= 0x77:
        length, cursor = initial & 0x1F, cursor + 1
    elif initial == 0x78 and cursor + 1 < len(payload):
        length, cursor = payload[cursor + 1], cursor + 2
    elif initial == 0x79 and cursor + 2 < len(payload):
        length = int.from_bytes(payload[cursor + 1 : cursor + 3], "big")
        cursor += 3
    elif initial == 0x7A and cursor + 4 < len(payload):
        length = int.from_bytes(payload[cursor + 1 : cursor + 5], "big")
        cursor += 5
    else:
        return None
    if length > MAX_CBOR_TEXT_BYTES:
        return None
    raw = payload[cursor : cursor + length]
    if len(raw) != length:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return raw.decode("latin1", errors="replace")


def _ordered_matches(buffer: bytes, registry: dict[bytes, str]) -> list[str]:
    return list(dict.fromkeys(label for token, label in registry.items() if token in buffer))


def _looks_like_c2pa(payload: bytes) -> bool:
    lowered = payload.lower()
    return (
        any(signature in payload for signature in C2PA_SIGNATURES)
        or b"c2pa" in lowered
        or b"jumb" in lowered
    )


_PNG_HEADER = struct.Struct(">I4s")
_TIMESTAMP_RE = re.compile(rb"\d{14}Z")


def find_c2pa_chunk(path: Path) -> tuple[bytes | None, str | None]:
    """Return ``(payload, None)`` for the first recognizable caBX chunk, or ``(None, reason)``.

    Walks PNG chunk headers with seeks and bounds: a chunk whose declared length
    exceeds :data:`MAX_CABX_CHUNK_BYTES` is reported, never read.
    """
    try:
        stream = path.open("rb")
    except OSError:
        return None, "unreadable file"
    with stream:
        if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return None, "not a PNG"
        file_size = stream.seek(0, 2)
        stream.seek(len(PNG_SIGNATURE))
        while True:
            header = stream.read(_PNG_HEADER.size)
            if len(header) != _PNG_HEADER.size:
                return None, None
            length, kind = _PNG_HEADER.unpack(header)
            if length + 4 > file_size - stream.tell():
                return None, None
            if kind == C2PA_CHUNK_TYPE:
                if length > MAX_CABX_CHUNK_BYTES:
                    return None, f"caBX chunk exceeds the {MAX_CABX_CHUNK_BYTES} byte scan cap"
                payload = stream.read(length)
                if _looks_like_c2pa(payload):
                    return payload, None
            else:
                stream.seek(length + 4, 1)
            if kind == b"IEND":
                return None, None


def scan_c2pa_hints(path: Path) -> dict[str, Any]:
    """Bounded stdlib C2PA hint scan for a PNG (fallback when the wheel is absent).

    Byte-level registry matches over the caBX chunk payload, anchored
    definite-length text extraction for claim-generator/spec strings, and one
    strict bounded CBOR walk recorded with its stable outcome. No finding is
    returned unless a registry token matched or an anchored extraction
    succeeded; the scan never guesses.
    """
    base: dict[str, Any] = {
        "has_cabx_chunk": False,
        "found": False,
        "claim_generator": None,
        "c2pa_spec": None,
        "timestamp": None,
        "issuers": [],
        "ai_tools": [],
        "claim_generator_platforms": [],
        "actions": [],
        "soft_binding_algorithms": [],
        "chunk_bytes": None,
        "cbor": {"status": "skipped", "reason": "no caBX chunk"},
        "rejected_reason": None,
    }
    payload, reject = find_c2pa_chunk(path)
    if reject is not None:
        base["rejected_reason"] = reject
        base["cbor"] = {"status": "rejected", "reason": reject}
        return base
    if payload is None:
        return base
    base["has_cabx_chunk"] = True
    base["chunk_bytes"] = len(payload)
    base["issuers"] = _ordered_matches(payload, C2PA_ISSUERS)
    base["ai_tools"] = _ordered_matches(payload, C2PA_AI_TOOLS)
    base["actions"] = _ordered_matches(payload, C2PA_ACTIONS)
    for key, output_key in (
        (b"claim_generator", "claim_generator"),
        (b"name", "claim_generator"),
        (b"specVersion", "c2pa_spec"),
    ):
        if base[output_key] is None:
            value = cbor_text_after(payload, key)
            if value and value.isprintable():
                base[output_key] = value
    generator = base["claim_generator"]
    if generator:
        lowered = generator.lower()
        base["claim_generator_platforms"] = [
            platform for token, platform in C2PA_CLAIM_GENERATOR_PLATFORMS if token in lowered
        ]
    algorithm = cbor_text_after(payload, b"alg")
    if algorithm:
        base["soft_binding_algorithms"] = [
            {"algorithm": algorithm, "label": soft_binding_label(algorithm)}
        ]
    timestamps = _TIMESTAMP_RE.findall(payload[:MAX_CABX_CHUNK_BYTES])
    if timestamps:
        base["timestamp"] = timestamps[0].decode("ascii")
    scan = decode_cbor_texts(payload)
    base["cbor"] = {"status": scan.status, "reason": scan.reason}
    base["found"] = bool(
        base["issuers"]
        or base["ai_tools"]
        or base["claim_generator_platforms"]
        or base["actions"]
        or generator
    )
    return base
