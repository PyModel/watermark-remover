"""Inspect/clean AI provenance metadata in non-raster containers.

Formats: SVG, PDF (best-effort), DOCX, XLSX, PPTX, ODT, EPUB, HTML,
Markdown frontmatter. Stdlib-first; PDF prefers optional exiftool/c2patool
when present.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import external_command
from common import atomic_write_bytes, atomic_write_text, classify_finding_confidence, which
from image_meta import (
    AI_META_HINTS,
    C2PA_MARKERS,
    detect_format,
    inspect_avif,
    inspect_bmp,
    inspect_gif,
    inspect_heic,
    inspect_jpeg,
    inspect_png,
    inspect_tiff,
    inspect_webp,
    run_optional_tools,
    strip_avif,
    strip_bmp,
    strip_gif,
    strip_heic,
    strip_jpeg,
    strip_png,
    strip_tiff,
    strip_webp,
)

run_command = external_command.run_command

# Frontmatter / meta keys that often carry AI provenance
AI_FRONTMATTER_KEYS = frozenset(
    {
        "generator",
        "ai",
        "ai_generated",
        "ai-generated",
        "claude",
        "anthropic",
        "openai",
        "gemini",
        "synthid",
        "c2pa",
        "content_credentials",
        "contentcredentials",
        "provenance",
        "digital_source_type",
        "digitalsourcetype",
        "created_with",
        "createdwith",
        "model",
        "llm",
    }
)

AI_META_NAME_RE = re.compile(
    r"generator|ai[-_ ]?generated|claude|anthropic|openai|gemini|synthid|"
    r"c2pa|content.?credential|provenance|digital.?source|aigc",
    re.I,
)


@dataclass
class ContainerInspectReport:
    path: str
    format: str
    has_c2pa: bool
    has_ai_metadata: bool
    findings: list[str] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    layer_a_total: int = 0
    layer_a_hits: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "has_c2pa": self.has_c2pa,
            "has_ai_metadata": self.has_ai_metadata,
            "findings": self.findings,
            "tools": self.tools,
            "details": self.details,
            "findings_confidence": [classify_finding_confidence(f) for f in self.findings],
            "notes": self.notes,
            "suspicious_total": self.layer_a_total,
            "layer_a_hits": self.layer_a_hits,
        }


def detect_container_format(path: Path, data: bytes | None = None) -> str:
    ext = path.suffix.lower()
    if ext in (".svg",):
        return "svg"
    if ext in (".pdf",):
        return "pdf"
    if ext in (".docx",):
        return "docx"
    if ext in (".xlsx",):
        return "xlsx"
    if ext in (".pptx",):
        return "pptx"
    if ext in (".odt",):
        return "odt"
    if ext in (".epub",):
        return "epub"
    if ext in (".html", ".htm"):
        return "html"
    if ext in (".md", ".markdown", ".mdx"):
        return "markdown"
    if data is not None:
        if data[:4] == b"%PDF":
            return "pdf"
        if data[:100].lstrip().startswith(b"<") and b"svg" in data[:500].lower():
            return "svg"
        if data[:2] == b"PK":
            # zip-based; sniff. A bounded prefix can truncate the central
            # directory, so when the in-memory parse fails, fall back to the
            # on-disk archive — a metadata-only read of the tail directory.
            sources: list[io.BytesIO | Path] = [io.BytesIO(data)]
            if path.is_file():
                sources.append(path)
            for source in sources:
                try:
                    with zipfile.ZipFile(source) as zf:
                        _validate_zip(zf)
                        names = set(zf.namelist())
                        if "word/document.xml" in names:
                            return "docx"
                        if "xl/workbook.xml" in names:
                            return "xlsx"
                        if "ppt/presentation.xml" in names:
                            return "pptx"
                        if "content.xml" in names and "meta.xml" in names:
                            return "odt"
                        if "META-INF/container.xml" in names and any(
                            n.endswith(".opf") for n in names
                        ):
                            return "epub"
                except (zipfile.BadZipFile, ValueError, RuntimeError, OSError):
                    continue
    if data is not None:
        head = data[:1024].lstrip()
        if (head.startswith(b"<") and b"<html" in head.lower()) or head.startswith(b"<!doctype"):
            return "html"
        if head.startswith(b"<svg") or (head.startswith(b"<") and b"svg" in head.lower()):
            return "svg"
        if data[:1024].startswith(b"---") or b"\n---\n" in data[:1024]:
            return "markdown"
        if b"# " in data[:1024] or b"\n# " in data[:1024]:
            return "markdown"
    return "unknown"


def _blob_hits(blob: bytes) -> tuple[bool, bool, list[str]]:
    lower = blob.lower()
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    for n in C2PA_MARKERS:
        if n.lower() in lower:
            has_c2pa = True
            findings.append(f"marker:{n.decode('ascii', errors='replace')}")
    for n in AI_META_HINTS:
        if n.lower() in lower:
            has_ai = True
            label = n.decode("ascii", errors="replace")
            if label not in {f.split(":", 1)[-1] for f in findings}:
                findings.append(f"ai:{label}")
    return has_c2pa, has_ai or has_c2pa, findings[:30]


# ---------------------------------------------------------------------------
# Embedded data URIs (SVG, HTML, Markdown)
# ---------------------------------------------------------------------------

# data:image/<format>;base64,<payload> — the payload may hold AI provenance
# (e.g. a PNG with an e-ORIGIN/CC chunk) that byte-scanning the raw container
# cannot see.
RE_DATA_IMAGE_URI = re.compile(
    r"data:image\/(?P<mime>[a-zA-Z0-9\+\-\-\.]+)(?P<params>;[^\s\"'\(\)<>]+)?,(?P<payload>[A-Za-z0-9+/=\s%]+)",
    re.I,
)


def _media_strip_succeeded(sub_actions: list[str], cleaned: bytes, raw: bytes) -> bool:
    """True when a media stripper changed bytes while reporting a removal.

    Most raster strippers "drop" chunks/segments; heif_meta neutralizes in
    place ("neutralized"/"zeroed") to preserve offsets, so accept both
    vocabularies. The no-op case always returns the input bytes unchanged.
    """
    if cleaned == raw:
        return False
    verbs = ("drop", "neutraliz", "zero")
    return any(verb in action.lower() for action in sub_actions for verb in verbs)


def _inspect_embedded_data_uris(text: str) -> tuple[bool, bool, list[str]]:
    has_c2pa = False
    has_ai = False
    findings: list[str] = []

    import base64
    import urllib.parse

    for m in RE_DATA_IMAGE_URI.finditer(text):
        mime = m.group("mime").lower()
        params = (m.group("params") or "").lower()
        payload = m.group("payload")
        is_b64 = "base64" in params

        try:
            if is_b64:
                raw_b64 = re.sub(r"\s+", "", payload)
                pad = len(raw_b64) % 4
                if pad:
                    raw_b64 += "=" * (4 - pad)
                data = base64.b64decode(raw_b64)
            else:
                data = urllib.parse.unquote_to_bytes(payload)
        except Exception:  # noqa: S112 — skip malformed payloads, keep scanning
            continue

        if not data:
            continue

        fmt = detect_format(data)
        if fmt == "png":
            sub_c2pa, sub_ai, sub_findings = inspect_png(data)
        elif fmt == "jpeg":
            sub_c2pa, sub_ai, sub_findings = inspect_jpeg(data)
        elif fmt == "webp":
            sub_c2pa, sub_ai, sub_findings = inspect_webp(data)
        elif fmt == "avif":
            sub_c2pa, sub_ai, sub_findings = inspect_avif(data)
        elif fmt == "heif":
            sub_c2pa, sub_ai, sub_findings = inspect_heic(data)
        elif "svg" in mime or data.lstrip().startswith(b"<"):
            sub_c2pa, sub_ai, sub_findings, _ = inspect_svg(data)
        else:
            sub_c2pa, sub_ai, sub_findings = _blob_hits(data)

        if sub_c2pa:
            has_c2pa = True
        if sub_ai or sub_c2pa:
            has_ai = True
        for f in sub_findings:
            findings.append(f"embedded data:image/{mime}: {f}")

    return has_c2pa, has_ai, findings


def _clean_embedded_data_uris(
    text: str, *, strip_all_metadata: bool = True
) -> tuple[str, list[str]]:
    import base64
    import urllib.parse

    actions: list[str] = []

    def _replace_uri(m: re.Match[str]) -> str:
        full_match = m.group(0)
        mime = m.group("mime")
        params = m.group("params") or ""
        payload = m.group("payload")
        is_b64 = "base64" in params.lower()

        try:
            if is_b64:
                raw_b64 = re.sub(r"\s+", "", payload)
                pad = len(raw_b64) % 4
                if pad:
                    raw_b64 += "=" * (4 - pad)
                data = base64.b64decode(raw_b64)
            else:
                data = urllib.parse.unquote_to_bytes(payload)
        except Exception:
            return full_match

        if not data:
            return full_match

        fmt = detect_format(data)
        sub_actions: list[str] = []
        cleaned_bytes = data

        try:
            if fmt == "png":
                cleaned_bytes, sub_actions = strip_png(data, strip_all_text=strip_all_metadata)
            elif fmt == "jpeg":
                cleaned_bytes, sub_actions = strip_jpeg(data, strip_all_app=strip_all_metadata)
            elif fmt == "webp":
                cleaned_bytes, sub_actions = strip_webp(data, strip_all_metadata=strip_all_metadata)
            elif fmt == "avif":
                cleaned_bytes, sub_actions = strip_avif(data, strip_all=strip_all_metadata)
            elif fmt == "heif":
                cleaned_bytes, sub_actions = strip_heic(data, strip_all=strip_all_metadata)
            elif "svg" in mime.lower() or data.lstrip().startswith(b"<"):
                cleaned_bytes, sub_actions = clean_svg(data)
        except Exception:
            return full_match

        if not _media_strip_succeeded(sub_actions, cleaned_bytes, data):
            return full_match

        actions.append(f"cleaned embedded data:image/{mime} ({', '.join(sub_actions[:2])})")

        if is_b64:
            new_b64 = base64.b64encode(cleaned_bytes).decode("ascii")
            return f"data:image/{mime}{params},{new_b64}"
        else:
            new_payload = urllib.parse.quote_from_bytes(cleaned_bytes)
            return f"data:image/{mime}{params},{new_payload}"

    out = RE_DATA_IMAGE_URI.sub(_replace_uri, text)
    return out, actions


# CMS generator allowlist: a plain <meta name="generator" content="WordPress">
# is CMS provenance, not AI-generator metadata, so it is kept and reported as
# an informational finding rather than dropped.
_GENERATOR_AI_RE = re.compile(
    r"claude|anthropic|openai|chatgpt|gemini|synthid|copilot|midjourney|dall.?e|stable.?diffusion",
    re.I,
)
_META_ATTR_RE = re.compile(
    r"""(name|property|content|generator)s*=s*["']([^"']*)["']""",
    re.I,
)


def _meta_attrs(tag: str) -> dict[str, str]:
    return {name.lower(): value for name, value in _META_ATTR_RE.findall(tag)}


def _is_cms_generator_meta(tag: str) -> bool:
    """Return True for a generator meta tag that is CMS provenance, not AI."""
    attrs = _meta_attrs(tag)
    name_or_prop = (
        attrs.get("name") or attrs.get("property") or attrs.get("generator") or ""
    ).lower()
    if name_or_prop != "generator":
        return False
    return not (_GENERATOR_AI_RE.search(attrs.get("content", "")) or _GENERATOR_AI_RE.search(tag))


# ---------------------------------------------------------------------------
# Markdown frontmatter
# ---------------------------------------------------------------------------

_FM_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def _parse_simple_yaml_keys(block: str) -> list[tuple[str, str, int]]:
    """Return list of (key, full_line, line_index) for top-level keys only."""
    rows: list[tuple[str, str, int]] = []
    for i, line in enumerate(block.splitlines()):
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line[0] in (" ", "\t", "-"):
            continue  # nested / list — leave alone
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if m:
            rows.append((m.group(1), line, i))
    return rows


def inspect_markdown(text: str) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_ai = False
    has_fm = False
    keys = []
    m = _FM_RE.match(text)
    if m:
        has_fm = True
        block = m.group(1)
        for key, _line, _i in _parse_simple_yaml_keys(block):
            keys.append(key)
            if key.lower() in AI_FRONTMATTER_KEYS or AI_META_NAME_RE.search(key):
                has_ai = True
                findings.append(f"frontmatter key: {key}")
            # also check value
            val = _line.split(":", 1)[1] if ":" in _line else ""
            if AI_META_NAME_RE.search(val):
                has_ai = True
                findings.append(f"frontmatter value hit on {key}")

    uri_c2pa, uri_ai, uri_findings = _inspect_embedded_data_uris(text)
    if uri_c2pa:
        has_ai = True
    if uri_ai:
        has_ai = True
    findings.extend(uri_findings)

    c2pa = uri_c2pa or any("c2pa" in f.lower() or "content" in f.lower() for f in findings)
    return c2pa, has_ai, findings, {"has_frontmatter": has_fm, "keys": keys}


def clean_markdown(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    m = _FM_RE.match(text)
    if not m:
        # no frontmatter: still scrub embedded data URIs in the body
        out, uri_actions = _clean_embedded_data_uris(text)
        if uri_actions:
            actions.extend(uri_actions)
        if not actions:
            actions.append("no AI frontmatter keys or embedded data URIs removed")
        return out, actions
    block = m.group(1)
    body = text[m.end() :]
    kept: list[str] = []
    dropping_parent = False
    for line in block.splitlines():
        stripped = line.strip()
        nested = bool(line) and line[0] in (" ", "\t", "-")
        if nested:
            if not dropping_parent:
                kept.append(line)
            continue
        if not stripped or stripped.startswith("#"):
            # Comments/blanks do not end a parent's nested block.
            if not dropping_parent:
                kept.append(line)
            continue

        dropping_parent = False
        km = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if not km:
            kept.append(line)
            continue
        key = km.group(1)
        val = line.split(":", 1)[1] if ":" in line else ""
        if key.lower() in AI_FRONTMATTER_KEYS or AI_META_NAME_RE.search(key):
            actions.append(f"drop frontmatter key: {key}")
            dropping_parent = True
            continue
        if AI_META_NAME_RE.search(val):
            actions.append(f"drop frontmatter key (value hit): {key}")
            dropping_parent = True
            continue
        kept.append(line)
    if not actions:
        actions.append("no AI frontmatter keys removed")
    # strip trailing empty nested orphans already handled
    new_block = "\n".join(kept).strip("\n")
    if new_block:
        out = f"---\n{new_block}\n---\n{body}"
    else:
        out = body.lstrip("\n")
        actions.append("removed empty frontmatter block")

    out, uri_actions = _clean_embedded_data_uris(out)
    if uri_actions:
        actions.extend(uri_actions)

    if not actions:
        actions.append("no AI frontmatter keys or embedded data URIs removed")
    return out, actions


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_META_TAG_RE = re.compile(
    r"<meta\b[^>]*>",
    re.I,
)
_JSONLD_RE = re.compile(
    r"<script\b[^>]*type\s*=\s*[\"']application/ld\+json[\"'][^>]*>.*?</script>",
    re.I | re.DOTALL,
)


def inspect_html(text: str) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_ai = False
    has_c2pa = False
    for tag in _META_TAG_RE.findall(text):
        if re.search(r"c2pa|content.?credential", tag, re.I):
            has_c2pa = True
        if _is_cms_generator_meta(tag):
            findings.append(f"info: cms generator: {tag[:120]}")
            continue
        if AI_META_NAME_RE.search(tag) or any(
            h.decode("ascii", "ignore").lower() in tag.lower() for h in AI_META_HINTS[:12]
        ):
            has_ai = True
            findings.append(f"meta: {tag[:120]}")
    for m in _JSONLD_RE.finditer(text):
        blob = m.group(0)
        if AI_META_NAME_RE.search(blob) or re.search(
            r"DigitalSourceType|trainedAlgorithmicMedia|SoftwareAgent", blob, re.I
        ):
            has_ai = True
            findings.append("json-ld provenance-like block")
            if re.search(r"c2pa|contentcredential", blob, re.I):
                has_c2pa = True
    # data-ai* attributes
    for m in re.finditer(r"\bdata-ai[\w-]*\s*=\s*[\"'][^\"']*[\"']", text, re.I):
        has_ai = True
        findings.append(f"attr: {m.group(0)[:80]}")

    uri_c2pa, uri_ai, uri_findings = _inspect_embedded_data_uris(text)
    if uri_c2pa:
        has_c2pa = True
    if uri_ai:
        has_ai = True
    findings.extend(uri_findings)

    return has_c2pa, has_ai, findings, {}


def clean_html(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []

    def _meta_sub(m: re.Match[str]) -> str:
        tag = m.group(0)
        if _is_cms_generator_meta(tag):
            return tag
        if AI_META_NAME_RE.search(tag) or re.search(
            r"generator|claude|anthropic|openai|gemini|synthid|c2pa|aigc", tag, re.I
        ):
            actions.append(f"drop meta: {tag[:80]}")
            return ""
        return tag

    out = _META_TAG_RE.sub(_meta_sub, text)

    def _jsonld_sub(m: re.Match[str]) -> str:
        blob = m.group(0)
        if AI_META_NAME_RE.search(blob) or re.search(
            r"DigitalSourceType|trainedAlgorithmicMedia|SoftwareAgent", blob, re.I
        ):
            actions.append("drop json-ld provenance-like script")
            return ""
        return blob

    out = _JSONLD_RE.sub(_jsonld_sub, out)
    out2, n = re.subn(r"\sdata-ai[\w-]*\s*=\s*[\"'][^\"']*[\"']", "", out, flags=re.I)
    if n:
        actions.append(f"drop data-ai* attributes x{n}")
        out = out2
    out, uri_actions = _clean_embedded_data_uris(out)
    actions.extend(uri_actions)
    if not actions:
        actions.append("no HTML AI meta removed")
    return out, actions


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def inspect_svg(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa, has_ai, hits = _blob_hits(data)
    findings.extend(hits)
    try:
        text = data.decode("utf-8", errors="replace")
        if re.search(r"<metadata[\s>]", text, re.I):
            findings.append("svg <metadata> present")
            has_ai = True  # often XMP; treat as inspect signal
        if re.search(r"xmpmeta|rdf:RDF|contentcredentials", text, re.I):
            has_ai = True
            findings.append("XMP/RDF-like content in SVG")
        if re.search(r"c2pa|jumbf", text, re.I):
            has_c2pa = True

        uri_c2pa, uri_ai, uri_findings = _inspect_embedded_data_uris(text)
        if uri_c2pa:
            has_c2pa = True
        if uri_ai:
            has_ai = True
        findings.extend(uri_findings)
    except Exception as e:
        findings.append(f"svg decode note: {e}")
    return has_c2pa, has_ai or has_c2pa, findings, {}


def clean_svg(data: bytes) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    text = data.decode("utf-8", errors="surrogateescape")
    # Drop metadata blocks
    new, n = re.subn(
        r"<metadata\b[^>]*>.*?</metadata\s*>",
        "",
        text,
        flags=re.I | re.DOTALL,
    )
    if n:
        actions.append(f"drop <metadata> x{n}")
        text = new
    # Drop adobe xmp packets
    new, n = re.subn(
        r"<x:xmpmeta\b[^>]*>.*?</x:xmpmeta\s*>",
        "",
        text,
        flags=re.I | re.DOTALL,
    )
    if n:
        actions.append(f"drop xmpmeta x{n}")
        text = new

    # Drop comments that look like provenance
    def _cmt(m: re.Match[str]) -> str:
        body = m.group(0)
        if AI_META_NAME_RE.search(body):
            actions.append("drop SVG comment with AI markers")
            return ""
        return body

    text = re.sub(r"<!--.*?-->", _cmt, text, flags=re.DOTALL)

    # Clean embedded data URIs
    text, uri_actions = _clean_embedded_data_uris(text)
    if uri_actions:
        actions.extend(uri_actions)

    if not actions:
        # still strip generator attribute on root if present
        new, n = re.subn(
            r'\s(inkscape:version|sodipodi:docname|generator)\s*=\s*"[^"]*"',
            "",
            text,
            flags=re.I,
        )
        if n:
            actions.append(f"drop generator-like attrs x{n}")
            text = new
    if not actions:
        actions.append("no SVG metadata removed")
    return text.encode("utf-8", errors="surrogateescape"), actions


# ---------------------------------------------------------------------------
# DOCX / ODT (zip + XML)
# ---------------------------------------------------------------------------

DOCX_META_PARTS = (
    "docProps/core.xml",
    "docProps/app.xml",
    "docProps/custom.xml",
)
MAX_ZIP_ENTRIES = 5_000
MAX_ZIP_ENTRY_BYTES = 128 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ZIP_RATIO = 1_000


def _validate_zip(zf: zipfile.ZipFile) -> None:
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise ValueError(f"archive has too many entries ({len(infos)} > {MAX_ZIP_ENTRIES})")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("archive contains duplicate entry names")
    total = 0
    for info in infos:
        if info.file_size > MAX_ZIP_ENTRY_BYTES:
            raise ValueError(f"archive entry too large: {info.filename}")
        total += info.file_size
        if total > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("archive uncompressed size exceeds safety limit")
        if info.file_size and not info.compress_size:
            raise ValueError(f"invalid zero compressed size: {info.filename}")
        if info.compress_size and info.file_size / info.compress_size > MAX_ZIP_RATIO:
            raise ValueError(f"suspicious compression ratio: {info.filename}")


class ZipBudgetExceeded(Exception):
    """A zip's declared decompressed size exceeds the processing cap.

    Kept separate from the parse errors below so a refused zip bomb keeps
    propagating (as it already does out of the clean_* helpers) instead of
    being reported as an unparseable container.
    """


import zlib

# A corrupt, truncated, encrypted, or unsupported-compression zip surfaces as
# more than just BadZipFile: reading a member can raise NotImplementedError
# (unknown compression/version), RuntimeError (encrypted), zlib.error (bad
# deflate stream), EOFError (truncated), OSError (invalid stream), or
# ValueError (e.g. a negative seek).
_ZIP_PARSE_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    NotImplementedError,
    RuntimeError,
    EOFError,
    OSError,
    ValueError,
    zlib.error,
)

MAX_ZIP_DECOMPRESSED_BYTES = 128 * 1024 * 1024


def _check_zip_budget(info: zipfile.ZipInfo, budget: list[int]) -> None:
    """Fast-path zip-bomb guard on the declared member size.

    A single member whose *declared* size already exceeds the cap is
    rejected before any decompression. The authoritative accounting lives in
    _read_zip_member, which charges **actual** decompressed bytes to the
    shared budget: ZipInfo.file_size comes from the archive central
    directory and is attacker-controlled, so trusting it for the cumulative
    limit would let a crafted archive declare a tiny size and still expand
    to gigabytes.
    """
    if info.file_size > MAX_ZIP_DECOMPRESSED_BYTES:
        raise ZipBudgetExceeded(
            "zip decompressed size exceeds cap "
            f"({MAX_ZIP_DECOMPRESSED_BYTES} bytes); refusing to process"
        )


def _read_zip_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, budget: list[int]) -> bytes:
    """Read one zip member, charging real decompressed bytes to the budget.

    Streams the member in chunks so the cumulative cap is enforced on bytes
    actually produced, not on the declared ``file_size``; raises
    ``ZipBudgetExceeded`` the moment the cap is crossed, before the whole
    member is buffered.
    """
    _check_zip_budget(info, budget)
    with zf.open(info) as stream:
        chunks: list[bytes] = []
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                break
            budget[0] += len(chunk)
            if budget[0] > MAX_ZIP_DECOMPRESSED_BYTES:
                raise ZipBudgetExceeded(
                    "zip decompressed size exceeds cap "
                    f"({MAX_ZIP_DECOMPRESSED_BYTES} bytes); refusing to process"
                )
            chunks.append(chunk)
    return b"".join(chunks)


DOCX_META_PARTS = (
    "docProps/core.xml",
    "docProps/app.xml",
    "docProps/custom.xml",
)
DOCX_CUSTOM_PREFIXES = (
    "customXml/",
    "docProps/",
)

# Provenance fields in docProps/core.xml and docProps/app.xml that always come
# out empty. dc:title is deliberately not listed: it is the document's own
# heading, not provenance.
DOCX_SCRUB_FIELDS = (
    ("dc:creator", "dc:creator"),
    ("cp:lastModifiedBy", "cp:lastModifiedBy"),
    ("dc:description", "dc:description"),
    ("cp:keywords", "cp:keywords"),
    ("dc:subject", "dc:subject"),
    ("cp:category", "cp:category"),
    ("Application", "Application"),
    ("AppVersion", "AppVersion"),
    ("Company", "Company"),
    ("Manager", "Manager"),
)


def _is_docx_meta_part(name: str) -> bool:
    """Return True for DOCX/XLSX/PPTX parts that carry provenance, not visible content."""
    return name.startswith(("docProps/", "customXml/"))


def _inspect_ooxml_zip(data: bytes, fmt: str) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    parts: list[str] = []
    budget = [0]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            try:
                _validate_zip(zf)
            except ValueError as ve:
                if "compression ratio" in str(ve):
                    findings.append("unsafe/invalid compression ratio")
                else:
                    raise
            parts = zf.namelist()
            for info in zf.infolist():
                _check_zip_budget(info, budget)
                if (
                    info.file_size
                    and info.compress_size
                    and info.file_size / info.compress_size > MAX_ZIP_RATIO
                ):
                    findings.append(f"unsafe/invalid compression ratio: {info.filename}")
                name = info.filename
                # Check media parts for C2PA/AI metadata
                if re.search(
                    r"^(?:word|xl|ppt)/media/.+\.(png|jpe?g|webp|avif|heic|gif|bmp|tiff?|svg)$",
                    name,
                    re.I,
                ):
                    raw = _read_zip_member(zf, info, budget)
                    img_fmt = detect_format(raw)
                    sub_c2pa, sub_ai, sub_findings = False, False, []
                    if img_fmt == "png":
                        sub_c2pa, sub_ai, sub_findings = inspect_png(raw)
                    elif img_fmt == "jpeg":
                        sub_c2pa, sub_ai, sub_findings = inspect_jpeg(raw)
                    elif img_fmt == "webp":
                        sub_c2pa, sub_ai, sub_findings = inspect_webp(raw)
                    elif img_fmt == "avif":
                        sub_c2pa, sub_ai, sub_findings = inspect_avif(raw)
                    elif img_fmt == "heif":
                        sub_c2pa, sub_ai, sub_findings = inspect_heic(raw)
                    elif img_fmt == "gif":
                        sub_c2pa, sub_ai, sub_findings = inspect_gif(raw)
                    elif img_fmt == "tiff":
                        sub_c2pa, sub_ai, sub_findings = inspect_tiff(raw)
                    elif img_fmt == "bmp":
                        sub_c2pa, sub_ai, sub_findings = inspect_bmp(raw)
                    elif name.lower().endswith(".svg") or raw.lstrip().startswith(b"<"):
                        sub_c2pa, sub_ai, sub_findings, _ = inspect_svg(raw)
                    if sub_c2pa:
                        has_c2pa = True
                    if sub_ai or sub_c2pa:
                        has_ai = True
                    for sf in sub_findings:
                        findings.append(f"{name}: {sf}")
                    continue

                # Only metadata/provenance parts carry AI markers. The visible
                # body (word/*.xml, xl/*.xml, ppt/*.xml) may legitimately mention
                # vendor names such as "Claude" without being AI-generated metadata.
                if not _is_docx_meta_part(name):
                    continue
                raw = _read_zip_member(zf, info, budget)
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    findings.append(f"{name}: {', '.join(hits[:6])}")
            # always flag customXml presence lightly
            custom = [n for n in parts if n.startswith("customXml/")]
            if custom:
                findings.append(f"customXml parts: {len(custom)}")
    except _ZIP_PARSE_ERRORS:
        return False, False, [f"not a valid {fmt.upper()} zip"], {}
    return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(parts)}


def inspect_docx(data: bytes) -> tuple[bool, bool, list[str], dict]:
    return _inspect_ooxml_zip(data, "docx")


def inspect_xlsx(data: bytes) -> tuple[bool, bool, list[str], dict]:
    return _inspect_ooxml_zip(data, "xlsx")


def inspect_pptx(data: bytes) -> tuple[bool, bool, list[str], dict]:
    return _inspect_ooxml_zip(data, "pptx")


def _scrub_docx_text(xml_text: str) -> tuple[str, int, int]:
    """Run Layer A over the w:t text runs of a DOCX part (xml:space preserved)."""
    from text_unicode import clean_text  # local import to avoid cycles

    removed = 0
    replaced = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal removed, replaced
        open_tag, inner, close_tag = m.group(1), m.group(2), m.group(3)
        new_inner, stats = clean_text(inner)
        if not (stats["removed_count"] or stats["replaced_count"]):
            return m.group(0)
        removed += stats["removed_count"]
        replaced += stats["replaced_count"]
        if (new_inner[:1].isspace() or new_inner[-1:].isspace()) and "xml:space" not in open_tag:
            open_tag = open_tag[:-1] + ' xml:space="preserve">'
        return open_tag + new_inner + close_tag

    new = re.sub(r"(<w:t\b[^>]*>)(.*?)(</w:t>)", _repl, xml_text, flags=re.S)
    return new, removed, replaced


def _scrub_xlsx_text(xml_text: str) -> tuple[str, int, int]:
    """Run Layer A over the t text elements of an XLSX part."""
    from text_unicode import clean_text  # local import to avoid cycles

    removed = 0
    replaced = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal removed, replaced
        open_tag, inner, close_tag = m.group(1), m.group(2), m.group(3)
        new_inner, stats = clean_text(inner)
        if not (stats["removed_count"] or stats["replaced_count"]):
            return m.group(0)
        removed += stats["removed_count"]
        replaced += stats["replaced_count"]
        if (new_inner[:1].isspace() or new_inner[-1:].isspace()) and "xml:space" not in open_tag:
            open_tag = open_tag[:-1] + ' xml:space="preserve">'
        return open_tag + new_inner + close_tag

    new = re.sub(r"(<t\b[^>]*>)(.*?)(</t>)", _repl, xml_text, flags=re.S)
    return new, removed, replaced


def _scrub_pptx_text(xml_text: str) -> tuple[str, int, int]:
    """Run Layer A over the a:t text elements of a PPTX part."""
    from text_unicode import clean_text  # local import to avoid cycles

    removed = 0
    replaced = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal removed, replaced
        open_tag, inner, close_tag = m.group(1), m.group(2), m.group(3)
        new_inner, stats = clean_text(inner)
        if not (stats["removed_count"] or stats["replaced_count"]):
            return m.group(0)
        removed += stats["removed_count"]
        replaced += stats["replaced_count"]
        return open_tag + new_inner + close_tag

    new = re.sub(r"(<a:t\b[^>]*>)(.*?)(</a:t>)", _repl, xml_text, flags=re.S)
    return new, removed, replaced


def _scrub_odt_text(xml_text: str) -> tuple[str, int, int]:
    """Run Layer A over ODF paragraph text (text:p content, incl. spans)."""
    from text_unicode import clean_text  # local import to avoid cycles

    removed = 0
    replaced = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal removed, replaced
        open_tag, inner, close_tag = m.group(1), m.group(2), m.group(3)
        new_inner, stats = clean_text(inner)
        if not (stats["removed_count"] or stats["replaced_count"]):
            return m.group(0)
        removed += stats["removed_count"]
        replaced += stats["replaced_count"]
        return open_tag + new_inner + close_tag

    new = re.sub(r"(<text:p\b[^>]*>)(.*?)(</text:p>)", _repl, xml_text, flags=re.S)
    return new, removed, replaced


def _prune_dangling_relationships(
    rels_name: str, raw: bytes, kept_names: set[str]
) -> tuple[bytes, int]:
    """Drop <Relationship> entries whose internal target part no longer exists."""
    base = posixpath.dirname(posixpath.dirname(rels_name))
    text = raw.decode("utf-8", errors="replace")
    dropped = [0]

    def _target_attr(tag: str) -> str:
        m = re.search(r"\bTarget\s*=\s*\"([^\"]*)\"", tag, re.I)
        return m.group(1) if m else ""

    def _drop(m: re.Match[str]) -> str:
        tag = m.group(0)
        if re.search(r"\bTargetMode\s*=", tag, re.I):
            return tag  # external (http / mailto / ...) — never pruned
        target = _target_attr(tag)
        if target.startswith("/"):
            resolved = posixpath.normpath(target.lstrip("/"))
        else:
            resolved = posixpath.normpath(posixpath.join(base, target))
        if resolved in ("", "."):
            return tag  # points at the package root
        if resolved in kept_names:
            return tag
        dropped[0] += 1
        return ""

    new = re.sub(r"<Relationship\b[^>]*/>", _drop, text, flags=re.I)
    return new.encode("utf-8"), dropped[0]


def _scrub_ooxml_zip(
    data: bytes, fmt: str, *, also_layer_a_text: bool = True
) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    budget = [0]
    layer_removed = 0
    layer_replaced = 0
    kept: list[tuple[zipfile.ZipInfo, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        _validate_zip(zin)
        for info in zin.infolist():
            _check_zip_budget(info, budget)
            name = info.filename
            raw = _read_zip_member(zin, info, budget)

            # 1. Clean embedded media (PNG, JPEG, WebP, AVIF, HEIC, GIF, BMP, TIFF, SVG)
            if re.search(
                r"^(?:word|xl|ppt)/media/.+\.(png|jpe?g|webp|avif|heic|gif|bmp|tiff?|svg)$",
                name,
                re.I,
            ):
                img_fmt = detect_format(raw)
                sub_actions: list[str] = []
                cleaned_bytes = raw
                try:
                    if img_fmt == "png":
                        cleaned_bytes, sub_actions = strip_png(raw, strip_all_text=True)
                    elif img_fmt == "jpeg":
                        cleaned_bytes, sub_actions = strip_jpeg(raw, strip_all_app=True)
                    elif img_fmt == "webp":
                        cleaned_bytes, sub_actions = strip_webp(raw, strip_all_metadata=True)
                    elif img_fmt == "avif":
                        cleaned_bytes, sub_actions = strip_avif(raw, strip_all=True)
                    elif img_fmt == "heif":
                        cleaned_bytes, sub_actions = strip_heic(raw, strip_all=True)
                    elif img_fmt == "gif":
                        cleaned_bytes, sub_actions = strip_gif(raw, strip_all_metadata=True)
                    elif img_fmt == "bmp":
                        cleaned_bytes, sub_actions = strip_bmp(raw, strip_all_metadata=True)
                    elif img_fmt == "tiff":
                        cleaned_bytes, sub_actions = strip_tiff(raw, strip_all_metadata=True)
                    elif name.lower().endswith(".svg") or raw.lstrip().startswith(b"<"):
                        cleaned_bytes, sub_actions = clean_svg(raw)
                except Exception:  # noqa: S110
                    pass
                if _media_strip_succeeded(sub_actions, cleaned_bytes, raw):
                    actions.append(f"clean embedded media in {name} ({', '.join(sub_actions[:2])})")
                    raw = cleaned_bytes
                kept.append((info, raw))
                continue

            # 2. Drop customXml trees
            if name.startswith("customXml/"):
                actions.append(f"drop part {name}")
                continue

            # 3. docProps/ provenance
            if name in DOCX_META_PARTS or name.startswith("docProps/"):
                if name.endswith("custom.xml"):
                    actions.append(f"drop part {name}")
                    continue
                text = raw.decode("utf-8", errors="replace")
                new = text
                for tag, label in DOCX_SCRUB_FIELDS:
                    pat = rf"(<{tag}\b[^>]*>)(.*?)(</{tag}>)"

                    def _empty(m: re.Match[str], _label=label, _name=name) -> str:
                        if m.group(2):
                            actions.append(f"scrub {_name} field {_label}")
                        return m.group(1) + m.group(3)

                    new = re.sub(pat, _empty, new, flags=re.I | re.DOTALL)
                raw = new.encode("utf-8")
            # 4. [Content_Types].xml overrides
            if name == "[Content_Types].xml":
                text = raw.decode("utf-8", errors="replace")
                new, n = re.subn(
                    r"<Override\b[^>]*PartName=\"/customXml/[^\"]*\"[^>]*/>",
                    "",
                    text,
                )
                if n:
                    actions.append(f"drop Content_Types customXml overrides x{n}")
                    raw = new.encode("utf-8")
                new, n = re.subn(
                    r"<Override\b[^>]*PartName=\"/docProps/custom\.xml\"[^>]*/>",
                    "",
                    raw.decode("utf-8", errors="replace"),
                )
                if n:
                    actions.append(f"drop Content_Types custom.xml override x{n}")
                    raw = new.encode("utf-8")

            # 5. Layer A text runs
            if also_layer_a_text and name.endswith(".xml"):
                if fmt == "docx" and name.startswith("word/"):
                    text = raw.decode("utf-8", errors="replace")
                    new, r, rp = _scrub_docx_text(text)
                    if r or rp:
                        layer_removed += r
                        layer_replaced += rp
                        raw = new.encode("utf-8")
                elif fmt == "xlsx" and name.startswith("xl/"):
                    text = raw.decode("utf-8", errors="replace")
                    new, r, rp = _scrub_xlsx_text(text)
                    if r or rp:
                        layer_removed += r
                        layer_replaced += rp
                        raw = new.encode("utf-8")
                elif fmt == "pptx" and name.startswith("ppt/"):
                    text = raw.decode("utf-8", errors="replace")
                    new, r, rp = _scrub_pptx_text(text)
                    if r or rp:
                        layer_removed += r
                        layer_replaced += rp
                        raw = new.encode("utf-8")

            kept.append((info, raw))

    kept_names = {info.filename for info, _ in kept}
    final: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, raw in kept:
        part_raw = raw
        if info.filename.endswith(".rels"):
            part_raw, n = _prune_dangling_relationships(info.filename, raw, kept_names)
            if n:
                actions.append(f"prune dangling relationships x{n} in {info.filename}")
        final.append((info, part_raw))

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, raw in final:
            zout.writestr(info, raw)
    if layer_removed or layer_replaced:
        actions.append(f"layer A text: removed={layer_removed} replaced={layer_replaced}")
    if not actions:
        actions.append(f"no {fmt.upper()} metadata parts removed")
    return out_buf.getvalue(), actions


def clean_docx(data: bytes, *, also_layer_a_text: bool = True) -> tuple[bytes, list[str]]:
    return _scrub_ooxml_zip(data, "docx", also_layer_a_text=also_layer_a_text)


def clean_xlsx(data: bytes, *, also_layer_a_text: bool = True) -> tuple[bytes, list[str]]:
    return _scrub_ooxml_zip(data, "xlsx", also_layer_a_text=also_layer_a_text)


def clean_pptx(data: bytes, *, also_layer_a_text: bool = True) -> tuple[bytes, list[str]]:
    return _scrub_ooxml_zip(data, "pptx", also_layer_a_text=also_layer_a_text)


def _prune_odt_manifest_entries(raw: bytes, dropped: set[str]) -> tuple[bytes, int]:
    """Remove manifest file-entry elements pointing at dropped parts."""
    text = raw.decode("utf-8", errors="replace")
    removed = [0]

    def _full_path(tag: str) -> str:
        m = re.search(r'\bfull-path\s*=\s*"([^"]*)"', tag, re.I)
        return m.group(1) if m else ""

    def _drop(m: re.Match[str]) -> str:
        tag = m.group(0)
        path = posixpath.normpath(_full_path(tag).lstrip("/"))
        if path in ("", "."):
            return tag  # package root — never pruned
        if path in dropped:
            removed[0] += 1
            return ""
        return tag

    new = re.sub(r"<manifest:file-entry\b[^>]*/>", _drop, text, flags=re.I)
    return new.encode("utf-8"), removed[0]


def inspect_odt(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    budget = [0]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _validate_zip(zf)
            for info in zf.infolist():
                _check_zip_budget(info, budget)
                raw = _read_zip_member(zf, info, budget)
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    findings.append(f"{info.filename}: {', '.join(hits[:6])}")
            if "meta.xml" in zf.namelist():
                meta = _read_zip_member(zf, zf.getinfo("meta.xml"), budget).decode(
                    "utf-8", errors="replace"
                )
                if re.search(r"generator|claude|openai|anthropic|gemini", meta, re.I):
                    has_ai = True
                    findings.append("meta.xml generator-like fields")
    except _ZIP_PARSE_ERRORS:
        return False, False, ["not a valid ODT zip"], {}
    return has_c2pa, has_ai or has_c2pa, findings, {}


def clean_odt(data: bytes, *, also_layer_a_text: bool = True) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    budget = [0]
    layer_removed = 0
    layer_replaced = 0
    kept: list[tuple[zipfile.ZipInfo, bytes]] = []
    dropped: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        _validate_zip(zin)
        for info in zin.infolist():
            _check_zip_budget(info, budget)
            name = info.filename
            raw = _read_zip_member(zin, info, budget)
            if name == "meta.xml":
                text = raw.decode("utf-8", errors="replace")
                new, n = re.subn(
                    r"<meta:generator\b[^>]*>.*?</meta:generator\s*>",
                    "",
                    text,
                    flags=re.I | re.DOTALL,
                )
                if n:
                    actions.append("drop meta:generator")
                    text = new

                # scrub creator-like if AI
                def _creator(m: re.Match[str]) -> str:
                    if AI_META_NAME_RE.search(m.group(0)):
                        actions.append("scrub creator-like meta")
                        return ""
                    return m.group(0)

                text = re.sub(
                    r"<dc:creator\b[^>]*>.*?</dc:creator\s*>",
                    _creator,
                    text,
                    flags=re.I | re.DOTALL,
                )
                raw = text.encode("utf-8")
            else:
                c2, ai, _ = _blob_hits(raw)
                if (c2 or ai) and name not in (
                    "content.xml",
                    "styles.xml",
                    "mimetype",
                    "META-INF/manifest.xml",
                ):
                    actions.append(f"drop part {name} (AI/C2PA markers)")
                    dropped.add(name)
                    continue
            # Layer A over the visible paragraph text of the body part.
            if also_layer_a_text and name == "content.xml":
                text = raw.decode("utf-8", errors="replace")
                new, r, rp = _scrub_odt_text(text)
                if r or rp:
                    layer_removed += r
                    layer_replaced += rp
                    raw = new.encode("utf-8")
            kept.append((info, raw))

    # Two-pass: rewrite META-INF/manifest.xml now that the dropped set is
    # known. The manifest precedes the dropped parts in the archive, so a
    # single pass cannot remove entries for parts dropped later in the loop.
    if dropped:
        rewritten: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info, part_raw in kept:
            out_raw = part_raw
            if info.filename == "META-INF/manifest.xml":
                pruned, n = _prune_odt_manifest_entries(part_raw, dropped)
                if n:
                    actions.append(f"drop manifest entries x{n}")
                    out_raw = pruned
            rewritten.append((info, out_raw))
        kept = rewritten

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, raw in kept:
            zout.writestr(info, raw)
    if layer_removed or layer_replaced:
        actions.append(f"layer A text: removed={layer_removed} replaced={layer_replaced}")
    if not actions:
        actions.append("no ODT metadata removed")
    return out_buf.getvalue(), actions


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

_EPUB_MEDIA_RE = re.compile(r"\.(png|jpe?g|webp|avif|heic|gif|bmp|tiff?|svg)$", re.I)


def _epub_content_part(name: str) -> bool:
    """True for parts that carry visible content or structure and must never be dropped."""
    low = name.lower()
    return bool(
        low == "mimetype"
        or low.endswith(".rels")
        or low
        in (
            "meta-inf/container.xml",
            "meta-inf/encryption.xml",
            "meta-inf/signatures.xml",
            "meta-inf/rights.xml",
        )
        or re.search(
            r"\.(xhtml|html?|css|js|ncx|opf|svg|png|jpe?g|webp|avif|heic|gif|bmp|tiff?|ttf|otf|woff2?)$",
            low,
        )
    )


def _epub_encrypted_parts(data: bytes) -> set[str]:
    """Return part names listed as encrypted in META-INF/encryption.xml (OCF)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "META-INF/encryption.xml" not in zf.namelist():
                return set()
            budget: list[int] = [0]
            xml = _read_zip_member(zf, zf.getinfo("META-INF/encryption.xml"), budget).decode(
                "utf-8", errors="replace"
            )
    except (zipfile.BadZipFile, KeyError):
        return set()
    names: set[str] = set()
    for uri in re.findall(r'CipherReference\s+URI="([^"]+)"', xml, re.I):
        names.add(posixpath.normpath(posixpath.join("META-INF", uri)))
        names.add(posixpath.normpath(uri))
        if uri.startswith("../"):
            names.add(posixpath.normpath(uri[3:]))
    return names


def inspect_epub(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    budget = [0]
    encrypted = _epub_encrypted_parts(data)
    names: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            for info in zf.infolist():
                _check_zip_budget(info, budget)
                if (
                    info.file_size
                    and info.compress_size
                    and info.file_size / info.compress_size > MAX_ZIP_RATIO
                ):
                    findings.append(f"unsafe/invalid compression ratio: {info.filename}")
                name = info.filename
                if name in encrypted:
                    findings.append(f"{name}: encrypted content (skipped)")
                    continue
                raw = _read_zip_member(zf, info, budget)
                if name.lower().endswith((".xhtml", ".html", ".htm")):
                    text = raw.decode("utf-8", errors="surrogateescape")
                    c2, ai, sub, _ = inspect_html(text)
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    for f in sub:
                        findings.append(f"{name}: {f}")
                    continue
                if name.lower().endswith(".opf"):
                    text = raw.decode("utf-8", errors="surrogateescape")
                    if AI_META_NAME_RE.search(text):
                        has_ai = True
                        findings.append(f"{name}: AI-ish metadata in package document")
                    c2, ai, hits = _blob_hits(raw)
                    if c2 or ai:
                        has_c2pa = has_c2pa or c2
                        has_ai = has_ai or ai
                        findings.append(f"{name}: {', '.join(hits[:6])}")
                    continue
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    has_c2pa = has_c2pa or c2
                    has_ai = has_ai or ai
                    findings.append(f"{name}: {', '.join(hits[:6])}")
    except zipfile.BadZipFile:
        return False, False, ["not a valid EPUB zip"], {}
    return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(names)}


def _prune_opf_manifest(raw: bytes, opf_name: str, dropped: set[str]) -> tuple[bytes, int]:
    """Remove OPF <item> entries pointing at dropped parts (and spine refs)."""
    base = posixpath.dirname(opf_name)
    text = raw.decode("utf-8", errors="replace")
    removed = [0]
    removed_ids: set[str] = set()

    def _attr(tag: str, name: str) -> str:
        m = re.search(rf'\b{name}\s*=\s*"([^"]*)"', tag, re.I)
        return m.group(1) if m else ""

    def _drop_item(m: re.Match[str]) -> str:
        tag = m.group(0)
        href = _attr(tag, "href")
        if not href:
            return tag
        resolved = posixpath.normpath(posixpath.join(base, href))
        if resolved in dropped:
            removed[0] += 1
            item_id = _attr(tag, "id")
            if item_id:
                removed_ids.add(item_id)
            return ""
        return tag

    new = re.sub(r"<item\b[^>]*/>", _drop_item, text, flags=re.I)

    if removed_ids:

        def _drop_itemref(m: re.Match[str]) -> str:
            tag = m.group(0)
            if _attr(tag, "idref") in removed_ids:
                removed[0] += 1
                return ""
            return tag

        new = re.sub(r"<itemref\b[^>]*/>", _drop_itemref, new, flags=re.I)
    return new.encode("utf-8"), removed[0]


def _scrub_epub_opf(text: str) -> tuple[str, list[str]]:
    """Scrub AI-ish metadata from the EPUB package document (OPF)."""
    actions: list[str] = []

    def _meta(m: re.Match[str]) -> str:
        tag = m.group(0)
        if AI_META_NAME_RE.search(tag):
            actions.append("drop OPF meta tag")
            return ""
        return tag

    new = re.sub(r"<meta\b[^>]*/>", _meta, text, flags=re.I)
    new = re.sub(r"<meta\b[^>]*>.*?</meta\s*>", _meta, new, flags=re.I | re.DOTALL)

    def _dc(m: re.Match[str]) -> str:
        if AI_META_NAME_RE.search(m.group(0)):
            actions.append(f"scrub {m.group(1)} (AI vendor name)")
            return f"<{m.group(1)}/>"
        return m.group(0)

    new = re.sub(
        r"<(dc:(?:creator|contributor|publisher|description|rights|source))\b[^>]*>.*?</\1\s*>",
        _dc,
        new,
        flags=re.I | re.DOTALL,
    )

    if not actions:
        actions.append("no OPF metadata removed")
    return new, actions


def clean_epub(data: bytes, *, also_layer_a_text: bool = True) -> tuple[bytes, list[str]]:
    """Rewrite the EPUB: scrub OPF metadata, XHTML meta/JSON-LD, and Layer A."""
    from text_unicode import clean_text  # local import to avoid cycles

    actions: list[str] = []
    budget = [0]
    layer_removed = 0
    layer_replaced = 0
    encrypted = _epub_encrypted_parts(data)
    kept: list[tuple[zipfile.ZipInfo, bytes]] = []
    dropped: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        for info in zin.infolist():
            _check_zip_budget(info, budget)
            name = info.filename
            raw = _read_zip_member(zin, info, budget)
            low = name.lower()

            # Encrypted parts are opaque ciphertext: pass through untouched.
            if name in encrypted:
                kept.append((info, raw))
                continue

            # 1. Embedded raster / vector media: strip metadata
            if _EPUB_MEDIA_RE.search(low):
                img_fmt = detect_format(raw)
                sub_actions: list[str] = []
                cleaned = raw
                try:
                    if img_fmt == "png":
                        cleaned, sub_actions = strip_png(raw, strip_all_text=True)
                    elif img_fmt == "jpeg":
                        cleaned, sub_actions = strip_jpeg(raw, strip_all_app=True)
                    elif img_fmt == "webp":
                        cleaned, sub_actions = strip_webp(raw, strip_all_metadata=True)
                    elif img_fmt == "avif":
                        cleaned, sub_actions = strip_avif(raw, strip_all=True)
                    elif img_fmt == "heif":
                        cleaned, sub_actions = strip_heic(raw, strip_all=True)
                    elif img_fmt == "gif":
                        cleaned, sub_actions = strip_gif(raw, strip_all_metadata=True)
                    elif img_fmt == "bmp":
                        cleaned, sub_actions = strip_bmp(raw, strip_all_metadata=True)
                    elif img_fmt == "tiff":
                        cleaned, sub_actions = strip_tiff(raw, strip_all_metadata=True)
                    elif low.endswith(".svg") or raw.lstrip().startswith(b"<"):
                        cleaned, sub_actions = clean_svg(raw)
                except Exception:  # noqa: S110
                    pass
                if _media_strip_succeeded(sub_actions, cleaned, raw):
                    actions.append(f"clean embedded media in {name} ({', '.join(sub_actions[:2])})")
                    raw = cleaned
                kept.append((info, raw))
                continue

            # 2. XHTML content: strip AI meta/JSON-LD, then Layer A
            if low.endswith((".xhtml", ".html", ".htm")):
                text = raw.decode("utf-8", errors="surrogateescape")
                text, sub_actions = clean_html(text)
                if sub_actions and sub_actions != ["no HTML AI meta removed"]:
                    actions.append(f"{name}: {', '.join(sub_actions[:2])}")
                if also_layer_a_text:
                    text2, stats = clean_text(text)
                    if stats["removed_count"] or stats["replaced_count"]:
                        layer_removed += stats["removed_count"]
                        layer_replaced += stats["replaced_count"]
                        text = text2
                raw = text.encode("utf-8", errors="surrogateescape")
                kept.append((info, raw))
                continue

            # 3. Package document (OPF): scrub AI-ish metadata
            if low.endswith(".opf"):
                text = raw.decode("utf-8", errors="surrogateescape")
                new_text, sub_actions = _scrub_epub_opf(text)
                if sub_actions and sub_actions != ["no OPF metadata removed"]:
                    actions.extend(f"{name}: {a}" for a in sub_actions)
                raw = new_text.encode("utf-8", errors="surrogateescape")
                kept.append((info, raw))
                continue

            # 4. Other parts: drop non-content parts carrying AI/C2PA markers
            c2, ai, _hits = _blob_hits(raw)
            if (c2 or ai) and not _epub_content_part(name):
                actions.append(f"drop part {name} (AI/C2PA markers)")
                dropped.add(name)
                continue

            # 5. mimetype must stay first and stored — it already is, since we
            #    write in the original entry order; force stored compression.
            if low == "mimetype":
                info.compress_type = zipfile.ZIP_STORED
            kept.append((info, raw))

    # Two-pass: prune the OPF manifest now that the dropped set is known.
    if dropped:
        rewritten: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info, part_raw in kept:
            out_raw = part_raw
            if info.filename.lower().endswith(".opf"):
                pruned, n = _prune_opf_manifest(part_raw, info.filename, dropped)
                if n:
                    actions.append(f"prune OPF manifest entries x{n}")
                    out_raw = pruned
            rewritten.append((info, out_raw))
        kept = rewritten

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, raw in kept:
            zout.writestr(info, raw)

    if layer_removed or layer_replaced:
        actions.append(f"layer A text: removed={layer_removed} replaced={layer_replaced}")
    if not actions:
        actions.append("no EPUB metadata removed")
    return out_buf.getvalue(), actions


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


_XMP_PACKET_RE = re.compile(
    rb"<\?xpacket begin.*?<\?xpacket end[^?]*\?>",
    re.I | re.DOTALL,
)


def _pdf_structured_blob(data: bytes) -> bytes:
    """Return PDF bytes with stream payloads removed, plus XMP packets."""
    no_streams = re.sub(
        rb"stream\r?\n.*?endstream",
        b"stream endstream",
        data,
        flags=re.DOTALL,
    )
    xmp = b"\n".join(_XMP_PACKET_RE.findall(data))
    return no_streams + b"\n" + xmp


def inspect_pdf(path: Path, data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa, has_ai, hits = _blob_hits(_pdf_structured_blob(data))
    findings.extend(f"pdf-structured:{h}" for h in hits)
    xmp_blob = b"\n".join(_XMP_PACKET_RE.findall(data))
    if xmp_blob:
        findings.append("XMP packet present")
        has_ai = has_ai or bool(
            re.search(
                rb"digitalSourceType|trainedAlgorithmicMedia|SoftwareAgent|c2pa",
                xmp_blob,
                re.I,
            )
        )
    tools = run_optional_tools(path)
    ct = tools.get("c2patool") or {}
    if ct.get("has_manifest"):
        has_c2pa = True
        findings.append("c2patool reports C2PA-related manifest")
    return has_c2pa, has_ai or has_c2pa, findings, {"tools": tools}


def clean_pdf_pypdf(
    path: Path, dest: Path, *, skip_exiftool: bool = False
) -> tuple[list[str], dict]:
    """Clean PDF metadata. exiftool > full-document pypdf clone > unchanged copy.

    *skip_exiftool* is used by clean_pdf when exiftool already ran and failed,
    so the fallback does not invoke the same failing command a second time.
    """
    actions: list[str] = []
    data = path.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)

    exiftool = which("exiftool")

    # Strategy 1: exiftool (most reliable)
    if exiftool and not skip_exiftool:
        dest.write_bytes(data)
        try:
            result = external_command.run_command(
                [exiftool, "-all=", "-overwrite_original", str(dest)],
                timeout=60,
                output_limit=2 * 1024 * 1024,
            )
            actions.append(f"exiftool -all= (rc={result.returncode})")
            if result.returncode == 0 and not (result.stdout_truncated or result.stderr_truncated):
                return actions, {"mode": "exiftool", "degraded": False}
            if result.stdout_truncated or result.stderr_truncated:
                actions.append("exiftool output exceeded safety limit; trying pypdf")
            else:
                actions.append(f"exiftool degraded (rc={result.returncode}); trying pypdf")
        except Exception as error:
            actions.append(f"exiftool failed: {error}; trying pypdf")

    # Strategy 2: clone the complete document graph, then remove only metadata.
    # Copying pages alone loses outlines, forms, attachments, labels, and viewer state.
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        PdfReader = None  # type: ignore[assignment]
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                if reader.decrypt("") == 0:
                    actions.append("encrypted PDF (password required); copied as-is")
                    dest.write_bytes(data)
                    return actions, {"mode": "copy-encrypted", "degraded": True}
                actions.append("decrypted with empty password")
            writer = PdfWriter()
            writer.clone_document_from_reader(reader)
            page_metadata = 0
            for page in writer.pages:
                if "/Metadata" in page:
                    del page["/Metadata"]
                    page_metadata += 1
            if page_metadata:
                actions.append(f"pypdf: drop per-page /Metadata x{page_metadata}")
            if reader.metadata:
                actions.append("pypdf: drop document info dictionary")
            if reader.xmp_metadata is not None or "/Metadata" in writer.root_object:
                actions.append("pypdf: drop catalog XMP packet")
            writer.metadata = None
            writer.xmp_metadata = None
            if "/Metadata" in writer.root_object:
                del writer.root_object["/Metadata"]
            buf = io.BytesIO()
            writer.write(buf)
            # Publish only after a complete in-memory rewrite.
            atomic_write_bytes(dest, buf.getvalue())
            actions.append("pypdf: cloned full document graph; removed docinfo/XMP")
            return actions, {"mode": "pypdf", "degraded": False}
        except Exception as e:
            actions.append(f"pypdf failed: {e}; copied unchanged")
    else:
        actions.append("pypdf not installed; copied unchanged")

    # Never delete bytes from a PDF without rebuilding xref/object offsets.
    atomic_write_bytes(dest, data)
    actions.append("no structural PDF cleaner succeeded; copied unchanged")
    return actions, {"mode": "copy", "degraded": True}


def _pdf_structural_rewrite(dest: Path, actions: list[str]) -> bool:
    """Rebuild a PDF so unreferenced objects are dropped (qpdf --linearize)."""
    qpdf = which("qpdf")
    if not qpdf:
        actions.append(
            "warning: exiftool PDF edits are incremental — the original metadata "
            "bytes remain recoverable; install qpdf for a structural rewrite"
        )
        return False

    tmp = dest.with_name(dest.name + ".qpdf-tmp")
    try:
        r = run_command(
            [qpdf, "--linearize", "--", str(dest), str(tmp)],
            timeout=120,
            output_limit=2 * 1024 * 1024,
        )
    except Exception as e:
        tmp.unlink(missing_ok=True)
        actions.append(f"qpdf rewrite failed: {e}; metadata bytes may remain recoverable")
        return False

    if r.returncode in (0, 3) and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        actions.append(f"qpdf --linearize structural rewrite (rc={r.returncode})")
        return True

    tmp.unlink(missing_ok=True)
    actions.append(
        f"qpdf rewrite skipped (rc={r.returncode}); metadata bytes may remain recoverable"
    )
    return False


def clean_pdf(path: Path, dest: Path) -> tuple[list[str], dict]:
    """Best-effort PDF clean. Prefers exiftool + qpdf; falls back to pypdf."""
    actions: list[str] = []
    data = path.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)

    exiftool = which("exiftool")
    if exiftool:
        atomic_write_bytes(dest, data)
        exiftool_ok = False
        try:
            r = run_command(
                [exiftool, "-all=", "-overwrite_original", str(dest)],
                timeout=60,
                output_limit=2 * 1024 * 1024,
            )
            actions.append(f"exiftool -all= (rc={r.returncode})")
            truncated = bool(
                getattr(r, "stdout_truncated", False) or getattr(r, "stderr_truncated", False)
            )
            exiftool_ok = r.returncode == 0 and not truncated
            if r.returncode != 0:
                actions.append(f"exiftool degraded (rc={r.returncode})")
            elif truncated:
                actions.append("exiftool output exceeded safety limit")
        except Exception as e:
            actions.append(f"exiftool failed: {e}")
        if not exiftool_ok:
            # exiftool ran but did not strip; dest still holds the original
            # bytes. Hand off to the pypdf path rather than publishing
            # unstripped output under mode "exiftool" with no degraded flag.
            actions.append("trying pypdf fallback")
            fallback_actions, fallback_meta = clean_pdf_pypdf(path, dest, skip_exiftool=True)
            return actions + fallback_actions, fallback_meta
        rewritten = _pdf_structural_rewrite(dest, actions)
        c2patool = which("c2patool")
        if c2patool:
            actions.append("c2patool available for inspect; strip via exiftool/re-export")
        return actions, {"mode": "exiftool", "structural_rewrite": rewritten}

    return clean_pdf_pypdf(path, dest)


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


def inspect_container(path: Path) -> ContainerInspectReport:
    data = path.read_bytes()
    fmt = detect_container_format(path, data)
    tools: dict[str, Any] = {}
    details: dict[str, Any] = {}
    layer_a_total = 0
    layer_a_hits: list[dict] = []

    if fmt == "svg":
        has_c2pa, has_ai, findings, details = inspect_svg(data)
    elif fmt == "pdf":
        has_c2pa, has_ai, findings, details = inspect_pdf(path, data)
        tools = details.pop("tools", {})
    elif fmt == "docx":
        has_c2pa, has_ai, findings, details = inspect_docx(data)
    elif fmt == "xlsx":
        has_c2pa, has_ai, findings, details = inspect_xlsx(data)
    elif fmt == "pptx":
        has_c2pa, has_ai, findings, details = inspect_pptx(data)
    elif fmt == "odt":
        has_c2pa, has_ai, findings, details = inspect_odt(data)
    elif fmt == "epub":
        has_c2pa, has_ai, findings, details = inspect_epub(data)
    elif fmt == "html":
        body = data.decode("utf-8", errors="surrogateescape")
        has_c2pa, has_ai, findings, details = inspect_html(body)
        from text_unicode import inspect_text  # local import to avoid cycles

        ta = inspect_text(body).to_dict()
        layer_a_total = ta["suspicious_total"]
        layer_a_hits = ta["hits"]
        for h in layer_a_hits:
            findings.append(f"layer-a: {h['codepoint']} {h['label']} x{h['count']} ({h['kind']})")
    elif fmt == "markdown":
        body = data.decode("utf-8", errors="surrogateescape")
        has_c2pa, has_ai, findings, details = inspect_markdown(body)
        from text_unicode import inspect_text  # local import to avoid cycles

        ta = inspect_text(body).to_dict()
        layer_a_total = ta["suspicious_total"]
        layer_a_hits = ta["hits"]
        for h in layer_a_hits:
            findings.append(f"layer-a: {h['codepoint']} {h['label']} x{h['count']} ({h['kind']})")
    else:
        has_c2pa, has_ai, findings = False, False, [f"unsupported container: {fmt}"]

    if fmt == "epub":
        from text_unicode import inspect_text  # local import to avoid cycles

        encrypted = _epub_encrypted_parts(data)
        budget: list[int] = [0]
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.filename in encrypted:
                        continue
                    if info.filename.lower().endswith((".xhtml", ".html", ".htm")):
                        ta = inspect_text(
                            _read_zip_member(zf, info, budget).decode(
                                "utf-8", errors="surrogateescape"
                            )
                        ).to_dict()
                        layer_a_total += ta["suspicious_total"]
                        for h in ta["hits"]:
                            layer_a_hits.append(h)
                            findings.append(
                                f"layer-a ({info.filename}): {h['codepoint']} {h['label']} "
                                f"x{h['count']} ({h['kind']})"
                            )
        except zipfile.BadZipFile:
            pass

    notes: list[str] = []
    if fmt == "pdf":
        notes.append(
            "PDF inspection is best-effort; exiftool/c2patool give more reliable metadata detection"
        )
    elif fmt in ("docx", "xlsx", "pptx"):
        notes.append(f"{fmt.upper()}: metadata/provenance and embedded media are scanned")
    elif fmt == "epub":
        notes.append(
            "EPUB: package-document metadata, XHTML meta/JSON-LD, and embedded media are scanned"
        )
    if "unsupported" in details:
        notes.append(f"format not fully inspected: {fmt}")
    if layer_a_total:
        notes.append(
            f"layer A: {layer_a_total} invisible/format codepoint(s) in body text; "
            "clean removes these"
        )

    if fmt in ("svg", "pdf", "docx", "xlsx", "pptx") and not tools:
        tools = run_optional_tools(path)

    return ContainerInspectReport(
        path=str(path),
        format=fmt,
        has_c2pa=has_c2pa,
        has_ai_metadata=has_ai,
        findings=findings,
        tools=tools,
        details=details,
        notes=notes,
        layer_a_total=layer_a_total,
        layer_a_hits=layer_a_hits,
    )


def clean_container(
    path: Path,
    dest: Path,
    fmt: str | None = None,
    *,
    also_layer_a_text: bool = True,
) -> dict[str, Any]:
    """Clean container metadata; optionally Layer-A scrub text bodies for md/html."""
    from text_unicode import clean_text  # local import to avoid cycles

    data = path.read_bytes()
    fmt = fmt or detect_container_format(path, data)
    actions: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"format": fmt}

    if fmt == "svg":
        cleaned, actions = clean_svg(data)
        atomic_write_bytes(dest, cleaned)
    elif fmt == "pdf":
        actions, meta_extra = clean_pdf(path, dest)
        meta.update(meta_extra)
    elif fmt == "docx":
        cleaned, actions = clean_docx(data, also_layer_a_text=also_layer_a_text)
        atomic_write_bytes(dest, cleaned)
    elif fmt == "xlsx":
        cleaned, actions = clean_xlsx(data, also_layer_a_text=also_layer_a_text)
        atomic_write_bytes(dest, cleaned)
    elif fmt == "pptx":
        cleaned, actions = clean_pptx(data, also_layer_a_text=also_layer_a_text)
        atomic_write_bytes(dest, cleaned)
    elif fmt == "odt":
        cleaned, actions = clean_odt(data, also_layer_a_text=also_layer_a_text)
        atomic_write_bytes(dest, cleaned)
    elif fmt == "epub":
        cleaned, actions = clean_epub(data, also_layer_a_text=also_layer_a_text)
        atomic_write_bytes(dest, cleaned)
    elif fmt == "html":
        text = data.decode("utf-8", errors="surrogateescape")
        text, actions = clean_html(text)
        if also_layer_a_text:
            text2, stats = clean_text(text)
            if stats["removed_count"] or stats["replaced_count"]:
                actions.append(
                    f"layer A text: removed={stats['removed_count']} replaced={stats['replaced_count']}"
                )
                text = text2
        atomic_write_text(dest, text)
    elif fmt == "markdown":
        text = data.decode("utf-8", errors="surrogateescape")
        text, actions = clean_markdown(text)
        if also_layer_a_text:
            text2, stats = clean_text(text)
            if stats["removed_count"] or stats["replaced_count"]:
                actions.append(
                    f"layer A text: removed={stats['removed_count']} replaced={stats['replaced_count']}"
                )
                text = text2
        atomic_write_text(dest, text)
    else:
        raise ValueError(f"unsupported container format: {fmt}")

    after = inspect_container(dest)
    return {
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        "actions": actions,
        "bytes_in": len(data),
        "bytes_out": dest.stat().st_size,
        "still_has_c2pa": after.has_c2pa,
        "still_has_ai_metadata": after.has_ai_metadata,
        "post_findings": after.findings,
        "meta": meta,
    }
