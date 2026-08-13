"""Inspect/clean AI provenance metadata in non-raster containers.

Formats: SVG, PDF (best-effort), DOCX, ODT, HTML, Markdown frontmatter.
Stdlib-first; PDF prefers optional exiftool/c2patool when present.
"""

from __future__ import annotations

import io
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import atomic_write_bytes, atomic_write_text, which
from image_meta import AI_META_HINTS, C2PA_MARKERS, run_optional_tools

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

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "has_c2pa": self.has_c2pa,
            "has_ai_metadata": self.has_ai_metadata,
            "findings": self.findings,
            "tools": self.tools,
            "details": self.details,
        }


def detect_container_format(path: Path, data: bytes | None = None) -> str:
    ext = path.suffix.lower()
    if ext in (".svg",):
        return "svg"
    if ext in (".pdf",):
        return "pdf"
    if ext in (".docx",):
        return "docx"
    if ext in (".odt",):
        return "odt"
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
            # zip-based; sniff
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    _validate_zip(zf)
                    names = set(zf.namelist())
                    if "word/document.xml" in names:
                        return "docx"
                    if "content.xml" in names and "meta.xml" in names:
                        return "odt"
            except (zipfile.BadZipFile, ValueError, RuntimeError):
                pass
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
    m = _FM_RE.match(text)
    if not m:
        return False, False, [], {"has_frontmatter": False}
    block = m.group(1)
    keys = []
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
    c2pa = any("c2pa" in f.lower() or "content" in f.lower() for f in findings)
    return c2pa, has_ai, findings, {"has_frontmatter": True, "keys": keys}


def clean_markdown(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    m = _FM_RE.match(text)
    if not m:
        return text, ["no YAML frontmatter"]
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
        if AI_META_NAME_RE.search(tag) or any(
            h.decode("ascii", "ignore").lower() in tag.lower() for h in AI_META_HINTS[:12]
        ):
            has_ai = True
            findings.append(f"meta: {tag[:120]}")
            if re.search(r"c2pa|content.?credential", tag, re.I):
                has_c2pa = True
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
    return has_c2pa, has_ai, findings, {}


def clean_html(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []

    def _meta_sub(m: re.Match[str]) -> str:
        tag = m.group(0)
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


def inspect_docx(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    parts: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _validate_zip(zf)
            parts = zf.namelist()
            for info in zf.infolist():
                name = info.filename
                raw = zf.read(info)
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
    except (zipfile.BadZipFile, ValueError, RuntimeError) as error:
        return False, True, [f"unsafe/invalid DOCX zip: {error}"], {"malformed": True}
    return has_c2pa, has_ai or has_c2pa, findings, {"parts": len(parts)}


def clean_docx(data: bytes) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    out_buf = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as zin,
        zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        _validate_zip(zin)
        for info in zin.infolist():
            name = info.filename
            raw = zin.read(info)
            # customXml can back content controls/business data. Preserve it; the
            # post-clean inspection reports any provenance-like residue.
            if name in DOCX_META_PARTS or name.startswith("docProps/"):
                text = raw.decode("utf-8", errors="replace")
                # Scrub known AI generator fields via simple regex on XML text nodes
                new = text
                for pattern, label in (
                    (
                        r"(<dc:creator[^>]*>)(.*?)(</dc:creator>)",
                        "dc:creator",
                    ),
                    (
                        r"(<cp:lastModifiedBy[^>]*>)(.*?)(</cp:lastModifiedBy>)",
                        "cp:lastModifiedBy",
                    ),
                    (
                        r"(<Application[^>]*>)(.*?)(</Application>)",
                        "Application",
                    ),
                    (
                        r"(<AppVersion[^>]*>)(.*?)(</AppVersion>)",
                        "AppVersion",
                    ),
                ):

                    def _sub(
                        m: re.Match[str],
                        *,
                        _label: str = label,
                        _part_name: str = name,
                    ) -> str:
                        inner = m.group(2)
                        if AI_META_NAME_RE.search(inner) or AI_META_NAME_RE.search(_label):
                            actions.append(f"scrub {_part_name} field {_label}")
                            return m.group(1) + m.group(3)
                        # Always clear Application if it looks like AI
                        if _label in ("Application", "AppVersion") and re.search(
                            r"claude|openai|anthropic|gemini|chatgpt|synthid|copilot",
                            inner,
                            re.I,
                        ):
                            actions.append(f"scrub {_part_name} field {_label}")
                            return m.group(1) + m.group(3)
                        return m.group(0)

                    new = re.sub(pattern, _sub, new, flags=re.I | re.DOTALL)
                raw = new.encode("utf-8")
            zout.writestr(info, raw)
    if not actions:
        actions.append("no DOCX metadata parts removed")
    return out_buf.getvalue(), actions


def inspect_odt(data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa = False
    has_ai = False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _validate_zip(zf)
            for info in zf.infolist():
                name = info.filename
                raw = zf.read(info)
                c2, ai, hits = _blob_hits(raw)
                if c2 or ai:
                    if c2:
                        has_c2pa = True
                    if ai:
                        has_ai = True
                    findings.append(f"{name}: {', '.join(hits[:6])}")
            if "meta.xml" in zf.namelist():
                meta = zf.read("meta.xml").decode("utf-8", errors="replace")
                if re.search(r"generator|claude|openai|anthropic|gemini", meta, re.I):
                    has_ai = True
                    findings.append("meta.xml generator-like fields")
    except (zipfile.BadZipFile, ValueError, RuntimeError) as error:
        return False, True, [f"unsafe/invalid ODT zip: {error}"], {"malformed": True}
    return has_c2pa, has_ai or has_c2pa, findings, {}


def clean_odt(data: bytes) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    out_buf = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as zin,
        zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        _validate_zip(zin)
        for info in zin.infolist():
            name = info.filename
            raw = zin.read(info)
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
            # Preserve every non-metadata part. Embedded objects, thumbnails,
            # and custom package data may legitimately contain vendor-like text;
            # inspection reports residual markers instead of deleting content.
            zout.writestr(info, raw)
    if not actions:
        actions.append("no ODT metadata removed")
    return out_buf.getvalue(), actions


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def inspect_pdf(path: Path, data: bytes) -> tuple[bool, bool, list[str], dict]:
    findings: list[str] = []
    has_c2pa, has_ai, hits = _blob_hits(data)
    findings.extend(f"pdf-bytes:{h}" for h in hits)
    # XMP packet scan
    if b"<x:xmpmeta" in data or b"application/rdf+xml" in data:
        findings.append("XMP packet present")
        has_ai = has_ai or bool(
            re.search(
                rb"digitalSourceType|trainedAlgorithmicMedia|SoftwareAgent|c2pa",
                data,
                re.I,
            )
        )
    tools = run_optional_tools(path)
    ct = tools.get("c2patool") or {}
    if ct.get("has_manifest"):
        has_c2pa = True
        findings.append("c2patool reports C2PA-related manifest")
    return has_c2pa, has_ai or has_c2pa, findings, {"tools": tools}


def clean_pdf_pypdf(path: Path, dest: Path) -> tuple[list[str], dict]:
    """Clean PDF metadata. exiftool > full-document pypdf clone > unchanged copy."""
    actions: list[str] = []
    data = path.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)

    exiftool = which("exiftool")

    # Strategy 1: exiftool (most reliable)
    if exiftool:
        dest.write_bytes(data)
        try:
            r = subprocess.run(
                [exiftool, "-all=", "-overwrite_original", str(dest)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            actions.append(f"exiftool -all= (rc={r.returncode})")
            if r.returncode == 0:
                return actions, {"mode": "exiftool", "degraded": False}
            actions.append(f"exiftool degraded (rc={r.returncode}); trying pypdf")
        except Exception as e:
            actions.append(f"exiftool failed: {e}; trying pypdf")

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
            dest.write_bytes(buf.getvalue())
            actions.append("pypdf: cloned full document graph; removed docinfo/XMP")
            return actions, {"mode": "pypdf", "degraded": False}
        except Exception as e:
            actions.append(f"pypdf failed: {e}; copied unchanged")
    else:
        actions.append("pypdf not installed; copied unchanged")

    # Never delete bytes from a PDF without rebuilding xref/object offsets.
    dest.write_bytes(data)
    actions.append("no structural PDF cleaner succeeded; copied unchanged")
    return actions, {"mode": "copy", "degraded": True}


def clean_pdf(path: Path, dest: Path) -> tuple[list[str], dict]:
    return clean_pdf_pypdf(path, dest)


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


def inspect_container(path: Path) -> ContainerInspectReport:
    data = path.read_bytes()
    fmt = detect_container_format(path, data)
    tools: dict[str, Any] = {}
    details: dict[str, Any] = {}

    if fmt == "svg":
        has_c2pa, has_ai, findings, details = inspect_svg(data)
    elif fmt == "pdf":
        has_c2pa, has_ai, findings, details = inspect_pdf(path, data)
        tools = details.pop("tools", {})
    elif fmt == "docx":
        has_c2pa, has_ai, findings, details = inspect_docx(data)
    elif fmt == "odt":
        has_c2pa, has_ai, findings, details = inspect_odt(data)
    elif fmt == "html":
        text = data.decode("utf-8", errors="replace")
        has_c2pa, has_ai, findings, details = inspect_html(text)
    elif fmt == "markdown":
        text = data.decode("utf-8", errors="replace")
        has_c2pa, has_ai, findings, details = inspect_markdown(text)
    else:
        has_c2pa, has_ai, findings = False, False, [f"unsupported container: {fmt}"]

    if fmt in ("svg", "pdf", "docx") and not tools:
        tools = run_optional_tools(path)

    return ContainerInspectReport(
        path=str(path),
        format=fmt,
        has_c2pa=has_c2pa,
        has_ai_metadata=has_ai,
        findings=findings,
        tools=tools,
        details=details,
    )


def clean_container(
    path: Path,
    dest: Path,
    *,
    also_layer_a_text: bool = True,
) -> dict[str, Any]:
    """Clean container metadata; optionally Layer-A scrub text bodies for md/html."""
    from text_unicode import clean_text  # local import to avoid cycles

    data = path.read_bytes()
    fmt = detect_container_format(path, data)
    actions: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"format": fmt}

    if fmt == "svg":
        cleaned, actions = clean_svg(data)
        dest.write_bytes(cleaned)
    elif fmt == "pdf":
        actions, meta_extra = clean_pdf(path, dest)
        meta.update(meta_extra)
    elif fmt == "docx":
        cleaned, actions = clean_docx(data)
        dest.write_bytes(cleaned)
    elif fmt == "odt":
        cleaned, actions = clean_odt(data)
        dest.write_bytes(cleaned)
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
        dest.write_text(text, encoding="utf-8", errors="surrogateescape")
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
        dest.write_text(text, encoding="utf-8", errors="surrogateescape")
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
