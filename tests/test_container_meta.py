"""Tests for SVG/HTML/MD/DOCX/ODT container cleaners."""

from __future__ import annotations

import io
import json
import posixpath
import re
import struct
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import container_meta
from container_meta import (
    MAX_ZIP_DECOMPRESSED_BYTES,
    ZipBudgetExceeded,
    _check_zip_budget,
    _read_zip_member,
    clean_container,
    clean_docx,
    clean_html,
    clean_markdown,
    clean_odt,
    clean_svg,
    detect_container_format,
    inspect_container,
    inspect_docx,
    inspect_html,
    inspect_markdown,
    inspect_odt,
    inspect_svg,
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_markdown_frontmatter():
    text = """---
title: Hello
generator: Claude
ai_generated: true
---
Body\u200b text.
"""
    _c2, has_ai, findings, _notes, _d = inspect_markdown(text)
    assert has_ai
    assert any("generator" in f or "ai" in f.lower() for f in findings)
    cleaned, actions = clean_markdown(text)
    assert "generator:" not in cleaned
    assert "ai_generated:" not in cleaned
    assert "title: Hello" in cleaned
    assert any("drop" in a.text for a in actions)


def test_markdown_frontmatter_with_blank_line_does_not_crash():
    """Regression: a blank line inside frontmatter used to raise IndexError."""
    text = "---\ntitle: Demo\n\nauthor: you\n---\nBody\n"
    cleaned, actions = clean_markdown(text)
    assert "title: Demo" in cleaned
    assert "author: you" in cleaned
    assert actions


def test_markdown_drops_nested_children_of_dropped_key():
    """Regression: nested values under a dropped AI key survived the clean."""
    text = "---\ntitle: Demo\nmodel:\n  name: claude-opus\n  version: 4\nauthor: you\n---\nBody\n"
    cleaned, actions = clean_markdown(text)
    assert "claude-opus" not in cleaned  # the leak
    assert "version: 4" not in cleaned
    assert "title: Demo" in cleaned  # siblings untouched
    assert "author: you" in cleaned
    assert any("drop frontmatter key: model" in a.text for a in actions)


def test_markdown_clean_output_is_no_longer_flagged():
    """Round-trip: re-inspecting a cleaned document reports nothing AI-ish."""
    text = "---\ntitle: Demo\nmodel:\n  name: claude-opus\ngenerator: Claude\n---\nBody\n"
    cleaned, _ = clean_markdown(text)
    _c2, has_ai, findings, _notes, _d = inspect_markdown(cleaned)
    assert not has_ai, findings


def test_markdown_preserves_comments_and_non_ai_keys():
    text = "---\n# editorial notes\ntitle: Demo\ntags:\n  - one\n  - two\n---\nBody\n"
    cleaned, _ = clean_markdown(text)
    assert "# editorial notes" in cleaned
    assert "- one" in cleaned and "- two" in cleaned


def test_html_meta_strip():
    html = """<html><head>
<meta name="generator" content="ChatGPT">
<meta name="viewport" content="width=device-width">
<meta name="description" content="ok">
</head><body data-ai-model="gpt">Hi</body></html>"""
    _c2, has_ai, _findings, _notes, _ = inspect_html(html)
    assert has_ai
    cleaned, actions = clean_html(html)
    assert "ChatGPT" not in cleaned
    assert "viewport" in cleaned
    assert "data-ai-model" not in cleaned
    assert any("drop" in a.text for a in actions)


def test_html_cms_generator_not_ai():
    html = '<meta name="generator" content="WordPress 6.0">'
    has_c2pa, has_ai, findings, notes, _ = inspect_html(html)
    assert not has_c2pa
    assert not has_ai
    assert findings == []
    assert any("cms" in note for note in notes)


def test_html_cms_generator_preserved_by_clean():
    html = '<html><head><meta name="generator" content="WordPress 6.0"><meta name="viewport" content="width=device-width"></head></html>'
    cleaned, _actions = clean_html(html)
    assert "WordPress" in cleaned
    assert "viewport" in cleaned


def test_html_cms_generator_attribute_names_are_case_insensitive():
    for html in (
        '<META NAME="generator" CONTENT="WordPress 6.0">',
        '<meta Name="generator" Content="WordPress 6.0">',
    ):
        has_c2pa, has_ai, findings, notes, _ = inspect_html(html)
        assert not has_c2pa
        assert not has_ai
        assert findings == []
        assert any("cms" in note for note in notes)
        assert clean_html(html)[0] == html

    ai_html = '<META NAME="generator" CONTENT="Claude">'
    assert inspect_html(ai_html)[1]
    assert clean_html(ai_html)[0] == ""


def test_html_ai_generator_still_dropped():
    html = '<meta name="generator" content="Claude">'
    cleaned, actions = clean_html(html)
    assert "Claude" not in cleaned
    assert any("drop" in a.text for a in actions)


def test_pdf_stream_byte_collision_not_ai(tmp_path: Path):
    from container_meta import inspect_pdf

    pdf = b"%PDF-1.4\n1 0 obj<< /Length 4 >>stream\nAIGC\nendstream\nendobj\n%%EOF\n"
    src = tmp_path / "collision.pdf"
    src.write_bytes(pdf)
    has_c2pa, has_ai, _findings, _notes, _ = inspect_pdf(src, pdf)
    assert not has_c2pa
    assert not has_ai


def test_svg_metadata():
    svg = b"""<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <metadata>c2pa contentcredentials Anthropic</metadata>
  <circle cx="1" cy="1" r="1"/>
</svg>"""
    has_c2pa, has_ai, _findings, _notes, _ = inspect_svg(svg)
    assert has_c2pa or has_ai
    cleaned, actions = clean_svg(svg)
    assert b"<metadata" not in cleaned.lower() or b"c2pa" not in cleaned.lower()
    assert b"<circle" in cleaned
    assert any("metadata" in a.text or "drop" in a.text for a in actions)


def _make_docx_with_app(app_name: str = "Claude AI Writer") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/customXml/item1.xml" ContentType="application/xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "docProps/app.xml",
            f'<?xml version="1.0"?><Properties><Application>{app_name}</Application></Properties>',
        )
        zf.writestr(
            "customXml/item1.xml",
            '<?xml version="1.0"?><root>c2pa contentcredentials</root>',
        )
    return buf.getvalue()


def test_docx_strips_app_and_customxml(tmp_path: Path):
    data = _make_docx_with_app()
    cleaned, actions = clean_docx(data)
    assert any(
        "customXml" in a.text or "Application" in a.text or "drop" in a.text for a in actions
    )
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert "word/document.xml" in names
        assert not any(n.startswith("customXml/") for n in names)
        app = zf.read("docProps/app.xml").decode()
        assert "Claude" not in app


def test_docx_thumbnail_part_is_dropped_not_corrupted():
    """docProps/thumbnail.jpeg is binary; it must be dropped, not UTF-8 round-tripped."""
    buf = io.BytesIO()
    thumbnail = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01binary-preview-bytes\x00\x01\x01"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="jpeg" ContentType="image/jpeg"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/docProps/thumbnail.jpeg" ContentType="image/jpeg"/>'
            "</Types>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>',
        )
        zf.writestr("docProps/thumbnail.jpeg", thumbnail)
    cleaned, actions = clean_docx(buf.getvalue())
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert not any(n.startswith("docProps/thumbnail") for n in names), names
        types = zf.read("[Content_Types].xml").decode()
        assert "/docProps/thumbnail.jpeg" not in types
    assert any("thumbnail" in a.text for a in actions)


def test_html_jsonld_ai_creator_dropped():
    text = (
        "<html><head>"
        '<script type="application/ld+json">{"creator":"ChatGPT","@type":"Article"}</script>'
        "</head><body>ok</body></html>"
    )
    cleaned, actions = clean_html(text)
    assert "ChatGPT" not in cleaned
    # Surgical granularity: the provenance key goes, the block's own payload
    # survives — the same field-level precision image/ODT cleaning uses.
    assert "ld+json" in cleaned
    assert '"@type": "Article"' in cleaned
    assert any(a.code == "scrub_field" and a.params.get("part") == "json-ld" for a in actions)


@pytest.mark.parametrize(
    "vendor_phrase",
    [
        "made with Midjourney",
        "generated by Midjourney",
        "made with DALL-E 3",
        "produced by Copilot",
        "powered by Sora",
        "generated by ChatGPT",
    ],
)
def test_html_vendor_comment_dropped(vendor_phrase):
    """Generation relations name the generator — the shared
    AI_PROVENANCE_TEXT_RE vocabulary, not an inline brand list."""
    comment = f"<!-- {vendor_phrase} -->"
    cleaned, actions = clean_html(f"<body>{comment}visible</body>")
    assert comment not in cleaned
    assert "visible" in cleaned
    assert any(a.code == "drop_html_comment" for a in actions)


@pytest.mark.parametrize(
    "human_phrase",
    [
        "stable diffusion output",
        "firefly render",
        "wire up the copilot dropdown",
        "sora: heading scale",
        "stable diffusion banner, paid placement",
    ],
)
def test_html_bare_vendor_comment_is_human_text(human_phrase):
    """Bare vendor words are ordinary language (image_meta invariant: only a
    generator field or generation relation makes them evidence). A comment
    mentioning Copilot/Sora/Firefly without one survives, and inspect agrees
    with clean so the preview never over-reports."""
    comment = f"<!-- {human_phrase} -->"
    cleaned, actions = clean_html(f"<body>{comment}visible</body>")
    assert comment in cleaned
    assert not any(a.code == "drop_html_comment" for a in actions)
    _c2pa, ai, findings, _notes, _ = inspect_html(f"<body>{comment}visible</body>")
    assert not ai
    assert not any("comment" in f for f in findings)


@pytest.mark.parametrize(
    "vendor_phrase",
    ["made with Midjourney", "made with DALL-E 3"],
)
def test_svg_vendor_comment_dropped(vendor_phrase):
    """SVG comments use the same shared rule as HTML (no drift)."""
    comment = f"<!-- {vendor_phrase} -->"
    cleaned, actions = clean_svg(
        f'<svg xmlns="http://www.w3.org/2000/svg">{comment}<rect/></svg>'.encode()
    )
    assert comment not in cleaned.decode()
    assert "<rect" in cleaned.decode()
    assert any(a.code == "drop_svg_comment" for a in actions)


def test_svg_bare_vendor_comment_is_human_text():
    """Bare vendor words in SVG comments are ordinary text, same as HTML."""
    comment = "<!-- firefly render -->"
    cleaned, actions = clean_svg(
        f'<svg xmlns="http://www.w3.org/2000/svg">{comment}<rect/></svg>'.encode()
    )
    assert comment in cleaned.decode()
    assert not any(a.code == "drop_svg_comment" for a in actions)


def test_html_comment_inside_script_is_payload_not_a_comment():
    """<!-- inside <script>/<style>/<pre> is ordinary text, never rewritten."""
    text = '<script>var s="<!-- AI GENERATED -->";var k=1;</script>'
    cleaned, _actions = clean_html(text)
    assert text in cleaned


def test_html_provenance_comment_dropped():
    """Marker-bearing comments are provenance noise (SVG already strips them)."""
    text = "<body><!-- AI-GENERATED do not remove -->visible text</body>"
    cleaned, _actions = clean_html(text)
    assert "AI-GENERATED" not in cleaned
    assert "visible text" in cleaned


def test_html_benign_comment_preserved():
    text = "<body><!-- section nav -->visible text</body>"
    cleaned, _actions = clean_html(text)
    assert "<!-- section nav -->" in cleaned


def test_html_jsonld_human_author_also_dropped():
    """Debrand consistency: images and ODT strip ALL author/date provenance by
    default, so a JSON-LD provenance key is dropped whatever its value — a
    human author name is an identity fingerprint, not content. Granularity is
    field-level, as for images/ODT: the key goes, the block's own payload
    (headline) survives."""
    text = '<script type="application/ld+json">{"author":"Jane Smith","headline":"Hello"}</script>'
    cleaned, actions = clean_html(text)
    assert "Jane Smith" not in cleaned
    assert "ld+json" in cleaned
    assert "headline" in cleaned
    assert any(a.code == "scrub_field" and a.params.get("field") == "author" for a in actions)


def test_html_jsonld_human_newsarticle_keeps_payload_drops_publisher():
    """A human NewsArticle with only a publisher loses the provenance key —
    not the whole structured-data payload (headline, datePublished stay)."""
    text = (
        '<script type="application/ld+json">{"@type":"NewsArticle",'
        '"headline":"Big story","datePublished":"2026-01-01","publisher":"NYT"}</script>'
    )
    cleaned, _actions = clean_html(text)
    assert "publisher" not in cleaned
    assert "NYT" not in cleaned
    assert "headline" in cleaned and "datePublished" in cleaned
    _c2pa, ai, findings, _notes, _ = inspect_html(text)
    assert ai and any("json-ld" in f for f in findings)


def test_html_jsonld_bare_vendor_word_in_value_keeps_block():
    """A description that merely mentions a vendor is not provenance: the
    block survives verbatim and inspect does not flag it."""
    text = (
        '<script type="application/ld+json">{"@type":"Article",'
        '"description":"an OpenAI announcement","headline":"Hello"}</script>'
    )
    cleaned, actions = clean_html(text)
    assert cleaned == text
    assert not any(a.effect.value == "change" for a in actions)
    assert not inspect_html(text)[1]


def test_html_meta_bare_vendor_word_in_content_keeps_tag():
    """<meta name="description" content="...the firefly"> is human text:
    the tag survives and inspect agrees (name vocabulary, not content scan)."""
    tag = '<meta name="description" content="A nature essay about the firefly">'
    text = f"<head>{tag}</head>"
    cleaned, actions = clean_html(text)
    assert tag in cleaned
    assert not any(a.code == "drop_html_meta" for a in actions)
    _c2pa, ai, _findings, _notes, _ = inspect_html(text)
    assert not ai


def test_html_meta_generator_content_relation_drops_tag():
    """A description whose content states a generation relation is evidence."""
    text = '<meta name="description" content="Hero image generated by DALL-E 3">'
    cleaned, actions = clean_html(text)
    assert "DALL-E" not in cleaned
    assert any(a.code == "drop_html_meta" for a in actions)
    assert inspect_html("<head>" + text + "</head>")[1]


def test_svg_generator_attributes_stripped_even_after_metadata_drop():
    text = (
        '<svg xmlns="http://www.w3.org/2000/svg" inkscape:version="1.3" '
        'sodipodi:docname="ai.svg"><metadata><rdf:RDF/></metadata>'
        '<rect width="1" height="1"/></svg>'
    )
    cleaned, actions = clean_svg(text.encode())
    out = cleaned.decode()
    assert "inkscape:version" not in out
    assert "sodipodi:docname" not in out
    assert "<rect" in out
    assert any("generator" in a.text.lower() or "svg" in a.text.lower() for a in actions)


def _dangling_rels(zip_bytes: bytes) -> list[str]:
    """Return every internal relationship whose target part is missing."""
    bad: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        for rels in (n for n in names if n.endswith(".rels")):
            base = posixpath.dirname(posixpath.dirname(rels))
            text = zf.read(rels).decode()
            for m in re.finditer(r'<Relationship\b[^>]*Target="([^"]*)"[^>]*/>', text, re.I):
                target, tag = m.group(1), m.group(0)
                if re.search(r"\bTargetMode\s*=", tag, re.I):
                    continue  # external
                if target.startswith("/"):
                    resolved = posixpath.normpath(target.lstrip("/"))
                else:
                    resolved = posixpath.normpath(posixpath.join(base, target))
                if resolved not in ("", ".") and resolved not in names:
                    bad.append(f"{rels} -> {target}")
    return bad


def _make_docx_with_rels() -> bytes:
    """DOCX whose document rels reference customXml, a kept part and a URL."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/customXml/item1.xml" ContentType="application/xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "customXml/item1.xml",
            '<?xml version="1.0"?><root>c2pa contentcredentials</root>',
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/customXml" Target="../customXml/item1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com/" TargetMode="External"/>
</Relationships>""",
        )
    return buf.getvalue()


def test_docx_dropped_customxml_prunes_dangling_relationships():
    data = _make_docx_with_rels()
    assert _dangling_rels(data) == []
    cleaned, actions = clean_docx(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert not any(n.startswith("customXml/") for n in names)
        rels = zf.read("word/_rels/document.xml.rels").decode()
        assert "../customXml/item1.xml" not in rels
        assert 'Target="document.xml"' in rels
        assert 'TargetMode="External"' in rels
    assert _dangling_rels(cleaned) == []
    assert any("prune dangling relationships" in a.text for a in actions)


def _make_docx_with_body_text(body_text: str = "Claude wrote this.") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>'
            + body_text
            + "</w:t></w:r></w:p></w:body></w:document>",
        )
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"></cp:coreProperties>',
        )
    return buf.getvalue()


def test_docx_body_vendor_word_is_not_ai_metadata():
    from container_meta import inspect_docx

    data = _make_docx_with_body_text()
    has_c2pa, has_ai, findings, _notes, _ = inspect_docx(data)
    assert not has_c2pa
    assert not has_ai
    assert not any("Claude" in f for f in findings)


def test_docx_metadata_vendor_word_is_still_flagged():
    from container_meta import inspect_docx

    data = _make_docx_with_app("Claude AI Writer")
    _has_c2pa, has_ai, findings, _notes, _ = inspect_docx(data)
    assert has_ai
    assert any("Claude" in f for f in findings)


def _make_docx_with_docprops() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/custom.xml" ContentType="application/vnd.openxmlformats-officedocument.custom-properties+xml"/>
</Types>""",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>',
        )
        zf.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">
  <dc:title>My Document</dc:title>
  <dc:creator>ChatGPT</dc:creator>
  <cp:lastModifiedBy>Claude</cp:lastModifiedBy>
  <dc:description>Generated by AI</dc:description>
  <cp:keywords>ai, model</cp:keywords>
  <dc:subject>artificial intelligence</dc:subject>
  <cp:category>report</cp:category>
</cp:coreProperties>""",
        )
        zf.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>ChatGPT</Application>
  <AppVersion>16.0</AppVersion>
  <Company>OpenAI</Company>
  <Manager>Someone</Manager>
</Properties>""",
        )
        zf.writestr(
            "docProps/custom.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" name="Client"><vt:lpwstr>Acme</vt:lpwstr></property>
</Properties>""",
        )
    return buf.getvalue()


def _make_docx_with_invisible_body() -> bytes:
    buf = io.BytesIO()
    document = (
        '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Hello\u200b world\u2060 with\u00a0space </w:t></w:r></w:p>"
        '<w:p><w:r><w:instrText xml:space="preserve"> FIELD \u200b KEEP </w:instrText></w:r></w:p>'
        '<w:p><w:r><w:t xml:space="preserve">keep\u200bme </w:t></w:r></w:p>'
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>""",
        )
        zf.writestr("word/document.xml", document)
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"></cp:coreProperties>',
        )
    return buf.getvalue()


def test_docx_scrubs_docprops_provenance_fields_unconditionally():
    import xml.etree.ElementTree as ET

    data = _make_docx_with_docprops()
    cleaned, actions = clean_docx(data)

    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        names = zf.namelist()
        assert "docProps/core.xml" in names
        assert "docProps/app.xml" in names
        assert "docProps/custom.xml" not in names
        ct = zf.read("[Content_Types].xml").decode()
        assert 'PartName="/docProps/custom.xml"' not in ct

        core = zf.read("docProps/core.xml").decode()
        app = zf.read("docProps/app.xml").decode()

    for field in (
        "dc:creator",
        "cp:lastModifiedBy",
        "dc:description",
        "cp:keywords",
        "dc:subject",
        "cp:category",
    ):
        assert f"<{field}></{field}>" in core or f"<{field}/>" in core
    for field in ("Application", "AppVersion", "Company", "Manager"):
        assert f"<{field}></{field}>" in app or f"<{field}/>" in app
    assert "<dc:title>My Document</dc:title>" in core
    assert "ChatGPT" not in core
    assert "Generated by AI" not in core
    assert "OpenAI" not in app

    ET.fromstring(core)  # noqa: S314
    ET.fromstring(app)  # noqa: S314

    assert any("scrub docProps/core.xml field dc:creator" in a.text for a in actions)
    assert any("drop part docProps/custom.xml" in a.text for a in actions)


def test_docx_docprops_scrub_clears_residual_warning(tmp_path: Path):
    src = tmp_path / "in.docx"
    src.write_bytes(_make_docx_with_docprops())
    dest = tmp_path / "out.docx"
    result = clean_container(src, dest)
    assert result["format"] == "docx"
    assert not result["still_has_ai_metadata"]
    assert not any("docProps/core.xml" in f for f in result["post_findings"])


def _make_docx_with_parts(core_xml: str, app_xml: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            "</Types>",
        )
        zf.writestr("word/document.xml", '<?xml version="1.0"?><w:document/>')
        zf.writestr("docProps/core.xml", core_xml)
        if app_xml is not None:
            zf.writestr("docProps/app.xml", app_xml)
    return buf.getvalue()


_CORE_NS = (
    'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
)


def test_docx_dates_revision_and_brand_title_are_scrubbed():
    """Debrand parity with ODT: dcterms:created/modified and cp:revision go
    empty, and a dc:title naming a product is scrubbed by the conditional
    pass — the brand string must never survive a VERIFIED-clean verdict."""
    core = (
        f'<?xml version="1.0"?><cp:coreProperties {_CORE_NS}>'
        "<dc:title>ChatGPT draft</dc:title>"
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-01-01T10:00:00Z</dcterms:created>'
        "<dcterms:modified>2026-01-02T10:00:00Z</dcterms:modified>"
        "<cp:revision>3</cp:revision>"
        "</cp:coreProperties>"
    )
    data = _make_docx_with_parts(core)
    _c2pa, ai, _findings, _notes, _ = inspect_docx(data)
    assert ai  # the residual scan sees ChatGPT in docProps
    cleaned, actions = clean_docx(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("docProps/core.xml").decode()
    assert "ChatGPT" not in out
    assert "2026-01-01" not in out and "2026-01-02" not in out
    assert "<dc:title>" in out  # neutral titles survive; this one is emptied, tag kept
    assert any("dc:title" in a.text for a in actions)
    assert any("dcterms:created" in a.text for a in actions)
    assert any("cp:revision" in a.text for a in actions)
    assert not inspect_docx(cleaned)[1]  # re-inspect: clean
    import xml.etree.ElementTree as ET

    ET.fromstring(out)  # noqa: S314


def test_docx_residual_product_name_in_exotic_docprops_field():
    """AI_GENERATOR_PRODUCTS joins the docProps residual scan, and the clean's
    conditional catch-all empties any exotic field carrying a product name —
    so no brand string can hide outside the known scrub list."""
    core = (
        f'<?xml version="1.0"?><cp:coreProperties {_CORE_NS}>'
        "<cp:contentStatus>ChatGPT final pass</cp:contentStatus>"
        "<dc:title>My Document</dc:title>"
        "</cp:coreProperties>"
    )
    app = (
        '<?xml version="1.0"?><Properties '
        'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        "<Application>Office</Application>"
        "</Properties>"
    )
    data = _make_docx_with_parts(core, app)
    assert inspect_docx(data)[1]
    cleaned, _actions = clean_docx(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("docProps/core.xml").decode()
    assert "ChatGPT" not in out
    assert "<dc:title>My Document</dc:title>" in out  # neutral title untouched
    assert not inspect_docx(cleaned)[1]


def test_docx_layer_a_strips_invisible_body_chars():
    data = _make_docx_with_invisible_body()
    cleaned, actions = clean_docx(data)
    assert any(a.text.startswith("layer A text: removed=") for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        doc = zf.read("word/document.xml").decode()
        assert "Hello\u200b" not in doc
        assert "world\u2060" not in doc
        assert "with\u00a0space" not in doc
        assert '<w:t xml:space="preserve">Hello world with space </w:t>' in doc
        assert " FIELD \u200b KEEP " in doc
        assert '<w:t xml:space="preserve">keepme </w:t>' in doc


def test_docx_layer_a_can_be_disabled():
    data = _make_docx_with_invisible_body()
    cleaned, actions = clean_docx(data, also_layer_a_text=False)
    assert not any(a.text.startswith("layer A text:") for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert "Hello\u200b" in zf.read("word/document.xml").decode()


def test_docx_layer_a_via_clean_container(tmp_path: Path):
    src = tmp_path / "in.docx"
    src.write_bytes(_make_docx_with_invisible_body())
    dest = tmp_path / "out.docx"
    result = clean_container(src, dest)
    assert any(a.startswith("layer A text: removed=") for a in result["actions"])
    # Each human line has a structured twin, in order, so no consumer parses text.
    details = result["action_details"]
    assert [d["text"] for d in details] == result["actions"]
    [layer_a] = [d for d in details if d["code"] == "layer_a_text"]
    assert layer_a["effect"] == "change"
    assert layer_a["params"] == {"removed": 3, "replaced": 1}
    with zipfile.ZipFile(dest) as zf:
        doc = zf.read("word/document.xml").decode()
        assert "Hello\u200b" not in doc
        assert " FIELD \u200b KEEP " in doc


def _make_odt_with_invisible_text() -> bytes:
    buf = io.BytesIO()
    content = (
        '<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        "<office:body><office:text>"
        '<text:p text:style-name="P1">Hello\u200b <text:span>world\u2060</text:span>!</text:p>'
        "</office:text></office:body></office:document-content>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("meta.xml", '<?xml version="1.0"?><office:document-meta/>')
        zf.writestr("content.xml", content)
        zf.writestr("META-INF/manifest.xml", '<?xml version="1.0"?><manifest:manifest/>')
    return buf.getvalue()


def test_odt_layer_a_strips_invisible_text():
    data = _make_odt_with_invisible_text()
    cleaned, actions = clean_odt(data)
    assert any(a.text.startswith("layer A text: removed=") for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        content = zf.read("content.xml").decode()
        assert "\u200b" not in content
        assert "\u2060" not in content
        assert "<text:span>world</text:span>" in content


def _make_odt(generator: str = "Anthropic Claude") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr(
            "meta.xml",
            f'<?xml version="1.0"?><office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"><meta:generator>{generator}</meta:generator></office:document-meta>',
        )
        zf.writestr(
            "content.xml",
            '<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"/>',
        )
        zf.writestr(
            "META-INF/manifest.xml",
            '<?xml version="1.0"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>',
        )
    return buf.getvalue()


def test_odt_drops_generator(tmp_path: Path):
    data = _make_odt()
    cleaned, actions = clean_odt(data)
    assert any("generator" in a.text for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        meta = zf.read("meta.xml").decode()
        assert "Claude" not in meta
        assert "meta:generator" not in meta or "Anthropic" not in meta


def test_odt_scrubs_neutral_creator_and_dates():
    """Debrand consistency: images strip all metadata by default, so ODT
    author/date provenance must not survive just because it isn't AI-ish."""
    buf = io.BytesIO()
    meta = (
        '<?xml version="1.0"?><office:document-meta '
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<meta:generator>SomeApp/1.0</meta:generator>"
        "<dc:creator>ai-bot</dc:creator>"
        "<meta:initial-creator>ai-bot</meta:initial-creator>"
        "<dc:date>2025-01-01T00:00:00</dc:date>"
        "<meta:creation-date>2025-01-01T00:00:00</meta:creation-date>"
        "</office:document-meta>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("meta.xml", meta)
        zf.writestr(
            "content.xml",
            '<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><text:p>keep me</text:p></office:text></office:body></office:document-content>',
        )
        zf.writestr("META-INF/manifest.xml", '<?xml version="1.0"?><manifest:manifest/>')
    cleaned, _actions = clean_odt(buf.getvalue())
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("meta.xml").decode()
    for gone in ("ai-bot", "meta:initial-creator", "dc:date", "meta:creation-date"):
        assert gone not in out
    assert "keep me" in zipfile.ZipFile(io.BytesIO(cleaned)).read("content.xml").decode()


def _make_odt_meta_probe(meta: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("meta.xml", meta)
        zf.writestr("content.xml", '<?xml version="1.0"?><office:document-content/>')
        zf.writestr("META-INF/manifest.xml", '<?xml version="1.0"?><manifest:manifest/>')
    return buf.getvalue()


_ODT_META_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:xlink="http://www.w3.org/1999/xlink"'
)


def test_odt_removes_template_user_defined_and_description_provenance():
    """inspect_odt flags what the widened scan sees; clean_odt must remove
    exactly that (meta:template, AI user-defined fields, AI description), so
    a re-inspect reaches a clean verdict."""
    meta = (
        f'<?xml version="1.0"?><office:document-meta {_ODT_META_NS}>'
        "<meta:generator>LibreOffice</meta:generator>"
        '<meta:template xlink:href="file:///Users/janedoe/Documents/ChatGPT-draft.odt"/>'
        '<meta:user-defined meta:name="ai:generator">ChatGPT-4o</meta:user-defined>'
        '<meta:user-defined meta:name="EditingCycles">7</meta:user-defined>'
        "<dc:description>Written with Claude</dc:description>"
        "</office:document-meta>"
    )
    data = _make_odt_meta_probe(meta)
    _c2pa, ai, findings, _notes, _ = inspect_odt(data)
    assert ai
    assert any("meta:template" in f for f in findings)
    assert any("ai:generator" in f for f in findings)
    cleaned, actions = clean_odt(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("meta.xml").decode()
    assert "meta:template" not in out and "janedoe" not in out
    assert "ai:generator" not in out and "ChatGPT-4o" not in out
    assert "Written with Claude" not in out
    assert 'meta:name="EditingCycles"' in out  # neutral field survives
    assert any("meta:template" in a.text for a in actions)
    assert not inspect_odt(cleaned)[1]  # re-inspect: clean


def test_odt_template_path_pii_removed_without_ai_word():
    """meta:template leaks a filesystem path even when it names no AI tool;
    it is creator-template provenance and always goes."""
    meta = (
        f'<?xml version="1.0"?><office:document-meta {_ODT_META_NS}>'
        '<meta:template xlink:href="file:///Users/janedoe/Documents/contract.odt"/>'
        "</office:document-meta>"
    )
    cleaned, _actions = clean_odt(_make_odt_meta_probe(meta))
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("meta.xml").decode()
    assert "meta:template" not in out and "janedoe" not in out


def test_odt_self_closing_creator_does_not_eat_siblings():
    """<dc:creator/> must not match as an open tag: the old pattern ate
    everything up to the next close of any scrubbed element, losing
    meta:editing-duration and meta:document-statistic with it."""
    meta = (
        f'<?xml version="1.0"?><office:document-meta {_ODT_META_NS}>'
        "<dc:creator/>"
        "<meta:editing-duration>PT1H30M</meta:editing-duration>"
        '<meta:document-statistic meta:table-count="0" meta:image-count="1"/>'
        "</office:document-meta>"
    )
    cleaned, _actions = clean_odt(_make_odt_meta_probe(meta))
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("meta.xml").decode()
    assert "dc:creator" not in out
    assert "meta:editing-duration" in out
    assert "meta:document-statistic" in out
    assert "image-count" in out


def test_odt_vendor_word_in_kept_field_is_scrubbed_and_reported():
    """A kept field whose value names a product (dc:title "ChatGPT draft") is
    flagged by the meta.xml byte-scan, so the clean must empty it — otherwise
    the verdict stays dirty forever."""
    meta = (
        f'<?xml version="1.0"?><office:document-meta {_ODT_META_NS}>'
        "<dc:title>ChatGPT draft</dc:title>"
        "</office:document-meta>"
    )
    data = _make_odt_meta_probe(meta)
    assert inspect_odt(data)[1]
    cleaned, _actions = clean_odt(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        out = zf.read("meta.xml").decode()
    assert "ChatGPT" not in out and "<dc:title>" in out
    assert not inspect_odt(cleaned)[1]


def test_odt_drops_marked_non_metadata_part_and_prunes_manifest():
    """A non-metadata part carrying AI/C2PA markers is dropped and its
    manifest file-entry pruned (THEIRS-aligned), not silently preserved."""
    manifest_entries = [
        '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>',
        '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>',
        '<manifest:file-entry manifest:full-path="Pictures/vendor-note.txt" manifest:media-type="text/plain"/>',
    ]
    data = _make_odt_with_manifest(
        manifest_entries,
        {"Pictures/vendor-note.txt": b"OpenAI Claude c2pa business data"},
    )
    cleaned, actions = clean_odt(data)
    assert any("drop part Pictures/vendor-note.txt" in a.text for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert "Pictures/vendor-note.txt" not in zf.namelist()
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert "Pictures/vendor-note.txt" not in manifest


def _make_odt_with_manifest(manifest_entries: list[str], extra_parts: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    manifest = (
        '<?xml version="1.0"?><manifest:manifest '
        'xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
        + "".join(manifest_entries)
        + "</manifest:manifest>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", b'<?xml version="1.0"?><office:document-content/>')
        zf.writestr("META-INF/manifest.xml", manifest)
        for name, payload in extra_parts.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_odt_dropped_part_removes_manifest_entry():
    """Dropping a marker-bearing part must also drop its manifest entry,
    or readers flag the package as damaged."""
    data = _make_odt_with_manifest(
        [
            '<manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.text" manifest:full-path="/"/>',
            '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>',
            '<manifest:file-entry manifest:media-type="application/octet-stream" manifest:full-path="custommeta.xml"/>',
        ],
        extra_parts={"custommeta.xml": b"<meta><creator>Anthropic Claude</creator></meta>"},
    )
    cleaned, actions = clean_odt(data)
    assert any("drop part custommeta.xml" in a.text for a in actions)
    assert any("drop manifest entries" in a.text for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        assert zf.namelist().count("META-INF/manifest.xml") == 1
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert "custommeta.xml" not in manifest
        assert 'full-path="content.xml"' in manifest
        assert 'full-path="/"' in manifest


def test_odt_manifest_rewrite_is_attribute_order_independent():
    """The dropped entry has full-path before media-type; matching must not
    depend on attribute order."""
    data = _make_odt_with_manifest(
        [
            '<manifest:file-entry manifest:full-path="/"/>',
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>',
            '<manifest:file-entry manifest:full-path="custommeta.xml" manifest:media-type="application/octet-stream"/>',
        ],
        extra_parts={"custommeta.xml": b"<meta><creator>Anthropic Claude</creator></meta>"},
    )
    cleaned, _actions = clean_odt(data)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert "custommeta.xml" not in manifest
        assert 'full-path="content.xml"' in manifest


def test_odt_manifest_untouched_when_nothing_dropped():
    data = _make_odt_with_manifest(
        [
            '<manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.text" manifest:full-path="/"/>',
            '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>',
        ],
        extra_parts={},
    )
    cleaned, actions = clean_odt(data)
    assert not any("manifest" in a.text for a in actions)
    with zipfile.ZipFile(io.BytesIO(cleaned)) as zf:
        manifest = zf.read("META-INF/manifest.xml").decode()
        assert 'full-path="content.xml"' in manifest


def test_clean_container_markdown_file(tmp_path: Path):
    src = tmp_path / "x.md"
    src.write_text("---\ngenerator: OpenAI\n---\nHi\u200b\n", encoding="utf-8")
    dest = tmp_path / "x.cleaned.md"
    result = clean_container(src, dest)
    assert dest.is_file()
    body = dest.read_text(encoding="utf-8")
    assert "generator" not in body
    assert "\u200b" not in body
    assert result["format"] == "markdown"


def test_inspect_container_reports_layer_a_body_text(tmp_path: Path):
    """inspect must flag invisible carriers that clean would strip.

    Regression: markdown/html routed to the container inspector, which never
    ran the Layer A scan, so identical bytes were reported suspicious as .txt
    and clean as .md while clean_container() went on to remove them.
    """
    body = "Hello\u200bworld\u200c test\u2060end.\n"
    for name in ("x.md", "x.html"):
        src = tmp_path / name
        src.write_text(body, encoding="utf-8")
        report = inspect_container(src)
        assert report.layer_a_total == 3, name
        assert report.to_dict()["suspicious_total"] == 3, name
        codepoints = {h["codepoint"] for h in report.layer_a_hits}
        assert {"U+200B", "U+200C", "U+2060"} <= codepoints, name
        assert any(f.startswith("layer-a:") for f in report.findings), name


def test_inspect_container_clean_leaves_no_layer_a(tmp_path: Path):
    """The post-clean re-inspect must come back with nothing left."""
    src = tmp_path / "x.md"
    src.write_text("Hi\u200b\u200cthere\n", encoding="utf-8")
    dest = tmp_path / "x.cleaned.md"
    clean_container(src, dest)
    assert inspect_container(dest).layer_a_total == 0


def test_inspect_container_clean_file_stays_clean(tmp_path: Path):
    """No false positives on ordinary prose."""
    src = tmp_path / "x.md"
    src.write_text("# Title\n\nOrdinary prose, nothing hidden.\n", encoding="utf-8")
    report = inspect_container(src)
    assert report.layer_a_total == 0
    assert report.layer_a_hits == []


def test_inspect_container_svg(tmp_path: Path):
    src = tmp_path / "a.svg"
    src.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"><metadata>c2pa</metadata></svg>')
    report = inspect_container(src)
    assert report.format == "svg"
    assert report.has_c2pa or report.has_ai_metadata


def test_fixtures_md_html_svg_roundtrip(tmp_path: Path):
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    for name in ("sample_ai.md", "sample_ai.html", "sample_meta.svg"):
        src = root / name
        dest = tmp_path / f"{name}.cleaned{src.suffix}"
        result = clean_container(src, dest)
        assert dest.is_file()
        assert result["format"] in ("markdown", "html", "svg")
        body = dest.read_bytes().lower()
        assert b"chatgpt" not in body
        assert b"generator: claude" not in body


def test_zip_containers_reject_duplicate_names():
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "first")
            zf.writestr("word/document.xml", "second")
    try:
        clean_docx(buf.getvalue())
    except ValueError as error:
        assert "duplicate entry names" in str(error)
    else:
        raise AssertionError("expected duplicate-name rejection")


def test_docx_rejects_suspicious_compression_ratio():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"x" * 5_000_000)
    data = buf.getvalue()
    try:
        clean_docx(data)
    except ValueError as error:
        assert "compression ratio" in str(error)
    else:
        raise AssertionError("expected zip-bomb rejection")
    _c2pa, _ai, findings, _notes, _ = inspect_docx(data)
    assert any("unsafe/invalid" in finding for finding in findings)


def _patch_zip_declared_size(data: bytes, new_size: int) -> bytes:
    """Rewrite the uncompressed-size field of every central-directory entry.

    Produces the crafted-archive shape zip-bomb guards must defend
    against: the central directory (which backs ZipInfo.file_size) claims
    ``new_size`` decompressed bytes while the real deflate payload and its
    CRC stay intact. The local headers are left untouched.
    """
    out = bytearray(data)
    pos = 0
    while True:
        pos = out.find(b"PK\x01\x02", pos)
        if pos == -1:
            break
        out[pos + 24 : pos + 28] = struct.pack("<I", new_size)
        pos += 4
    return bytes(out)


def test_zip_budget_rejects_oversized_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        info = zf.infolist()[0]
    info.file_size = MAX_ZIP_DECOMPRESSED_BYTES + 1
    raised = False
    try:
        _check_zip_budget(info, [0])
    except ZipBudgetExceeded:
        raised = True
    assert raised


def test_zip_budget_charges_real_bytes_not_declared(monkeypatch):
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1 << 20)
    payload = b"y" * (1 << 17)  # 128 KiB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("word/document.xml", payload)
    crafted = _patch_zip_declared_size(buf.getvalue(), (1 << 17) + (1 << 16))
    with zipfile.ZipFile(io.BytesIO(crafted)) as zf:
        info = zf.infolist()[0]
        assert info.file_size == (1 << 17) + (1 << 16)  # declared claim
        budget = [0]
        out = _read_zip_member(zf, info, budget)
    assert out == payload
    assert budget[0] == len(payload)  # real bytes, not the declared claim


def test_zip_budget_accumulates_real_bytes_across_members(monkeypatch):
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1 << 20)
    payload = b"a" * (1 << 19)  # 512 KiB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.xml", payload)
        zf.writestr("b.xml", payload)
        zf.writestr("c.xml", payload)  # 3 x 512 KiB > 1 MiB cap
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        budget = [0]
        infos = zf.infolist()
        _read_zip_member(zf, infos[0], budget)
        _read_zip_member(zf, infos[1], budget)
        with pytest.raises(ZipBudgetExceeded):
            _read_zip_member(zf, infos[2], budget)


def test_inspect_container_truncated_zip():
    truncated = bytes([0x50, 0x4B, 0x03, 0x04]) + bytes(8)
    garbage = b"not a zip at all"
    for data in (truncated, garbage):
        assert inspect_docx(data) == (False, False, ["not a valid DOCX zip"], [], {})
        assert inspect_odt(data) == (False, False, ["not a valid ODT zip"], [], {})
        assert detect_container_format(Path("x.bin"), data) == "unknown"


def test_zip_budget_rejection_propagates_from_inspect(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    monkeypatch.setattr(container_meta, "MAX_ZIP_DECOMPRESSED_BYTES", 1)
    with pytest.raises(container_meta.ZipBudgetExceeded):
        inspect_docx(buf.getvalue())


def test_pdf_degraded_clean_without_crash(tmp_path: Path):
    """Minimal PDF with an XMP packet; clean should not raise (may be degraded)."""
    from container_meta import clean_pdf, inspect_pdf

    xmp = (
        b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        b"<rdf:Description>"
        b"<digitalSourceType>trainedAlgorithmicMedia</digitalSourceType>"
        b"</rdf:Description></rdf:RDF></x:xmpmeta>"
        b"<?xpacket end='w'?>"
    )
    pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n" + xmp + b"\n%%EOF\n"
    src = tmp_path / "t.pdf"
    dest = tmp_path / "t.cleaned.pdf"
    src.write_bytes(pdf)
    has_c2pa, has_ai, findings, _notes, _ = inspect_pdf(src, pdf)
    assert has_ai or has_c2pa or findings
    actions, meta = clean_pdf(src, dest)
    assert dest.is_file()
    assert actions
    assert meta.get("mode") in ("exiftool", "pypdf", "copy")
    if meta.get("mode") == "copy":
        assert dest.read_bytes() == pdf


def test_clean_container_accepts_explicit_fmt(tmp_path: Path):
    """A .bak source with no detectable magic bytes still cleans when fmt is pinned."""
    src = tmp_path / "backup.md.bak"
    src.write_text("---\ngenerator: OpenAI\n---\nHi\u200b\n", encoding="utf-8")
    dest = tmp_path / "out.md"
    result = clean_container(src, dest, fmt="markdown")
    assert result["format"] == "markdown"
    body = dest.read_text(encoding="utf-8")
    assert "generator" not in body
    assert "\u200b" not in body


@pytest.mark.parametrize(
    "ext,make_bytes",
    [
        (
            "md",
            (Path(__file__).resolve().parent / "fixtures" / "sample_ai.md").read_bytes,
        ),
        (
            "html",
            (Path(__file__).resolve().parent / "fixtures" / "sample_ai.html").read_bytes,
        ),
        (
            "svg",
            (Path(__file__).resolve().parent / "fixtures" / "sample_meta.svg").read_bytes,
        ),
        ("docx", _make_docx_with_app),
        ("odt", _make_odt),
    ],
)
def test_clean_file_in_place_for_every_container_ext(tmp_path: Path, ext: str, make_bytes):
    path = tmp_path / f"a.{ext}"
    path.write_bytes(make_bytes())
    r = _run("clean_file.py", str(path), "--in-place", "--json")
    assert r.returncode == 0, r.stderr
    assert path.with_suffix(path.suffix + ".bak").is_file()
    data = json.loads(r.stdout)
    assert data["kind"] == "container"
    assert (
        data["format"]
        == {"docx": "docx", "odt": "odt", "svg": "svg", "md": "markdown", "html": "html"}[ext]
    )


def test_cms_generator_meta_with_spaced_attributes_is_kept() -> None:
    page = '<html><head><meta name = "generator" content = "WordPress 6.5"></head></html>'
    _c2pa, has_ai, findings, notes, _details = container_meta.inspect_html(page)
    assert not has_ai
    assert findings == []
    assert any("cms generator" in note for note in notes)
    cleaned, _actions = clean_html(page)
    assert 'content = "WordPress 6.5"' in cleaned


def test_markdown_word_content_in_a_key_is_not_c2pa() -> None:
    doc = "---\ncontent_generator: ChatGPT\ntitle: x\n---\nbody\n"
    c2pa, has_ai, findings, _notes, _details = container_meta.inspect_markdown(doc)
    assert has_ai and findings
    assert c2pa is False


def test_markdown_content_credentials_key_is_c2pa() -> None:
    doc = "---\ncontent_credentials: yes\n---\nbody\n"
    c2pa, _has_ai, _findings, _notes, _details = container_meta.inspect_markdown(doc)
    assert c2pa is True


def test_markdown_key_matching_name_and_value_counts_once() -> None:
    doc = "---\ngenerator: Claude\n---\nbody\n"
    _c2pa, _has_ai, findings, _notes, _details = container_meta.inspect_markdown(doc)
    assert findings == ["frontmatter key: generator"]


def test_marker_hits_list_each_marker_once_regardless_of_case() -> None:
    has_c2pa, has_ai, findings = container_meta._blob_hits(
        b"c2pa C2PA jumb JUMB contentcredentials"
    )
    assert has_c2pa and has_ai
    assert findings == ["marker:c2pa", "marker:jumb", "marker:contentcredentials"]
