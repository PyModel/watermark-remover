"""Tests for finding confidence and aggregate audit scripts (adapted to OURS)."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_lib import aggregate, format_sarif, is_actionable, scan_file
from audit_website import guess_kind, parse_sitemap
from common import classify_finding_confidence
from container_meta import inspect_container
from text_unicode import inspect_text


def test_classify_finding_confidence_buckets():
    cases = {
        "c2patool reports a C2PA-related manifest": "confirmed",
        "PNG chunk caBX (possible C2PA container)": "confirmed",
        "JPEG APP11 segment (JUMBF/C2PA common)": "confirmed",
        "pdf-structured:ai:digitalSourceType": "confirmed",
        "pdf-structured:ai:AIGC": "probable",
        "PNG tEXt: c2pa, contentcredentials": "probable",
        "frontmatter key: generator": "probable",
        'info: cms generator: <meta name="generator" content="WordPress">': "informational",
        "customXml parts: 1": "informational",
        "unsupported container: woff2": "informational",
        "svg <metadata> present": "informational",
        "byte-scan C2PA markers: c2pa": "likely_false_positive",
    }
    for finding, expected in cases.items():
        assert classify_finding_confidence(finding) == expected, finding


def test_text_report_hit_confidence():
    report = inspect_text("a" + chr(0x200B) + "b")
    assert report.to_dict()["hits"][0]["confidence"] == "probable"

    report = inspect_text("a" + chr(0x2003) + "b")
    assert report.to_dict()["hits"][0]["confidence"] == "informational"


def test_container_report_findings_confidence(tmp_path: Path):
    src = tmp_path / "cms.html"
    src.write_text(
        '<html><head><meta name="generator" content="WordPress 6.0"></head></html>',
        encoding="utf-8",
    )
    report = inspect_container(src)
    assert report.to_dict()["findings_confidence"] == ["informational"]


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _minimal_png_with_text() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def test_image_report_findings_confidence(tmp_path: Path):
    from image_meta import inspect_image

    src = tmp_path / "t.png"
    src.write_bytes(_minimal_png_with_text())
    report = inspect_image(src)
    d = report.to_dict()
    assert "findings_confidence" in d
    assert any(c in ("probable", "confirmed") for c in d["findings_confidence"])


def test_scan_file_text_and_html(tmp_path: Path):
    text = tmp_path / "a.txt"
    text.write_text("Hello" + chr(0x200B) + "World\n", encoding="utf-8")
    item = scan_file(text)
    assert item["kind"] == "text"
    assert is_actionable(item)

    html = tmp_path / "b.html"
    html.write_text(
        '<html><head><meta name="generator" content="WordPress"></head><body>ok</body></html>',
        encoding="utf-8",
    )
    item = scan_file(html)
    assert item["kind"] == "html"
    assert not is_actionable(item)


def test_scan_file_container_layer_a_reported_once(tmp_path: Path):
    md = tmp_path / "post.md"
    md.write_text("# Title\n\nHello" + chr(0x200B) + "World\n", encoding="utf-8")
    item = scan_file(md)
    layer_a = [f for f in item["findings"] if "layer-a" in f]
    assert len(layer_a) == 1
    assert item["suspicious_total"] == 1
    assert is_actionable(item)


def test_aggregate_summary(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.write_text("Hello" + chr(0x200B) + "World", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("# Plain markdown\n\nNo invisible carriers.", encoding="utf-8")
    items = [scan_file(a), scan_file(b)]
    summary = aggregate(items)
    assert summary["total"] == 2
    assert summary["actionable_files"] == 1
    assert summary["with_suspicious_text"] == 1


def test_format_sarif_shape(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.write_text("Hello" + chr(0x200B) + "World", encoding="utf-8")
    item = scan_file(a)
    sarif = format_sarif({"root": str(tmp_path), "files": [item]})
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["tool"]["driver"]["name"] == "watermark-remover"
    assert sarif["runs"][0]["results"]


def test_guess_kind():
    assert guess_kind("https://ex.com/a", b"", content_type="text/html; charset=utf-8") == "html"
    assert guess_kind("https://ex.com/a.png", b"") == "png"
    assert guess_kind("https://ex.com/a.pdf", b"") == "pdf"


def test_parse_sitemap_gz_payload():
    import gzip

    xml = b"<urlset><url><loc>https://example.com/a.html</loc></url></urlset>"
    kind, urls = parse_sitemap(gzip.compress(xml))
    assert kind == "urlset" or "urlset" in str(kind)
    assert urls and "https://example.com/a.html" in urls
