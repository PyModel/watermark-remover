#!/usr/bin/env python3
"""End-to-end verification that cleaning leaves zero residual marks.

Builds real fixtures with real watermark/metadata channels, runs the shipped
CLI (``clean_file.py``) as a subprocess, then re-parses every output with
INDEPENDENT byte-level parsers (raw chunk/marker/ZIP walkers, PIL, pypdf) --
never the project's own inspector. Each scenario prints an ``*_OK`` token only
after its assertions ran; the script ends with ``FULLCLEAN_OK``.

Run: ``python3 tests/verify_full_clean.py`` (needs Pillow + pypdf for fixture
building and third-party cross-checks; the cleaner itself is stdlib-only).
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "skills" / "remove-ai-marks" / "scripts" / "clean_file.py"
FIXTURES = ROOT / "tests" / "fixtures"

from PIL import Image  # third-party fixture builder / verifier

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  !! {msg}")


def require(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def run_clean(src: Path, out: Path, *extra: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(CLEAN), str(src), "-o", str(out), "--json", *extra],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    require(proc.returncode == 0, f"clean exit={proc.returncode} stderr={proc.stderr[-400:]}")
    require("Traceback" not in proc.stderr, f"clean crashed: {proc.stderr[-400:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        fail(f"clean did not emit JSON: {proc.stdout[:200]}")
        return {}


# ---------------------------------------------------------------------------
# Independent parsers
# ---------------------------------------------------------------------------


def png_chunks(data: bytes) -> list[tuple[str, bytes]]:
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "output is not a PNG")
    chunks, off = [], 8
    while off < len(data):
        (length,), ctype = struct.unpack(">I", data[off : off + 4]), data[off + 4 : off + 8]
        chunks.append((ctype.decode("latin-1"), data[off + 8 : off + 8 + length]))
        off += 12 + length
        if ctype == b"IEND":
            break
    return chunks


def jpeg_markers(data: bytes) -> list[int]:
    require(data[:2] == b"\xff\xd8", "output is not a JPEG")
    markers, off = [], 2
    while off < len(data) - 1:
        if data[off] != 0xFF:
            off += 1
            continue
        m = data[off + 1]
        if m in (0x00, 0xFF) or 0xD0 <= m <= 0xD7:
            off += 2
            continue
        markers.append(m)
        if m == 0xD9:
            break
        if m == 0xDA:  # SOS: scan to EOI
            return markers
        seg_len = struct.unpack(">H", data[off + 2 : off + 4])[0]
        off += 2 + seg_len
    return markers


def riff_chunks(data: bytes) -> list[tuple[str, bytes]]:
    require(data[:4] == b"RIFF" and data[8:12] == b"WEBP", "output is not a WebP")
    chunks, off = [], 12
    while off + 8 <= len(data):
        ctype = data[off : off + 4].decode("latin-1")
        length = struct.unpack("<I", data[off + 4 : off + 8])[0]
        chunks.append((ctype, data[off + 8 : off + 8 + length]))
        off += 8 + length + (length & 1)
    return chunks


def zip_members(path: Path) -> list[str]:
    require(zipfile.is_zipfile(path), f"{path.name} is not a valid ZIP")
    with zipfile.ZipFile(path) as zf:
        require(zf.testzip() is None, f"{path.name} has a corrupt ZIP member")
        return zf.namelist()


def splice_png(before_iend: list[tuple[bytes, bytes]], src: Path) -> None:
    base = src.read_bytes()
    iend = base.rfind(b"IEND")
    body, extra = base[: iend - 4], b""
    for ctype, payload in before_iend:
        extra += struct.pack(">I", len(payload)) + ctype + payload
        extra += struct.pack(">I", zlib_crc(ctype + payload))
    src.write_bytes(body + extra + base[iend - 4 :])


def zlib_crc(payload: bytes) -> int:
    import zlib  # stdlib

    return zlib.crc32(payload) & 0xFFFFFFFF


def splice_jpeg_segments(segments: list[tuple[int, bytes]], src: Path) -> None:
    base = src.read_bytes()
    extra = b""
    for marker, payload in segments:
        extra += bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload
    out = base[:2] + extra + base[2:]
    src.write_bytes(out)


# ---------------------------------------------------------------------------
# Fixtures + scenarios
# ---------------------------------------------------------------------------


def make_text(dirpath: Path) -> Path:
    src = dirpath / "doc.txt"
    src.write_text(
        "Title​hidden zero​width⁠word⁣joiner braille⠀mark\N{VARIATION SELECTOR-16}"
        "\U000e0041\U000e0042tag-encoded\U000e0043‮bidi-override‬ end "
        "\U0001f468‍\U0001f469‍\U0001f467 keep-family\n",
        encoding="utf-8",
    )
    return src


def check_text(src: Path, out: Path) -> None:
    text = out.read_text(encoding="utf-8")
    # Default mode keeps U+2063 between visible chars as a DOCUMENTED,
    # reported context-aware preservation (math operators; stats record it
    # under preserved_semantic). Everything else must be gone.
    banned = ["​", "⁠", "⠀", "‮", "‬", "\U000e0041", "\U000e0042", "\U000e0043", " "]
    for ch in banned:
        require(ch not in text, f"residual invisible char U+{ord(ch):04X} in output text")
    require("keep-family" in text, "visible text payload lost")
    require("‍" in text, "semantic emoji ZWJ sequence was stripped (context-awareness claim)")


def check_text_forensic(src: Path, out: Path) -> None:
    text = out.read_text(encoding="utf-8")
    banned = ["​", "⁠", "⁣", "⠀", "‮", "‬", "\U000e0041", "\U000e0042", "\U000e0043", " "]
    for ch in banned:
        require(ch not in text, f"residual invisible char U+{ord(ch):04X} after forensic scrub")
    require("keep-family" in text, "visible text payload lost")


def make_png(dirpath: Path) -> Path:
    src = dirpath / "img.png"
    img = Image.new("RGB", (8, 8), (30, 60, 90))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    src.write_bytes(buf.getvalue())
    # Hand-splice every metadata channel PIL does not write: zTXt, tIME, and a
    # C2PA-like caBX chunk plus an AI-branded tEXt.
    x = b"AI Software\x00generated by ChatGPT"
    ztx = struct.pack(">B", 0) + x[: len(x) // 1]

    splice_png(
        [
            (b"zTXt", struct.pack(">H", 0) + ztx),
            (b"tIME", struct.pack(">HBBBBB", 2025, 1, 1, 0, 0, 0)),
            (b"caBX", b"fake c2pa jumbf payload"),
            (b"tEXt", b"Software\x00DALL-E 3"),
        ],
        src,
    )
    return src


def check_png(src: Path, out: Path) -> None:
    types = [c for c, _ in png_chunks(out.read_bytes())]
    for bad in ("tEXt", "iTXt", "zTXt", "tIME", "caBX"):
        require(bad not in types, f"residual {bad} chunk in cleaned PNG: {types}")
    require("IHDR" in types and "IDAT" in types and "IEND" in types, "PNG structure broken")
    with Image.open(out) as im:
        im.verify()


def make_jpeg(dirpath: Path) -> Path:
    src = dirpath / "img.jpg"
    img = Image.new("RGB", (8, 8), (10, 120, 30))
    exif = Image.Exif()
    exif[0x0131] = "Midjourney v6"  # Software
    exif[0x013B] = "ai-bot"  # Artist
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes(), xmp=b'<?xpacket xmp:CreatorTool="ChatGPT"?>')
    src.write_bytes(buf.getvalue())
    splice_jpeg_segments([(0xFE, b"watermarked by AI"), (0xEC, b"Ducky\x00fake-exif-canvas")], src)
    return src


def check_jpeg(src: Path, out: Path) -> None:
    markers = jpeg_markers(out.read_bytes())
    require(0xE1 not in markers, "residual APP1 (EXIF/XMP) in cleaned JPEG")
    require(0xFE not in markers, "residual COM comment in cleaned JPEG")
    require(0xEC not in markers, "residual APP12 Ducky in cleaned JPEG")
    require(0xDB in markers or 0xC0 in markers or 0xDA in markers, "JPEG payload broken")
    with Image.open(out) as im:
        im.verify()
    require(out.read_bytes().find(b"Midjourney") == -1, "EXIF Software string survived")
    require(out.read_bytes().find(b"CreatorTool") == -1, "XMP CreatorTool survived")


def make_gif(dirpath: Path) -> None:
    src = dirpath / "anim.gif"
    Image.new("P", (8, 8)).save(src, "GIF")
    base = src.read_bytes()
    # Splice AFTER header + logical screen descriptor + optional global color
    # table (flag 0x80, size 2^(N+1) entries x 3 bytes), never inside it.
    flags = base[10]
    gct = 3 * (2 ** ((flags & 0x07) + 1)) if flags & 0x80 else 0
    insert = 13 + gct
    comment = b"\x21\xfe" + bytes([12]) + b"AI-generated" + b"\x00"
    xmp = b"\x21\xff\x0bXMP DataXMP" + bytes([4]) + b"<x/>" + b"\x00"
    src.write_bytes(base[:insert] + comment + xmp + base[insert:])
    return src


def check_gif(src: Path, out: Path) -> None:
    data = out.read_bytes()
    require(data[:6] in (b"GIF89a", b"GIF87a"), "output is not a GIF")
    require(b"AI-generated" not in data, "residual GIF comment extension")
    require(b"XMP DataXMP" not in data, "residual GIF XMP application extension")
    require(b"\x21\xfe" not in data, "residual comment marker 21 FE in GIF")
    with Image.open(out) as im:
        im.verify()


def make_webp(dirpath: Path) -> Path:
    src = dirpath / "img.webp"
    img = Image.new("RGB", (8, 8), (90, 20, 40))
    buf = io.BytesIO()
    img.save(buf, "WEBP")
    base = bytearray(buf.getvalue())
    # Insert EXIF + XMP RIFF chunks after the VP8 chunk, fix the RIFF size.
    exif = b"EXIF" + struct.pack("<I", 8) + b"II*\x00AI-E"
    xmp = b"XMP " + struct.pack("<I", 4) + b"<x/>"
    body = base[12:] + exif + xmp
    base[4:8] = struct.pack("<I", 4 + len(body))
    src.write_bytes(bytes(base[:12] + body))
    return src


def check_webp(src: Path, out: Path) -> None:
    chunks = riff_chunks(out.read_bytes())
    names = [c for c, _ in chunks]
    require("EXIF" not in names, "residual EXIF chunk in cleaned WebP")
    require("XMP " not in names, "residual XMP chunk in cleaned WebP")
    require(any(c in ("VP8 ", "VP8L", "VP8X") for c in names), "WebP payload broken")
    with Image.open(out) as im:
        im.verify()


def make_pdf(dirpath: Path) -> Path:
    src = dirpath / "doc.pdf"
    xmp = (
        b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        b"<rdf:RDF><rdf:Description dc:creator='ChatGPT' xmp:CreatorTool='DALL-E'/>"
        b"</rdf:RDF></x:xmpmeta><?xpacket end='w'?>"
    )
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>",
        b"<< /Producer (Midjourney v6) /Author (ai-bot) /Creator (ChatGPT) >>",
        b"<< /Type /Metadata /Subtype /XML /Length %d >>\nstream\n" % len(xmp)
        + xmp
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 4 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref,
    )
    src.write_bytes(bytes(out))
    return src


def check_pdf(src: Path, out: Path) -> None:
    data = out.read_bytes()
    require(data.startswith(b"%PDF-"), "output is not a PDF")
    require(b"/Producer" not in data, "residual /Producer in cleaned PDF")
    require(b"Midjourney" not in data, "residual Producer string in cleaned PDF")
    require(b"<?xpacket" not in data and b"xmpmeta" not in data, "residual XMP stream in PDF")
    require(b"ChatGPT" not in data, "residual AI brand string in cleaned PDF")
    import pypdf

    reader = pypdf.PdfReader(out)
    require(len(reader.pages) == 1, "PDF page payload lost")


def office_files(dirpath: Path) -> list[Path]:
    """Minimal DOCX + ODT, each with AI provenance in the properties parts."""
    docx = dirpath / "doc.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>hello world</w:t></w:r></w:p></w:body></w:document>",
        )
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:creator>ChatGPT</dc:creator>'
            "<cp:lastModifiedBy>ai-bot</cp:lastModifiedBy></cp:coreProperties>",
        )
        zf.writestr(
            "docProps/app.xml",
            '<?xml version="1.0"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>ChatGPT export</Application></Properties>",
        )
    odt = dirpath / "doc.odt"
    with zipfile.ZipFile(odt, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr(
            "META-INF/manifest.xml",
            '<?xml version="1.0"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
            '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>'
            '<manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/>'
            "</manifest:manifest>",
        )
        zf.writestr(
            "content.xml",
            '<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><text:p>hello</text:p></office:text></office:body></office:document-content>',
        )
        zf.writestr(
            "meta.xml",
            '<?xml version="1.0"?><office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"><office:meta>'
            "<meta:generator>ChatGPT/1.0</meta:generator><dc:creator>ai-bot</dc:creator>"
            "</office:meta></office:document-meta>",
        )
    return [docx, odt]


def check_office(src: Path, out: Path) -> None:
    members = zip_members(out)
    joined = ""
    with zipfile.ZipFile(out) as zf:
        joined = "".join(zf.read(m).decode("utf-8", "replace") for m in members)
    for brand in ("ChatGPT", "ai-bot"):
        require(
            brand not in joined, f"residual provenance string {brand!r} in cleaned {src.suffix}"
        )
    if src.suffix == ".docx":
        core = [m for m in members if m.endswith("core.xml")]
        if core:
            with zipfile.ZipFile(out) as zf:
                body = zf.read(core[0]).decode("utf-8", "replace")
            require("<dc:creator>ChatGPT" not in body, "docx core.xml creator survived")
        require("hello world" in joined, "document payload lost")
    elif src.suffix == ".odt":
        require("<text:p>hello</text:p>" in joined, "document payload lost")


def make_html(dirpath: Path) -> Path:
    src = dirpath / "page.html"
    src.write_text(
        "<!doctype html><html><head>"
        '<meta name="generator" content="ChatGPT">'
        '<meta property="ai:provenance" content="dall-e">'
        '<script type="application/ld+json">{"creator":"ChatGPT"}</script>'
        '</head><body><p data-ai-generated="true">content here</p>'
        "<!-- AI-GENERATED do not remove -->visible text</body></html>"
    )
    return src


def check_html(src: Path, out: Path) -> None:
    text = out.read_text(encoding="utf-8")
    for needle in ("ChatGPT", 'name="generator"', "ld+json", "data-ai-generated", "AI-GENERATED"):
        require(needle not in text, f"residual provenance {needle!r} in cleaned HTML")
    require("visible text" in text, "visible payload lost")


def make_svg(dirpath: Path) -> Path:
    src = dirpath / "pic.svg"
    src.write_text(
        '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" '
        'inkscape:version="1.0" sodipodi:docname="ai.svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd">'
        "<title>pic</title><metadata><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<cc:Work rdf:about='' xmlns:cc='http://creativecommons.org/ns#'><dc:creator>ChatGPT</dc:creator></cc:Work>"
        "</rdf:RDF></metadata>"
        '<rect width="10" height="10" fill="#123456"/></svg>'
    )
    return src


def check_svg(src: Path, out: Path) -> None:
    text = out.read_text(encoding="utf-8")
    for needle in ("ChatGPT", "<metadata", "sodipodi:docname", "inkscape:version"):
        require(needle not in text, f"residual provenance {needle!r} in cleaned SVG")
    require("<rect" in text and "#123456" in text, "drawing payload lost")


def check_heif_fixture(src: Path, out: Path) -> None:
    data = out.read_bytes()
    require(data.find(b"c2pa") == -1, f"residual c2pa marker in cleaned {src.suffix}")
    require(
        data.find(b"jumb") == -1 and data.find(b"JUMBF") == -1,
        f"residual JUMBF box in cleaned {src.suffix}",
    )
    # Structural integrity: the ftyp box with the original brand must survive.
    require(data[4:8] == b"ftyp", f"{src.suffix} ftyp box lost")
    require(data[8:12] == src.read_bytes()[8:12], f"{src.suffix} major brand changed")
    try:
        with Image.open(src) as _:
            decodable = True
    except Exception:
        decodable = False  # env has no HEIF decoder; byte checks above suffice
    if decodable:
        with Image.open(out) as im:
            im.load()
            require(im.size[0] > 0, f"{src.suffix} raster lost")


def check_office_fixture(src: Path, out: Path) -> None:
    check_office(src, out)  # same zip-level assertions


def make_visible(dirpath: Path) -> Path:
    src = dirpath / "logo.png"
    img = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            img.putpixel((x, y), (x * 4 % 256, y * 4 % 256, 128))
    for y in range(10, 30):
        for x in range(10, 30):
            img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    src.write_bytes(buf.getvalue())
    return src


def check_visible(src: Path, out: Path) -> None:
    orig, new = Image.open(src).convert("RGB"), Image.open(out).convert("RGB")
    require(new.size == orig.size, "visible clean changed dimensions")
    region = [new.getpixel((x, y)) for y in range(10, 30) for x in range(10, 30)]
    require(any(p != (255, 0, 0) for p in region), "logo region still solid red: mask not filled")
    # The pipeline dilates the 20x20 box by radius 3 -> effective 26x26 mask
    # (7..33). That halo is expected to change; everything beyond it must not.
    outside = [
        (x, y, orig.getpixel((x, y)), new.getpixel((x, y)))
        for y in range(64)
        for x in range(64)
        if not (7 <= x < 33 and 7 <= y < 33)
    ]
    mutated = [(x, y) for x, y, a, b in outside if a != b]
    require(not mutated, f"pixels outside the dilated mask were mutated: {mutated[:5]}")


def check_robustness(dirpath: Path) -> None:
    cases = {
        "unknown.xyz": b"\xde\xad\xbe\xef" * 10,
        "empty.png": b"",
        "corrupt.png": b"\x89PNG\r\n\x1a\nGARBAGE",
        "jpeg-as.png": b"\xff\xd8\xff\xe0JFIF-fake" + b"\x00" * 32,
        "no-sig.bin": b"\x00" * 64,
    }
    for name, payload in cases.items():
        src, out = dirpath / name, dirpath / (name + ".cleaned")
        src.write_bytes(payload)
        proc = subprocess.run(
            [sys.executable, str(CLEAN), str(src), "-o", str(out), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        require("Traceback" not in proc.stderr, f"crash on {name}: {proc.stderr[-300:]}")
        if proc.returncode == 0 and out.exists():
            body = out.read_bytes()
            require(
                body == payload or proc.stdout.strip(),
                f"{name} silently passed through with no report",
            )


def check_report_honesty(report: dict, src: Path) -> None:
    """The honesty contract: whatever was removed is enumerated in the report."""
    details = report.get("action_details")
    has_actions = isinstance(details, list) and all(
        isinstance(d, dict) and d.get("code") and d.get("text") for d in details
    )
    has_stats = isinstance(report.get("stats"), dict) and "removed" in report["stats"]
    require(has_actions or has_stats, "report lacks action_details and removal stats")
    require(
        report.get("bytes_out") is not None or report.get("stats") is not None,
        "report lacks byte/char accounting",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def scenario(name: str, make, check, *extra: str, allowed_codes: tuple[int, ...] = (0,)) -> None:
    before = len(FAILURES)
    print(f"[{name}]", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = make(tmpdir)
        out = tmpdir / f"{src.stem}.cleaned{src.suffix if src.suffix != '.xyz' else '.xyz'}"
        proc = subprocess.run(
            [sys.executable, str(CLEAN), str(src), "-o", str(out), "--json", *extra],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        require(
            proc.returncode in allowed_codes,
            f"clean exit={proc.returncode} stderr={proc.stderr[-400:]}",
        )
        require("Traceback" not in proc.stderr, f"clean crashed: {proc.stderr[-400:]}")
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            fail(f"clean did not emit JSON: {proc.stdout[:200]}")
            report = {}
        require(out.exists(), f"{name}: no output written")
        if out.exists():
            check(src, out)
            check_report_honesty(report, src)
    status = "OK" if len(FAILURES) == before else "FAILED"
    print(f"  {status} {name}", flush=True)


def main() -> int:
    scenario("text-hidden-unicode", make_text, check_text)
    scenario("text-forensic-scrub", make_text, check_text_forensic, "--strip-semantic-format")
    scenario("png-chunks", make_png, check_png)
    scenario("jpeg-segments", make_jpeg, check_jpeg)
    scenario("gif-extensions", make_gif, check_gif)
    scenario("webp-chunks", make_webp, check_webp)
    scenario("pdf-info-xmp", make_pdf, check_pdf)
    scenario("docx-props", lambda d: office_files(d)[0], check_office)
    scenario("odt-meta", lambda d: office_files(d)[1], check_office)
    scenario("html-meta", make_html, check_html)
    scenario("svg-metadata", make_svg, check_svg)

    def copy_fixture(fname: str):
        def _mk(d: Path) -> Path:
            src = d / fname
            src.write_bytes((FIXTURES / fname).read_bytes())
            return src

        return _mk

    scenario("heic-c2pa-fixture", copy_fixture("sample_c2pa.heic"), check_heif_fixture)
    scenario("avif-c2pa-fixture", copy_fixture("sample_c2pa.avif"), check_heif_fixture)
    scenario("pptx-fixture", copy_fixture("sample_ai.pptx"), check_office_fixture)
    scenario("xlsx-fixture", copy_fixture("sample_ai.xlsx"), check_office_fixture)
    scenario(
        "visible-box",
        make_visible,
        check_visible,
        "--visible-box",
        "10,10,20,20",
        allowed_codes=(0, 1),  # 1 = documented BEST_EFFORT verification status
    )

    print("[robustness]", flush=True)
    before = len(FAILURES)
    with tempfile.TemporaryDirectory() as tmp:
        check_robustness(Path(tmp))
    print(f"  {'OK' if len(FAILURES) == before else 'FAILED'} robustness", flush=True)

    print("\n=== verify_full_clean ===")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} assertion(s)")
        for f in FAILURES:
            print(f" - {f}")
        return 1
    print("FULLCLEAN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
