"""Tests for the HTTP service entrypoint (server.py).
Adapted from THEIRS test_http_server.py to the OURS paths. The server is
loaded as a module and bound to a free port via ThreadingHTTPServer.
"""

import base64
import http.client
import json
import struct
import sys
import threading
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server


def _png_chunk(ctype, payload):
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _watermarked_png():
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa test contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _post(conn, path, payload):
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


def _get(conn, path):
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


@pytest.fixture(scope="module")
def conn():
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_health(conn):
    status, body = _get(conn, "/health")
    assert status == 200
    assert body["ok"] is True
    assert "version" in body


def test_capabilities(conn):
    status, body = _get(conn, "/capabilities")
    assert status == 200
    assert set(body["tools"]) == {"c2patool", "exiftool", "qpdf"}
    assert "pixel_backends" in body
    assert "scorers" in body
    assert "harnesses" in body


def test_openapi_spec_covers_all_endpoints(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    assert body["openapi"] == "3.0.3"
    assert body["info"]["title"] == "watermarks-remover service"
    expected = {
        "/health": {"get"},
        "/capabilities": {"get"},
        "/openapi.json": {"get"},
        "/inspect": {"post"},
        "/clean": {"post"},
    }
    for path, methods in expected.items():
        assert path in body["paths"]
        for method in methods:
            assert method in body["paths"][path]
            assert "responses" in body["paths"][path][method]


def test_openapi_spec_describes_request_bodies(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    clean = body["paths"]["/clean"]["post"]
    assert clean["requestBody"]["required"] is True
    schema = clean["requestBody"]["content"]["application/json"]["schema"]
    assert "file" in schema["properties"]
    assert "options" in schema["properties"]
    inspect = body["paths"]["/inspect"]["post"]
    assert "file" in inspect["requestBody"]["content"]["application/json"]["schema"]["properties"]


def test_openapi_spec_reflects_auth(conn, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "sekret")
    conn.request("GET", "/openapi.json", headers={"Authorization": "Bearer sekret"})
    resp = conn.getresponse()
    body = json.loads(resp.read())
    assert resp.status == 200
    assert "bearerAuth" in body["components"]["securitySchemes"]
    assert body["security"] == [{"bearerAuth": []}]


def test_inspect_text_finds_watermark(conn):
    data = "Hello​World­!".encode()
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": "note.txt"})
    assert status == 200
    assert body["kind"] == "text"
    assert body["suspicious"] is True
    assert body["report"]["suspicious_total"] >= 1


def test_clean_text_roundtrip(conn):
    data = "Hello​World­!".encode()
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "note.txt"})
    assert status == 200
    assert body["kind"] == "text"
    cleaned = base64.b64decode(body["cleaned"]).decode("utf-8")
    assert cleaned == "HelloWorld!"
    assert body["report"]["stats"]["removed_count"] >= 1


def test_clean_png_strips_metadata(conn):
    data = _watermarked_png()
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "shot.png"})
    assert status == 200
    assert body["kind"] == "image"
    cleaned = base64.b64decode(body["cleaned"])
    assert b"c2pa" not in cleaned.lower()
    assert body["report"]["format"] == "png"


def test_clean_markdown_container(conn):
    data = (ROOT / "tests" / "fixtures" / "sample_ai.md").read_bytes()
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "note.md"})
    assert status == 200
    assert body["kind"] == "container"
    cleaned = base64.b64decode(body["cleaned"]).decode("utf-8")
    assert "generator: Claude" not in cleaned
    assert body["report"]["format"] == "markdown"


def test_unknown_option_rejected(conn):
    status, body = _post(
        conn, "/clean", {"file": _b64(b"x"), "name": "x.txt", "options": {"nope": 1}}
    )
    assert status == 400
    assert "unknown option" in body["error"]


def test_option_wrong_type_nfkc(conn):
    status, body = _post(
        conn, "/clean", {"file": _b64(b"x"), "name": "x.txt", "options": {"nfkc": "false"}}
    )
    assert status == 400
    assert "must be" in body["error"]


def test_option_wrong_type_remove_pixel(conn):
    status, body = _post(
        conn, "/clean", {"file": _b64(b"x"), "name": "x.txt", "options": {"remove_pixel": False}}
    )
    assert status == 400
    assert "must be" in body["error"]


def test_bad_base64_rejected(conn):
    status, _body = _post(conn, "/inspect", {"file": "!!!not-base64!!!"})
    assert status == 400


def test_binary_named_as_text_rejected(conn):
    status, _body = _post(conn, "/clean", {"file": _b64(_watermarked_png()), "name": "x.txt"})
    assert status == 400


def test_clean_unknown_format_rejected(conn):
    data = b"no magic, no extension"
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "input"})
    assert status == 400
    assert "unrecognized file format" in body["error"]


def test_inspect_unknown_format_reports_kind(conn):
    data = b"no magic, no extension"
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": "input"})
    assert status == 200
    assert body["kind"] == "unknown"
    assert body["suspicious"] is False
    assert "note" in body["report"]


def test_missing_file_field_rejected(conn):
    status, _body = _post(conn, "/inspect", {"name": "x.txt"})
    assert status == 400


def test_safe_name_sanitizes_traversal():
    assert server._safe_name("../../etc/passwd") == "passwd"
    assert server._safe_name("a/b/c/notes.md") == "notes.md"
    assert server._safe_name("..\\..\\win.txt") == "win.txt"
    assert server._safe_name("..") == "input"
    assert server._safe_name(".") == "input"
    assert server._safe_name("") == "input"
    assert server._safe_name("report.docx") == "report.docx"


def test_oversized_body_413(conn, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 64)
    data = "x" * 200
    status, _body = _post(conn, "/inspect", {"file": _b64(data.encode())})
    assert status == 413


def test_auth_required(conn, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "sekret")
    status, _ = _post(conn, "/health", {"anything": 1})
    assert status == 401
    conn.request("GET", "/health", headers={"Authorization": "Bearer sekret"})
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read()


def test_404(conn):
    status, _body = _get(conn, "/nope")
    assert status == 404
    status, _body = _post(conn, "/nope", {"file": _b64(b"x")})
    assert status == 404
