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
    assert body["info"]["title"] == "watermark-remover service"
    expected = {
        "/health": {"get"},
        "/capabilities": {"get"},
        "/openapi.json": {"get"},
        "/inspect": {"post"},
        "/clean": {"post"},
        "/detect": {"post"},
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
    response = inspect["responses"]["200"]["content"]["application/json"]["schema"]
    assert response["properties"]["detection_status"]["enum"] == [
        "DETECTED",
        "INCONCLUSIVE",
        "NOT_DETECTED",
        "NOT_RUN",
    ]


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
    assert body["detection_status"] == "NOT_RUN"
    assert body["report"]["suspicious_total"] >= 1


def test_inspect_text_treats_inconclusive_detector_as_suspicious(conn, monkeypatch):
    monkeypatch.setattr(
        server,
        "run_all_text_detectors",
        lambda _text: [
            {
                "detector": "markllm",
                "available": True,
                "verdict": "INCONCLUSIVE",
                "is_watermarked": False,
            }
        ],
    )
    status, body = _post(
        conn,
        "/inspect",
        {"file": _b64(b"ordinary text"), "name": "note.txt", "detect": True},
    )
    assert status == 200
    assert body["suspicious"] is True
    assert body["detection_status"] == "INCONCLUSIVE"


def test_inspect_text_treats_failed_configured_detector_as_suspicious(conn, monkeypatch):
    """A configured detector that ran and failed is unresolved, not 'never ran'."""
    monkeypatch.setattr(
        server,
        "run_all_text_detectors",
        lambda _text: [
            {
                "detector": "markllm",
                "available": False,
                "configured": True,
                "error": "MarkLLM detection timed out",
            }
        ],
    )
    status, body = _post(
        conn,
        "/inspect",
        {"file": _b64(b"ordinary text"), "name": "note.txt", "detect": True},
    )
    assert status == 200
    assert body["detection_status"] == "INCONCLUSIVE"
    assert body["suspicious"] is True


@pytest.mark.parametrize(
    ("reports", "expected"),
    [
        ([{"available": True, "verdict": "DETECTED", "is_watermarked": True}], "DETECTED"),
        (
            [
                {"available": True, "verdict": "INCONCLUSIVE", "is_watermarked": False},
                {"available": True, "verdict": "NOT_DETECTED", "is_watermarked": False},
            ],
            "INCONCLUSIVE",
        ),
        ([{"available": True, "verdict": "UNSUPPORTED"}], "INCONCLUSIVE"),
        ([{"available": True, "error": "detector failed"}], "INCONCLUSIVE"),
        ([{"available": True, "verdict": "NOT_DETECTED", "is_watermarked": False}], "NOT_DETECTED"),
        ([{"available": False, "configured": False, "error": "not configured"}], "NOT_RUN"),
        ([{"available": False, "error": "not configured"}], "NOT_RUN"),
        (
            [{"available": False, "configured": True, "error": "MarkLLM detection timed out"}],
            "INCONCLUSIVE",
        ),
        (
            [
                {"available": False, "configured": True, "error": "MarkLLM detection timed out"},
                {"available": True, "verdict": "NOT_DETECTED", "is_watermarked": False},
            ],
            "INCONCLUSIVE",
        ),
        ([], "NOT_RUN"),
    ],
)
def test_text_detection_status(reports, expected):
    assert server._text_detection_status(reports) == expected


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
    assert body["detection_status"] == "NOT_RUN"
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


def test_executor_failure_is_a_sanitized_500(conn, monkeypatch):
    """An executor error payload is internal, not a client-visible 400.

    _run_planned re-raised every executor error payload as ValueError, so a
    failure inside clean_asset surfaced as 400 with the raw exception text.
    Only planner/preflight refusals (ValueError out of plan_work) are client
    errors; executor failures must map back to the sanitized 500 the direct
    calls this endpoint replaced produced.
    """

    def _boom(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "kind": "unknown",
            "error": "s3cr3t internal detail: /tmp/wm-clean-xyz/path.py:1",
            "exit_code": 1,
        }

    monkeypatch.setattr(server, "run_clean_item", _boom)
    status, body = _post(
        conn, "/clean", {"file": _b64("Hello\u200bWorld!".encode()), "name": "note.txt"}
    )
    assert status == 500
    assert body == {"ok": False, "error": "internal error"}
    assert "s3cr3t" not in json.dumps(body)


def test_clean_refuses_oversized_input_like_the_cli(conn, monkeypatch):
    """The plan path's input cap gates /clean exactly as it gates the CLI.

    A body under the HTTP cap can still decode to a file the planner refuses;
    the direct-cleaning path this endpoint used to run had no size check, so a
    request like this cleaned fine where ``wm`` would have refused.
    """
    import clean_request

    monkeypatch.setattr(clean_request, "MAX_INPUT_BYTES", 64)
    data = b"x" * 200
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "x.txt"})
    assert status == 400
    assert "refusing input larger than 64 bytes" in body["error"]


def test_clean_refuses_oversized_input_with_layer_a_skipped(conn, monkeypatch):
    """The also_layer_a_text=False container branch inherits the shared preflight.

    It used to call clean_container directly, skipping plan_work: a body under
    the HTTP cap that decodes over the input cap cleaned with 200 where every
    other branch returns the planner's refusal.
    """
    import clean_request

    monkeypatch.setattr(clean_request, "MAX_INPUT_BYTES", 64)
    data = (ROOT / "tests" / "fixtures" / "sample_ai.md").read_bytes()
    assert len(data) > 64
    status, body = _post(
        conn,
        "/clean",
        {"file": _b64(data), "name": "note.md", "options": {"also_layer_a_text": False}},
    )
    assert status == 400
    assert "refusing input larger than 64 bytes" in body["error"]


def test_clean_container_layer_a_skip_still_honored_after_preflight(conn):
    """Under the cap, also_layer_a_text=False keeps skipping the Layer-A scrub.

    The branch now clears plan_work before cleaning; the option must keep its
    meaning — the hidden text survives — where the default branch still scrubs
    it. This pins the executor gap the direct clean_container call exists for:
    the canonical plan cannot express the skip, so folding this branch's
    execution into run_clean_item would silently run the scrub.
    """
    data = "---\ngenerator: Claude\n---\n\nhidden\u200bmark\n".encode("utf-8")
    status, skipped = _post(
        conn,
        "/clean",
        {"file": _b64(data), "name": "note.md", "options": {"also_layer_a_text": False}},
    )
    assert status == 200
    assert skipped["kind"] == "container"
    cleaned = base64.b64decode(skipped["cleaned"]).decode("utf-8")
    assert "\u200b" in cleaned  # Layer-A text scrub skipped
    assert "generator: Claude" not in cleaned  # metadata layer still ran
    status, default = _post(conn, "/clean", {"file": _b64(data), "name": "note.md"})
    assert status == 200
    assert "\u200b" not in base64.b64decode(default["cleaned"]).decode("utf-8")


def test_clean_text_report_key_order(conn):
    status, body = _post(
        conn, "/clean", {"file": _b64("Hello\u200bWorld\u00ad!".encode()), "name": "note.txt"}
    )
    assert status == 200
    assert list(body["report"]) == ["kind", "stats", "length"]


def test_clean_image_report_key_order(conn):
    status, body = _post(conn, "/clean", {"file": _b64(_watermarked_png()), "name": "shot.png"})
    assert status == 200
    assert list(body["report"]) == [
        "kind",
        "format",
        "actions",
        "action_details",
        "bytes_in",
        "bytes_out",
        "still_has_c2pa",
        "still_has_ai_metadata",
        "post_findings",
        "synthid_before",
        "synthid_after",
        "synthid_removal",
        "wmct_marker",
    ]


def test_clean_container_report_key_order(conn):
    data = (ROOT / "tests" / "fixtures" / "sample_ai.md").read_bytes()
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "note.md"})
    assert status == 200
    assert list(body["report"]) == [
        "kind",
        "format",
        "actions",
        "action_details",
        "bytes_in",
        "bytes_out",
        "still_has_c2pa",
        "still_has_ai_metadata",
        "post_findings",
        "meta",
    ]


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


def test_capabilities_survives_an_unparseable_remote_opt_in(monkeypatch):
    """/capabilities is a report; a bad env value must not take it down.

    ``read_bool_env`` raises on a value like "maybe", and ``do_GET`` does not
    catch it, so the whole capability report failed over one misspelt flag —
    while the rewrite path it describes would simply have denied.
    """
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "maybe")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://example.com:11434")

    report = server.capabilities()
    assert report["layer_b"]["endpoint_allowed"] is False
    assert report["layer_b"]["loopback"] is False
