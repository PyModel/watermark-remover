"""Focused regression tests for the release-hardening stability slice.

Covers:
  - synthid_score_server.py: exceptions raised by score_file are contained
    at the request boundary and answered with the fail-soft HTTP 200 JSON
    contract ({"available": false, "error": ...}) instead of killing the
    request with an unstructured traceback. Success and auth paths kept.
  - score_synthid.py: the external checkout is inserted into sys.path at
    most once, and the shared process-global work (codebook load /
    extraction / stdout redirection) is serialized under a module-level
    threading.Lock so concurrent HTTP requests cannot corrupt it.
  - server.py: /detect answers unrecognized assets with the documented
    HTTP 200 note (kind "unknown") instead of falling through to the
    container branch / a 500, the OpenAPI spec documents the unknown
    shape, and /clean keeps refusing unknown formats.
  - compose-check.sh: the wr-core health check curl carries bounded
    connect/max timeouts; endpoint and status assertions are preserved.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import re
import struct
import sys
import threading
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Ensure the real cv2 (when installed) wins over the stub upstream's cv2.py:
# score_file inserts the extraction dir at sys.path[0] before importing cv2,
# so without this the stub would shadow the real module for the whole test
# session. When cv2 is absent the stub satisfies the import instead.
with contextlib.suppress(ImportError):
    import cv2  # noqa: F401

import score_synthid
import server
import synthid_score_server


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _tiny_png() -> bytes:
    """A real 1x1 RGB PNG, decodable by either the stub or real cv2."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")  # filter byte + 1 RGB pixel
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _post(conn, path, payload, headers=None):
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
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


@pytest.fixture(scope="module")
def sidecar():
    srv = synthid_score_server.ThreadingHTTPServer(("127.0.0.1", 0), synthid_score_server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


# --- score_synthid.py: one sys.path insertion + serialized shared state ---


def _write_stub_upstream(root: Path) -> Path:
    """Stub reverse-SynthID checkout with a noisy, importable extractor.

    Mirrors the stub in test_synthid_stdout_purity.py: the upstream prints
    progress straight to stdout (which redirect_stdout must divert) and
    carries its own cv2 module so the import works without OpenCV.
    """
    ext = root / "src" / "extraction"
    ext.mkdir(parents=True)
    (root / "artifacts").mkdir()
    (root / "artifacts" / "spectral_codebook_v4.npz").touch()

    (ext / "cv2.py").write_text(
        "COLOR_BGR2RGB = 4\n"
        "def imread(path):\n"
        "    return object()\n"
        "def cvtColor(img, code):\n"
        "    return img\n"
    )
    (ext / "synthid_bypass_v4.py").write_text(
        "class SpectralCodebookV4:\n"
        "    def load(self, path):\n"
        '        print(f"CodebookV4 loaded: {path}")\n'
    )
    (ext / "robust_extractor.py").write_text(
        "from types import SimpleNamespace\n"
        "class RobustSynthIDExtractor:\n"
        "    def detect_from_v4_codebook(self, rgb, codebook, model=None):\n"
        "        return SimpleNamespace(\n"
        "            details={'profile_key': 'stub', 'exact_match': False,\n"
        "                     'per_channel_scores': [0.1], 'per_channel_n': [1]},\n"
        "            is_watermarked=False, confidence=0.42,\n"
        "            phase_match=0.0, multi_scale_consistency=0.0,\n"
        "        )\n"
    )
    return root


def test_score_file_inserts_upstream_path_once(tmp_path):
    upstream = _write_stub_upstream(tmp_path / "upstream")
    img = tmp_path / "img.png"
    img.write_bytes(_tiny_png())
    extraction = str(upstream / "src" / "extraction")
    assert extraction not in sys.path

    first = score_synthid.score_file(img, upstream_dir=str(upstream))
    second = score_synthid.score_file(img, upstream_dir=str(upstream))

    assert first[0] == 0, first
    assert second[0] == 0, second
    assert sys.path.count(extraction) == 1


def test_score_file_shared_state_is_guarded_by_module_lock():
    lock = score_synthid._SCORE_LOCK
    assert callable(getattr(lock, "acquire", None))
    assert callable(getattr(lock, "release", None))


def test_concurrent_score_file_serializes_shared_state(tmp_path):
    upstream = _write_stub_upstream(tmp_path / "upstream")
    img = tmp_path / "img.png"
    img.write_bytes(_tiny_png())
    extraction = str(upstream / "src" / "extraction")
    results: list = []
    errors: list = []

    def worker() -> None:
        try:
            results.append(score_synthid.score_file(img, upstream_dir=str(upstream)))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    assert len(results) == 8
    assert all(code == 0 and payload["confidence"] == 0.42 for code, payload in results)
    assert sys.path.count(extraction) == 1


# --- synthid sidecar: score_file exceptions become fail-soft 200s ---


def test_sidecar_contains_score_file_exception_as_fail_soft_200(sidecar, monkeypatch):
    def boom(path, *, model=None):
        raise RuntimeError("codebook exploded")

    monkeypatch.setattr(synthid_score_server, "score_file", boom)
    status, body = _post(sidecar, "/score", {"file": _b64(_tiny_png())})

    assert status == 200
    assert body["available"] is False
    assert isinstance(body["error"], str) and body["error"]
    assert "Traceback" not in body["error"]


def test_sidecar_score_success_payload_passthrough(sidecar, monkeypatch):
    payload = {"available": True, "is_watermarked": False, "confidence": 0.1}
    monkeypatch.setattr(
        synthid_score_server, "score_file", lambda path, *, model=None: (0, payload)
    )
    status, body = _post(sidecar, "/score", {"file": _b64(_tiny_png())})

    assert status == 200
    assert body == payload


def test_sidecar_auth_still_enforced(sidecar, monkeypatch):
    monkeypatch.setattr(synthid_score_server, "API_KEY", "sekret")
    monkeypatch.setattr(
        synthid_score_server, "score_file", lambda path, *, model=None: (0, {"available": True})
    )
    status, _ = _post(sidecar, "/score", {"file": _b64(_tiny_png())})
    assert status == 401
    status, body = _post(
        sidecar,
        "/score",
        {"file": _b64(_tiny_png())},
        headers={"Authorization": "Bearer sekret"},
    )
    assert status == 200
    assert body["available"] is True


# --- server.py: /detect unknown note, OpenAPI shape, /clean refusal ---


def test_detect_unknown_format_returns_200_note(conn):
    data = b"no magic, no extension"
    status, body = _post(conn, "/detect", {"file": _b64(data), "name": "input"})

    assert status == 200
    assert body["ok"] is True
    assert body["kind"] == "unknown"
    assert body["detections"] == []
    assert "note" in body["report"]


def test_detect_openapi_documents_unknown_kind(conn):
    status, body = _get(conn, "/openapi.json")
    assert status == 200
    detect = body["paths"]["/detect"]["post"]
    schema = detect["responses"]["200"]["content"]["application/json"]["schema"]
    properties = schema["properties"]
    assert "unknown" in properties["kind"]["enum"]
    assert "report" in properties


def test_clean_unknown_format_still_refused(conn):
    data = b"no magic, no extension"
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": "input"})
    assert status == 400
    assert "unrecognized file format" in body["error"]


# --- compose-check.sh: bounded health-check curl ---


def test_compose_check_health_curl_has_bounded_timeouts():
    script = (ROOT / "compose-check.sh").read_text(encoding="utf-8")
    match = re.search(r"curl\s+([^\n]*)\$BASE_URL/health", script)
    assert match, "health check curl invocation not found"
    flags = match.group(1)
    assert "-fsS" in flags
    assert "--connect-timeout 2" in flags
    assert "--max-time 10" in flags
    assert 'echo "wr-core: OK"' in script
    assert 'echo "wr-core: FAIL' in script
    assert "for svc in wr-markllm wr-markdiffusion wr-ctrlregen wr-synthid; do" in script
    assert 'exit "$FAIL"' in script
