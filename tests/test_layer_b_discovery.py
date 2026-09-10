"""Layer B endpoint policy and model discovery.

Discovery targets an operator-supplied URL, so it is held to the same rules as
the rewrite itself: default-deny for non-loopback hosts, http(s) only, no
redirect off-origin, and bounded responses.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from layer_b_discovery import (
    MODEL_ROUTES,
    classify_endpoint,
    layer_b_status,
    probe_backend,
)


class _Handler(BaseHTTPRequestHandler):
    payloads: ClassVar[dict[str, object]] = {}

    def do_GET(self):
        payload = self.payloads.get(self.path)
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return
        if payload == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.test/v1/models")
            self.end_headers()
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


# --- endpoint policy ---------------------------------------------------------


def test_loopback_is_allowed_without_an_opt_in():
    for url in ("http://127.0.0.1:11434", "http://localhost:8000", "http://[::1]:9000"):
        policy = classify_endpoint(url)
        assert policy.allowed, url
        assert policy.loopback
        assert policy.warning is None


def test_remote_is_denied_by_default_and_says_why():
    policy = classify_endpoint("https://api.example.test")
    assert not policy.allowed
    assert not policy.loopback
    assert "not loopback" in policy.reason
    # The warning is carried even while denied, so a UI can show what opting in means.
    assert "content will leave this machine" in policy.warning


def test_remote_with_an_explicit_opt_in_is_allowed_but_still_warns():
    policy = classify_endpoint("https://api.example.test", allow_remote=True)
    assert policy.allowed
    assert "content will leave this machine" in policy.warning


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "gopher://h"])
def test_non_http_schemes_are_always_refused(url):
    policy = classify_endpoint(url, allow_remote=True)
    assert not policy.allowed
    assert "http(s)" in policy.reason


def test_missing_url_is_not_allowed():
    assert not classify_endpoint(None).allowed
    assert not classify_endpoint("").allowed


# --- probing -----------------------------------------------------------------


def test_probe_refuses_a_remote_host_before_making_any_request():
    probe = probe_backend("ollama", "https://api.example.test")
    assert not probe.reachable
    assert "not loopback" in probe.error


def test_probe_lists_ollama_models(stub_server):
    _Handler.payloads = {
        "/api/tags": {"models": [{"name": "qwen3:14b"}, {"model": "llama3"}]},
    }
    probe = probe_backend("ollama", stub_server)
    assert probe.reachable
    assert probe.models == ("llama3", "qwen3:14b")
    assert "2 model(s)" in probe.summary


def test_probe_lists_openai_compatible_models(stub_server):
    _Handler.payloads = {"/v1/models": {"data": [{"id": "gpt-x"}, {"id": "gpt-y"}]}}
    probe = probe_backend("openai-compatible", stub_server)
    assert probe.reachable
    assert probe.models == ("gpt-x", "gpt-y")


def test_probe_survives_a_malformed_payload(stub_server):
    """Endpoint output is untrusted: unexpected shapes yield no models, not a crash."""
    _Handler.payloads = {"/api/tags": {"models": ["not-a-dict", {"no_name": 1}, None]}}
    probe = probe_backend("ollama", stub_server)
    assert probe.reachable
    assert probe.models == ()
    assert "no models listed" in probe.summary


def test_probe_refuses_an_off_origin_redirect(stub_server):
    """An Authorization header must never be replayed to another host."""
    _Handler.payloads = {"/v1/models": "redirect"}
    probe = probe_backend("openai-compatible", stub_server, api_key="unit-test-key-not-real")
    assert not probe.reachable


def test_probe_reports_an_unreachable_endpoint_without_raising():
    probe = probe_backend("ollama", "http://127.0.0.1:1")
    assert not probe.reachable
    assert probe.error


def test_every_live_backend_has_a_discovery_route():
    from rewrite_text import LIVE_REWRITE_BACKENDS

    assert set(MODEL_ROUTES) == set(LIVE_REWRITE_BACKENDS)


# --- capability reporting ----------------------------------------------------


def test_status_is_configuration_only_and_makes_no_request():
    status = layer_b_status("ollama", "http://127.0.0.1:11434")
    assert status["configured"] is True
    assert status["loopback"] is True
    assert status["endpoint_allowed"] is True
    # No model list: /capabilities is polled and must not become an outbound call.
    assert "models" not in status


def test_status_reports_an_unconfigured_backend():
    status = layer_b_status(None, None)
    assert status["configured"] is False
    assert status["endpoint_allowed"] is False


def test_capabilities_exposes_extras_and_layer_b():
    import server

    capabilities = server.capabilities()
    assert set(capabilities["extras"]) >= {"visible", "quality", "ai", "provenance", "tui"}
    assert "endpoint_allowed" in capabilities["layer_b"]
    for entry in capabilities["extras"].values():
        assert "available" in entry
        assert entry["hint"]
