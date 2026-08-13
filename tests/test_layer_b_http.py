"""Behavioral contract for Layer B's shared HTTP transport."""

from __future__ import annotations

import io
import math
import sys
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import layer_b_http
from layer_b_http import LayerBHTTPError, request_json

FAKE_SECRET = "unit-test-secret-never-real"


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_length: int | str | None = None,
        content_type: str | None = "application/json",
    ) -> None:
        self.body = body
        self._offset = 0
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, limit: int = -1) -> bytes:
        if self._offset >= len(self.body):
            return b""
        if limit < 0:
            limit = len(self.body) - self._offset
        end = min(self._offset + limit, len(self.body))
        chunk = self.body[self._offset : end]
        self._offset = end
        return chunk


class RecordingOpener:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[urllib.request.Request, float]] = []

    def open(self, request: urllib.request.Request, *, timeout: float):
        self.calls.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def call(
    opener: RecordingOpener,
    *,
    endpoint: str = "http://127.0.0.1:11434",
    route: str = "/v1/chat/completions",
    headers: dict[str, str] | None = None,
    timeout: float = 3.5,
    response_limit: int = 1024,
):
    return request_json(
        endpoint,
        route,
        {"model": "test-model"},
        headers=headers,
        timeout=timeout,
        response_limit=response_limit,
        opener=opener,
    )


@pytest.mark.parametrize(
    ("endpoint", "expected_url"),
    [
        ("http://example.test", "http://example.test/v1/chat/completions"),
        ("https://example.test/", "https://example.test/v1/chat/completions"),
        ("http://localhost:11434", "http://localhost:11434/v1/chat/completions"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://[::1]:11434", "http://[::1]:11434/v1/chat/completions"),
        ("https://example.test/api", "https://example.test/api/v1/chat/completions"),
        ("https://example.test/api/", "https://example.test/api/v1/chat/completions"),
    ],
)
def test_request_json_accepts_supported_endpoints_and_preserves_path_prefix(
    endpoint: str, expected_url: str
):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}'))

    assert call(opener, endpoint=endpoint) == {"ok": True}
    request, timeout = opener.calls[0]
    assert request.full_url == expected_url
    assert timeout == 3.5


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "not a url",
        "http:///missing-host",
        "file:///tmp/provider.sock",
        "http://user:pass@example.test",
        "http://example.test?mode=test",
        "http://example.test/#fragment",
        "http://example.test:bad-port",
        "http://example.test:0",
        "http://example.test\\evil",
        " http://example.test",
    ],
)
def test_invalid_endpoints_fail_before_network_access(endpoint: str):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}'))

    with pytest.raises(LayerBHTTPError, match="endpoint") as raised:
        call(opener, endpoint=endpoint, headers={"Authorization": f"Bearer {FAKE_SECRET}"})

    assert opener.calls == []
    assert FAKE_SECRET not in str(raised.value)


@pytest.mark.parametrize(
    "route",
    [
        "",
        "relative",
        "//other.test/path",
        "/x?query=1",
        "/x#part",
        "/x\\y",
        "/x\ny",
        "/x\x7fy",
        "/x/../y",
        "/./x",
    ],
)
def test_invalid_provider_routes_fail_before_network_access(route: str):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}'))

    with pytest.raises(LayerBHTTPError, match="route"):
        call(opener, route=route)

    assert opener.calls == []


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_timeouts_fail_before_network_access(timeout: float):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}'))

    with pytest.raises(LayerBHTTPError, match="timeout"):
        call(opener, timeout=timeout)

    assert opener.calls == []


@pytest.mark.parametrize("response_limit", [True, -1, 1.5, sys.maxsize, 10**100])
def test_invalid_response_limits_fail_before_network_access(response_limit):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}'))

    with pytest.raises(LayerBHTTPError, match="response limit"):
        call(opener, response_limit=response_limit)

    assert opener.calls == []


def test_request_json_encodes_payload_and_supplied_headers():
    opener = RecordingOpener(FakeResponse(b'{"ok":true}'))

    call(opener, headers={"Authorization": f"Bearer {FAKE_SECRET}"})

    request, _timeout = opener.calls[0]
    assert request.data == b'{"model": "test-model"}'
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") == f"Bearer {FAKE_SECRET}"


@pytest.mark.parametrize("content_length", [None, 11])
def test_response_allows_missing_or_accurate_content_length(content_length: int | None):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}', content_length=content_length))

    assert call(opener) == {"ok": True}


@pytest.mark.parametrize(
    ("body", "content_length", "match"),
    [
        (b'{"ok":true}', 4, "Content-Length"),
        (b'{"ok":true}', 50, "Content-Length"),
        (b'{"ok":true}', 2048, "safety limit"),
        (b"x" * 1025, None, "safety limit"),
        (b"", None, "valid UTF-8 JSON"),
        (b'{"ok":', None, "valid UTF-8 JSON"),
        (b'{"ok":NaN}', None, "valid UTF-8 JSON"),
        (b"\xff", None, "valid UTF-8 JSON"),
        (b"[]", None, "JSON object"),
    ],
)
def test_response_body_contract(body: bytes, content_length: int | None, match: str):
    opener = RecordingOpener(FakeResponse(body, content_length=content_length))

    with pytest.raises(LayerBHTTPError, match=match):
        call(opener)


def test_response_rejects_invalid_content_length():
    opener = RecordingOpener(FakeResponse(b'{"ok":true}', content_length="eleven"))

    with pytest.raises(LayerBHTTPError, match="Content-Length"):
        call(opener)


def test_response_rejects_conflicting_content_length_values():
    response = FakeResponse(b'{"ok":true}', content_length=11)
    response.headers["Content-Length"] = "12"
    opener = RecordingOpener(response)

    with pytest.raises(LayerBHTTPError, match="conflicting Content-Length"):
        call(opener)


def test_response_rejects_huge_numeric_content_length_without_integer_conversion():
    opener = RecordingOpener(FakeResponse(b'{"ok":true}', content_length="9" * 5000))

    with pytest.raises(LayerBHTTPError, match="safety limit"):
        call(opener)


def test_response_rejects_present_non_json_content_type():
    opener = RecordingOpener(FakeResponse(b'{"ok":true}', content_type="text/html"))

    with pytest.raises(LayerBHTTPError, match="content type"):
        call(opener)


def test_response_body_reads_until_eof_after_short_reads():
    class ShortReadResponse(FakeResponse):
        def __init__(self, body: bytes) -> None:
            super().__init__(body)
            self.offset = 0

        def read(self, limit: int = -1) -> bytes:
            if self.offset >= len(self.body):
                return b""
            width = 2 if limit < 0 else min(2, limit)
            chunk = self.body[self.offset : self.offset + width]
            self.offset += len(chunk)
            return chunk

    opener = RecordingOpener(ShortReadResponse(b'{"ok":true}'))

    assert call(opener) == {"ok": True}


@pytest.mark.parametrize(
    "content_type",
    ["application/json; charset=utf-8", "application/problem+json", "APPLICATION/JSON"],
)
def test_response_allows_json_suffix_and_parameterized_content_types(content_type: str):
    opener = RecordingOpener(FakeResponse(b'{"ok":true}', content_type=content_type))

    assert call(opener) == {"ok": True}


def test_response_allows_missing_content_type_for_compatible_local_servers():
    opener = RecordingOpener(FakeResponse(b'{"ok":true}', content_type=None))

    assert call(opener) == {"ok": True}


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (urllib.error.URLError("connection refused"), "connection failed"),
        (urllib.error.URLError(TimeoutError("timed out")), "timed out"),
        (TimeoutError("timed out"), "timed out"),
        (
            urllib.error.HTTPError(
                "https://example.test/secret",
                503,
                "upstream failed",
                Message(),
                io.BytesIO(b"x" * 4096),
            ),
            "HTTP 503",
        ),
    ],
)
def test_expected_transport_failures_are_safe_and_bounded(failure: Exception, match: str):
    opener = RecordingOpener(failure)

    with pytest.raises(LayerBHTTPError, match=match) as raised:
        call(opener, headers={"Authorization": f"Bearer {FAKE_SECRET}"})

    message = str(raised.value)
    assert FAKE_SECRET not in message
    assert "example.test/secret" not in message
    assert "x" * 100 not in message


def test_unexpected_programming_errors_propagate():
    failure = AssertionError("programming defect")
    opener = RecordingOpener(failure)

    with pytest.raises(AssertionError, match="programming defect"):
        call(opener)


@pytest.mark.parametrize(
    ("code", "expected_method", "expected_data"),
    [
        (301, "GET", None),
        (302, "GET", None),
        (303, "GET", None),
        (307, "POST", b"{}"),
        (308, "POST", b"{}"),
    ],
)
def test_redirect_handler_closes_body_without_reading_before_following(
    code, expected_method, expected_data
):
    class RedirectBody:
        closed = False

        def read(self, *_args):
            raise AssertionError("redirect body must not be read")

        def close(self):
            self.closed = True

    class Parent:
        def __init__(self):
            self.calls = []

        def open(self, request, *, timeout):
            self.calls.append((request, timeout))
            return "target response"

    handler = layer_b_http.SameOriginRedirectHandler()
    parent = Parent()
    handler.add_parent(parent)
    original = urllib.request.Request(
        "https://example.test/api/start",
        data=b"{}",
        method="POST",
    )
    original.timeout = 3.5
    headers = Message()
    headers["Location"] = "/api/next"
    body = RedirectBody()

    handle_redirect = getattr(handler, f"http_error_{code}")
    assert handle_redirect(original, body, code, "redirect", headers) == "target response"
    assert body.closed
    redirected, timeout = parent.calls[0]
    assert redirected.full_url == "https://example.test/api/next"
    assert redirected.get_method() == expected_method
    assert redirected.data == expected_data
    assert timeout == 3.5


def test_redirect_handler_allows_only_same_origin_and_forwards_credentials():
    handler = layer_b_http.SameOriginRedirectHandler()
    original = urllib.request.Request(
        "https://example.test/api/start",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {FAKE_SECRET}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    redirected = handler.redirect_request(
        original,
        None,
        307,
        "redirect",
        Message(),
        "https://example.test/api/next",
    )
    assert redirected is not None
    assert redirected.full_url == "https://example.test/api/next"
    assert redirected.get_method() == "POST"
    assert redirected.data == b"{}"
    assert redirected.get_header("Authorization") == f"Bearer {FAKE_SECRET}"

    with pytest.raises(LayerBHTTPError, match="cross-origin redirect") as raised:
        handler.redirect_request(
            original,
            None,
            307,
            "redirect",
            Message(),
            "https://other.test/api/next",
        )
    assert FAKE_SECRET not in str(raised.value)


def test_redirect_handler_allows_same_origin_query_string():
    handler = layer_b_http.SameOriginRedirectHandler()
    original = urllib.request.Request(
        "https://example.test/api/start?cursor=before",
        data=b"{}",
        method="POST",
    )

    redirected = handler.redirect_request(
        original,
        None,
        307,
        "redirect",
        Message(),
        "https://example.test/api/next?token=abc",
    )

    assert redirected is not None
    assert redirected.full_url == "https://example.test/api/next?token=abc"
