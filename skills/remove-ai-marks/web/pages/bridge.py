"""wm-serve without a socket, for the Pyodide build.

The page sends the same requests it sends to wm-serve; this file feeds them to
``server.Handler`` so validation, limits and answers stay byte-identical. Only
``_respond`` (no socket to write to) and Layer B's HTTP transport (Pyodide has
no sockets; a worker may use synchronous XHR) are replaced.
"""

from __future__ import annotations

import email.message
import io
import json
import os
import urllib.error
import urllib.request
import urllib.response

import layer_b_http
import server

# ponytail: the file, its base64 JSON copy and the wasm heap all live in one tab; 100 MB is wm-serve's
# limit for a machine with swap. Raise once memory64 Pyodide makes the heap ceiling >4 GB.
server.MAX_INPUT_BYTES = 40 * 1024 * 1024
server.MAX_BODY_BYTES = server.MAX_INPUT_BYTES + (server.MAX_INPUT_BYTES >> 1)

REWRITE_ENV = {
    "backend": "WATERMARKS_REWRITE_BACKEND",
    "model": "WATERMARKS_REWRITE_MODEL",
    "base_url": "WATERMARKS_REWRITE_BASE_URL",
    "api_key": "WATERMARKS_REWRITE_API_KEY",
}


class _Call(server.Handler):
    def __init__(self, path: str, body: bytes | None) -> None:  # no super(): there is no socket
        self.path = path
        self.headers = {} if body is None else {"Content-Length": str(len(body))}
        self.rfile = io.BytesIO(body or b"")
        self.client_address = ("browser", 0)
        self.answer: tuple[int, str] = (500, '{"ok": false, "error": "no answer"}')

    def _respond(self, status, payload, retry_after=None) -> None:
        self.answer = (int(status), server._json_ok(payload).decode("utf-8"))


def handle(method: str, path: str, body: str | None = None) -> list:
    # JS null arrives as a JsNull proxy, not None.
    call = _Call(path, body.encode("utf-8") if isinstance(body, str) else None)
    (call.do_POST if method == "POST" else call.do_GET)()
    return list(call.answer)


def configure(settings: str) -> None:
    """Rewrite settings from the page (the visitor's own model server) -> the env Layer B reads."""
    values = json.loads(settings)
    for key, name in REWRITE_ENV.items():
        value = str(values.get(key) or "").strip()
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)


class XHRHandler(urllib.request.HTTPHandler):
    """urllib -> synchronous XMLHttpRequest. Failures surface as URLError, so Layer B's
    existing error mapping (connection failed / timed out / HTTP n) is unchanged."""

    handler_order = 100

    def http_open(self, req: urllib.request.Request):
        import js  # Pyodide only
        from pyodide.ffi import JsException, to_js

        xhr = js.XMLHttpRequest.new()
        xhr.open(req.get_method(), req.full_url, False)
        xhr.responseType = "arraybuffer"
        if req.timeout:
            xhr.timeout = int(req.timeout * 1000)
        for name, value in req.header_items():
            if name.lower() not in {"host", "content-length", "connection", "user-agent"}:
                xhr.setRequestHeader(name, value)
        try:
            xhr.send(None if req.data is None else to_js(bytes(req.data)))
        except JsException as error:
            reason = TimeoutError() if "timeout" in str(error).lower() else str(error)
            raise urllib.error.URLError(reason) from None
        if xhr.status == 0:
            raise urllib.error.URLError("blocked or unreachable")
        headers = email.message.Message()
        for line in str(xhr.getAllResponseHeaders()).splitlines():
            name, _, value = line.partition(":")
            if name.strip():
                headers[name.strip()] = value.strip()
        body = js.Uint8Array.new(xhr.response).to_py().tobytes()
        response = urllib.response.addinfourl(io.BytesIO(body), headers, req.full_url, xhr.status)
        response.msg = str(xhr.statusText)
        return response

    https_open = http_open


layer_b_http._OPENER = urllib.request.build_opener(
    layer_b_http.SameOriginRedirectHandler(), XHRHandler()
)
