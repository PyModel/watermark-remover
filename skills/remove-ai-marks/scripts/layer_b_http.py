"""Shared, bounded HTTP+JSON transport for live Layer B operations."""

from __future__ import annotations

import http.client
import json
import math
import re
import string
import sys
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import SplitResult, quote, urljoin, urlsplit, urlunsplit

from common import DEFAULT_HTTP_JSON_LIMIT

_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


class LayerBHTTPError(RuntimeError):
    """Expected endpoint, transport, or HTTP-response failure."""


def _parse_endpoint(endpoint: str, *, allow_query: bool = False) -> SplitResult:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint != endpoint.strip()
        or "\\" in endpoint
        or any(ord(char) < 32 or ord(char) == 127 for char in endpoint)
    ):
        raise LayerBHTTPError("invalid Layer B HTTP endpoint")
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise LayerBHTTPError("invalid Layer B HTTP endpoint") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or (port is None and ":" in parsed.netloc and parsed.netloc.endswith(":"))
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.query and not allow_query)
        or parsed.fragment
    ):
        raise LayerBHTTPError("invalid Layer B HTTP endpoint")
    return parsed


def _parse_route(route: str) -> str:
    if (
        not isinstance(route, str)
        or not route.startswith("/")
        or route.startswith("//")
        or "?" in route
        or "#" in route
        or "\\" in route
        or any(segment in {".", ".."} for segment in route.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in route)
    ):
        raise LayerBHTTPError("invalid Layer B HTTP route")
    return route


def _join_route(endpoint: str, route: str) -> str:
    parsed = _parse_endpoint(endpoint)
    route = _parse_route(route)
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/{route.lstrip('/')}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise LayerBHTTPError("Layer B HTTP timeout must be a finite positive number")
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise LayerBHTTPError("Layer B HTTP timeout must be a finite positive number")
    return value


def _validate_response_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 or limit >= sys.maxsize:
        raise LayerBHTTPError("invalid Layer B HTTP response limit")
    return limit


def _validate_headers(headers: dict[str, str] | None) -> dict[str, str]:
    validated = {"Content-Type": "application/json"}
    for name, value in (headers or {}).items():
        if (
            not isinstance(name, str)
            or not _HEADER_NAME_RE.fullmatch(name)
            or not isinstance(value, str)
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or name.lower() in {"content-length", "content-type", "host"}
        ):
            raise LayerBHTTPError("invalid Layer B HTTP headers")
        validated[name] = value
    return validated


def _origin(parsed: SplitResult) -> tuple[str, str, int]:
    hostname = parsed.hostname
    if hostname is None:
        raise LayerBHTTPError("invalid Layer B HTTP endpoint")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, hostname.lower(), parsed.port if parsed.port is not None else default_port


class SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve JSON POSTs only across same-origin redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code not in _REDIRECT_CODES:
            return None
        original = _parse_endpoint(req.full_url, allow_query=True)
        target = _parse_endpoint(newurl, allow_query=True)
        if _origin(original) != _origin(target):
            raise LayerBHTTPError("Layer B HTTP cross-origin redirect refused")
        forwarded = {**req.headers, **req.unredirected_hdrs}
        method = req.get_method()
        data = req.data
        if code in {301, 302, 303} and method == "POST":
            # RFC 9110 follows these redirects with a bodyless GET; 307/308
            # preserve the original method and body.
            method = "GET"
            data = None
            forwarded = {
                name: value
                for name, value in forwarded.items()
                if name.lower() not in {"content-length", "content-type"}
            }
        return urllib.request.Request(  # noqa: S310
            newurl,
            data=data,
            headers=forwarded,
            method=method,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
        )

    def http_error_302(self, req, fp, code, msg, headers):
        location = headers.get("Location") or headers.get("URI")
        if not isinstance(location, str) or not location:
            return None
        try:
            try:
                encoded = quote(location, encoding="iso-8859-1", safe=string.punctuation)
                newurl = urljoin(req.full_url, encoded)
            except (UnicodeError, ValueError) as error:
                raise LayerBHTTPError("invalid Layer B HTTP redirect") from error

            redirected = self.redirect_request(req, fp, code, msg, headers, newurl)
            if redirected is None:
                return None
            if hasattr(req, "redirect_dict"):
                visited = redirected.redirect_dict = req.redirect_dict
                if (
                    visited.get(newurl, 0) >= self.max_repeats
                    or len(visited) >= self.max_redirections
                ):
                    raise urllib.error.HTTPError(
                        req.full_url,
                        code,
                        self.inf_msg + msg,
                        headers,
                        fp,
                    )
            else:
                visited = redirected.redirect_dict = req.redirect_dict = {}
            visited[newurl] = visited.get(newurl, 0) + 1
        finally:
            fp.close()

        return self.parent.open(redirected, timeout=req.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


_OPENER = urllib.request.build_opener(SameOriginRedirectHandler())


def _header_values(headers: Any, name: str) -> list[str]:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        raw = get_all(name)
        values = raw if isinstance(raw, (list, tuple)) else []
    else:
        value = headers.get(name) if hasattr(headers, "get") else None
        values = [] if value is None else [value]
    return [value for value in values if isinstance(value, str)]


def _reject_json_constant(_value: str) -> Any:
    raise ValueError


def _read_json_object(response: Any, limit: int) -> dict[str, Any]:
    content_lengths = _header_values(response.headers, "Content-Length")
    if content_lengths:
        normalized = {value.strip() for value in content_lengths}
        if len(normalized) != 1:
            raise LayerBHTTPError("Layer B HTTP response has conflicting Content-Length values")
        declared_text = normalized.pop()
        if not declared_text or any(char < "0" or char > "9" for char in declared_text):
            raise LayerBHTTPError("Layer B HTTP response has invalid Content-Length")
        digits = declared_text.lstrip("0") or "0"
        limit_text = str(limit)
        if len(digits) > len(limit_text) or (
            len(digits) == len(limit_text) and digits > limit_text
        ):
            raise LayerBHTTPError(f"Layer B HTTP response exceeds safety limit of {limit:,} bytes")
        declared = int(digits)
    else:
        declared = None

    content_types = _header_values(response.headers, "Content-Type")
    if content_types:
        media_types = {value.split(";", 1)[0].strip().lower() for value in content_types}
        if any(
            value != "application/json" and not value.endswith("+json") for value in media_types
        ):
            raise LayerBHTTPError("Layer B HTTP response has unexpected content type")

    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        chunk = response.read(remaining)
        if not isinstance(chunk, bytes):
            raise LayerBHTTPError("Layer B HTTP response body must be bytes")
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > limit:
        raise LayerBHTTPError(f"Layer B HTTP response exceeds safety limit of {limit:,} bytes")
    if declared is not None and declared != len(raw):
        raise LayerBHTTPError("Layer B HTTP response Content-Length does not match its body")
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise LayerBHTTPError("Layer B HTTP response is not valid UTF-8 JSON") from error
    if not isinstance(data, dict):
        raise LayerBHTTPError("Layer B HTTP response must be a JSON object")
    return data


def request_json(
    endpoint: str,
    route: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float,
    response_limit: int = DEFAULT_HTTP_JSON_LIMIT,
    opener: Any = None,
) -> dict[str, Any]:
    """POST JSON and return one bounded JSON object.

    Routes are provider-owned absolute paths appended beneath any configured
    endpoint path prefix. Expected network and response failures are normalized
    without embedding endpoint URLs, headers, or response bodies in diagnostics.
    """
    url = _join_route(endpoint, route)
    timeout = _validate_timeout(timeout)
    response_limit = _validate_response_limit(response_limit)
    request_headers = _validate_headers(headers)
    body = json.dumps(payload).encode("utf-8")
    try:
        request = urllib.request.Request(  # noqa: S310
            url,
            data=body,
            headers=request_headers,
            method="POST",
        )
    except (TypeError, ValueError) as error:
        raise LayerBHTTPError("invalid Layer B HTTP request") from error

    client = _OPENER if opener is None else opener
    try:
        with client.open(request, timeout=timeout) as response:
            return _read_json_object(response, response_limit)
    except LayerBHTTPError:
        raise
    except urllib.error.HTTPError as error:
        error.close()
        raise LayerBHTTPError(f"Layer B HTTP request failed with HTTP {error.code}") from None
    except TimeoutError:
        raise LayerBHTTPError("Layer B HTTP request timed out") from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise LayerBHTTPError("Layer B HTTP request timed out") from None
        raise LayerBHTTPError("Layer B HTTP connection failed") from None
    except (http.client.HTTPException, OSError):
        raise LayerBHTTPError("Layer B HTTP connection failed") from None
