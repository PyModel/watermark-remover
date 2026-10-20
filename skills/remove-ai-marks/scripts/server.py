#!/usr/bin/env python3
"""HTTP service exposing the watermark-remover cleaning pipeline.

Stdlib-only. The agent skill and any web app can call it over HTTP instead of
running the CLI scripts locally.

Endpoints:
    GET  /               -> the web UI (built from ../web; static files skip auth)
    GET  /health         -> {"ok": true, "version": ...}
    GET  /capabilities   -> which optional tools / pixel backends are present
    GET  /openapi.json   -> dynamically generated OpenAPI 3.0.3 spec
    POST /inspect        -> {"file": <base64>, "name": "x.png"} -> findings JSON
    POST /detect         -> {"file": <base64>, "name": "x.txt"} -> watermark detector reports
    POST /clean          -> {"file": <base64>, "name": "x.png", "options": {...}}
                         -> {"cleaned": <base64>, "report": {...}}
    POST /rewrite        -> {"file": <base64>, "name": "x.txt", "strength": "paraphrase"}
                         -> Layer B rewrite with the model configured in the
                            server's environment (never chosen by the caller)
    POST /visible        -> {"file": <base64>, "name": "x.png", "box": [x, y, w, h]}
                         -> visible-mark inpainting inside the box, PNG out

Hardening mirrors the CLIs: input size caps, binary-as-text guard, atomic
writes, loopback-only bind by default, optional bearer API key. Every
deployment also gets a Host allowlist (DNS rebinding), a same-origin check on
POST (CSRF), a socket timeout and a cap on concurrent jobs.
WATERMARKS_SERVER_PUBLIC=1 adds a per-client rate limit and turns off the
endpoints that spend the operator's model or vendor keys (rewrite, detect).
Run it as an unprivileged user (the Docker image does). Intended for a trusted network;
expose through a reverse proxy if reachable from untrusted clients.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import os
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_kind import classify_bytes
from clean_pipeline import execute
from clean_request import CleanRequest
from common import (
    MAX_INPUT_BYTES,
    eprint,
    looks_binary,
    read_flag_env,
    which,
)
from identify import identify_file, unknown_verdict
from inspect_file import inspect_asset
from layer_b_discovery import layer_b_status
from layer_b_http import LayerBHTTPError
from morphomod import VisiblePlan, remove_visible
from optional_deps import extras_status
from pipeline_actions import action_sentences
from rewrite_text import REWRITE_STRENGTHS, RewriteConfigurationError, RewritePlan, rewrite
from score_stylometry import score_text_stylometry
from text_detectors import detector_status, run_all_text_detectors, run_text_detectors


def run_synthid_score(path: Path) -> dict[str, Any] | None:
    return import_module("image_meta").run_synthid_score(path)


VERSION = os.environ.get("WATERMARKS_SERVER_VERSION", "dev")

# Optional bearer token: when set, every request must send
# `Authorization: Bearer <key>`. Empty means no auth (default).
API_KEY = os.environ.get("WATERMARKS_SERVER_API_KEY", "").strip()

# Deployment policy, filled from the environment by main(); tests set them directly.
#: Public mode: rate limit per client and no rewrite/detect, which would spend the
#: operator's model and vendor API keys on anyone's behalf.
PUBLIC = False
#: Host header values this server answers to (lower-case, no port). Empty = any.
#: main() pins loopback binds to loopback names, which defeats DNS rebinding.
ALLOWED_HOSTS: frozenset[str] = frozenset()
#: Read the client address from the last X-Forwarded-For hop (set behind a proxy).
TRUST_PROXY = False
#: Features refused with 403 in public mode.
PUBLIC_DISABLED = ("rewrite", "detect")
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
# A slow client can't hold a worker thread forever.
SOCKET_TIMEOUT_SECONDS = 60

# Body cap for the JSON envelope. Base64 inflates by 4/3, so the decoded file
# stays well under MAX_INPUT_BYTES for the same cap.
MAX_BODY_BYTES = MAX_INPUT_BYTES + (MAX_INPUT_BYTES >> 1)

TEXT_DETECTION_STATUSES = ("DETECTED", "INCONCLUSIVE", "NOT_DETECTED", "NOT_RUN")
_UNRESOLVED_TEXT_VERDICTS = frozenset({"INCONCLUSIVE", "UNSUPPORTED", "ERROR"})

# The web UI, built by `bun run build` in ../web. Only the flat files the build
# wrote are served, and they are public: the API calls they make still need the
# bearer key.
WEB_ROOT = Path(__file__).resolve().parent.parent / "web" / "dist"
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}
# The page talks to this origin only: no fonts, CDNs or analytics. blob: is the
# image preview and the cleaned-file download, both built in the browser.
_UI_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class RateLimiter:
    """Token bucket per client: *per_minute* requests, refilled continuously."""

    # ponytail: in-memory and per-process; move to Redis if the service runs more than one replica.
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        rate = self.per_minute / 60.0
        with self._lock:
            tokens, stamp = self._buckets.get(client, (float(self.per_minute), now))
            tokens = min(float(self.per_minute), tokens + (now - stamp) * rate)
            allowed = tokens >= 1.0
            self._buckets[client] = (tokens - 1.0 if allowed else tokens, now)
            if len(self._buckets) > 10_000:
                # Full buckets carry no state worth keeping; drop them.
                full = [
                    k
                    for k, (t, ts) in self._buckets.items()
                    if t + (now - ts) * rate >= self.per_minute
                ]
                for key in full:
                    del self._buckets[key]
            return allowed


RATE_LIMITER = RateLimiter(0)
#: Concurrent POST jobs; past this the server answers 503 instead of queueing threads.
JOBS = threading.BoundedSemaphore(int(os.environ.get("WATERMARKS_SERVER_MAX_JOBS", "4")))


def host_name(header: str | None) -> str:
    """Host header -> lower-case name without port; handles [::1]:8765."""
    value = (header or "").strip().lower()
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def static_file(path: str) -> Path | None:
    """Map a GET path to a built UI file, or None. Never leaves WEB_ROOT."""
    name = "index.html" if path == "/" else path[1:]
    # ":" also blocks Windows drive-relative names: WEB_ROOT / "C:x.js" is C:x.js.
    if not name or any(c in name for c in "/\\:") or name.startswith("."):
        return None
    if Path(name).suffix not in _STATIC_TYPES:
        return None
    candidate = WEB_ROOT / name
    if candidate.parent != WEB_ROOT:
        return None
    # codeql[py/path-injection] suppress the name passed the charset/suffix/parent confinement above; only a file directly under WEB_ROOT can match
    return candidate if candidate.is_file() else None


CLEAN_OPTION_TRANSLATIONS = {
    "nfkc": ("nfkc", bool),
    "aggressive_homoglyphs": ("aggressive_homoglyphs", bool),
    "keep_non_ai_metadata": ("keep_non_ai_metadata", bool),
    "also_layer_a_text": ("also_layer_a_text", bool),
    "strip_all_metadata": ("keep_non_ai_metadata", bool),
    "detect_before": ("detect_before", bool),
    "detect_after": ("detect_after", bool),
    "remove_synthid": ("remove_synthid", bool),
    "wmct_marker": ("wmct_marker", bool),
    "remove_pixel": ("pixel_backend", str),
}


# Inpainting backends the page may pick. "external" and a detector command run
# shell templates, so they stay CLI-only.
WEB_VISIBLE_BACKENDS = ("texture", "simple")


class UpstreamError(Exception):
    """The configured Layer B model endpoint failed; answered as 502."""


def _only_keys(body: dict[str, Any], allowed: set[str]) -> None:
    """Refuse unknown body keys, so no caller can smuggle in an endpoint or model."""
    for key in body:
        if key not in allowed:
            raise ValueError(f"unknown field: {key}")


def _json_ok(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _text_detection_status(reports: list[dict[str, Any]]) -> str:
    """Aggregate available text-detector reports without treating unknown as clean."""
    unresolved = False
    not_detected = False
    for report in reports:
        if not isinstance(report, dict):
            continue
        if report.get("available") is not True:
            # A detector that was configured to run but reported unavailable
            # ran and failed (timeout, crash, bad output). That is unresolved
            # evidence, not "detection was never requested".
            if report.get("configured") is True:
                unresolved = True
            continue
        verdict = report.get("verdict")
        if report.get("is_watermarked") is True or verdict == "DETECTED":
            return "DETECTED"
        if verdict in _UNRESOLVED_TEXT_VERDICTS:
            unresolved = True
        elif verdict == "NOT_DETECTED" or report.get("is_watermarked") is False:
            not_detected = True
        else:
            # An available detector that produced no recognized decision did
            # run, but cannot support a clean result.
            unresolved = True
    if unresolved:
        return "INCONCLUSIVE"
    if not_detected:
        return "NOT_DETECTED"
    return "NOT_RUN"


def capabilities() -> dict[str, Any]:
    return {
        "version": VERSION,
        "public": PUBLIC,
        "disabled": list(PUBLIC_DISABLED) if PUBLIC else [],
        "max_input_bytes": MAX_INPUT_BYTES,
        "tools": {
            "c2patool": which("c2patool") is not None,
            "exiftool": which("exiftool") is not None,
            "qpdf": which("qpdf") is not None,
        },
        "pixel_backends": {
            "ctrlregen": bool(os.environ.get("NOAI_WATERMARK_DIR")),
            "diffusion": bool(os.environ.get("MARKDIFFUSION_DIR")),
        },
        "scorers": {
            "synthid": bool(os.environ.get("REVERSE_SYNTHID_DIR")),
            "synthid_http": bool(os.environ.get("WATERMARKS_SYNTHID_SCORER_URL")),
            "stylometry": True,
        },
        "text_detectors": detector_status(),
        "harnesses": {
            "markllm": bool(os.environ.get("MARKLLM_DIR")),
        },
        # Optional extras and Layer B configuration, so a front end can grey out
        # what is unavailable instead of failing a run to find out.  Layer B is
        # reported from configuration only: /capabilities is polled and must not
        # become an outbound request per call.
        "extras": extras_status(),
        "layer_b": layer_b_status(
            os.environ.get("WATERMARKS_REWRITE_BACKEND"),
            os.environ.get("WATERMARKS_REWRITE_BASE_URL"),
            # ``read_flag_env``, not ``read_bool_env``: /capabilities is a
            # report, and it must say what the rewrite path would enforce.
            # An unparseable opt-in denies there, so it reads as False here
            # rather than failing the whole capability report.
            allow_remote=read_flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE"),
        ),
    }


# OpenAPI generation. The spec is built from this single declarative table
# plus live runtime values (version, auth, allowed options), so it can never
# drift from the endpoints the handler actually serves. Served at /openapi.json.


def _schema(**props: Any) -> dict[str, Any]:
    return props


def _file_request(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "required": ["file"],
        "properties": {
            "file": {
                "type": "string",
                "description": "Base64-encoded file bytes",
                "example": "SGVsbG8gd29ybGQ=",
            },
            "name": {
                "type": "string",
                "description": "Original filename (extension drives format routing)",
                "example": "notes.md",
            },
        },
    }
    if extra:
        schema["properties"].update(extra["properties"])
        schema["required"] = schema["required"] + extra.get("required", [])
    return schema


def _clean_request_schema() -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key, (_field, kind) in CLEAN_OPTION_TRANSLATIONS.items():
        if kind is bool:
            options[key] = _schema(type="boolean")
        else:
            options[key] = _schema(type="string")
    return _file_request(
        {
            "properties": {
                "options": _schema(type="object", properties=options, additionalProperties=False)
            },
        }
    )


_OPENAPI_PATHS: dict[str, dict[str, Any]] = {
    "/health": {
        "get": {
            "summary": "Liveness and version",
            "responses": {
                "200": _schema(
                    type="object",
                    properties={"ok": _schema(type="boolean"), "version": _schema(type="string")},
                )
            },
        }
    },
    "/capabilities": {
        "get": {
            "summary": "Which optional tools and heavy backends are available",
            "responses": {
                "200": _schema(
                    type="object",
                    properties={
                        "ok": _schema(type="boolean"),
                        "version": _schema(type="string"),
                        "tools": _schema(
                            type="object",
                            properties={
                                k: _schema(type="boolean") for k in ("c2patool", "exiftool", "qpdf")
                            },
                        ),
                        "pixel_backends": _schema(
                            type="object",
                            properties={
                                k: _schema(type="boolean") for k in ("ctrlregen", "diffusion")
                            },
                        ),
                        "scorers": _schema(
                            type="object",
                            properties={
                                "synthid": _schema(type="boolean"),
                                "synthid_http": _schema(type="boolean"),
                                "stylometry": _schema(type="boolean"),
                            },
                        ),
                        "harnesses": _schema(
                            type="object", properties={"markllm": _schema(type="boolean")}
                        ),
                        "text_detectors": _schema(
                            type="object",
                            additionalProperties=_schema(type="boolean"),
                        ),
                    },
                )
            },
        }
    },
    "/openapi.json": {
        "get": {
            "summary": "This OpenAPI 3.0.3 document, generated dynamically",
            "responses": {
                "200": _schema(type="object", description="An OpenAPI 3.0.3 document"),
            },
        }
    },
    "/inspect": {
        "post": {
            "summary": "Inspect a file for AI provenance marks (text / image / container auto-routed)",
            "parameters": [
                {
                    "name": "identify",
                    "in": "query",
                    "required": False,
                    "schema": _schema(type="boolean"),
                    "description": (
                        "Also return one provenance verdict (tri-state AI-ness, "
                        "platform, source kind, integrity clashes, C2PA validation) "
                        "under ``identify``. Metadata only; classifiers are never run"
                    ),
                }
            ],
            "requestBody": _schema(
                required=True,
                content={
                    "application/json": _schema(
                        schema=_file_request(
                            {
                                "properties": {
                                    "detect": _schema(
                                        type="boolean",
                                        description=(
                                            "Also run configured text watermark detectors "
                                            "(opt-in; may call vendor APIs and send text "
                                            "to them)"
                                        ),
                                    )
                                },
                                "required": [],
                            }
                        )
                    )
                },
            ),
            "responses": {
                "200": _schema(
                    type="object",
                    properties={
                        "ok": _schema(type="boolean"),
                        "kind": _schema(
                            type="string", enum=["text", "image", "container", "unknown"]
                        ),
                        "suspicious": _schema(type="boolean"),
                        "detection_status": _schema(
                            type="string",
                            enum=list(TEXT_DETECTION_STATUSES),
                            description=(
                                "Aggregate text-watermark detector state. INCONCLUSIVE means "
                                "the service cannot rule out a watermark; it is not a confirmed "
                                "detection. A configured detector that ran and failed reports "
                                "INCONCLUSIVE, not NOT_RUN."
                            ),
                        ),
                        "report": _schema(type="object"),
                        "identify": _schema(
                            type="object",
                            description=(
                                "Present only when ?identify=true. The six-field "
                                "provenance verdict; is_ai_generated is true or null "
                                "(unknown) — never asserted clean"
                            ),
                        ),
                    },
                )
            },
        }
    },
    "/clean": {
        "post": {
            "summary": "Clean a file; returns the cleaned bytes and an actions/stats report",
            "requestBody": _schema(
                required=True,
                content={"application/json": _schema(schema=_clean_request_schema())},
            ),
            "responses": {
                "200": _schema(
                    type="object",
                    properties={
                        "ok": _schema(type="boolean"),
                        "kind": _schema(type="string", enum=["text", "image", "container"]),
                        "cleaned": _schema(
                            type="string", description="Base64-encoded cleaned file bytes"
                        ),
                        "report": _schema(type="object"),
                    },
                )
            },
        }
    },
    "/rewrite": {
        "post": {
            "summary": "Layer B rewrite of a text file with the server-configured model (best-effort)",
            "requestBody": _schema(
                required=True,
                content={
                    "application/json": _schema(
                        schema=_file_request(
                            {
                                "properties": {
                                    "strength": _schema(
                                        type="string", enum=list(REWRITE_STRENGTHS)
                                    ),
                                    "lang": _schema(
                                        type="string",
                                        description="Pivot language for backtranslate",
                                    ),
                                },
                                "required": ["strength"],
                            }
                        )
                    )
                },
            ),
            "responses": {
                "200": _schema(
                    type="object",
                    properties={
                        "ok": _schema(type="boolean"),
                        "kind": _schema(type="string", enum=["text"]),
                        "cleaned": _schema(type="string", description="Base64 rewritten text"),
                        "report": _schema(type="object"),
                    },
                )
            },
        }
    },
    "/visible": {
        "post": {
            "summary": "Inpaint a visible mark inside a box (PNG/JPEG in, PNG out; best-effort)",
            "requestBody": _schema(
                required=True,
                content={
                    "application/json": _schema(
                        schema=_file_request(
                            {
                                "properties": {
                                    "box": _schema(
                                        type="array",
                                        items=_schema(type="integer", minimum=0),
                                        minItems=4,
                                        maxItems=4,
                                        description="x, y, width, height in image pixels",
                                    ),
                                    "backend": _schema(
                                        type="string", enum=list(WEB_VISIBLE_BACKENDS)
                                    ),
                                },
                                "required": ["box"],
                            }
                        )
                    )
                },
            ),
            "responses": {
                "200": _schema(
                    type="object",
                    properties={
                        "ok": _schema(type="boolean"),
                        "kind": _schema(type="string", enum=["image"]),
                        "cleaned": _schema(type="string", description="Base64 PNG"),
                        "report": _schema(type="object"),
                    },
                )
            },
        }
    },
    "/detect": {
        "post": {
            "summary": "Run watermark detectors on a file (text: vendor/statistical; image: SynthID score, open DWT-DCT payload, pixel-evidence families, forensic-metadata dump)",
            "requestBody": _schema(
                required=True,
                content={"application/json": _schema(schema=_file_request())},
            ),
            "responses": {
                "200": _schema(
                    type="object",
                    properties={
                        "ok": _schema(type="boolean"),
                        "kind": _schema(
                            type="string", enum=["text", "image", "container", "unknown"]
                        ),
                        "detections": _schema(type="array", items=_schema(type="object")),
                        "report": _schema(type="object"),
                    },
                )
            },
        }
    },
}

_ERROR_SCHEMA = _schema(
    type="object",
    properties={"ok": _schema(type="boolean", enum=[False]), "error": _schema(type="string")},
)
_COMMON_ERRORS = {
    "400": {
        "description": "Bad request",
        "content": {"application/json": {"schema": _ERROR_SCHEMA}},
    },
    "401": {
        "description": "Missing/invalid bearer token",
        "content": {"application/json": {"schema": _ERROR_SCHEMA}},
    },
    "404": {"description": "Not found", "content": {"application/json": {"schema": _ERROR_SCHEMA}}},
    "502": {
        "description": "The configured rewrite model failed",
        "content": {"application/json": {"schema": _ERROR_SCHEMA}},
    },
    "413": {
        "description": "Request body too large",
        "content": {"application/json": {"schema": _ERROR_SCHEMA}},
    },
    "500": {
        "description": "Internal error",
        "content": {"application/json": {"schema": _ERROR_SCHEMA}},
    },
}


def openapi_spec() -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for path, ops in _OPENAPI_PATHS.items():
        for method, op in ops.items():
            responses = dict(_COMMON_ERRORS)
            for status, body in op["responses"].items():
                responses[status] = {
                    "description": "Success",
                    "content": {"application/json": {"schema": body}},
                }
            paths.setdefault(path, {})[method] = {
                "summary": op["summary"],
                "responses": responses,
                **((op.get("requestBody") and {"requestBody": op["requestBody"]}) or {}),
                **((op.get("parameters") and {"parameters": op["parameters"]}) or {}),
            }

    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {
            "title": "watermark-remover service",
            "version": VERSION,
            "description": "Strip multi-vendor AI provenance marks (Unicode, C2PA/EXIF/XMP, containers). "
            "Files are passed base64-encoded in JSON; cleaned bytes come back base64-encoded.",
        },
        "paths": paths,
    }
    if API_KEY:
        spec["components"] = {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer"},
            }
        }
        spec["security"] = [{"bearerAuth": []}]
    return spec


def _safe_name(name: str) -> str:
    """Reduce a client-supplied filename to a bare basename safe for temp use.

    CodeQL (uncontrolled data in path expression): a name like '../../x'
    would otherwise let the write below escape the request temp dir. Fold
    Windows separators too, and fall back to a neutral name for '.', '..' or
    empty results.
    """
    base = Path(name.replace("\\", "/")).name
    if base in ("", ".", ".."):
        return "input"
    return base


def _tmp_path(tmpdir: Path, *parts: str) -> Path:
    """Join *parts* under *tmpdir* and refuse anything that escapes it.

    Defense-in-depth for the CodeQL "uncontrolled data in path expression"
    findings: even if a caller slips a separator through, the write can never
    land outside the request temp dir.
    """
    path = tmpdir.joinpath(*parts)
    if path.parent != tmpdir:
        raise ValueError("unsafe filename")
    return path


def _decode_input(body: dict[str, Any]) -> tuple[bytes, str]:
    raw = body.get("file")
    if not isinstance(raw, str):
        raise ValueError("missing string field 'file' (base64-encoded bytes)")
    name = body.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("'name' must be a string")
    try:
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("'file' is not valid base64") from None
    return data, _safe_name(name or "")


# Payload keys the canonical executor adds to its payload for the CLI/TUI
# that the /clean response has never exposed: the executor's input/output
# paths, its exit_code, and the note naming text-only options a non-text
# asset skips. Dropping exactly these (and nothing else) keeps the
# endpoint's report byte-identical to the pre-plan shape.
_PAYLOAD_BOOKKEEPING = frozenset({"input", "output", "exit_code", "skipped_text_transforms"})
# Historical /clean report order. The pipeline payload carries the same fields
# in its own order; the endpoint keeps the order clients already pin.
_IMAGE_REPORT_ORDER = (
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
)
_CONTAINER_REPORT_ORDER = (
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
)


def _ordered_report(payload: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    stripped = {key: value for key, value in payload.items() if key not in _PAYLOAD_BOOKKEEPING}
    if "actions" not in stripped and "action_details" in stripped:
        stripped["actions"] = action_sentences(stripped["action_details"])
    ordered = {key: stripped[key] for key in order if key in stripped}
    ordered.update({key: value for key, value in stripped.items() if key not in ordered})
    return ordered


def _identify_query(target: str) -> bool:
    """The ``?identify=`` query flag of a request target (opt-in, truthy spellings)."""
    parsed = parse_qs(urlparse(target).query)
    return parsed.get("identify", [""])[0].strip().lower() in ("true", "1", "yes")


def _invisible_watermark_detection(path: Path) -> dict[str, Any]:
    """The open DWT-DCT detector entry for ``/detect`` images: fail-soft.

    Runs ``invisible_watermark.detect_invisible_watermark`` when the
    ``detect`` extra is installed and reports what it found through the
    identify signal vocabulary. Every failure mode is an explicit entry — an
    absent extra or a crashed decoder never produces a negative finding, and
    a clean-looking decode carries the "absence is never clean" note.
    """
    import invisible_watermark
    from identify import invisible_watermark_signal

    entry: dict[str, Any] = {"detector": "invisible_watermark"}
    try:
        if not invisible_watermark.is_available():
            entry["available"] = False
            entry["error"] = (
                "detect extra not installed (numpy/cv2/pywt); "
                "install watermark-remover[detect] to decode open DWT-DCT payloads"
            )
            return entry
        entry["available"] = True
        scheme = invisible_watermark.detect_invisible_watermark(path)
    except Exception as exc:
        entry["available"] = True
        entry["error"] = f"invisible-watermark detection failed: {exc!r}"
        return entry
    entry.update(invisible_watermark_signal(scheme))
    entry["found"] = scheme is not None
    return entry


def _pixel_evidence_detection(path: Path) -> dict[str, Any]:
    """The pixel-forensics evidence entry for ``/detect`` images: fail-soft.

    Collects the six statistic families through the ``pixel_evidence``
    collector — bounded by construction (the 2048px analysis cap bounds every
    array) — and reports the identify signal vocabulary plus the versioned
    payload. The image-identifying artifacts layer is never requested here: a
    ``/detect`` response carries aggregates, not image content. Every failure
    mode is an explicit entry — an absent numeric stack or a crashed collector
    never produces a negative finding, and nothing measured is never claimed.
    """
    import pixel_evidence
    from identify import pixel_evidence_signal

    entry: dict[str, Any] = {"detector": "pixel_evidence"}
    try:
        if not pixel_evidence.is_available():
            entry["available"] = False
            entry["error"] = (
                "pixel numeric stack not installed (numpy); install "
                "watermark-remover[pixel] to collect pixel evidence"
            )
            return entry
        entry["available"] = True
        evidence = pixel_evidence.extract_pixel_evidence(path)
    except Exception as exc:
        entry["error"] = f"pixel evidence collection failed: {exc!r}"
        return entry
    payload = evidence.to_dict()
    entry.update(pixel_evidence_signal(payload))
    entry["evidence"] = payload
    return entry


def _forensic_metadata_detection(path: Path) -> dict[str, Any]:
    """The raw forensic-metadata dump for ``/detect`` images: fail-soft.

    ``forensic_metadata.collect_forensic_metadata`` (04's collector) is
    consumed read-only: the entry carries the identify signal summary plus the
    versioned record verbatim — the dump is forwarded, not re-interpreted. A
    collection failure is an explicit error entry; it is never a negative
    finding and never a clean verdict.
    """
    import forensic_metadata
    from identify import forensic_metadata_signal

    entry: dict[str, Any] = {"detector": "forensic_metadata"}
    try:
        record = forensic_metadata.collect_forensic_metadata(path)
    except Exception as exc:
        entry["error"] = f"forensic metadata collection failed: {exc!r}"
        return entry
    entry.update(forensic_metadata_signal(record))
    entry["record"] = record
    return entry


class Handler(BaseHTTPRequestHandler):
    server_version = f"watermark-remover/{VERSION}"
    timeout = SOCKET_TIMEOUT_SECONDS

    def log_message(self, fmt: str, *args: object) -> None:
        eprint(f"{self.address_string()} - {fmt % args}")

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header.encode(), f"Bearer {API_KEY}".encode())

    def _host_allowed(self) -> bool:
        return not ALLOWED_HOSTS or host_name(self.headers.get("Host")) in ALLOWED_HOSTS

    def _same_origin(self) -> bool:
        """A browser POST from another site carries its Origin; refuse it (CSRF)."""
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # curl, scripts, same-origin GET-less clients
        return urlparse(origin).netloc.lower() == (self.headers.get("Host") or "").strip().lower()

    def _client(self) -> str:
        if TRUST_PROXY:
            forwarded = self.headers.get("X-Forwarded-For", "")
            hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
            if hops:
                return hops[-1]  # the hop the proxy saw; earlier ones are client-supplied
        return self.client_address[0]

    def _read_json(self) -> dict[str, Any] | None:
        raw = self.headers.get("Content-Length")
        if raw is None or not raw.isdigit():
            return None
        length = int(raw)
        if length > MAX_BODY_BYTES:
            return None
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(body, dict):
            return None
        return body

    def _respond(
        self, status: int, payload: dict[str, Any], retry_after: int | None = None
    ) -> None:
        data = _json_ok(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, file: Path) -> None:
        # codeql[py/path-injection] suppress file is the WEB_ROOT-confined answer of the static resolver; the raw request line never reaches read_bytes
        data = file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _STATIC_TYPES[file.suffix])
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Security-Policy", _UI_CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._respond(HTTPStatus.MISDIRECTED_REQUEST, {"ok": False, "error": "unknown host"})
            return
        path = urlparse(self.path).path
        file = static_file(path)
        if file is not None:
            self._send_static(file)
            return
        if path == "/":
            self._respond(
                HTTPStatus.NOT_FOUND,
                {
                    "ok": False,
                    # No absolute path: this answer needs no auth.
                    "error": "web UI not built: run `bun run build` in skills/remove-ai-marks/web",
                },
            )
            return
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return
        if path == "/health":
            self._respond(HTTPStatus.OK, {"ok": True, "version": VERSION})
        elif path == "/capabilities":
            self._respond(HTTPStatus.OK, {"ok": True, **capabilities()})
        elif path == "/openapi.json":
            self._respond(HTTPStatus.OK, openapi_spec())
        else:
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self._host_allowed():
            self._respond(HTTPStatus.MISDIRECTED_REQUEST, {"ok": False, "error": "unknown host"})
            return
        if not self._same_origin():
            self._respond(
                HTTPStatus.FORBIDDEN, {"ok": False, "error": "cross-origin request refused"}
            )
            return
        path = urlparse(self.path).path
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return
        if path not in ("/inspect", "/clean", "/detect", "/rewrite", "/visible"):
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if PUBLIC and path.lstrip("/") in PUBLIC_DISABLED:
            self._respond(
                HTTPStatus.FORBIDDEN,
                {"ok": False, "error": f"{path.lstrip('/')} is turned off on this server"},
            )
            return
        if not RATE_LIMITER.allow(self._client()):
            self._respond(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"ok": False, "error": "too many requests; wait a minute"},
                60,
            )
            return
        if not JOBS.acquire(blocking=False):
            self._respond(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": "server is busy; try again shortly"},
                5,
            )
            return
        try:
            self._post(path)
        finally:
            JOBS.release()

    def _post(self, path: str) -> None:
        body = self._read_json()
        if body is None:
            raw_len = self.headers.get("Content-Length")
            oversized = raw_len is not None and raw_len.isdigit() and int(raw_len) > MAX_BODY_BYTES
            self._respond(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE if oversized else HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "invalid request body"},
            )
            return
        try:
            data, name = _decode_input(body)
        except ValueError as e:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
            return
        try:
            if path == "/inspect":
                identify = _identify_query(self.path)
                self._handle_inspect(data, name, body, identify=identify)
            elif path == "/detect":
                self._handle_detect(data, name)
            elif path == "/rewrite":
                self._handle_rewrite(data, name, body)
            elif path == "/visible":
                self._handle_visible(data, name, body)
            else:
                self._handle_clean(data, name, body)
        except ValueError as e:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
        except UpstreamError as e:
            self._respond(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(e)})
        except Exception as e:
            eprint(f"error handling {path}: {e!r}")
            self._respond(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "internal error"}
            )

    def _handle_inspect(
        self, data: bytes, name: str, body: dict[str, Any], *, identify: bool = False
    ) -> None:
        kind = classify_bytes(data, Path(name).suffix)
        if kind == "unknown":
            response: dict[str, Any] = {
                "ok": True,
                "kind": "unknown",
                "report": {
                    "note": "unrecognized format; use a filename with a known extension",
                },
                "suspicious": False,
                "detection_status": "NOT_RUN",
            }
            if identify:
                response["identify"] = unknown_verdict().to_dict()
            self._respond(HTTPStatus.OK, response)
            return
        run_detect = body.get("detect") is True
        if run_detect and PUBLIC:
            raise ValueError("detectors are turned off on this server")
        verdict = None
        with tempfile.TemporaryDirectory(prefix="wm-inspect-") as tmp:
            path = _tmp_path(Path(tmp), name or "input")
            path.write_bytes(data)
            if kind == "text":
                if looks_binary(data):
                    raise ValueError(
                        "refusing to inspect bytes that look like a binary container as text"
                    )
                raw_text = data.decode("utf-8", errors="surrogateescape")
                inspected = inspect_asset(path, force_type=kind)
                report = {
                    key: value
                    for key, value in inspected.items()
                    if key not in {"kind", "path", "suspicious"}
                }
                report["stylometry"] = score_text_stylometry(
                    raw_text, path=name or "<text>"
                ).to_dict()
                if run_detect:
                    report["text_detectors"] = run_all_text_detectors(raw_text)
                if identify:
                    # Provenance identification is a metadata question about
                    # binary assets; a text asset carries none — unknown.
                    verdict = unknown_verdict()
            else:
                inspected = inspect_asset(path, force_type=kind)
                report = {
                    key: value
                    for key, value in inspected.items()
                    if key not in {"kind", "path", "suspicious"}
                }
                if identify:
                    verdict = identify_file(path)
        detection_status = _text_detection_status(report.get("text_detectors") or [])
        suspicious = (
            bool(report.get("suspicious_total"))
            or bool(report.get("has_c2pa") or report.get("has_ai_metadata"))
            or bool(report.get("stylometry", {}).get("score", 0.0) >= 0.65)
            or detection_status in {"DETECTED", "INCONCLUSIVE"}
        )
        response = {
            "ok": True,
            "kind": kind,
            "report": report,
            "suspicious": suspicious,
            "detection_status": detection_status,
        }
        if identify:
            # One provenance verdict from the already-collected evidence;
            # classifiers stay opt-in and are never triggered by the param.
            response["identify"] = (verdict or unknown_verdict()).to_dict()
        self._respond(HTTPStatus.OK, response)

    def _handle_detect(self, data: bytes, name: str) -> None:
        kind = classify_bytes(data, Path(name).suffix)
        if kind == "unknown":
            self._respond(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "kind": "unknown",
                    "detections": [],
                    "report": {
                        "note": "unrecognized format; use a filename with a known extension",
                    },
                },
            )
            return
        with tempfile.TemporaryDirectory(prefix="wm-detect-") as tmp:
            path = _tmp_path(Path(tmp), name or "input")
            path.write_bytes(data)
            if kind == "text":
                if looks_binary(data):
                    raise ValueError(
                        "refusing to detect bytes that look like a binary container as text"
                    )
                raw_text = data.decode("utf-8", errors="surrogateescape")
                detections: list[dict[str, Any]] = run_all_text_detectors(raw_text)
                s_rep = score_text_stylometry(raw_text, path=name or "<text>")
                detections.append({"detector": "stylometry", "available": True, **s_rep.to_dict()})
            elif kind == "image":
                score = run_synthid_score(path)
                if score is None:
                    score = {
                        "detector": "synthid",
                        "available": False,
                        "error": (
                            "no SynthID scorer configured (set "
                            "WATERMARKS_SYNTHID_SCORER_URL or REVERSE_SYNTHID_DIR)"
                        ),
                    }
                else:
                    score.setdefault("detector", "synthid")
                detections = [
                    score,
                    _invisible_watermark_detection(path),
                    _pixel_evidence_detection(path),
                    _forensic_metadata_detection(path),
                ]
            else:
                detections = []
                report = inspect_asset(path)
                self._respond(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "kind": kind,
                        "detections": detections,
                        "report": report,
                    },
                )
                return
        self._respond(HTTPStatus.OK, {"ok": True, "kind": kind, "detections": detections})

    def _handle_rewrite(self, data: bytes, name: str, body: dict[str, Any]) -> None:
        _only_keys(body, {"file", "name", "strength", "lang"})
        strength = body.get("strength")
        if strength not in REWRITE_STRENGTHS:
            raise ValueError(f"strength must be one of: {', '.join(REWRITE_STRENGTHS)}")
        lang = body.get("lang")
        if lang is not None and (
            not isinstance(lang, str)
            or not 0 < len(lang) <= 40
            or not lang.replace(" ", "").isalpha()
        ):
            raise ValueError("lang must be a language name such as French")
        if classify_bytes(data, Path(name).suffix) != "text" or looks_binary(data):
            raise ValueError("rewrite takes a text file")
        text = data.decode("utf-8", errors="surrogateescape")
        # Backend, model, endpoint and key come from the server's environment
        # only; the remote opt-in is enforced inside rewrite().
        try:
            plan = RewritePlan.live_from_environment(strength, lang=lang)
            out, info = rewrite(text, plan)
        except RewriteConfigurationError as e:
            raise ValueError(str(e)) from None
        except LayerBHTTPError as e:
            raise UpstreamError(f"rewrite model failed: {e}") from None
        info.pop("base_url", None)
        self._respond(
            HTTPStatus.OK,
            {
                "ok": True,
                "kind": "text",
                "cleaned": base64.b64encode(out.encode("utf-8", errors="surrogateescape")).decode(
                    "ascii"
                ),
                "report": {"kind": "text", "rewrite": info, "length": len(out)},
            },
        )

    def _handle_visible(self, data: bytes, name: str, body: dict[str, Any]) -> None:
        _only_keys(body, {"file", "name", "box", "backend"})
        box = body.get("box")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in box)
            or box[2] == 0
            or box[3] == 0
        ):
            raise ValueError("box must be four non-negative integers [x, y, width, height]")
        backend = body.get("backend", "texture")
        if backend not in WEB_VISIBLE_BACKENDS:
            raise ValueError(f"backend must be one of: {', '.join(WEB_VISIBLE_BACKENDS)}")
        if classify_bytes(data, Path(name).suffix) != "image":
            raise ValueError("visible-mark removal takes a PNG or JPEG image")
        # Input and output get separate temp dirs, so no upload name can alias the output.
        with (
            tempfile.TemporaryDirectory(prefix="wm-visible-") as tmp,
            tempfile.TemporaryDirectory(prefix="wm-visible-out-") as out_dir,
        ):
            src = _tmp_path(Path(tmp), name or "input")
            src.write_bytes(data)
            dest = Path(out_dir) / "cleaned.png"
            plan = VisiblePlan(
                box=(box[0], box[1], box[2], box[3]), backend=backend, publish_mask=False
            )
            result = remove_visible(src, dest, plan)
            # codeql[py/path-injection] suppress dest is a fixed-name file inside a server-owned request tempdir; no upload path is composed into it
            cleaned_bytes = dest.read_bytes()
        for key in ("input", "output", "mask"):
            result.pop(key, None)
        self._respond(
            HTTPStatus.OK,
            {
                "ok": True,
                "kind": "image",
                "cleaned": base64.b64encode(cleaned_bytes).decode("ascii"),
                "report": {"kind": "image", **result},
            },
        )

    def _handle_clean(self, data: bytes, name: str, body: dict[str, Any]) -> None:
        kind = classify_bytes(data, Path(name).suffix)
        if kind == "unknown":
            raise ValueError(
                "unrecognized file format; use a filename with a known extension "
                "(e.g. notes.txt) or a supported image/container name"
            )
        options = body.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object")
        if PUBLIC and (options.get("detect_before") or options.get("detect_after")):
            raise ValueError("detectors are turned off on this server")
        values: dict[str, Any] = {}
        for key, value in options.items():
            translation = CLEAN_OPTION_TRANSLATIONS.get(key)
            if translation is None:
                raise ValueError(f"unknown option: {key}")
            field, expected_type = translation
            if not isinstance(value, expected_type):
                type_name = "boolean" if expected_type is bool else "string"
                raise ValueError(f"option {key!r} must be a {type_name}")
            if key == "strip_all_metadata":
                values["keep_non_ai_metadata"] = not value
            elif field == "pixel_backend":
                values[field] = value
            else:
                values[field] = value
        if "strip_all_metadata" in options:
            # Historical contract: strip_all_metadata wins over keep_non_ai_metadata.
            values["keep_non_ai_metadata"] = not options["strip_all_metadata"]
        pixel = values.pop("pixel_backend", None)
        if pixel is not None:
            available = bool(capabilities()["pixel_backends"].get(pixel, False))
            self._respond(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "kind": kind,
                    "capability": {
                        "requested": pixel,
                        "supported": pixel in ("ctrlregen", "diffusion"),
                        "available": available,
                        "executed": False,
                        "reason": (
                            "pixel backend is unavailable"
                            if not available
                            else "pixel backend is not wired into the canonical pipeline"
                        ),
                    },
                },
            )
            return
        with tempfile.TemporaryDirectory(prefix="wm-clean-") as tmp:
            tmpdir = Path(tmp)
            src = _tmp_path(tmpdir, name or "input")
            src.write_bytes(data)
            # The cleaned output is staged in its own directory so a client
            # filename can never alias it, whatever the asset kind.
            outdir = Path(tempfile.mkdtemp(dir=tmpdir, prefix="out-"))
            if kind == "text":
                dest_name = "out.txt"
            elif kind == "image":
                dest_name = "out.png"
            else:
                dest_name = f"out{Path(name).suffix}"
            dest = outdir / dest_name
            if kind == "text" and looks_binary(data):
                raise ValueError(
                    "refusing to clean bytes that look like a binary container as text"
                )
            detector_reports: dict[str, Any] = {}
            if kind == "text" and values.get("detect_before"):
                detector_reports["before"] = run_text_detectors(
                    data.decode("utf-8", errors="surrogateescape")
                )
            request = CleanRequest(
                paths=(src,),
                output=dest,
                force_type=kind,
                **values,
                json=True,
            )
            payload = execute(request).payload
            if payload.get("error"):
                # Preflight refusals raise ValueError inside execute and stay
                # 400. An error payload is an execution failure: re-raise as
                # RuntimeError so the handler keeps the sanitized 500.
                raise RuntimeError(payload["error"])
            # codeql[py/path-injection] suppress dest is a fixed-name file inside the request-scoped outdir; the upload suffix cannot carry separators into that composition
            cleaned_bytes = dest.read_bytes()
            if kind == "text":
                if values.get("detect_after"):
                    detector_reports["after"] = run_text_detectors(
                        cleaned_bytes.decode("utf-8", errors="surrogateescape")
                    )
                result = {
                    "kind": "text",
                    "stats": payload["stats"],
                    "length": len(cleaned_bytes.decode("utf-8", errors="surrogateescape")),
                }
                if detector_reports:
                    result["text_detectors"] = detector_reports
            else:
                order = _IMAGE_REPORT_ORDER if kind == "image" else _CONTAINER_REPORT_ORDER
                result = _ordered_report(payload, order)
                if (
                    kind == "image"
                    and values.get("detect_before")
                    and result.get("synthid_before") is None
                ):
                    result["synthid_before"] = run_synthid_score(src)
                if (
                    kind == "image"
                    and values.get("detect_after")
                    and result.get("synthid_after") is None
                ):
                    result["synthid_after"] = run_synthid_score(dest)
        self._respond(
            HTTPStatus.OK,
            {
                "ok": True,
                "kind": kind,
                "cleaned": base64.b64encode(cleaned_bytes).decode("ascii"),
                "report": result,
            },
        )


def configure_policy(bind_host: str) -> None:
    """Set the deployment policy from the environment for a server bound to *bind_host*."""
    global PUBLIC, ALLOWED_HOSTS, RATE_LIMITER, TRUST_PROXY  # noqa: PLW0603
    PUBLIC = read_flag_env("WATERMARKS_SERVER_PUBLIC")
    TRUST_PROXY = PUBLIC or read_flag_env("WATERMARKS_SERVER_TRUST_PROXY")
    hosts = os.environ.get("WATERMARKS_SERVER_ALLOWED_HOSTS", "")
    ALLOWED_HOSTS = frozenset(h.strip().lower() for h in hosts.split(",") if h.strip())
    if not ALLOWED_HOSTS and bind_host in LOOPBACK_NAMES:
        ALLOWED_HOSTS = LOOPBACK_NAMES
    default_limit = "30" if PUBLIC else "0"
    RATE_LIMITER = RateLimiter(int(os.environ.get("WATERMARKS_SERVER_RATE_LIMIT", default_limit)))


def main() -> int:
    global API_KEY  # noqa: PLW0603
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("WATERMARKS_SERVER_HOST", "127.0.0.1"))
    # PORT is what Railway, Heroku and Cloud Run hand the process.
    default_port = os.environ.get("WATERMARKS_SERVER_PORT") or os.environ.get("PORT") or "8765"
    p.add_argument("--port", type=int, default=int(default_port))
    p.add_argument("--api-key", default=API_KEY, help="require this bearer token (default: none)")
    p.add_argument("-V", "--version", action="store_true", help="print version and exit")
    args = p.parse_args()

    if args.version:
        print(VERSION)
        return 0

    API_KEY = args.api_key
    configure_policy(args.host)
    if PUBLIC:
        eprint(
            f"public mode: {RATE_LIMITER.per_minute} requests/min per client, "
            f"{', '.join(PUBLIC_DISABLED)} off"
        )
        if not ALLOWED_HOSTS:
            eprint("warning: public mode without WATERMARKS_SERVER_ALLOWED_HOSTS answers any Host")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        eprint(f"warning: binding {args.host} — intended for a trusted network only")
    if API_KEY:
        eprint("API key required for requests")
    else:
        eprint("warning: no API key set — only bind to loopback or a trusted network")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    eprint(f"watermark-remover service {VERSION} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        eprint("shutting down")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
