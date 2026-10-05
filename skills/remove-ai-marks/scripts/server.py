#!/usr/bin/env python3
"""HTTP service exposing the watermark-remover cleaning pipeline.

Stdlib-only. The agent skill and any web app can call it over HTTP instead of
running the CLI scripts locally.

Endpoints:
    GET  /health         -> {"ok": true, "version": ...}
    GET  /capabilities   -> which optional tools / pixel backends are present
    GET  /openapi.json   -> dynamically generated OpenAPI 3.0.3 spec
    POST /inspect        -> {"file": <base64>, "name": "x.png"} -> findings JSON
    POST /detect         -> {"file": <base64>, "name": "x.txt"} -> watermark detector reports
    POST /clean          -> {"file": <base64>, "name": "x.png", "options": {...}}
                         -> {"cleaned": <base64>, "report": {...}}

Hardening mirrors the CLIs: input size caps, binary-as-text guard, atomic
writes, loopback-only bind by default, optional bearer API key. Run it as an
unprivileged user (the Docker image does). Intended for a trusted network;
expose through a reverse proxy if reachable from untrusted clients.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_kind import classify_bytes
from batch_inputs import InputItem
from clean_file import run_clean_item
from clean_request import CleanRequest, plan_work
from common import (
    MAX_INPUT_BYTES,
    eprint,
    looks_binary,
    read_flag_env,
    which,
)
from container_meta import clean_container, inspect_container
from identify import identify_file, unknown_verdict
from image_meta import inspect_image, run_synthid_score
from layer_b_discovery import layer_b_status
from optional_deps import extras_status
from score_stylometry import score_text_stylometry
from text_detectors import detector_status, run_all_text_detectors, run_text_detectors
from text_unicode import inspect_text

VERSION = os.environ.get("WATERMARKS_SERVER_VERSION", "dev")

# Optional bearer token: when set, every request must send
# `Authorization: Bearer <key>`. Empty means no auth (default).
API_KEY = os.environ.get("WATERMARKS_SERVER_API_KEY", "").strip()

# Body cap for the JSON envelope. Base64 inflates by 4/3, so the decoded file
# stays well under MAX_INPUT_BYTES for the same cap.
MAX_BODY_BYTES = MAX_INPUT_BYTES + (MAX_INPUT_BYTES >> 1)

TEXT_DETECTION_STATUSES = ("DETECTED", "INCONCLUSIVE", "NOT_DETECTED", "NOT_RUN")
_UNRESOLVED_TEXT_VERDICTS = frozenset({"INCONCLUSIVE", "UNSUPPORTED", "ERROR"})

ALLOWED_CLEAN_OPTIONS = {
    "nfkc": bool,
    "aggressive_homoglyphs": bool,
    "keep_non_ai_metadata": bool,
    "also_layer_a_text": bool,
    "remove_pixel": str,
    "strip_all_metadata": bool,
    "detect_before": bool,
    "detect_after": bool,
    # OURS-only additive keys (Phase 9 plan)
    "remove_synthid": bool,
    "wmct_marker": bool,
}


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
    for key, kind in ALLOWED_CLEAN_OPTIONS.items():
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
    "/detect": {
        "post": {
            "summary": "Run watermark detectors on a file (text: vendor/statistical; image: SynthID score)",
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


def _compose_clean_request(options: dict[str, Any], dest: Path, kind: str) -> CleanRequest:
    """Map the whitelisted /clean options onto the canonical ``CleanRequest``.

    The whitelist above is the whole accepted option surface, so the request
    built here is the same object ``clean_file`` would build for these flags;
    every other field stays at its CLI default. ``strip_all_metadata`` keeps
    its historical meaning of an explicit override of
    ``keep_non_ai_metadata`` (the endpoint has always let it win), expressed
    through the single canonical field the planner derives strip-all from.
    ``force_type`` carries the router's decision so the planner re-classifies
    nothing and cannot disagree with ``classify_bytes`` above.
    """
    keep = bool(options.get("keep_non_ai_metadata"))
    if "strip_all_metadata" in options:
        keep = not bool(options["strip_all_metadata"])
    return CleanRequest(
        output=dest,
        force_type=kind,
        nfkc=bool(options.get("nfkc")),
        aggressive_homoglyphs=bool(options.get("aggressive_homoglyphs")),
        keep_non_ai_metadata=keep,
        remove_synthid=bool(options.get("remove_synthid")),
        wmct_marker=bool(options.get("wmct_marker")),
        json=True,  # the HTTP handler owns presentation; the executor stays silent
    )


def _run_planned(src: Path, dest: Path, request: CleanRequest) -> dict[str, Any]:
    """Preflight and clean one asset through the canonical plan executor.

    ``plan_work`` resolves the destination and refuses oversized inputs,
    aliased outputs, and binary-as-text before the first write — the same
    refusals the CLI gets, now inherited here. The executor reports failures
    as error payloads; the direct calls this replaced raised ``ValueError``
    into the handler's 400 mapping, so the message is re-raised unchanged.
    """
    item = InputItem(path=src, relative=Path(src.name))
    ((_, output, plan),) = plan_work([item], request, batch=False)
    payload = run_clean_item(item.path, output, request, plan)
    if "error" in payload:
        raise ValueError(payload["error"])
    return payload


def _plan_preflight(src: Path, dest: Path, request: CleanRequest) -> None:
    """Run the canonical preflight for one asset without executing it.

    The ``also_layer_a_text=False`` container branch cannot hand the executor
    the client's value (``CleanPlan`` has no container option yet), but its
    requests must still clear the same ``plan_work`` refusals — the input
    size cap, aliased outputs — that ``_run_planned`` gives every other
    branch. Raises ``ValueError`` with the planner's message, exactly like
    the executed paths.
    """
    item = InputItem(path=src, relative=Path(src.name))
    plan_work([item], request, batch=False)


def _identify_query(target: str) -> bool:
    """The ``?identify=`` query flag of a request target (opt-in, truthy spellings)."""
    parsed = parse_qs(urlparse(target).query)
    return parsed.get("identify", [""])[0].strip().lower() in ("true", "1", "yes")


class Handler(BaseHTTPRequestHandler):
    server_version = f"watermark-remover/{VERSION}"

    def log_message(self, fmt: str, *args: object) -> None:
        eprint(f"{self.address_string()} - {fmt % args}")

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {API_KEY}"

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

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        data = _json_ok(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
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
        path = urlparse(self.path).path
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return
        if path not in ("/inspect", "/clean", "/detect"):
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
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
            else:
                self._handle_clean(data, name, body)
        except ValueError as e:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
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
                report = inspect_text(raw_text).to_dict()
                s_rep = score_text_stylometry(raw_text, path=name or "<text>")
                report["stylometry"] = s_rep.to_dict()
                if run_detect:
                    report["text_detectors"] = run_all_text_detectors(raw_text)
                if identify:
                    # Provenance identification is a metadata question about
                    # binary assets; a text asset carries none — unknown.
                    verdict = unknown_verdict()
            elif kind == "image":
                report = inspect_image(path).to_dict()
            else:
                report = inspect_container(path).to_dict()
            if identify and verdict is None:
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
                detections = [score]
            else:
                detections = []
                report = inspect_container(path).to_dict()
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

    def _handle_clean(self, data: bytes, name: str, body: dict[str, Any]) -> None:
        kind = classify_bytes(data, Path(name).suffix)
        if kind == "unknown":
            raise ValueError(
                "unrecognized file format; use a filename with a known extension "
                "(e.g. notes.txt) or a supported image/container name"
            )
        options = body.get("options")
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object")
        for key, value in options.items():
            if key not in ALLOWED_CLEAN_OPTIONS:
                raise ValueError(f"unknown option: {key}")
            expected_type = ALLOWED_CLEAN_OPTIONS[key]
            if not isinstance(value, expected_type):
                type_name = "boolean" if expected_type is bool else "string"
                raise ValueError(f"option {key!r} must be a {type_name}")

        with tempfile.TemporaryDirectory(prefix="wm-clean-") as tmp:
            tmpdir = Path(tmp)
            src = _tmp_path(tmpdir, name or "input")
            src.write_bytes(data)
            # The cleaned output is staged in its own directory so a client
            # filename can never alias it, whatever the asset kind.
            outdir = Path(tempfile.mkdtemp(dir=tmpdir, prefix="out-"))
            if kind == "text":
                if looks_binary(data):
                    raise ValueError(
                        "refusing to clean bytes that look like a binary container as text"
                    )
                text = data.decode("utf-8", errors="surrogateescape")
                detector_reports: dict[str, Any] = {}
                if bool(options.get("detect_before")):
                    detector_reports["before"] = run_text_detectors(text)
                dest = outdir / "out.txt"
                request = _compose_clean_request(options, dest, kind)
                payload = _run_planned(src, dest, request)
                cleaned = dest.read_text(encoding="utf-8", errors="surrogateescape")
                report: dict[str, Any] = {
                    "kind": "text",
                    "stats": payload["stats"],
                    "length": len(cleaned),
                }
                if bool(options.get("detect_after")):
                    detector_reports["after"] = run_text_detectors(cleaned)
                if detector_reports:
                    report["text_detectors"] = detector_reports
                cleaned_bytes = dest.read_bytes()
            elif kind == "image":
                remove_pixel = options.get("remove_pixel")
                if remove_pixel not in (None, "ctrlregen", "diffusion"):
                    raise ValueError("remove_pixel must be one of: ctrlregen, diffusion")
                dest = outdir / "out.png"
                request = _compose_clean_request(options, dest, kind)
                payload = _run_planned(src, dest, request)
                report = {k: v for k, v in payload.items() if k not in _PAYLOAD_BOOKKEEPING}
                if remove_pixel:
                    # Pixel backends (ctrlregen / diffusion) are wired through
                    # the image_meta port; mark unavailable in the report
                    # rather than failing the request.
                    report["pixel_removal"] = {
                        "available": False,
                        "error": (
                            f"remove_pixel={remove_pixel!r} is reserved for the future "
                            "image_meta integration; not yet exposed via OURS "
                            "clean_image"
                        ),
                    }
                if bool(options.get("detect_before")) and report.get("synthid_before") is None:
                    report["synthid_before"] = run_synthid_score(src)
                if bool(options.get("detect_after")) and report.get("synthid_after") is None:
                    report["synthid_after"] = run_synthid_score(dest)
                cleaned_bytes = dest.read_bytes()
            else:
                dest = outdir / f"out{Path(name).suffix}"
                if options.get("also_layer_a_text") is False:
                    # The one whitelisted option the canonical CleanPlan cannot
                    # express yet: skipping the Layer-A text scrub of container
                    # bodies has no plan field, so the executor would run the
                    # scrub the client asked to skip. The request still clears
                    # the shared preflight — the same refusals _run_planned
                    # inherits — and only execution falls to clean_container
                    # with the client's value. When CleanPlan grows a container
                    # option, this branch folds into _run_planned like every
                    # other case.
                    _plan_preflight(src, dest, _compose_clean_request(options, dest, kind))
                    result = clean_container(src, dest, also_layer_a_text=False)
                    report = {"kind": "container", **result}
                else:
                    request = _compose_clean_request(options, dest, kind)
                    payload = _run_planned(src, dest, request)
                    report = {k: v for k, v in payload.items() if k not in _PAYLOAD_BOOKKEEPING}
                cleaned_bytes = dest.read_bytes()
            report.pop("input", None)
            report.pop("output", None)
        self._respond(
            HTTPStatus.OK,
            {
                "ok": True,
                "kind": kind,
                "cleaned": base64.b64encode(cleaned_bytes).decode("ascii"),
                "report": report,
            },
        )


def main() -> int:
    global API_KEY  # noqa: PLW0603
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("WATERMARKS_SERVER_HOST", "127.0.0.1"))
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("WATERMARKS_SERVER_PORT", "8765"))
    )
    p.add_argument("--api-key", default=API_KEY, help="require this bearer token (default: none)")
    p.add_argument("-V", "--version", action="store_true", help="print version and exit")
    args = p.parse_args()

    if args.version:
        print(VERSION)
        return 0

    API_KEY = args.api_key

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
