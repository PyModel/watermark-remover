#!/usr/bin/env python3
"""The Python half of wm-tui: a JSON-Lines server over stdin and stdout.

The TypeScript frontend (``tui/``) spawns this with ``$WM_TUI_PYTHON`` and
talks to it one JSON object per line; ``tui/PROTOCOL.md`` is the contract.
The frontend owns the terminal, and this process owns every decision about a
clean.  The decisions themselves live in ``tui_core`` so they can be tested
without a process; this module is the transport around them.

Three properties this module exists to hold:

* **Frames are never corrupted.**  The pipeline prints: warnings, progress, a
  stray debug line, sometimes while it is still being imported.  Before any
  pipeline import, fd 1 is duplicated for frames only, and fd 1 plus
  ``sys.stdout`` are pointed at stderr, so anything else that writes to
  "stdout", from Python or from C, lands in the frontend's log instead of in
  the middle of a frame.
* **The frontend never freezes.**  Requests run on worker threads, so a
  ``cancel`` or ``detect_endpoints`` is answered while a clean runs.  Only one
  clean or inspect runs at a time; a second is answered ``busy`` rather than
  queued, because a queued clean the operator cannot see is a surprise write.
* **The key never leaves.**  Nothing here puts the API key in a frame, and the
  frame writer redacts it from every payload as a last line of defence.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO


def claim_stdout() -> BinaryIO:
    """Keep fd 1 for frames; send every other write to stderr.

    ``os.dup2`` moves the *file descriptor*, so C extensions and subprocesses
    that inherit fd 1 are redirected too, not only Python's ``print``.
    """
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    frames_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(frames_fd, "wb")


# Claimed before the imports below: the pipeline probes optional libraries at
# import time, and one that prints would otherwise write ahead of ``ready``.
_FRAMES = claim_stdout() if __name__ == "__main__" else None

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tui as launcher
from asset_kind import classify_asset
from batch_inputs import InputItem, InputSelection
from clean_asset import CleanPlan
from clean_file import dry_run_payload, run_clean_item
from clean_request import CleanRequest, select_request_inputs
from inspect_file import inspect_asset
from rewrite_text import LIVE_REWRITE_BACKENDS
from tui_core import (
    API_KEY_ENV,
    PRESETS,
    BadRequest,
    InvalidOptions,
    apply_settings_update,
    compose_request,
    confirm_gates,
    describe_output,
    detect_local_endpoints,
    display_path,
    environment_checks,
    estimate_rewrite_seconds,
    failed_finding,
    finding_from_report,
    history_summary,
    layer_for_result,
    load_settings,
    plan_layer,
    plan_warnings,
    preflight_work,
    preview_command,
    result_class_for,
    result_lines,
    save_settings,
    settings_path,
    should_onboard,
    text_for_diff,
    unified_diff,
)

#: Minimum interval between two ``token`` frames.  A model can emit hundreds
#: of fragments a second; one frame each would spend the frontend's render
#: budget on JSON parsing.
TOKEN_INTERVAL = 0.05

#: Most files ``plan`` lists.  ``plan`` runs on every keystroke (debounced);
#: pointing it at a home directory must not produce a megabyte frame.
PLAN_FILES_LIMIT = 2000

#: How long exit waits for in-flight requests to answer.
DRAIN_SECONDS = 5.0

#: Keys shorter than this are not redacted: replacing every "ab" in every
#: frame would mangle the protocol, and a string that short is not a secret.
REDACT_MIN_LENGTH = 8
REDACTED = "[redacted]"


class BridgeError(Exception):
    """A request failed with a PROTOCOL error code."""

    def __init__(self, code: str, message: str, data: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = dict(data) if data is not None else None


# --- frames ------------------------------------------------------------------


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, REDACTED) if secret in value else value
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secret) for item in value]
    return value


def _error_frame(
    rid: Any, code: str, message: str, data: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = dict(data)
    return {"id": rid, "error": error}


class FrameWriter:
    """Serialises frames onto the protocol stream, one whole line at a time.

    Worker threads, the token flusher and the reader thread all write; the lock
    is what keeps two frames from interleaving mid-line.  ASCII-escaped JSON is
    still UTF-8, and it survives file names that are not valid Unicode (lone
    surrogates from ``surrogateescape``), which raw UTF-8 would refuse.
    """

    def __init__(self, stream: BinaryIO, *, redact: bool = True) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._redact = redact
        self._closed = False

    def write(self, frame: dict[str, Any]) -> None:
        if self._redact:
            secret = os.environ.get(API_KEY_ENV) or ""
            if len(secret) >= REDACT_MIN_LENGTH:
                # Payloads only: the envelope (id, event names) is ours, and
                # redacting a substring of "file_done" would break framing.
                frame = {
                    key: _redact(value, secret) if key in ("result", "error", "data") else value
                    for key, value in frame.items()
                }
        line = json.dumps(frame, ensure_ascii=True, default=str) + "\n"
        with self._lock:
            if self._closed:
                return
            try:
                self._stream.write(line.encode("ascii"))
                self._stream.flush()
            except (OSError, ValueError):
                # The frontend is gone; the reader will see EOF and exit.
                self._closed = True


class TokenCoalescer:
    """Batch Layer B fragments into at most one frame per ``TOKEN_INTERVAL``.

    Emission happens on one flusher thread only, so frames stay in order.
    ``close`` stops and joins that thread and then emits whatever is left,
    waiting out the interval first, so the last fragment is never lost, never
    sent after the file's ``file_done``, and never breaks the rate limit.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._parts: list[str] = []
        self._lock = threading.Lock()
        self._last = float("-inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="wm-tui-tokens", daemon=True)
        self._thread.start()

    def add(self, fragment: str) -> None:
        with self._lock:
            self._parts.append(fragment)

    def _take(self) -> str:
        with self._lock:
            text = "".join(self._parts)
            self._parts.clear()
        return text

    def _send(self, text: str) -> None:
        self._last = time.monotonic()
        self._emit(text)

    def _loop(self) -> None:
        while not self._stop.wait(TOKEN_INTERVAL / 5):
            if time.monotonic() - self._last >= TOKEN_INTERVAL and (text := self._take()):
                self._send(text)

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        text = self._take()
        if not text:
            return
        wait = TOKEN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._send(text)


# --- helpers -----------------------------------------------------------------


def _state(params: Mapping[str, Any]) -> Mapping[str, Any]:
    state = params.get("state")
    if not isinstance(state, Mapping):
        raise BadRequest("params.state must be an object")
    return state


def _select(request: CleanRequest) -> InputSelection:
    """The request's files, with a selector refusal reported as ``invalid_options``."""
    try:
        return select_request_inputs(request)
    except ValueError as error:
        raise InvalidOptions(str(error)) from error


def _file_done(
    index: int,
    display: str,
    layer: str,
    *,
    output: str | None,
    exit_code: int,
    before: Mapping[str, int],
    after: Mapping[str, int],
    lines: list[str],
    diff: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "index": index,
        "display": display,
        "output": output,
        "layer": layer,
        "result_class": result_class_for(layer),
        "exit_code": exit_code,
        "before": dict(before),
        "after": dict(after),
        "lines": lines,
        "diff": diff,
        "error": error,
    }


def _require_confirmed(gates: list[dict[str, str]], confirmed: list[str]) -> None:
    """Refuse with ``needs_confirm`` unless every gate was agreed to."""
    missing = [gate for gate in gates if gate["kind"] not in confirmed]
    if missing:
        raise BridgeError(
            "needs_confirm",
            f"confirm before cleaning: {', '.join(gate['kind'] for gate in missing)}",
            {"confirm": missing},
        )


def _kinds(work: list[tuple[InputItem, Path | None, CleanPlan]]) -> list[str]:
    """Each planned file's resolved kind (``plan_work`` records it on the plan)."""
    return [plan.forced_kind for _item, _output, plan in work]


def _parse_launch_argv(raw: str | None) -> Any:
    """Re-parse the launcher's arguments, falling back to defaults.

    The launcher already validated them; a failure here means the variable was
    set by hand, and the right response is a working UI on defaults plus a
    line in the log, not a dead bridge.
    """
    parser = launcher.build_parser()
    defaults = parser.parse_args([])
    if not raw:
        return defaults
    try:
        argv = json.loads(raw)
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("WM_TUI_ARGV must be a JSON list of strings")

        def fail(message: str) -> None:
            raise ValueError(message)

        parser.error = fail  # type: ignore[method-assign]
        return parser.parse_args(argv)
    except (ValueError, SystemExit) as error:
        print(f"wm-tui-bridge: ignoring WM_TUI_ARGV: {error}", file=sys.stderr)
        return defaults


# --- the server --------------------------------------------------------------


class Bridge:
    """Dispatches requests to handlers, one worker thread per request."""

    #: Methods answered on the reader thread: they must never wait behind work.
    INLINE = frozenset({"cancel", "shutdown"})
    #: Methods that touch many files; one at a time.
    EXCLUSIVE = frozenset({"clean", "inspect"})

    def __init__(self, writer: FrameWriter, *, launch_argv: str | None = None) -> None:
        self.writer = writer
        self.launch = _parse_launch_argv(launch_argv)
        self.cancel = threading.Event()
        self.stopping = False
        self._exclusive = threading.Lock()
        self._history: list[dict[str, str]] = []
        self._history_lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._workers_lock = threading.Lock()
        self._handlers: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {
            "hello": self.hello,
            "plan": self.plan,
            "inspect": self.inspect,
            "clean": self.clean,
            "cancel": self.cancel_request,
            "detect_endpoints": self.detect_endpoints,
            "checks": self.checks,
            "save_settings": self.save_settings,
            "history": self.history,
            "shutdown": self.shutdown,
        }

    # -- framing -----------------------------------------------------------

    def event(self, rid: Any, name: str, data: Mapping[str, Any]) -> None:
        self.writer.write({"id": rid, "event": name, "data": dict(data)})

    def fail(self, rid: Any, code: str, message: str) -> None:
        self.writer.write(_error_frame(rid, code, message))

    def ready(self) -> None:
        self.writer.write({"event": "ready", "data": {"version": launcher.package_version()}})

    # -- dispatch ----------------------------------------------------------

    def handle_line(self, line: str | bytes) -> threading.Thread | None:
        """Parse one request and start it. Returns the worker, for tests to join."""
        try:
            message = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            self.fail(None, "bad_request", f"not valid JSON: {error}")
            return None
        if not isinstance(message, dict):
            self.fail(None, "bad_request", "a request must be a JSON object")
            return None
        rid = message.get("id")
        if rid is not None and (isinstance(rid, bool) or not isinstance(rid, (int, str))):
            self.fail(None, "bad_request", "id must be a number, a string or null")
            return None
        method = message.get("method")
        handler = self._handlers.get(method) if isinstance(method, str) else None
        if handler is None:
            self.fail(rid, "bad_request", f"unknown method: {method!r}")
            return None
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self.fail(rid, "bad_request", "params must be an object")
            return None

        if method in self.INLINE:
            self.writer.write(self._answer(rid, handler, params))
            return None
        exclusive = method in self.EXCLUSIVE
        if exclusive:
            # Claimed here, on the reader thread, not in the worker: otherwise
            # a cancel sent right after the clean could be cleared by a worker
            # that had not started yet, and two cleans could both get in.
            if not self._exclusive.acquire(blocking=False):
                self.fail(rid, "busy", "a clean or inspect is already running")
                return None
            self.cancel.clear()
        worker = threading.Thread(
            target=self._work,
            args=(rid, handler, params, exclusive),
            name=f"wm-tui-{method}",
            daemon=True,
        )
        with self._workers_lock:
            self._workers.add(worker)
        worker.start()
        return worker

    def _answer(
        self, rid: Any, handler: Callable[[Any, Mapping[str, Any]], Any], params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run a handler and return its response frame. Never raises.

        ``BaseException`` on purpose: parts of the pipeline still raise
        ``SystemExit`` for a CLI refusal, and inside a thread that would end
        the worker silently, never answering and never releasing the
        exclusive lock, which leaves every later clean ``busy`` forever.
        """
        try:
            return {"id": rid, "result": handler(rid, params)}
        except BridgeError as error:
            return _error_frame(rid, error.code, error.message, error.data)
        except BadRequest as error:
            return _error_frame(rid, "bad_request", str(error))
        except InvalidOptions as error:
            return _error_frame(rid, "invalid_options", str(error))
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            return _error_frame(rid, "internal", f"{type(error).__name__}: {error}")

    def _work(
        self,
        rid: Any,
        handler: Callable[[Any, Mapping[str, Any]], Any],
        params: Mapping[str, Any],
        exclusive: bool,
    ) -> None:
        frame = self._answer(rid, handler, params)
        # Released before the response is written: a frontend that sends the
        # next clean the moment it reads this answer must not be told busy.
        if exclusive:
            self._exclusive.release()
        self.writer.write(frame)
        # Only now, so drain waits for the response to be written.
        with self._workers_lock:
            self._workers.discard(threading.current_thread())

    def serve(self, stdin: BinaryIO) -> int:
        """Read requests until EOF or ``shutdown``. Returns the exit code."""
        self.ready()
        while not self.stopping:
            line = stdin.readline()
            if not line:
                break
            if line.strip():
                self.handle_line(line)
        self.drain()
        return 0

    def drain(self, timeout: float = DRAIN_SECONDS) -> None:
        """Let in-flight requests answer before exiting, within a bound.

        A clean is cancelled first, so it stops after the file it is on: the
        frontend that asked for it is gone or leaving, and nobody is watching
        the rest.  The bound is there because an in-flight model call cannot
        be interrupted, and a bridge the frontend has let go of must not
        linger for a whole rewrite timeout.
        """
        self.cancel.set()
        deadline = time.monotonic() + timeout
        with self._workers_lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))

    # -- methods -----------------------------------------------------------

    def hello(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        target = settings_path()
        args = self.launch
        flags: list[str] = []
        if args.recursive:
            flags.append("--recursive")
        if args.glob != "*":
            flags += ["--glob", args.glob]
        if args.extensions:
            flags += ["--extensions", args.extensions]
        return {
            "version": launcher.package_version(),
            "presets": [preset.to_dict() for preset in PRESETS],
            "settings": load_settings(target).to_wire(),
            "settings_path": str(target),
            "onboard": should_onboard(
                settings_exist=target.exists(), force=args.setup, skip=args.no_setup
            ),
            "initial": {"paths": list(args.path or ["."]), "flags": flags},
            "backends": list(LIVE_REWRITE_BACKENDS),
        }

    def plan(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """The cheap preview. Reads, never writes."""
        state = _state(params)
        try:
            composition = compose_request(state)
        except InvalidOptions as error:
            return {
                "ok": False,
                "error": str(error),
                "command": preview_command(state),
                "files": [],
                "files_total": 0,
                "discover_error": None,
                "layer": None,
                "result_class": None,
                "output": None,
                "confirm": [],
                "estimate_seconds": 0,
                "warnings": [],
                "preflight_error": None,
            }
        request = composition.request
        preview, kinds = self._preview_selection(request)
        total = preview["files_total"]
        # Unknown kinds (nothing selected, or preflight refused) count every
        # step the options ask for, and every file as rewritten.
        layer = plan_layer(request, kinds or ())
        text_count = total if kinds is None else kinds.count("text")
        return {
            "ok": True,
            "error": None,
            "command": composition.command,
            **preview,
            "layer": layer,
            "result_class": result_class_for(layer),
            "output": describe_output(request, total),
            "confirm": confirm_gates(request, total, text_count),
            "estimate_seconds": estimate_rewrite_seconds(request, text_count),
            "warnings": plan_warnings(request, kinds),
        }

    def _preview_selection(self, request: CleanRequest) -> tuple[dict[str, Any], list[str] | None]:
        """``plan``'s file fields, plus each file's resolved kind when preflight passes."""
        try:
            selection = select_request_inputs(request)
        except ValueError as error:
            return {
                "files": [],
                "files_total": 0,
                "discover_error": str(error),
                "preflight_error": None,
            }, None
        preview: dict[str, Any] = {
            "files": [
                self._file_entry(item.path, request) for item in selection.items[:PLAN_FILES_LIMIT]
            ],
            "files_total": len(selection.items),
            "discover_error": None,
            "preflight_error": None,
        }
        try:
            return preview, _kinds(preflight_work(request, selection))
        except ValueError as error:
            preview["preflight_error"] = str(error)
            return preview, None

    @staticmethod
    def _file_entry(path: Path, request: CleanRequest) -> dict[str, Any]:
        try:
            kind = classify_asset(path, forced_kind=request.force_type)
            size = path.stat().st_size
        except (OSError, ValueError):
            kind, size = "unknown", None
        return {
            "path": str(path.resolve()),
            "display": display_path(path),
            "kind": kind,
            "size": size,
        }

    @staticmethod
    def _inspect_one(path: Path, force_type: str, soft: bool) -> dict[str, Any]:
        display = display_path(path)
        try:
            report = inspect_asset(path, force_type=force_type, soft_binding=soft)
        except (Exception, SystemExit) as error:
            return failed_finding(path, display, error)
        return finding_from_report(report, path, display)

    def inspect(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        state = _state(params)
        soft = params.get("soft", False)
        if not isinstance(soft, bool):
            raise BadRequest("params.soft must be a boolean")
        request = compose_request(state).request
        findings = []
        cancelled = False
        for item in _select(request).items:
            if self.cancel.is_set():
                cancelled = True
                break
            finding = self._inspect_one(item.path, request.force_type, soft)
            self.event(rid, "inspected", finding)
            findings.append(finding)
        return {"files": findings, "cancelled": cancelled}

    def clean(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        state = _state(params)
        confirmed = params.get("confirmed") or []
        if not isinstance(confirmed, list) or not all(isinstance(c, str) for c in confirmed):
            raise BadRequest("params.confirmed must be a list of strings")
        # A confirmed "remote" is this run's egress permission and nothing
        # more: it is composed in here and never saved anywhere.
        composition = compose_request(state, allow_remote="remote" in confirmed)
        request = composition.request
        selection = _select(request)
        try:
            work = preflight_work(request, selection)
        except ValueError as error:
            raise InvalidOptions(f"preflight failed, nothing was written: {error}") from error
        kinds = _kinds(work)
        _require_confirmed(confirm_gates(request, len(work), kinds.count("text")), confirmed)

        if request.dry_run:
            self._dry_run(rid, request, work)
            errors, cancelled = 0, False
        else:
            errors, cancelled = self._clean_all(rid, request, selection.batch, work)
        total = len(work)
        return {
            "total": total,
            "errors": errors,
            "cancelled": cancelled,
            "command": composition.command,
            "history": self._record(
                composition.command, total, errors, plan_layer(request, kinds), cancelled
            ),
        }

    def _clean_all(
        self,
        rid: Any,
        request: CleanRequest,
        batch: bool,
        work: list[tuple[InputItem, Path | None, CleanPlan]],
    ) -> tuple[int, bool]:
        """Clean file by file until done or cancelled. Returns (errors, cancelled)."""
        if batch and request.output and not request.in_place:
            request.output.mkdir(parents=True, exist_ok=True)
        errors = 0
        for index, (item, output, plan) in enumerate(work):
            if self.cancel.is_set():
                return errors, True
            done = self._clean_one(rid, index, len(work), item.path, output, request, plan)
            if done["error"] is not None or done["exit_code"] != 0:
                errors += 1
        return errors, False

    def _clean_one(
        self,
        rid: Any,
        index: int,
        total: int,
        path: Path,
        output: Path | None,
        request: CleanRequest,
        plan: CleanPlan,
    ) -> dict[str, Any]:
        display = display_path(path)
        self.event(rid, "file_start", {"index": index, "total": total, "display": display})
        before = self._inspect_one(path, request.force_type, False)
        before_text = text_for_diff(path)

        payload = self._run_item(rid, index, path, output, request, plan)

        error = payload.get("error")
        written = Path(payload["output"]) if error is None and payload.get("output") else None
        after_counts: dict[str, int] = {}
        diff = None
        if written is not None and written.is_file():
            after_counts = self._inspect_one(written, request.force_type, False)["counts"]
            diff = unified_diff(before_text, text_for_diff(written), display)
        done = _file_done(
            index,
            display,
            layer_for_result(request, str(payload.get("kind") or before["kind"])),
            output=str(written.resolve()) if written is not None else None,
            exit_code=int(payload.get("exit_code") or 0),
            before=before["counts"],
            after=after_counts,
            lines=result_lines(payload),
            diff=diff,
            error=str(error) if error is not None else None,
        )
        self.event(rid, "file_done", done)
        return done

    def _run_item(
        self,
        rid: Any,
        index: int,
        path: Path,
        output: Path | None,
        request: CleanRequest,
        plan: CleanPlan,
    ) -> dict[str, Any]:
        """``run_clean_item`` with its Layer B stream sent as ``token`` frames.

        One file failing is a row, not the end of the batch, so an exception
        becomes an error payload.
        """
        coalescer = None
        if plan.text.rewrite_plan is not None:
            coalescer = TokenCoalescer(
                lambda text: self.event(rid, "token", {"index": index, "text": text})
            )
        try:
            return run_clean_item(
                path,
                output,
                request,
                plan,
                on_token=coalescer.add if coalescer is not None else None,
            )
        except (Exception, SystemExit) as error:
            return {"exit_code": 1, "error": f"{type(error).__name__}: {error}"}
        finally:
            # Joined before file_done, so no token frame can follow it.
            if coalescer is not None:
                coalescer.close()

    def _dry_run(
        self, rid: Any, request: CleanRequest, work: list[tuple[InputItem, Path | None, CleanPlan]]
    ) -> None:
        """Describe a visible-mark clean without writing, as ``wm --dry-run`` does."""
        total = len(work)
        for index, (item, output, plan) in enumerate(work):
            display = display_path(item.path)
            self.event(rid, "file_start", {"index": index, "total": total, "display": display})
            payload = dry_run_payload(item.path, output, plan, request.in_place)
            done = _file_done(
                index,
                display,
                layer_for_result(request, payload["kind"]),
                output=None,
                exit_code=0,
                before={},
                after={},
                lines=[
                    f"Dry run: would write {payload['output']}.",
                    *result_lines(payload),
                ],
                diff=None,
                error=None,
            )
            self.event(rid, "file_done", done)

    def _record(
        self, command: str, total: int, errors: int, layer: str, cancelled: bool
    ) -> dict[str, str]:
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "command": command,
            "summary": history_summary(total, errors, layer, cancelled=cancelled),
        }
        with self._history_lock:
            self._history.insert(0, entry)
        return entry

    def cancel_request(self, rid: Any, params: Mapping[str, Any]) -> dict[str, bool]:
        self.cancel.set()
        return {"ok": True}

    def detect_endpoints(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        return {"endpoints": detect_local_endpoints()}

    def checks(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        return {"checks": [vars(check) for check in environment_checks()]}

    def save_settings(self, rid: Any, params: Mapping[str, Any]) -> dict[str, str]:
        target = settings_path()
        updated = apply_settings_update(load_settings(target), params.get("settings"))
        return {"path": str(save_settings(updated, target))}

    def history(self, rid: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        with self._history_lock:
            return {"entries": [dict(entry) for entry in self._history]}

    def shutdown(self, rid: Any, params: Mapping[str, Any]) -> dict[str, bool]:
        self.cancel.set()
        self.stopping = True
        return {"ok": True}


# --- process setup -----------------------------------------------------------


def main(frames: BinaryIO | None = None) -> int:
    """Serve on stdin until EOF or ``shutdown``.

    *frames* is the stream claimed before the pipeline import when this file
    runs as a script; any other caller has fd 1 claimed here.
    """
    stream = frames if frames is not None else claim_stdout()
    # The launcher ran the frontend from its own directory; the operator's
    # relative paths are relative to where wm-tui was started.
    cwd = os.environ.get("WM_TUI_CWD")
    if cwd:
        try:
            os.chdir(cwd)
        except OSError as error:
            print(f"wm-tui-bridge: cannot enter WM_TUI_CWD {cwd}: {error}", file=sys.stderr)
    bridge = Bridge(FrameWriter(stream), launch_argv=os.environ.get("WM_TUI_ARGV"))
    return bridge.serve(sys.stdin.buffer)


if __name__ == "__main__":
    code = main(_FRAMES)
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    # Skip interpreter teardown: a daemon worker mid-clean or a probe pool
    # waiting on a hung port must not keep a bridge the frontend already let
    # go of alive.
    os._exit(code)
