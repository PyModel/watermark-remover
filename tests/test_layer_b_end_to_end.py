"""Layer B end to end: a live endpoint actually rewrites, through every surface.

These drive a stub OpenAI-compatible server so the whole path — CleanRequest ->
RewritePlan -> rewrite() -> Layer A re-scrub -> written file — is exercised
without a model. They are the tests that prove "connect an LLM and clean the
watermark" works, rather than merely that the plumbing type-checks.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from clean_request import CleanRequest, build_rewrite_plan
from rewrite_text import generate_candidates, rewrite

ZWSP = "​"
#: Deliberately carries a zero-width character so the Layer A re-scrub after the
#: rewrite is observable in the output.
REWRITTEN = f"A plainly worded replacement sentence.{ZWSP}"
SOURCE = f"Delve into it.{ZWSP} Moreover, it is important to note this.\n"


class _QuietServer(ThreadingHTTPServer):
    """The streaming tests hang up mid-response on purpose.

    ``socketserver`` prints a traceback for the resulting broken pipe, which is
    expected here and only makes real failures harder to spot.
    """

    def handle_error(self, request, client_address):
        return


class _ChatHandler(BaseHTTPRequestHandler):
    replies: ClassVar[list[str]] = []
    seen: ClassVar[list[dict]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.seen.append(body)
        index = min(len(self.seen) - 1, len(self.replies) - 1)
        content = self.replies[index] if self.replies else REWRITTEN
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


@pytest.fixture
def chat_server():
    _ChatHandler.replies = []
    _ChatHandler.seen = []
    server = _QuietServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _request(chat_server: str, **overrides) -> CleanRequest:
    base = {
        "rewrite": "humanize",
        "rewrite_backend": "openai-compatible",
        "rewrite_model": "stub-model",
        "rewrite_base_url": chat_server,
        "rewrite_timeout": 10.0,
    }
    base.update(overrides)
    return CleanRequest(**base)


# --- the library path --------------------------------------------------------


def test_rewrite_replaces_the_text_and_rescrubs_layer_a(chat_server):
    plan = build_rewrite_plan(_request(chat_server))
    out, info = rewrite(SOURCE, plan)
    assert "plainly worded replacement" in out
    # Layer A runs after the model, so the model's own zero-width char is gone.
    assert ZWSP not in out
    assert info["mode"] == "rewritten"
    assert info["model"] == "stub-model"
    # The honesty note travels with the result.
    assert "best-effort" in info["note"]


@pytest.mark.parametrize(
    "strength", ["paraphrase", "humanize", "code", "backtranslate", "structural"]
)
def test_every_strength_reaches_the_endpoint(chat_server, strength):
    plan = build_rewrite_plan(_request(chat_server, rewrite=strength))
    out, _ = rewrite(SOURCE, plan)
    assert "plainly worded replacement" in out
    assert len(_ChatHandler.seen) == 1


def test_candidates_are_all_returned_and_scored(chat_server):
    _ChatHandler.replies = [
        "One completely different rendering of the original statement here.",
        "Delve into it. Moreover, it is important to note this.",
        "A third variant, worded in yet another way entirely, for contrast.",
    ]
    plan = build_rewrite_plan(_request(chat_server, rewrite_candidates=3))
    candidates = generate_candidates(SOURCE, plan)
    assert len(candidates) == 3
    assert sum(candidate.selected for candidate in candidates) == 1
    # The near-identical echo must not be the auto-pick.
    assert not candidates[1].selected
    # Candidate bodies are returned here, never smuggled through rewrite()'s info.
    _, info = rewrite(SOURCE, build_rewrite_plan(_request(chat_server, rewrite_candidates=2)))
    assert not any("text" in entry for entry in info.get("candidate_scores", []))


def test_a_remote_endpoint_is_refused_without_an_opt_in():
    plan = build_rewrite_plan(
        CleanRequest(
            rewrite="humanize",
            rewrite_backend="openai-compatible",
            rewrite_model="m",
            rewrite_base_url="https://api.example.test",
        )
    )
    with pytest.raises(Exception, match="loopback"):
        rewrite(SOURCE, plan)


# --- the CLI path ------------------------------------------------------------


def test_cli_rewrite_writes_the_model_output(chat_server, tmp_path: Path):
    source = tmp_path / "draft.txt"
    source.write_text(SOURCE, encoding="utf-8")
    destination = tmp_path / "draft.cleaned.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(source),
            "-o",
            str(destination),
            "--rewrite",
            "humanize",
            "--rewrite-backend",
            "openai-compatible",
            "--rewrite-model",
            "stub-model",
            "--rewrite-base-url",
            chat_server,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    written = destination.read_text(encoding="utf-8")
    assert "plainly worded replacement" in written
    assert ZWSP not in written


def test_cli_audit_records_the_rewrite_without_the_document_body(chat_server, tmp_path: Path):
    source = tmp_path / "draft.txt"
    source.write_text(SOURCE, encoding="utf-8")
    audit = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(source),
            "-o",
            str(tmp_path / "out.txt"),
            "--rewrite",
            "humanize",
            "--rewrite-backend",
            "openai-compatible",
            "--rewrite-model",
            "stub-model",
            "--rewrite-base-url",
            chat_server,
            "--audit",
            str(audit),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(audit.read_text(encoding="utf-8"))
    # The audit says a rewrite happened; it must not contain the rewritten body.
    assert "plainly worded replacement" not in json.dumps(report)


# --- the TUI path ------------------------------------------------------------


def test_tui_run_performs_a_live_rewrite(chat_server, tmp_path: Path):
    pytest.importorskip("textual")
    from textual.widgets import DataTable, Input, Select
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(SOURCE, encoding="utf-8")
    destination = tmp_path / "cleaned.txt"
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-output", Input).value = str(destination)
            app.query_one("#sel-rewrite", Select).value = "humanize"
            app.query_one("#sel-backend", Select).value = "openai-compatible"
            app.query_one("#in-base-url", Input).value = chat_server
            app.query_one("#in-model", Input).value = "stub-model"
            app.query_one("#in-rewrite-timeout", Input).value = "10"
            await pilot.pause()

            request = app.collect_request()
            assert request.rewrite_strength == "humanize"
            # A loopback endpoint needs no confirmation, so the run proceeds.
            app.action_run()
            for _ in range(200):
                await pilot.pause()
                if app.query_one("#run-table", DataTable).row_count:
                    break
            assert destination.is_file()
            written = destination.read_text(encoding="utf-8")
            assert "plainly worded replacement" in written
            assert ZWSP not in written

    asyncio.run(scenario())


def test_tui_records_the_run_in_history_without_a_secret(chat_server, tmp_path: Path):
    pytest.importorskip("textual")
    from textual.widgets import DataTable, Input, Select
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(SOURCE, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-output", Input).value = str(tmp_path / "out.txt")
            app.query_one("#sel-rewrite", Select).value = "humanize"
            app.query_one("#sel-backend", Select).value = "openai-compatible"
            app.query_one("#in-base-url", Input).value = chat_server
            app.query_one("#in-model", Input).value = "stub-model"
            await pilot.pause()
            app.action_run()
            for _ in range(200):
                await pilot.pause()
                if app.query_one("#run-table", DataTable).row_count:
                    break
            for _ in range(10):
                await pilot.pause()
            assert app.history
            entry = app.history[0]
            assert "--rewrite humanize" in entry.command
            assert "B(humanize)" in entry.summary
            assert "api-key" not in entry.command

    asyncio.run(scenario())


# --- streaming ---------------------------------------------------------------


class _StreamHandler(BaseHTTPRequestHandler):
    mode: ClassVar[str] = "openai"
    chunks: ClassVar[list[str]] = ["Hello ", "streamed ", "world."]

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        assert body.get("stream") is True, "streaming path must ask for a stream"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/event-stream" if self.mode == "openai" else "application/x-ndjson",
        )
        self.end_headers()
        for chunk in self.chunks:
            if self.mode == "openai":
                line = json.dumps({"choices": [{"delta": {"content": chunk}}]})
                self.wfile.write(f"data: {line}\n\n".encode())
            else:
                self.wfile.write(
                    (json.dumps({"message": {"content": chunk}, "done": False}) + "\n").encode()
                )
            self.wfile.flush()
        if self.mode == "openai":
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.wfile.write((json.dumps({"done": True}) + "\n").encode())
        self.wfile.flush()

    def log_message(self, *args):
        return


@pytest.fixture
def stream_server():
    server = _QuietServer(("127.0.0.1", 0), _StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_openai_stream_delivers_fragments_and_the_full_text(stream_server):
    _StreamHandler.mode = "openai"
    seen: list[str] = []
    plan = build_rewrite_plan(_request(stream_server))
    candidates = generate_candidates(SOURCE, plan, on_token=seen.append)
    assert seen == ["Hello ", "streamed ", "world."]
    assert candidates[0].text == "Hello streamed world."


def test_ollama_stream_delivers_fragments(stream_server):
    _StreamHandler.mode = "ollama"
    seen: list[str] = []
    plan = build_rewrite_plan(
        _request(stream_server, rewrite_backend="ollama", rewrite_model="stub")
    )
    candidates = generate_candidates(SOURCE, plan, on_token=seen.append)
    assert "".join(seen) == "Hello streamed world."
    assert candidates[0].text == "Hello streamed world."


def test_a_raising_token_sink_never_loses_the_generation(stream_server):
    """The sink runs on the reader thread; its failure must not kill the read."""
    _StreamHandler.mode = "openai"

    def hostile(_fragment: str) -> None:
        raise RuntimeError("callback exploded")

    plan = build_rewrite_plan(_request(stream_server))
    candidates = generate_candidates(SOURCE, plan, on_token=hostile)
    assert candidates[0].text == "Hello streamed world."


def test_streaming_without_a_sink_stays_on_the_non_streaming_path(chat_server):
    """No sink means no stream: the plain request path is unchanged."""
    plan = build_rewrite_plan(_request(chat_server))
    generate_candidates(SOURCE, plan)
    assert _ChatHandler.seen[0].get("stream") is not True


class _FloodHandler(BaseHTTPRequestHandler):
    """Emits one absurd line, then keeps going — an endpoint that lies about size."""

    line_bytes: ClassVar[int] = (1 << 20) + 64
    lines: ClassVar[int] = 1

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        filler = "x" * self.line_bytes
        for _ in range(self.lines):
            self.wfile.write((json.dumps({"message": {"content": filler}}) + "\n").encode())
            self.wfile.flush()

    def log_message(self, *args):
        return


@pytest.fixture
def flood_server():
    server = _QuietServer(("127.0.0.1", 0), _FloodHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_a_single_stream_line_over_the_cap_is_refused(flood_server):
    """A stream has no Content-Length, so the per-line cap is the only bound."""
    from layer_b_http import LayerBHTTPError, stream_json_lines

    _FloodHandler.line_bytes = (1 << 20) + 64
    _FloodHandler.lines = 1
    with pytest.raises(LayerBHTTPError, match="line exceeds safety limit"):
        list(stream_json_lines(flood_server, "/api/chat", {}, timeout=10.0))


def test_many_small_lines_still_hit_the_total_cap(flood_server):
    """Under the per-line cap, the running total is what stops a flood."""
    from layer_b_http import LayerBHTTPError, stream_json_lines

    _FloodHandler.line_bytes = 4096
    _FloodHandler.lines = 64
    with pytest.raises(LayerBHTTPError, match="stream exceeds safety limit"):
        list(
            stream_json_lines(
                flood_server,
                "/api/chat",
                {},
                timeout=10.0,
                response_limit=20_000,
            )
        )


def test_a_stream_survives_keepalives_and_garbage_lines(stream_server):
    """Providers interleave comments and blank lines; those are skipped, not fatal."""
    from layer_b_http import stream_json_lines

    _StreamHandler.mode = "openai"
    _StreamHandler.chunks = ["a", "b"]
    try:
        objects = list(
            stream_json_lines(stream_server, "/v1/chat/completions", {"stream": True}, timeout=10.0)
        )
    finally:
        _StreamHandler.chunks = ["Hello ", "streamed ", "world."]
    assert len(objects) == 2


def test_the_clean_pipeline_streams_when_given_a_sink(stream_server, tmp_path):
    """The Run pane's stream box is fed by the same seam the CLI cleans through."""
    from clean_file import run_clean_item
    from clean_request import build_clean_plan, resolve_kind

    _StreamHandler.mode = "openai"
    source = tmp_path / "note.txt"
    source.write_text(SOURCE, encoding="utf-8")
    request = _request(stream_server, paths=(source,))
    plan = build_clean_plan(request, tmp_path / "note.clean.txt", resolve_kind(source, request))

    seen: list[str] = []
    payload = run_clean_item(
        source, tmp_path / "note.clean.txt", request, plan, on_token=seen.append
    )

    assert "".join(seen) == "Hello streamed world."
    assert payload["exit_code"] == 0
    assert "Hello streamed world." in (tmp_path / "note.clean.txt").read_text(encoding="utf-8")


def test_the_clean_pipeline_does_not_stream_without_a_sink(chat_server, tmp_path):
    from clean_file import run_clean_item
    from clean_request import build_clean_plan, resolve_kind

    source = tmp_path / "note.txt"
    source.write_text(SOURCE, encoding="utf-8")
    request = _request(chat_server, paths=(source,))
    plan = build_clean_plan(request, tmp_path / "note.clean.txt", resolve_kind(source, request))
    run_clean_item(source, tmp_path / "note.clean.txt", request, plan)
    assert _ChatHandler.seen[0].get("stream") is not True


def test_a_backend_that_ignores_stream_still_produces_a_rewrite(chat_server, tmp_path):
    """Asking to watch must not change whether the rewrite works.

    ``_ChatHandler`` answers every request with a plain, non-streamed body — the
    behaviour of any endpoint without SSE support.  Passing a sink used to turn
    that into "empty content"; it now degrades to a single fragment.
    """
    from clean_file import run_clean_item
    from clean_request import build_clean_plan, resolve_kind

    source = tmp_path / "note.txt"
    source.write_text(SOURCE, encoding="utf-8")
    destination = tmp_path / "note.clean.txt"
    request = _request(chat_server, paths=(source,))
    plan = build_clean_plan(request, destination, resolve_kind(source, request))

    seen: list[str] = []
    payload = run_clean_item(source, destination, request, plan, on_token=seen.append)

    assert _ChatHandler.seen[0].get("stream") is True
    assert seen == [REWRITTEN]
    assert payload["exit_code"] == 0
    assert "plainly worded replacement" in destination.read_text(encoding="utf-8")


def test_a_stream_that_yields_nothing_names_streaming_in_the_error(flood_server):
    """The operator must not be sent hunting a working non-streaming endpoint."""
    from rewrite_text import _stream_openai_compatible

    _FloodHandler.line_bytes = 8
    _FloodHandler.lines = 1
    with pytest.raises(RuntimeError, match="may not support streaming"):
        _stream_openai_compatible(
            flood_server, "/v1/chat/completions", {}, {}, 10.0, lambda _: None
        )
