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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
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
