"""wm-tui's Python half: the pure core, the JSON-Lines bridge, and the launcher.

The frontend is TypeScript and is tested on its own; these tests hold the
contract in ``tui/PROTOCOL.md`` from the Python side.  They spawn the real
bridge where the property under test is about the process (stdout discipline,
EOF, framing) and drive it in-process where it is about decisions (busy,
cancel, gates), because a thread join is a better synchronisation point than
a sleep.

Every test runs with the settings file, cache and config roots in a temp dir
and with the ``WATERMARKS_REWRITE_*`` environment cleared: this machine may
have a live model server on a candidate port, and a test must not depend on
whether it is up.
"""

from __future__ import annotations

import ast
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pipeline_actions as act
import tui as launcher
import tui_bridge
import tui_core
from clean_request import CleanRequest
from layer_b_discovery import BackendProbe
from tui_bridge import Bridge, FrameWriter, TokenCoalescer
from tui_core import (
    BEST_EFFORT,
    COST_CONFIRM_SECONDS,
    LOCAL_ENDPOINT_CANDIDATES,
    PRESETS,
    VERIFIABLE,
    BadRequest,
    InvalidOptions,
    TuiSettings,
    apply_settings_update,
    compose_request,
    confirm_gates,
    detect_local_endpoints,
    endpoint_candidates,
    environment_checks,
    estimate_rewrite_seconds,
    finding_from_report,
    history_summary,
    layer_for_result,
    load_settings,
    plan_layer,
    plan_warnings,
    preset_for,
    result_class_for,
    result_lines,
    save_settings,
    should_onboard,
    unified_diff,
)

ZWSP = "​"
SENTINEL = "sk-SENTINEL-never-in-a-frame-0123456789"
BRIDGE = SCRIPTS / "tui_bridge.py"
REWRITTEN = f"A plainly worded replacement sentence.{ZWSP}"
SOURCE = f"Delve into it.{ZWSP} Moreover, it is important to note this.\n"


# --- isolation ---------------------------------------------------------------

SCRUBBED_PREFIXES = ("WATERMARKS_REWRITE_", "WATERMARKS_TUI_", "WM_TUI_")


def _scrubbed_environ() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items() if not key.startswith(SCRUBBED_PREFIXES)
    }


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(SCRUBBED_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setenv("WATERMARKS_TUI_SETTINGS", str(tmp_path / "config" / "tui.json"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


# --- in-process harness ------------------------------------------------------


class Sink(io.RawIOBase):
    """Collects protocol bytes exactly as the frontend would read them."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer = bytearray()
        self.lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        with self.lock:
            self.buffer.extend(data)
        return len(data)

    def raw(self) -> str:
        with self.lock:
            return self.buffer.decode("ascii")

    def frames(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.raw().splitlines()]


def make_bridge(*, redact: bool = True, argv: list[str] | None = None) -> tuple[Bridge, Sink]:
    sink = Sink()
    launch = json.dumps(argv) if argv is not None else None
    return Bridge(FrameWriter(sink, redact=redact), launch_argv=launch), sink


def call(bridge: Bridge, sink: Sink, method: str, params=None, rid: Any = 1) -> list[dict]:
    """Send one request, wait for its worker, return every frame for *rid*."""
    message: dict[str, Any] = {"id": rid, "method": method}
    if params is not None:
        message["params"] = params
    worker = bridge.handle_line(json.dumps(message))
    if worker is not None:
        worker.join(60)
        assert not worker.is_alive(), f"{method} did not finish"
    return [frame for frame in sink.frames() if frame.get("id") == rid]


def response(frames: list[dict]) -> dict:
    final = [frame for frame in frames if "result" in frame or "error" in frame]
    assert len(final) == 1, frames
    return final[0]


def state(paths, preset="hidden", flags=(), endpoint=None) -> dict:
    return {
        "paths": [str(path) for path in paths],
        "preset": preset,
        "flags": list(flags),
        "endpoint": endpoint,
    }


# --- subprocess harness ------------------------------------------------------


class BridgeProcess:
    """A real ``python tui_bridge.py``, read on a thread so a hang is a timeout."""

    def __init__(self, cwd: Path, env: dict[str, str], program: list[str] | None = None) -> None:
        self.stderr_path = cwd / "bridge.stderr"
        self._stderr = self.stderr_path.open("wb")
        self.proc = subprocess.Popen(
            program or [sys.executable, str(BRIDGE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            cwd=cwd,
            env=env,
        )
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        self.raw: list[bytes] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def next_frame(self, timeout: float = 30.0) -> dict:
        line = self.lines.get(timeout=timeout)
        assert line is not None, "bridge closed stdout early: " + self.stderr_text()
        self.raw.append(line)
        return json.loads(line)

    def send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, rid: Any, method: str, params=None) -> list[dict]:
        message: dict[str, Any] = {"id": rid, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        frames = []
        while True:
            frame = self.next_frame()
            if frame.get("id") != rid:
                continue
            frames.append(frame)
            if "result" in frame or "error" in frame:
                return frames

    def stderr_text(self) -> str:
        if not self._stderr.closed:
            self._stderr.flush()
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")

    def close(self) -> int:
        """Close stdin, wait for exit, and keep every line it wrote after that."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            code = self.proc.wait(timeout=15)
            while (line := self.lines.get(timeout=5)) is not None:
                self.raw.append(line)
            self.lines.put(None)
            return code
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self._stderr.close()


def bridge_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = _scrubbed_environ()
    env.update(
        {
            "WATERMARKS_TUI_SETTINGS": str(tmp_path / "config" / "tui.json"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


# --- stub Layer B server (the pattern from test_layer_b_end_to_end) ----------


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        return


class _ChatHandler(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []
    auth: ClassVar[list[str]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.seen.append(json.loads(self.rfile.read(length) or b"{}"))
        self.auth.append(self.headers.get("Authorization", ""))
        payload = json.dumps({"choices": [{"message": {"content": REWRITTEN}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


@pytest.fixture
def chat_server():
    _ChatHandler.seen = []
    _ChatHandler.auth = []
    server = _QuietServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def rewrite_state(source: Path, destination: Path, server: str) -> dict:
    return state(
        [source],
        preset="hidden",
        flags=["-o", str(destination), "--rewrite", "humanize", "--rewrite-timeout", "10"],
        endpoint={"backend": "openai-compatible", "base_url": server, "model": "stub-model"},
    )


# =============================================================================
# compose_request
# =============================================================================


def test_preset_flag_lists_are_exact():
    """The frontend shows these verbatim; a silent change is a contract change."""
    assert {preset.key: list(preset.flags) for preset in PRESETS} == {
        "hidden": [],
        "hidden-aggressive": ["--nfkc", "--aggressive-homoglyphs"],
        "rewrite": ["--rewrite", "paraphrase"],
        "image": ["--degrade", "freq-dct"],
    }
    assert [preset.key for preset in PRESETS if preset.requires_endpoint] == ["rewrite"]


def test_presets_reach_the_request(tmp_path):
    hidden = compose_request(state(["a.txt"])).request
    assert hidden.nfkc is False and hidden.rewrite is None and hidden.degrade is None

    aggressive = compose_request(state(["a.txt"], preset="hidden-aggressive")).request
    assert aggressive.nfkc and aggressive.aggressive_homoglyphs

    rewrite = compose_request(state(["a.txt"], preset="rewrite"))
    assert rewrite.request.rewrite_strength == "paraphrase"
    assert "--rewrite paraphrase" in rewrite.command

    image = compose_request(state(["a.png"], preset="image")).request
    assert image.degrade == "freq-dct"


def test_a_bare_hidden_clean_is_a_bare_wm():
    composed = compose_request(state(["a.txt"]))
    assert composed.command == "wm a.txt"
    # json/quiet are forced for the bridge but never shown in the preview.
    assert composed.request.json and composed.request.quiet
    assert "--json" not in composed.command and "--quiet" not in composed.command


def test_a_later_flag_wins():
    composed = compose_request(
        state(["a.txt"], preset="rewrite", flags=["--rewrite", "humanize"])
    ).request
    assert composed.rewrite == "humanize"
    globbed = compose_request(state(["d"], flags=["--glob", "*.md", "--glob", "*.txt"])).request
    assert globbed.glob == "*.txt"


def test_a_path_that_looks_like_a_flag_is_still_a_path():
    composed = compose_request(state(["-weird.md"])).request
    assert composed.paths == (Path("-weird.md"),)


def test_the_endpoint_is_applied_only_with_a_rewrite():
    endpoint = {"backend": "ollama", "base_url": "http://127.0.0.1:11434", "model": "m"}
    plain = compose_request(state(["a.txt"], endpoint=endpoint))
    assert plain.request.rewrite_backend is None
    assert plain.request.rewrite_base_url is None
    assert "--rewrite-base-url" not in plain.command

    rewriting = compose_request(state(["a.txt"], preset="rewrite", endpoint=endpoint))
    assert rewriting.request.rewrite_backend == "ollama"
    assert rewriting.request.rewrite_model == "m"
    assert "--rewrite-base-url http://127.0.0.1:11434" in rewriting.command


def test_empty_endpoint_fields_leave_the_environment_in_charge():
    composed = compose_request(
        state(["a.txt"], preset="rewrite", endpoint={"backend": "", "base_url": None, "model": ""})
    ).request
    assert composed.rewrite_backend is None and composed.rewrite_model is None


def test_remote_egress_is_only_a_per_run_argument():
    endpoint = {"backend": "ollama", "base_url": "http://gpu.example.test:11434", "model": "m"}
    default = compose_request(state(["a.txt"], preset="rewrite", endpoint=endpoint))
    assert not default.request.rewrite_allow_remote
    allowed = compose_request(
        state(["a.txt"], preset="rewrite", endpoint=endpoint), allow_remote=True
    )
    assert allowed.request.rewrite_allow_remote is True
    assert "--rewrite-allow-remote" in allowed.command
    # Without a rewrite there is nothing to send, so nothing to allow.
    assert not compose_request(state(["a.txt"]), allow_remote=True).request.rewrite_allow_remote


def test_no_flag_or_environment_grants_egress_without_the_run_confirmation(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    typed = compose_request(state(["a.txt"], preset="rewrite", flags=["--rewrite-allow-remote"]))
    # An explicit False, so the environment's standing yes is overruled too.
    assert typed.request.rewrite_allow_remote is False
    assert "--rewrite-allow-remote" not in typed.command
    plain = compose_request(state(["a.txt"], flags=["--rewrite-allow-remote"]))
    assert plain.request.rewrite_allow_remote is None
    assert "--rewrite-allow-remote" not in plain.command


@pytest.mark.parametrize(
    "field",
    ["api_key", "allow_remote", "reasoning_effort", "rewrite_api_key"],
)
def test_the_endpoint_accepts_only_backend_url_and_model(field):
    with pytest.raises(BadRequest, match=r"unknown state\.endpoint field"):
        compose_request(state(["a.txt"], preset="rewrite", endpoint={field: "x"}))


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--no-such-flag"], "unrecognized arguments"),
        (["--rewrite", "bogus"], "invalid choice"),
        (["--timeout", "0"], "timeout"),
        (["--help"], "help"),
        # An abbreviation reaches the same help action.
        (["--he"], "help"),
    ],
)
def test_refused_flags_are_invalid_options_with_the_parser_message(flags, match, capsys):
    with pytest.raises(InvalidOptions, match=match):
        compose_request(state(["a.txt"], flags=flags))
    # argparse's own usage or help text would land in the frontend's log for nothing.
    captured = capsys.readouterr()
    assert "usage:" not in captured.out + captured.err


def test_no_paths_and_unknown_presets_are_invalid_options():
    with pytest.raises(InvalidOptions, match="required"):
        compose_request(state([]))
    with pytest.raises(InvalidOptions, match="unknown preset"):
        compose_request(state(["a.txt"], preset="nope"))
    with pytest.raises(InvalidOptions, match="rewrite backend"):
        compose_request(state(["a.txt"], preset="rewrite", endpoint={"backend": "print-prompt"}))


@pytest.mark.parametrize(
    "bad",
    [
        {"paths": "a.txt"},
        {"paths": ["a.txt"], "flags": [1]},
        {"paths": ["a.txt"], "preset": 3},
        {"paths": ["a.txt"], "endpoint": "ollama"},
        {"paths": ["a.txt"], "preset": "rewrite", "endpoint": {"model": 3}},
    ],
)
def test_a_malformed_state_is_a_bad_request(bad):
    with pytest.raises(BadRequest):
        compose_request(bad)


def test_a_composition_never_carries_the_key(monkeypatch):
    """The rewrite reads the key itself at run time (see the chat_server tests)."""
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", SENTINEL)
    composed = compose_request(state(["a.txt"], preset="rewrite"))
    assert composed.request.rewrite_api_key is None
    assert SENTINEL not in composed.command
    assert SENTINEL not in repr(composed)


# =============================================================================
# honesty labels, gates, estimates
# =============================================================================


def test_best_effort_layers_are_never_labelled_verifiable():
    assert result_class_for("B") == BEST_EFFORT
    assert result_class_for("V") == BEST_EFFORT
    assert result_class_for("synthid") == BEST_EFFORT
    assert result_class_for("perturb") == BEST_EFFORT
    assert result_class_for("A") == VERIFIABLE
    assert result_class_for("M") == VERIFIABLE
    assert result_class_for("something-new") == BEST_EFFORT


def test_preset_lookup_never_guesses():
    assert preset_for("no-such-preset") is None
    assert preset_for(None) is None
    assert len({preset.key for preset in PRESETS}) == len(PRESETS)


def test_no_preset_arms_a_confirmation_gate():
    """A one-click preset must not reach for a flag with its own gate."""
    forbidden = {"--in-place", "--strip-semantic-format", "--dry-run", "--rewrite-allow-remote"}
    for preset in PRESETS:
        assert not set(preset.flags) & forbidden, preset.key
        request = compose_request(state(["a.txt"], preset=preset.key)).request
        kinds = {gate["kind"] for gate in confirm_gates(request, 1, 1)}
        assert not kinds & {"in_place", "semantic", "remote"}, preset.key


def test_pixel_and_perturbation_work_is_never_badged_verifiable():
    for request in (
        CleanRequest(degrade="freq-dct"),
        CleanRequest(morpho="grid"),
        CleanRequest(remove_synthid=True),
        CleanRequest(visible_box=(0, 0, 1, 1)),
    ):
        assert result_class_for(layer_for_result(request, "image")) == BEST_EFFORT
    assert result_class_for(layer_for_result(CleanRequest(), "image")) == VERIFIABLE
    assert result_class_for(layer_for_result(CleanRequest(char_perturb=True), "text")) == (
        BEST_EFFORT
    )
    assert result_class_for(layer_for_result(CleanRequest(nfkc=True), "text")) == VERIFIABLE
    assert layer_for_result(CleanRequest(rewrite="paraphrase"), "container") == "M"


def test_the_plan_layer_follows_the_work_the_selected_files_get():
    rewrite = CleanRequest(rewrite="paraphrase")
    assert plan_layer(rewrite) == "B"
    # A rewrite skips a file that is not text, so it cannot weaken that plan.
    assert plan_layer(rewrite, ["container", "container"]) == "A"
    assert plan_layer(rewrite, ["container", "text"]) == "B"
    assert plan_layer(CleanRequest(char_perturb=True)) == "V"
    assert plan_layer(CleanRequest(remove_synthid=True), ["image"]) == "V"
    assert plan_layer(CleanRequest(remove_synthid=True), ["text"]) == "A"
    assert plan_layer(CleanRequest()) == "A"


def test_the_cost_estimate_and_gate():
    assert estimate_rewrite_seconds(CleanRequest(), 10) == 0.0
    request = CleanRequest(rewrite="humanize", rewrite_candidates=3, rewrite_timeout=10.0)
    assert estimate_rewrite_seconds(request, 4) == 4 * 3 * 10.0
    tsapa = CleanRequest(
        rewrite="tsapa", tsapa_generations=3, tsapa_population=4, rewrite_timeout=10.0
    )
    assert estimate_rewrite_seconds(tsapa, 2) == 2 * 3 * 4 * 10.0
    # The default timeout is RewritePlan's own.
    assert estimate_rewrite_seconds(CleanRequest(rewrite="humanize"), 1) == 120.0
    assert estimate_rewrite_seconds(CleanRequest(rewrite="tsapa"), 1) > COST_CONFIRM_SECONDS

    single = CleanRequest(rewrite="humanize", rewrite_candidates=1, rewrite_timeout=120.0)
    assert gate_kinds(single, 1, 1) == []
    assert gate_kinds(single, 10, 10) == ["cost"]
    assert gate_kinds(CleanRequest(), 1000, 1000) == []


def gate_kinds(request: CleanRequest, file_count: int = 1, text_count: int = 1) -> list[str]:
    return [gate["kind"] for gate in confirm_gates(request, file_count, text_count)]


def test_gates_cover_remote_in_place_semantic_and_cost():
    remote = CleanRequest(rewrite="humanize", rewrite_base_url="http://gpu.example.test:8000")
    assert gate_kinds(remote) == ["remote"]
    [gate] = confirm_gates(remote, 2, 2)
    assert gate["message"] == (
        "Rewrite base URL host is 'gpu.example.test' (not localhost); content will leave "
        "this machine. The full text of 2 files will be sent to gpu.example.test."
    )
    # A standing opt-in does not remove the per-run question.
    assert gate_kinds(tui_core.replace(remote, rewrite_allow_remote=True)) == ["remote"]
    local = CleanRequest(rewrite="humanize", rewrite_base_url="http://127.0.0.1:8000")
    assert gate_kinds(local) == []
    everything = CleanRequest(in_place=True, strip_semantic_format=True, rewrite="humanize")
    assert gate_kinds(everything, 10, 10) == ["in_place", "semantic", "cost"]


def test_files_a_rewrite_skips_neither_leave_the_machine_nor_cost_time():
    remote = CleanRequest(rewrite="humanize", rewrite_base_url="http://gpu.example.test:8000")
    assert gate_kinds(remote, 1, 0) == []
    costly = CleanRequest(rewrite="humanize", in_place=True)
    assert gate_kinds(costly, 10, 0) == ["in_place"]


def test_a_malformed_endpoint_is_a_warning_not_a_gate():
    bad = CleanRequest(rewrite="humanize", rewrite_base_url="file:///etc/passwd")
    assert gate_kinds(bad) == []
    [warning] = plan_warnings(bad, None)
    assert warning.startswith("Layer B endpoint refused") and "http" in warning
    # Nothing is rewritten, so nothing is refused.
    assert not any("refused" in item for item in plan_warnings(bad, ["container"]))
    assert plan_warnings(CleanRequest(), None) == []


def test_a_rewrite_names_the_files_it_skips():
    rewrite = CleanRequest(rewrite="paraphrase")
    assert plan_warnings(rewrite, ["text"]) == []
    [warning] = plan_warnings(rewrite, ["container", "text", "container"])
    assert warning == (
        "Layer B rewrite (paraphrase) will skip 2 files that are not plain text "
        "(Markdown and other documents). Add --as text to force it."
    )
    [both] = plan_warnings(CleanRequest(nfkc=True, char_perturb=True), ["image"])
    assert both == (
        "Character perturbation and NFKC normalization will skip 1 file that is not "
        "plain text (images). Add --as text to force them."
    )


# =============================================================================
# findings and diffs
# =============================================================================


def test_a_container_finding_separates_hidden_carriers_from_metadata(tmp_path):
    report = {
        "kind": "container",
        "findings": ["layer-a: U+200B x3 (zwj_family)", "frontmatter key: generator"],
        "layer_a_hits": [{"codepoint": "U+200B", "count": 3, "kind": "zwj_family"}],
        "suspicious": True,
    }
    finding = finding_from_report(report, tmp_path / "a.md", "a.md")
    assert finding["suspicious"] is True
    assert finding["counts"] == {"hidden": 3, "metadata": 1}
    assert finding["lines"] == ["3 zero-width carriers (U+200B)", "Frontmatter key: generator"]


def test_a_markdown_file_with_hidden_carriers_inspects_as_suspicious(tmp_path):
    from inspect_file import inspect_asset

    source = tmp_path / "a.md"
    source.write_text(f"# title\n\nbody{ZWSP} text\n", encoding="utf-8")
    finding = finding_from_report(inspect_asset(source), source, "a.md")
    assert finding["suspicious"] is True and finding["counts"]["hidden"] == 1


def test_finding_lines_are_capped_at_twelve(tmp_path):
    hits = [{"codepoint": f"U+{n:04X}", "count": 1, "kind": "bidi"} for n in range(40)]
    finding = finding_from_report({"kind": "text", "hits": hits}, tmp_path / "a", "a")
    assert len(finding["lines"]) == 12
    assert finding["lines"][-1] == "29 more not shown."
    assert finding["counts"]["hidden"] == 40


def _payload(*steps: act.Action) -> dict[str, Any]:
    return {"kind": "container", **act.report_actions(steps), "exit_code": 0}


def test_result_lines_are_sentences_built_from_the_payload():
    # A real `.md` payload: the no-op step never reaches the operator.
    markdown = _payload(
        act.nothing_removed("markdown", "no AI frontmatter keys or embedded data URIs removed"),
        act.drop_frontmatter_key("generator"),
        act.layer_a_text(2, 0),
    )
    assert result_lines(markdown) == [
        "Dropped frontmatter key generator.",
        "Removed 2 hidden characters.",
    ]
    text = {"kind": "text", "stats": {"removed_count": 1, "replaced_count": 3}, "exit_code": 0}
    assert result_lines(text) == [
        "Removed 1 hidden character.",
        "Replaced 3 look-alike characters.",
    ]
    assert result_lines({"kind": "text", "stats": {"removed_count": 0}}) == []
    assert result_lines({"error": "boom"}) == ["Failed: boom"]


def test_result_lines_name_the_layer_b_rewrite():
    # stats["tsapa"] holds rewrite()'s info, or only the search record after TSAPA.
    info = {
        "backend": "ollama",
        "strength": "paraphrase",
        "model": "qwen3:14b",
        "mode": "rewritten",
    }
    paraphrase = {"kind": "text", "stats": {"removed_count": 0, "tsapa": info}}
    assert result_lines(paraphrase) == [
        "Rewrote the text with a Layer B paraphrase rewrite by qwen3:14b. "
        "Best-effort, no detector guarantee."
    ]
    record = {"chunks": 1, "generations": 3, "population": 4, "stats": {}}
    tsapa = {"kind": "text", "stats": {"removed_count": 0, "tsapa": record}}
    assert result_lines(tsapa) == [
        "Rewrote the text with a Layer B TSAPA search over 3 generations. "
        "Best-effort, no detector guarantee."
    ]


def test_result_lines_read_codes_never_the_human_action_lines():
    # The human lines are for the CLI.  A payload whose lines say one thing and
    # whose codes say another is rendered from the codes.
    payload = _payload(act.drop_part("docProps/custom.xml"))
    payload["actions"] = ["layer A text: removed=9 replaced=9"]
    assert result_lines(payload) == ["Dropped part docProps/custom.xml."]
    assert result_lines({"kind": "container", "actions": ["drop part x"]}) == []


@pytest.mark.parametrize(
    ("step", "line"),
    [
        (act.drop_data_ai_attributes(1), "Dropped 1 data-ai attribute."),
        (
            act.prune_relationships("_rels/.rels", 2),
            "Pruned 2 dangling relationships in _rels/.rels.",
        ),
        (
            act.scrub_field("docProps/core.xml", "dc:creator"),
            "Cleared dc:creator in docProps/core.xml.",
        ),
        (act.zero_uuid_payload(299), "Zeroed the XMP uuid box payload (299 bytes)."),
        (
            act.neutralize_box("jumb", 22),
            "Neutralized the 'jumb' C2PA box (22 bytes zeroed).",
        ),
        (act.drop_svg_metadata(1), "Dropped 1 SVG metadata block."),
        (act.prune_opf_manifest(3), "Pruned 3 OPF manifest entries."),
        (act.prune_odf_manifest(1), "Pruned 1 ODF manifest entry."),
        (
            act.clean_embedded_media(
                "ppt/media/image1.jpeg",
                [act.drop_jpeg_segment("APP11", reason="C2PA/JUMBF"), act.preserve_jpeg_scan()],
            ),
            "Cleaned embedded image ppt/media/image1.jpeg: "
            "dropped the JPEG APP11 segment (C2PA/JUMBF).",
        ),
        (
            act.clean_part("OEBPS/content.opf", [act.scrub_opf_field("dc:creator")]),
            "Cleared dc:creator (AI vendor name) in OEBPS/content.opf.",
        ),
        (
            act.drop_part("META-INF/x.xml", markers=True),
            "Dropped part META-INF/x.xml (AI/C2PA markers).",
        ),
        # A tool name never opens a sentence, so it is never re-cased.
        (act.exiftool_run(1), "Ran exiftool, which exited with code 1."),
        (
            act.exiftool_failed(7, "bad file"),
            "Could not use exiftool (exit code 7): bad file.",
        ),
        (
            act.tool_failed("exiftool", "x", returncode=1, fallback="pypdf"),
            "Could not use exiftool (exit code 1); trying pypdf instead.",
        ),
        (act.drop_pdf_page_metadata(2), "Dropped page metadata from 2 pages."),
        (act.drop_pdf_docinfo(), "Dropped the PDF document info dictionary."),
    ],
)
def test_pipeline_actions_read_as_done_without_renaming_anything(step, line):
    assert result_lines(_payload(step)) == [line]


@pytest.mark.parametrize(
    "step",
    [
        act.nothing_removed(
            "png", "no PNG metadata chunks removed (already clean or none matched)"
        ),
        act.preserve_jpeg_scan(),
        act.keep_bmp_trailer(),
        act.c2patool_hint(),
        act.layer_a_text(0, 0),
    ],
)
def test_steps_that_change_nothing_say_nothing(step):
    assert result_lines(_payload(step)) == []


def test_every_action_code_has_a_sentence():
    assert set(tui_core._ACTION_PHRASES) == set(act.ActionCode)


def test_a_dry_run_describes_each_planned_step():
    steps = [
        act.plan_localize("box:1,2,3,4"),
        act.plan_refine_mask(3),
        act.plan_inpaint("texture"),
        act.plan_strip_metadata(),
        act.plan_publish("/o/a.mask.pgm", "/o/a.png"),
    ]
    assert result_lines(_payload(*steps)) == [
        "Would find the visible mark from box:1,2,3,4.",
        "Would fill holes and dilate the mask by 3 px.",
        "Would inpaint with the texture backend.",
        "Would strip the requested metadata.",
        "Would write the mask to /o/a.mask.pgm and the image to /o/a.png.",
    ]


def test_finding_lines_capitalise_words_but_never_paths_keys_or_acronyms(tmp_path):
    findings = [
        "svg <metadata> present",
        "docProps/core.xml: ai:Claude",
        "ai:Anthropic",
        "frontmatter key: generator",
    ]
    finding = finding_from_report({"kind": "container", "findings": findings}, tmp_path / "a", "a")
    assert finding["lines"] == [
        "SVG <metadata> present",
        "docProps/core.xml: ai:Claude",
        "ai:Anthropic",
        "Frontmatter key: generator",
    ]


def test_history_summaries_are_sentences_labelled_by_result_class():
    assert history_summary(1, 0, "A") == "1 file, 0 errors. Verifiable."
    assert history_summary(3, 1, "B", cancelled=True) == "3 files, 1 error, cancelled. Best-effort."


def test_the_diff_shows_invisible_characters_and_is_capped():
    diff = unified_diff(f"a{ZWSP}b\n", "ab\n", "x.txt")
    assert diff is not None and "<U+200B>" in diff
    # A variation selector is a mark, not a format character, and just as invisible.
    selector = unified_diff("a\ufe0fb\n", "ab\n", "x.txt")
    assert selector is not None and "<U+FE0F>" in selector
    long_before = "\n".join(f"line {n}" for n in range(500))
    long_after = "\n".join(f"LINE {n}" for n in range(500))
    capped = unified_diff(long_before, long_after, "x.txt")
    assert capped is not None
    assert len(capped.splitlines()) == 200
    assert "truncated" in capped.splitlines()[-1]
    assert unified_diff("same", "same", "x") is None
    assert unified_diff(None, "x", "x") is None


# =============================================================================
# reveal: hidden characters in place
# =============================================================================

RLO = "\u202e"


def _reveal(path: Path) -> list[dict]:
    from inspect_file import inspect_asset

    return finding_from_report(inspect_asset(path), path, path.name)["reveal"]


def test_reveal_maps_offsets_to_the_right_columns(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text(f"Hello{ZWSP} world\nsecond {RLO}line\n", encoding="utf-8")
    assert _reveal(source) == [
        {"line": 1, "text": "Hello\u25c6 world", "marks": [[5, 6, "ZWSP"]]},
        {"line": 2, "text": "second \u25c6line", "marks": [[7, 8, "RLO"]]},
    ]


def test_reveal_merges_marks_on_one_line_and_reads_markdown(tmp_path):
    source = tmp_path / "a.md"
    source.write_text(f"---\ntitle: x\n---\na{ZWSP}b{RLO}c\td\n", encoding="utf-8")
    assert _reveal(source) == [
        {
            "line": 4,
            "text": "a\u25c6b\u25c6c d",
            "marks": [[1, 2, "ZWSP"], [3, 4, "RLO"]],
        }
    ]


def test_reveal_trims_a_long_line_around_its_first_mark(tmp_path):
    source = tmp_path / "long.txt"
    source.write_text("x" * 300 + ZWSP + "y" * 300 + "\n", encoding="utf-8")
    [excerpt] = _reveal(source)
    text = excerpt["text"]
    assert len(text) <= 100
    assert text.startswith("\u2026") and text.endswith("\u2026")
    [[start, end, name]] = excerpt["marks"]
    assert name == "ZWSP" and text[start:end] == "\u25c6"
    assert 40 <= start <= 60


def test_reveal_is_capped_at_six_excerpts(tmp_path):
    source = tmp_path / "many.txt"
    source.write_text("".join(f"line{ZWSP}{n}\n" for n in range(10)), encoding="utf-8")
    excerpts = _reveal(source)
    assert [item["line"] for item in excerpts] == [1, 2, 3, 4, 5, 6]


def test_a_binary_or_image_asset_reveals_nothing(tmp_path):
    image = ROOT / "tests" / "fixtures" / "sample_c2pa.avif"
    assert _reveal(image) == []
    blob = tmp_path / "blob.txt"
    blob.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    assert finding_from_report({"kind": "text", "hits": []}, blob, "blob.txt")["reveal"] == []


def test_informational_notes_are_neither_counted_nor_shown_as_marks(tmp_path):
    from inspect_file import inspect_asset

    image = ROOT / "tests" / "fixtures" / "sample_c2pa.heic"
    finding = finding_from_report(inspect_asset(image), image, "sample_c2pa.heic")
    assert finding["counts"] == {"hidden": 0, "metadata": 1}
    assert finding["lines"] == [
        "XMP uuid box @ 83: digitalSourceType, trainedAlgorithmicMedia, algorithmicMedia"
    ]


def test_a_cleaned_heif_reports_no_metadata_left(tmp_path):
    source = tmp_path / "sample_c2pa.avif"
    source.write_bytes((ROOT / "tests" / "fixtures" / "sample_c2pa.avif").read_bytes())
    bridge, sink = make_bridge()
    frames = call(bridge, sink, "clean", {"state": state([source])})
    [done] = [frame["data"] for frame in frames if frame.get("event") == "file_done"]
    assert done["before"]["metadata"] == 2 and done["before"]["c2pa"] == 1
    assert done["after"] == {"hidden": 0, "metadata": 0}
    assert done["lines"] == [
        "Neutralized the 'jumb' C2PA box (22 bytes zeroed).",
        "Neutralized the 'jumb' C2PA box (32 bytes zeroed).",
        "Zeroed the XMP uuid box payload (299 bytes).",
    ]


def test_short_names_fall_back_to_hex():
    assert tui_core.short_name(0x200B) == "ZWSP"
    assert tui_core.short_name(0xE0041) == "TAG"
    assert tui_core.short_name(0xFE0F) == "VS16"
    assert tui_core.short_name(0x2028) == "U+2028"


# =============================================================================
# settings
# =============================================================================


def test_the_saved_setup_has_nowhere_to_put_a_key_or_a_standing_opt_in():
    names = set(TuiSettings.__dataclass_fields__)
    assert names == {"preset", "rewrite_backend", "rewrite_base_url", "rewrite_model"}


def test_settings_round_trip(tmp_path):
    target = tmp_path / "tui.json"
    settings = TuiSettings(
        preset="rewrite",
        rewrite_backend="ollama",
        rewrite_base_url="http://127.0.0.1:11434",
        rewrite_model="qwen3",
    )
    save_settings(settings, target)
    assert load_settings(target) == settings
    assert settings.to_wire() == {
        "preset": "rewrite",
        "endpoint": {"backend": "ollama", "base_url": "http://127.0.0.1:11434", "model": "qwen3"},
    }


def test_a_settings_file_with_retired_keys_loads_and_sheds_them(tmp_path):
    target = tmp_path / "tui.json"
    target.write_text(
        json.dumps(
            {
                "preset": "rewrite",
                "rewrite_allow_remote": True,
                "rewrite_backend": "openai-compatible",
                "rewrite_base_url": "http://127.0.0.1:1234",
                "rewrite_model": "m",
                "rewrite_reasoning_effort": "low",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    loaded = load_settings(target)
    assert loaded == TuiSettings("rewrite", "openai-compatible", "http://127.0.0.1:1234", "m")
    save_settings(apply_settings_update(loaded, {"preset": "hidden"}), target)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == {
        "preset": "hidden",
        "rewrite_backend": "openai-compatible",
        "rewrite_base_url": "http://127.0.0.1:1234",
        "rewrite_model": "m",
    }


def test_an_old_file_with_a_dead_backend_or_preset_is_read_soft(tmp_path):
    """hello echoes settings into every state, so a bad value would fail every plan."""
    target = tmp_path / "tui.json"
    target.write_text(
        '{"rewrite_backend": "print-prompt", "preset": "gone", "rewrite_model": "m"}',
        encoding="utf-8",
    )
    loaded = load_settings(target)
    assert loaded == TuiSettings(rewrite_model="m")
    compose_request(state(["a.txt"], endpoint=loaded.to_wire()["endpoint"]))


def test_an_unreadable_setup_file_is_not_fatal(tmp_path):
    broken = tmp_path / "tui.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_settings(broken) == TuiSettings()
    assert load_settings(tmp_path / "absent.json") == TuiSettings()
    broken.write_text('{"rewrite_model": {"host": "x"}, "preset": 3, "rewrite_base_url": "u"}')
    assert load_settings(broken) == TuiSettings(rewrite_base_url="u")
    broken.write_text("[1, 2]")
    assert load_settings(broken) == TuiSettings()


@pytest.mark.parametrize(
    "update",
    [
        {"api_key": "x"},
        {"rewrite_api_key": "x"},
        {"token": "x"},
        {"secret": "x"},
        {"rewrite_model": "m"},
        {"rewrite_allow_remote": True},
        {"endpoint": {"backend": "ollama", "api_key": "x"}},
        {"endpoint": {"allow_remote": True}},
        {"endpoint": {"reasoning_effort": "low"}},
        {"endpoint": {"model": 3}},
        {"endpoint": {"backend": "print-prompt"}},
        {"endpoint": "ollama"},
        {"preset": "nope"},
        {"preset": 1},
        "not an object",
    ],
)
def test_save_settings_refuses_anything_off_the_allow_list(update):
    with pytest.raises(BadRequest):
        apply_settings_update(TuiSettings(), update)


def test_a_settings_update_merges_and_clears():
    current = TuiSettings("rewrite", "ollama", "http://127.0.0.1:11434", "m")
    assert apply_settings_update(current, {"preset": "image"}).rewrite_model == "m"
    cleared = apply_settings_update(current, {"endpoint": None})
    assert cleared == TuiSettings(preset="rewrite")
    partial = apply_settings_update(current, {"endpoint": {"backend": "ollama", "model": ""}})
    assert partial.rewrite_model is None and partial.rewrite_base_url is None


def test_save_settings_over_the_bridge(tmp_path):
    bridge, sink = make_bridge()
    target = Path(os.environ["WATERMARKS_TUI_SETTINGS"])
    refused = response(call(bridge, sink, "save_settings", {"settings": {"api_key": SENTINEL}}))
    assert refused["error"]["code"] == "bad_request"
    assert not target.exists()

    settings = {
        "preset": "rewrite",
        "endpoint": {"backend": "ollama", "base_url": "http://127.0.0.1:11434", "model": "m"},
    }
    saved = response(call(bridge, sink, "save_settings", {"settings": settings}, rid=2))
    assert saved["result"] == {"path": str(target)}
    hello = response(call(bridge, sink, "hello", rid=3))["result"]
    assert hello["settings"] == settings
    assert hello["onboard"] is False


# =============================================================================
# onboarding and the hello handshake
# =============================================================================


@pytest.mark.parametrize(
    ("exists", "force", "skip", "expected"),
    [
        (False, False, False, True),
        (True, False, False, False),
        (True, True, False, True),
        (False, False, True, False),
    ],
)
def test_setup_opens_only_on_a_first_run_unless_asked(exists, force, skip, expected):
    assert should_onboard(settings_exist=exists, force=force, skip=skip) is expected


def test_setup_and_no_setup_cannot_be_combined():
    with pytest.raises(SystemExit):
        launcher.build_parser().parse_args(["--setup", "--no-setup"])


def test_hello_reports_the_contract_fields_and_the_launch_arguments():
    bridge, sink = make_bridge(argv=["docs", "--recursive", "--glob", "*.md", "--setup"])
    result = response(call(bridge, sink, "hello"))["result"]
    assert set(result) == {
        "version",
        "presets",
        "settings",
        "settings_path",
        "onboard",
        "initial",
        "backends",
    }
    assert result["onboard"] is True
    assert result["initial"] == {"paths": ["docs"], "flags": ["--recursive", "--glob", "*.md"]}
    assert result["backends"] == ["ollama", "openai-compatible"]
    assert result["settings"] == {
        "preset": None,
        "endpoint": {"backend": None, "base_url": None, "model": None},
    }


def test_a_hand_set_bad_argv_falls_back_to_defaults():
    bridge, sink = make_bridge()
    bridge.launch = tui_bridge._parse_launch_argv('["--setup", "--no-setup"]')
    result = response(call(bridge, sink, "hello"))["result"]
    assert result["initial"] == {"paths": ["."], "flags": []}


# =============================================================================
# endpoint detection
# =============================================================================


def test_every_built_in_candidate_is_loopback():
    for _backend, base_url, _label in LOCAL_ENDPOINT_CANDIDATES:
        assert base_url.startswith(("http://127.0.0.1:", "http://localhost:"))


def test_the_environment_endpoint_is_asked_first_and_not_twice():
    env = {
        "WATERMARKS_REWRITE_BACKEND": "ollama",
        "WATERMARKS_REWRITE_BASE_URL": "http://127.0.0.1:11434/",
    }
    candidates = endpoint_candidates(env)
    assert candidates[0] == ("ollama", "http://127.0.0.1:11434", "from WATERMARKS_REWRITE_*")
    assert [c[:2] for c in candidates].count(("ollama", "http://127.0.0.1:11434")) == 1
    assert len(candidates) == len(LOCAL_ENDPOINT_CANDIDATES)


def test_an_unknown_environment_backend_is_not_probed():
    env = {"WATERMARKS_REWRITE_BACKEND": "print-prompt", "WATERMARKS_REWRITE_BASE_URL": "x"}
    assert endpoint_candidates(env) == list(LOCAL_ENDPOINT_CANDIDATES)


def test_detection_is_parallel_ordered_and_never_allows_remote(monkeypatch):
    candidates = list(LOCAL_ENDPOINT_CANDIDATES)
    # Every probe waits for all the others: a sequential scan deadlocks here.
    barrier = threading.Barrier(len(candidates), timeout=5)
    calls = []

    def probe(backend, base_url, *, allow_remote, timeout):
        calls.append(allow_remote)
        barrier.wait()
        return BackendProbe(backend=backend, base_url=base_url, reachable=True, models=("m",))

    monkeypatch.setattr(tui_core, "probe_backend", probe)
    found = detect_local_endpoints(candidates)
    assert [item["base_url"] for item in found] == [c[1] for c in candidates]
    assert calls == [False] * len(candidates)
    assert found[0] == {
        "label": "Ollama",
        "backend": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "reachable": True,
        "models": ["m"],
        "error": None,
    }


def test_a_remote_environment_endpoint_is_refused_without_a_request(monkeypatch):
    import layer_b_discovery

    def no_network(*args, **kwargs):
        raise AssertionError("a denied host was contacted")

    monkeypatch.setattr(layer_b_discovery, "get_json", no_network)
    [found] = detect_local_endpoints(
        [("ollama", "http://gpu.example.test:11434", "from WATERMARKS_REWRITE_*")]
    )
    assert found["reachable"] is False
    assert "loopback" in (found["error"] or "")


def test_detect_endpoints_over_the_bridge(monkeypatch):
    monkeypatch.setattr(
        tui_core,
        "probe_backend",
        lambda backend, base_url, **_: BackendProbe(backend, base_url, False, (), "down"),
    )
    bridge, sink = make_bridge()
    endpoints = response(call(bridge, sink, "detect_endpoints"))["result"]["endpoints"]
    assert [item["base_url"] for item in endpoints] == [c[1] for c in LOCAL_ENDPOINT_CANDIDATES]
    assert all(item["reachable"] is False and item["error"] == "down" for item in endpoints)


def test_there_is_no_probe_method():
    bridge, sink = make_bridge()
    refused = response(call(bridge, sink, "probe", {"backend": "ollama", "base_url": "x"}))
    assert refused["error"]["code"] == "bad_request"


# =============================================================================
# environment checks
# =============================================================================


def test_checks_say_whether_a_key_is_set_and_never_what_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tui_core,
        "check_optional",
        lambda extra: type("A", (), {"available": False, "hint": f"install {extra}"})(),
    )
    checks = environment_checks({"WATERMARKS_REWRITE_API_KEY": SENTINEL}, tmp_path / "t.json")
    rendered = json.dumps([vars(check) for check in checks])
    assert SENTINEL not in rendered
    key = next(check for check in checks if check.name == "Layer B API key")
    assert key.state == "Set" and key.good
    for check in checks:
        if check.name.startswith("Extra: "):
            assert check.fix.startswith("pip install") and not check.good and check.optional
    assert set(vars(checks[0])) == {"name", "state", "good", "detail", "fix", "optional"}


def test_terminal_app_is_warned_about_clipboard(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tui_core,
        "check_optional",
        lambda extra: type("A", (), {"available": True, "hint": ""})(),
    )
    checks = environment_checks({"TERM_PROGRAM": "Apple_Terminal"}, tmp_path / "t.json")
    assert any(check.name == "Clipboard" for check in checks)


# =============================================================================
# plan
# =============================================================================


def test_plan_lists_the_files_the_cli_would_touch(tmp_path):
    (tmp_path / "a.md").write_text(ZWSP, encoding="utf-8")
    (tmp_path / "b.txt").write_text(ZWSP, encoding="utf-8")
    (tmp_path / "b.cleaned.txt").write_text("done", encoding="utf-8")
    bridge, sink = make_bridge()
    plan = response(call(bridge, sink, "plan", {"state": state([tmp_path])}))["result"]
    assert plan["ok"] is True and plan["error"] is None
    assert sorted(Path(item["path"]).name for item in plan["files"]) == ["a.md", "b.txt"]
    assert plan["files_total"] == 2
    assert plan["layer"] == "A" and plan["result_class"] == VERIFIABLE
    assert plan["confirm"] == [] and plan["preflight_error"] is None
    assert plan["output"].endswith("Originals are never touched.")
    assert not list(tmp_path.glob("*.cleaned.md")), "plan must never write"


def test_plan_caps_its_listing_but_preflights_every_file(tmp_path, monkeypatch):
    for n in range(5):
        (tmp_path / f"f{n}.txt").write_text("x", encoding="utf-8")
    sizes: list[int] = []
    real = tui_bridge.preflight_work

    def spy(request, selection):
        sizes.append(len(selection.items))
        return real(request, selection)

    monkeypatch.setattr(tui_bridge, "preflight_work", spy)
    monkeypatch.setattr(tui_bridge, "PLAN_FILES_LIMIT", 3)
    bridge, sink = make_bridge()
    plan = response(call(bridge, sink, "plan", {"state": state([tmp_path])}))["result"]
    assert plan["files_total"] == 5 and len(plan["files"]) == 3
    assert sizes == [5]


def test_plan_labels_a_rewrite_over_markdown_alone_by_what_actually_runs(tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(f"# {name}{ZWSP}\n", encoding="utf-8")
    remote = {"backend": "ollama", "base_url": "http://gpu.example.test:11434", "model": "m"}
    bridge, sink = make_bridge()
    plan = response(
        call(bridge, sink, "plan", {"state": state([tmp_path], preset="rewrite", endpoint=remote)})
    )["result"]
    assert plan["layer"] == "A" and plan["result_class"] == VERIFIABLE
    # Nothing is rewritten, so nothing leaves the machine and nothing costs model time.
    assert plan["confirm"] == [] and plan["estimate_seconds"] == 0
    assert plan["warnings"] == [
        "Layer B rewrite (paraphrase) will skip 2 files that are not plain text "
        "(Markdown and other documents). Add --as text to force it."
    ]

    (tmp_path / "c.txt").write_text("plain\n", encoding="utf-8")
    mixed = response(
        call(
            bridge,
            sink,
            "plan",
            {"state": state([tmp_path], preset="rewrite", endpoint=remote)},
            rid=2,
        )
    )["result"]
    assert mixed["layer"] == "B" and [gate["kind"] for gate in mixed["confirm"]] == ["remote"]
    assert "1 file" in mixed["confirm"][0]["message"]


def test_plan_reports_refused_flags_and_still_shows_them(tmp_path):
    bridge, sink = make_bridge()
    plan = response(
        call(bridge, sink, "plan", {"state": state(["a.txt"], flags=["--bogus-flag"])})
    )["result"]
    assert plan["ok"] is False
    assert "--bogus-flag" in plan["error"]
    assert plan["command"] == "wm a.txt --bogus-flag"


def test_plan_reports_a_selection_error_and_a_preflight_error(tmp_path):
    bridge, sink = make_bridge()
    missing = response(call(bridge, sink, "plan", {"state": state([tmp_path / "gone"])}))
    assert missing["result"]["ok"] is True
    assert "not a regular file" in missing["result"]["discover_error"]

    source = tmp_path / "a.txt"
    source.write_text("x", encoding="utf-8")
    # A rewrite with no endpoint anywhere is refused by plan_work before any write.
    no_model = response(
        call(bridge, sink, "plan", {"state": state([source], preset="rewrite")}, rid=2)
    )["result"]
    assert no_model["ok"] is True
    assert "requires a live backend" in no_model["preflight_error"]
    assert no_model["layer"] == "B" and no_model["result_class"] == BEST_EFFORT


def test_plan_shows_the_remote_gate_and_clean_stops_at_it(tmp_path, monkeypatch):
    source = tmp_path / "a.txt"
    source.write_text(ZWSP, encoding="utf-8")
    endpoint = {"backend": "ollama", "base_url": "http://gpu.example.test:11434", "model": "m"}
    remote = state([source], preset="rewrite", flags=["--rewrite-allow-remote"], endpoint=endpoint)
    bridge, sink = make_bridge()
    plan = response(call(bridge, sink, "plan", {"state": remote}))["result"]
    assert [gate["kind"] for gate in plan["confirm"]] == ["remote"]
    assert "gpu.example.test" in plan["confirm"][0]["message"]

    refused = response(call(bridge, sink, "clean", {"state": remote}, rid=2))
    assert refused["error"]["code"] == "needs_confirm", refused
    assert [gate["kind"] for gate in refused["error"]["data"]["confirm"]] == ["remote"]
    assert not (tmp_path / "a.cleaned.txt").exists()

    seen: list[CleanRequest] = []

    def fake_run(path, output, request, plan, *, on_token=None):
        seen.append(request)
        return {"kind": "text", "output": str(output), "exit_code": 0}

    monkeypatch.setattr(tui_bridge, "run_clean_item", fake_run)
    done = response(call(bridge, sink, "clean", {"state": remote, "confirmed": ["remote"]}, rid=3))[
        "result"
    ]
    assert seen[0].rewrite_allow_remote is True
    assert "--rewrite-allow-remote" in done["command"]
    # The yes was for that run only: the next preview asks again.
    again = response(call(bridge, sink, "plan", {"state": remote}, rid=4))["result"]
    assert [gate["kind"] for gate in again["confirm"]] == ["remote"]
    assert not Path(os.environ["WATERMARKS_TUI_SETTINGS"]).exists()


def test_clean_refuses_until_every_gate_is_confirmed(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text(f"x{ZWSP}\n", encoding="utf-8")
    gated = state([source], flags=["--in-place", "--strip-semantic-format"])
    bridge, sink = make_bridge()
    partial = response(call(bridge, sink, "clean", {"state": gated, "confirmed": ["in_place"]}))
    assert partial["error"]["code"] == "needs_confirm"
    assert [g["kind"] for g in partial["error"]["data"]["confirm"]] == ["semantic"]
    assert source.read_text(encoding="utf-8") == f"x{ZWSP}\n"

    done = response(
        call(bridge, sink, "clean", {"state": gated, "confirmed": ["in_place", "semantic"]}, rid=2)
    )
    assert done["result"]["errors"] == 0
    assert ZWSP not in source.read_text(encoding="utf-8")


def test_clean_reports_a_preflight_refusal_before_asking_for_confirmation(tmp_path):
    """Gates count the files preflight resolved, so preflight runs first."""
    source = tmp_path / "a.txt"
    source.write_text(ZWSP, encoding="utf-8")
    (tmp_path / "a.txt.bak").write_text("earlier backup", encoding="utf-8")
    bridge, sink = make_bridge()
    refused = response(
        call(bridge, sink, "clean", {"state": state([source], flags=["--in-place"])})
    )
    assert refused["error"]["code"] == "invalid_options"
    assert "backup already exists" in refused["error"]["message"]


def test_bad_requests_are_answered_not_fatal():
    bridge, sink = make_bridge()
    bridge.handle_line("{not json")
    bridge.handle_line("[1, 2]")
    call(bridge, sink, "no_such_method", rid=5)
    call(bridge, sink, "plan", ["not", "an", "object"], rid=6)
    call(bridge, sink, "plan", {"state": "x"}, rid=7)
    errors = [frame["error"]["code"] for frame in sink.frames() if "error" in frame]
    assert errors == ["bad_request"] * 5


# =============================================================================
# busy, cancel, and a pipeline that exits
# =============================================================================


def _two_files(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("a.txt", "b.txt"):
        (folder / name).write_text(f"{name}{ZWSP}\n", encoding="utf-8")
    return folder


def test_a_second_clean_is_busy_and_cancel_stops_after_the_current_file(tmp_path, monkeypatch):
    folder = _two_files(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_run(path, output, request, plan, *, on_token=None):
        started.set()
        assert release.wait(10)
        return {"kind": "text", "output": str(output), "exit_code": 0}

    monkeypatch.setattr(tui_bridge, "run_clean_item", slow_run)
    bridge, sink = make_bridge()
    first = bridge.handle_line(
        json.dumps({"id": 1, "method": "clean", "params": {"state": state([folder])}})
    )
    assert first is not None and started.wait(10)

    assert (
        bridge.handle_line(
            json.dumps({"id": 2, "method": "clean", "params": {"state": state([folder])}})
        )
        is None
    )
    assert (
        bridge.handle_line(
            json.dumps({"id": 3, "method": "inspect", "params": {"state": state([folder])}})
        )
        is None
    )
    # Everything else stays responsive while the clean holds the lock.
    history = bridge.handle_line(json.dumps({"id": 4, "method": "history"}))
    assert history is not None
    history.join(5)
    bridge.handle_line(json.dumps({"id": 5, "method": "cancel"}))
    release.set()
    first.join(20)

    frames = sink.frames()
    by_id = {frame["id"]: frame for frame in frames if "result" in frame or "error" in frame}
    assert by_id[2]["error"]["code"] == "busy"
    assert by_id[3]["error"]["code"] == "busy"
    assert by_id[4]["result"] == {"entries": []}
    assert by_id[5]["result"] == {"ok": True}
    result = by_id[1]["result"]
    assert result["cancelled"] is True and result["total"] == 2
    assert len([f for f in frames if f.get("event") == "file_done"]) == 1
    assert "cancelled" in result["history"]["summary"]

    # The lock is released, and the cancel flag does not leak into the next run.
    again = response(call(bridge, sink, "clean", {"state": state([folder])}, rid=6))
    assert again["result"]["cancelled"] is False


def test_a_pipeline_system_exit_is_an_internal_error_not_a_wedged_bridge(tmp_path, monkeypatch):
    source = tmp_path / "a.txt"
    source.write_text(ZWSP, encoding="utf-8")

    def exits(*args, **kwargs):
        raise SystemExit(2)

    bridge, sink = make_bridge()
    with monkeypatch.context() as patched:
        patched.setattr(tui_bridge, "preflight_work", exits)
        failed = response(call(bridge, sink, "clean", {"state": state([source])}))
    assert failed["error"]["code"] == "internal"
    ok = response(call(bridge, sink, "clean", {"state": state([source])}, rid=2))
    assert ok["result"]["errors"] == 0


def test_a_file_whose_clean_raises_is_a_row_and_the_batch_continues(tmp_path, monkeypatch):
    folder = _two_files(tmp_path)
    real = tui_bridge.run_clean_item

    def flaky(path, output, request, plan, *, on_token=None):
        if path.name == "a.txt":
            raise SystemExit(2)
        return real(path, output, request, plan, on_token=on_token)

    monkeypatch.setattr(tui_bridge, "run_clean_item", flaky)
    bridge, sink = make_bridge()
    frames = call(bridge, sink, "clean", {"state": state([folder])})
    done = [frame["data"] for frame in frames if frame.get("event") == "file_done"]
    assert [item["display"].endswith("a.txt") for item in done] == [True, False]
    assert done[0]["error"] and "SystemExit" in done[0]["error"]
    assert done[1]["error"] is None and done[1]["after"]["hidden"] == 0
    assert response(frames)["result"]["errors"] == 1


def test_a_failed_file_reports_no_output(tmp_path, monkeypatch):
    """run_clean_item names the destination it meant to write even when it failed."""
    source = tmp_path / "a.txt"
    source.write_text(ZWSP, encoding="utf-8")

    def fails(path, output, request, plan, *, on_token=None):
        return {"kind": "text", "output": str(output), "error": "disk full", "exit_code": 1}

    monkeypatch.setattr(tui_bridge, "run_clean_item", fails)
    bridge, sink = make_bridge()
    frames = call(bridge, sink, "clean", {"state": state([source])})
    [done] = [frame["data"] for frame in frames if frame.get("event") == "file_done"]
    assert done["output"] is None and done["error"] == "disk full"
    assert done["lines"] == ["Failed: disk full"]


def test_a_bridge_dry_run_reads_as_sentences_not_key_value_dumps(tmp_path):
    from PIL import Image

    source = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(source)
    bridge, sink = make_bridge()
    flags = ["--dry-run", "--visible-box", "1,1,3,3", "--visible-backend", "texture"]
    frames = call(bridge, sink, "clean", {"state": state([source], flags=flags)})
    [done] = [frame["data"] for frame in frames if frame.get("event") == "file_done"]
    lines = done["lines"]
    assert lines[0].startswith("Dry run: would write ")
    assert "Would find the visible mark from box:1,1,3,3." in lines
    assert "Would inpaint with the texture backend." in lines
    assert not any("radius=" in line or "+" in line for line in lines)
    assert not source.with_name("a.cleaned.png").exists()


def test_a_visible_clean_says_what_it_did_to_the_pixels(tmp_path):
    from PIL import Image

    source = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(source)
    bridge, sink = make_bridge()
    flags = ["--visible-box", "2,2,2,2", "--dilate", "0", "--visible-backend", "simple"]
    frames = call(bridge, sink, "clean", {"state": state([source], flags=flags)})
    [done] = [frame["data"] for frame in frames if frame.get("event") == "file_done"]
    assert done["lines"][:2] == [
        "Filled holes and dilated the mask by 0 px (4 to 4 pixels).",
        "Inpainted the visible mark with the simple backend.",
    ]


def test_the_exclusive_lock_is_free_before_the_answer_is_written(tmp_path):
    """A frontend may send the next inspect the moment it reads this answer."""
    folder = _two_files(tmp_path)
    bridge, sink = make_bridge()
    held_at_answer: list[bool] = []
    write = sink.write

    def spying_write(data) -> int:
        if b'"result"' in bytes(data):
            held_at_answer.append(bridge._exclusive.locked())
        return write(data)

    sink.write = spying_write  # type: ignore[method-assign]
    call(bridge, sink, "inspect", {"state": state([folder])})
    assert held_at_answer == [False]


def test_inspect_streams_one_event_per_file(tmp_path):
    folder = _two_files(tmp_path)
    bridge, sink = make_bridge()
    frames = call(bridge, sink, "inspect", {"state": state([folder]), "soft": False})
    events = [frame["data"] for frame in frames if frame.get("event") == "inspected"]
    assert len(events) == 2
    assert all(item["suspicious"] and item["counts"]["hidden"] == 1 for item in events)
    assert response(frames)["result"]["files"] == events
    # Every event precedes the response for the same id.
    assert "result" in frames[-1]


# =============================================================================
# tokens
# =============================================================================


def test_tokens_are_coalesced_to_one_frame_per_interval():
    emitted: list[tuple[float, str]] = []
    coalescer = TokenCoalescer(lambda text: emitted.append((time.monotonic(), text)))
    fragments = [f"t{n} " for n in range(300)]
    for fragment in fragments:
        coalescer.add(fragment)
        time.sleep(0.001)
    coalescer.close()
    assert "".join(text for _, text in emitted) == "".join(fragments)
    assert len(emitted) < len(fragments) / 5
    gaps = [later - earlier for (earlier, _), (later, _) in pairwise(emitted)]
    assert all(gap >= 0.045 for gap in gaps), gaps


def test_a_close_with_nothing_buffered_emits_nothing():
    emitted: list[str] = []
    TokenCoalescer(emitted.append).close()
    assert emitted == []


# =============================================================================
# secrets
# =============================================================================


def test_the_writer_redacts_a_key_from_payloads_but_not_the_envelope(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "file_done-secret")
    sink = Sink()
    FrameWriter(sink).write({"id": 1, "event": "file_done", "data": {"x": "a file_done-secret"}})
    frame = sink.frames()[0]
    assert frame["event"] == "file_done"
    assert frame["data"] == {"x": "a [redacted]"}


def test_no_frame_ever_carries_the_key(tmp_path, monkeypatch, chat_server):
    """Redaction off: this proves the real paths never emit it, not the scrubber."""
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", SENTINEL)
    source = tmp_path / "draft.txt"
    source.write_text(SOURCE, encoding="utf-8")
    run = rewrite_state(source, tmp_path / "out.txt", chat_server)
    bridge, sink = make_bridge(redact=False)
    call(bridge, sink, "hello", rid=1)
    call(bridge, sink, "plan", {"state": run}, rid=2)
    cleaned = response(call(bridge, sink, "clean", {"state": run}, rid=3))
    call(bridge, sink, "history", rid=4)
    call(bridge, sink, "checks", rid=5)
    assert cleaned["result"]["errors"] == 0
    # The key really was used, at run time, for the request that needed it.
    assert _ChatHandler.auth == [f"Bearer {SENTINEL}"]
    assert SENTINEL not in sink.raw()


def test_history_is_newest_first(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text(ZWSP, encoding="utf-8")
    bridge, sink = make_bridge()
    call(bridge, sink, "clean", {"state": state([source])}, rid=1)
    call(bridge, sink, "clean", {"state": state([source], preset="hidden-aggressive")}, rid=2)
    entries = response(call(bridge, sink, "history", rid=3))["result"]["entries"]
    assert [entry["command"].split()[-1] for entry in entries] == [
        "--aggressive-homoglyphs",
        str(source),
    ]
    assert entries[0]["summary"] == "1 file, 0 errors. Verifiable."
    assert set(entries[0]) == {"time", "command", "summary"}


# =============================================================================
# the real process
# =============================================================================


def test_end_to_end_over_a_real_bridge_process(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text(f"hello{ZWSP} world\n", encoding="utf-8")
    env = bridge_env(
        tmp_path,
        WM_TUI_CWD=str(work),
        WM_TUI_ARGV=json.dumps(["a.txt"]),
        WATERMARKS_REWRITE_API_KEY=SENTINEL,
    )
    bridge = BridgeProcess(tmp_path, env)
    try:
        ready = bridge.next_frame()
        assert ready["event"] == "ready" and ready["data"]["version"]

        hello = response(bridge.request(1, "hello"))["result"]
        assert hello["onboard"] is True
        assert hello["initial"] == {"paths": ["a.txt"], "flags": []}

        run = state(hello["initial"]["paths"])
        plan = response(bridge.request(2, "plan", {"state": run}))["result"]
        assert plan["command"] == "wm a.txt"
        assert [item["display"] for item in plan["files"]] == ["a.txt"]

        inspected = bridge.request(3, "inspect", {"state": run})
        finding = inspected[0]
        assert finding["event"] == "inspected"
        assert finding["data"]["suspicious"] is True
        assert finding["data"]["counts"]["hidden"] == 1
        assert any("U+200B" in line for line in finding["data"]["lines"])
        assert finding["data"]["reveal"] == [
            {"line": 1, "text": "hello\u25c6 world", "marks": [[5, 6, "ZWSP"]]}
        ]

        cleaned = bridge.request(4, "clean", {"state": run})
        events = [frame["event"] for frame in cleaned if "event" in frame]
        assert events == ["file_start", "file_done"]
        done = cleaned[1]["data"]
        assert done["before"]["hidden"] == 1 and done["after"]["hidden"] == 0
        assert done["layer"] == "A" and done["result_class"] == VERIFIABLE
        assert done["exit_code"] == 0 and done["error"] is None
        assert "<U+200B>" in done["diff"]
        result = response(cleaned)["result"]
        assert result["total"] == 1 and result["errors"] == 0 and result["cancelled"] is False

        output = Path(done["output"])
        assert output == (work / "a.cleaned.txt").resolve()
        assert ZWSP not in output.read_text(encoding="utf-8")
        assert (work / "a.txt").read_text(encoding="utf-8") == f"hello{ZWSP} world\n"

        entries = response(bridge.request(5, "history"))["result"]["entries"]
        assert len(entries) == 1 and entries[0]["command"] == "wm a.txt"

        assert response(bridge.request(6, "shutdown"))["result"] == {"ok": True}
        assert bridge.close() == 0
    finally:
        bridge.close()
    assert SENTINEL.encode() not in b"".join(bridge.raw)


def test_a_stray_print_in_the_pipeline_never_corrupts_a_frame(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text(f"x{ZWSP}\n", encoding="utf-8")
    program = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import tui_bridge\n"
        "real = tui_bridge.inspect_asset\n"
        "def noisy(*args, **kwargs):\n"
        "    print('STRAY print')\n"
        "    print('STRAY dunder', file=sys.__stdout__, flush=True)\n"
        "    os.write(1, b'STRAY fd write\\n')\n"
        "    return real(*args, **kwargs)\n"
        "tui_bridge.inspect_asset = noisy\n"
        "code = tui_bridge.main()\n"
        "sys.stderr.flush()\n"
        "os._exit(code)\n"
    )
    env = bridge_env(tmp_path, WM_TUI_CWD=str(work))
    bridge = BridgeProcess(tmp_path, env, [sys.executable, "-c", program])
    try:
        bridge.next_frame()
        frames = bridge.request(1, "inspect", {"state": state(["a.txt"])})
        assert frames[0]["event"] == "inspected"
        # EOF alone ends the bridge, cleanly.
        assert bridge.close() == 0
    finally:
        bridge.close()
    for line in bridge.raw:
        json.loads(line)
    assert all(b"STRAY" not in line for line in bridge.raw)
    stderr = bridge.stderr_text()
    assert "STRAY print" in stderr and "STRAY fd write" in stderr and "STRAY dunder" in stderr


def test_a_print_while_the_pipeline_imports_never_reaches_the_frames(tmp_path):
    """optional_deps imports c2pa at import time; a noisy one must not precede ready."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "c2pa.py").write_text(
        "import os\nprint('STRAY import print')\nos.write(1, b'STRAY import fd\\n')\n",
        encoding="utf-8",
    )
    bridge = BridgeProcess(tmp_path, bridge_env(tmp_path, PYTHONPATH=str(shadow)))
    try:
        assert bridge.next_frame()["event"] == "ready"
        assert bridge.close() == 0
    finally:
        bridge.close()
    assert all(b"STRAY" not in line for line in bridge.raw)
    assert "STRAY import print" in bridge.stderr_text()


# =============================================================================
# Layer B live rewrite, over the bridge
# =============================================================================


def test_a_bridge_clean_performs_a_live_rewrite(chat_server, tmp_path):
    source = tmp_path / "draft.txt"
    source.write_text(SOURCE, encoding="utf-8")
    destination = tmp_path / "cleaned.txt"
    env = bridge_env(tmp_path, WM_TUI_CWD=str(tmp_path), WATERMARKS_REWRITE_API_KEY=SENTINEL)
    bridge = BridgeProcess(tmp_path, env)
    try:
        bridge.next_frame()
        run = rewrite_state(source, destination, chat_server)
        plan = response(bridge.request(1, "plan", {"state": run}))["result"]
        # A loopback endpoint needs no confirmation.
        assert plan["confirm"] == [] and plan["preflight_error"] is None
        assert plan["layer"] == "B" and plan["result_class"] == BEST_EFFORT

        frames = bridge.request(2, "clean", {"state": run})
        events = [frame["event"] for frame in frames if "event" in frame]
        assert events[0] == "file_start" and events[-1] == "file_done"
        assert "token" in events, "the rewrite stream must reach the frontend"
        assert events.index("token") < events.index("file_done")
        done = frames[-2]["data"]
        assert done["layer"] == "B" and done["result_class"] == BEST_EFFORT
        assert any("Layer B" in line for line in done["lines"])

        written = destination.read_text(encoding="utf-8")
        assert "plainly worded replacement" in written
        assert ZWSP not in written

        entry = response(frames)["result"]["history"]
        assert "--rewrite humanize" in entry["command"]
        assert entry["summary"] == "1 file, 0 errors. Best-effort."
        assert "api-key" not in entry["command"] and SENTINEL not in entry["command"]
        history = response(bridge.request(3, "history"))["result"]["entries"]
        assert history == [entry]
    finally:
        bridge.close()
    assert SENTINEL.encode() not in b"".join(bridge.raw)
    assert _ChatHandler.auth == [f"Bearer {SENTINEL}"]


# =============================================================================
# the launcher
# =============================================================================


def test_a_missing_path_is_refused_before_anything_else(monkeypatch, capsys, tmp_path):
    def never(*args, **kwargs):
        raise AssertionError("bun was looked up for a command that must fail first")

    monkeypatch.setattr(launcher.shutil, "which", never)
    code = launcher.main([str(tmp_path / "nope.md")])
    assert code == 2
    assert f"wm-tui: no such file or directory: {tmp_path / 'nope.md'}" in capsys.readouterr().err


def test_print_env_prints_only_the_wm_tui_variables(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", SENTINEL)
    monkeypatch.chdir(tmp_path)
    code = launcher.main([".", "--recursive", "--print-env"])
    assert code == 0
    out = capsys.readouterr().out
    assert SENTINEL not in out
    env = json.loads(out)
    assert set(env) == {"WM_TUI_PYTHON", "WM_TUI_BRIDGE", "WM_TUI_ARGV", "WM_TUI_LOG", "WM_TUI_CWD"}
    assert env["WM_TUI_PYTHON"] == sys.executable
    assert Path(env["WM_TUI_BRIDGE"]) == BRIDGE.resolve()
    assert json.loads(env["WM_TUI_ARGV"]) == [".", "--recursive"]
    assert env["WM_TUI_LOG"] == str(tmp_path / "cache" / "watermark-remover" / "tui.log")
    assert Path(env["WM_TUI_CWD"]) == tmp_path.resolve() or env["WM_TUI_CWD"] == str(tmp_path)


def test_a_missing_bun_prints_the_install_hint(monkeypatch, capsys):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    assert launcher.main(["."]) == 1
    err = capsys.readouterr().err
    assert "https://bun.sh" in err


def test_a_missing_frontend_is_a_clear_error(monkeypatch, capsys, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("WM_TUI_DIR", str(empty))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/fake/bun")
    assert launcher.main(["."]) == 1
    assert "WM_TUI_DIR" in capsys.readouterr().err


def _fake_frontend(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "src" / "index.tsx").write_text("", encoding="utf-8")
    return root


@pytest.mark.skipif(os.name == "nt", reason="exec path is POSIX-only")
def test_the_launcher_installs_once_then_execs_bun_in_the_frontend(monkeypatch, tmp_path):
    frontend = _fake_frontend(tmp_path / "frontend")
    installs: list[Path] = []
    execs: list[tuple] = []

    def fake_install(directory: Path, bun: str) -> int:
        installs.append(directory)
        (directory / "node_modules").mkdir()
        return 0

    def fake_exec(file, args, env):
        execs.append((file, args, env, Path.cwd()))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WM_TUI_DIR", str(frontend))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/fake/bun")
    monkeypatch.setattr(launcher, "_install", fake_install)
    monkeypatch.setattr(launcher.os, "execvpe", fake_exec)

    assert launcher.main(["."]) == 0
    assert installs == [frontend]
    file, args, env, cwd = execs[0]
    assert (file, args) == ("/fake/bun", ["/fake/bun", "run", "src/index.tsx"])
    assert cwd.resolve() == frontend.resolve()
    assert env["WM_TUI_PYTHON"] == sys.executable
    assert json.loads(env["WM_TUI_ARGV"]) == ["."]
    assert env["WM_TUI_CWD"] == str(tmp_path) or Path(env["WM_TUI_CWD"]) == tmp_path.resolve()
    assert Path(env["WM_TUI_LOG"]).parent.is_dir()

    # Installed now: the second start neither installs nor copies again.
    monkeypatch.chdir(tmp_path)
    installs.clear()
    assert launcher.main(["."]) == 0
    assert installs == []


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions that bind the current user",
)
def test_a_truly_read_only_frontend_is_staged_into_a_writable_copy(monkeypatch, tmp_path):
    frontend = _fake_frontend(tmp_path / "frontend")
    installs: list[Path] = []

    def fake_install(directory: Path, bun: str) -> int:
        installs.append(directory)
        (directory / "node_modules").mkdir()
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WM_TUI_DIR", str(frontend))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/fake/bun")
    monkeypatch.setattr(launcher, "_install", fake_install)
    run_dirs: list[Path] = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: run_dirs.append(Path.cwd()))
    paths = [frontend / "package.json", frontend / "src" / "index.tsx", frontend / "src", frontend]
    try:
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)
        assert launcher.main(["."]) == 0
        monkeypatch.chdir(tmp_path)
        # The second start refreshes the copy over itself: it must stay writable.
        assert launcher.main(["."]) == 0
    finally:
        for path in paths:
            path.chmod(0o755 if path.is_dir() else 0o644)
    staged = tmp_path / "cache" / "watermark-remover" / f"tui-{launcher.package_version()}"
    assert installs == [staged]
    assert (staged / "src" / "index.tsx").is_file()
    assert [path.resolve() for path in run_dirs] == [staged.resolve()] * 2
    assert os.access(staged, os.W_OK) and os.access(staged / "package.json", os.W_OK)


def test_a_failed_install_stops_the_launch(monkeypatch, tmp_path, capsys):
    frontend = _fake_frontend(tmp_path / "frontend")
    monkeypatch.setenv("WM_TUI_DIR", str(frontend))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/fake/bun")
    monkeypatch.setattr(launcher, "_install", lambda directory, bun: 1)
    monkeypatch.setattr(launcher.os, "execvpe", lambda *a: pytest.fail("exec after failure"))
    assert launcher.main(["."]) == 1


# =============================================================================
# static guards
# =============================================================================

TUI_MODULES = ("tui.py", "tui_core.py", "tui_bridge.py")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("module", TUI_MODULES)
def test_the_tui_never_speaks_http_itself(module):
    """Every request must inherit layer_b_http's hardening."""
    imported = _imported_modules(SCRIPTS / module)
    assert not imported & {"urllib", "http", "socket", "requests", "httpx"}


@pytest.mark.parametrize("module", TUI_MODULES)
def test_the_tui_never_builds_a_plan_itself(module):
    source = (SCRIPTS / module).read_text(encoding="utf-8")
    for constructor in ("CleanPlan(", "TextCleanPlan(", "VisiblePlan(", "RewritePlan("):
        assert constructor not in source, (module, constructor)


def test_the_launcher_imports_nothing_from_the_pipeline():
    """It runs on every start; the pipeline import belongs to the bridge."""
    imported = _imported_modules(SCRIPTS / "tui.py")
    local = {path.stem for path in SCRIPTS.glob("*.py")}
    assert not imported & local
