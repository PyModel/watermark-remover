"""Tests for Layer B rewrite_text hook (offline / print-prompt + client hardening)."""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import layer_b_http
import rewrite_text
from common import read_bool_env
from layer_b_http import LayerBHTTPError
from rewrite_text import (
    RewritePlan,
    _check_remote,
    _flag_env,
    _lexical_divergence,
    _select_candidate,
    build_prompt,
    rewrite,
)


def test_prompt_plan_is_immutable_and_has_no_transport_surface():
    plan = rewrite_text.RewritePlan.prompt("paraphrase")

    with pytest.raises(FrozenInstanceError):
        plan.strength = "structural"
    with pytest.raises((AttributeError, TypeError)):
        plan.unexpected = True
    with pytest.raises(TypeError):
        rewrite_text.RewritePlan.prompt(
            "paraphrase",
            base_url="https://example.test",
        )


def test_live_tsapa_plan_resolves_environment_and_hides_secret(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "test-model")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "https://example.test/prefix")
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "unit-test-secret-never-real")
    monkeypatch.setenv("WATERMARKS_REWRITE_DISABLE_THINKING", "true")

    plan = rewrite_text.RewritePlan.live_tsapa_from_environment(
        generations=3,
        population=4,
    )

    assert plan.backend == "openai-compatible"
    assert plan.model == "test-model"
    assert plan.base_url == "https://example.test/prefix"
    assert plan.strength == "tsapa"
    assert plan.generations == 3
    assert plan.population == 4
    assert plan.disable_thinking is True
    assert "unit-test-secret-never-real" not in repr(plan)


def test_live_tsapa_plan_requires_backend_and_model(monkeypatch):
    monkeypatch.delenv("WATERMARKS_REWRITE_BACKEND", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_MODEL", raising=False)

    with pytest.raises(ValueError, match="requires a live backend"):
        rewrite_text.RewritePlan.live_tsapa_from_environment(generations=1, population=2)

    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    with pytest.raises(ValueError, match="requires WATERMARKS_REWRITE_MODEL"):
        rewrite_text.RewritePlan.live_tsapa_from_environment(generations=1, population=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "unknown"}, "unknown backend"),
        ({"strength": "unknown"}, "unknown strength"),
        ({"backend": "ollama"}, "model required"),
        (
            {"backend": "openai-compatible", "model": "model", "base_url": None},
            "base-url required",
        ),
        ({"strength": "tsapa", "generations": -1}, "generations"),
        ({"strength": "tsapa", "population": 1}, "population"),
    ],
)
def test_rewrite_plan_rejects_invalid_mode_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RewritePlan(**kwargs)


def test_rewrite_plan_rejects_invalid_types_at_construction():
    with pytest.raises(TypeError, match="disable_thinking"):
        RewritePlan(disable_thinking="yes")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_rewrite_plan_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError, match="timeout"):
        RewritePlan(timeout=timeout)


def test_rewrite_plan_rejects_boolean_timeout():
    with pytest.raises(TypeError, match="timeout"):
        RewritePlan(timeout=True)


def test_build_prompt_paraphrase_contains_text():
    p = build_prompt("paraphrase", "Hello world facts 42.", lang="French", original_lang="English")
    assert "Hello world facts 42." in p
    assert "Rewrite" in p or "rewrite" in p.lower()


def test_print_prompt_backend_uses_one_plan():
    plan = rewrite_text.RewritePlan.prompt("paraphrase")

    out, info = rewrite("Sample prose about water marks.", plan)
    assert info["mode"] == "print-prompt"
    assert "Sample prose" in out
    assert info["backend"] == "print-prompt"


def test_openai_adapter_adds_disable_thinking_only_when_requested(monkeypatch):
    payloads = []

    def fake_http(_endpoint, _route, payload, *, headers=None, timeout):
        payloads.append(payload)
        return {"choices": [{"message": {"content": "rewritten"}}]}

    monkeypatch.setattr(layer_b_http, "request_json", fake_http)
    common = {
        "backend": "openai-compatible",
        "model": "model",
        "base_url": "http://localhost:8080",
        "timeout": 1.0,
        "layer_a_after": False,
    }
    assert rewrite("source", RewritePlan(**common))[0] == "rewritten"
    assert "chat_template_kwargs" not in payloads[-1]
    assert rewrite("source", RewritePlan(**common, disable_thinking=True))[0] == "rewritten"
    assert payloads[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_provider_adapters_keep_routes_auth_and_response_parsing(monkeypatch):
    calls = []

    def fake_request(endpoint, route, payload, *, headers=None, timeout):
        calls.append((endpoint, route, payload, headers, timeout))
        if route == "/api/chat":
            return {"message": {"content": "ollama rewrite"}}
        return {"choices": [{"message": {"content": "openai rewrite"}}]}

    monkeypatch.setattr(layer_b_http, "request_json", fake_request)

    ollama, _ = rewrite(
        "source",
        RewritePlan(
            backend="ollama",
            model="ollama-model",
            base_url="http://localhost:11434/api",
            timeout=7.0,
            layer_a_after=False,
        ),
    )
    openai, _ = rewrite(
        "source",
        RewritePlan(
            backend="openai-compatible",
            model="openai-model",
            base_url="https://example.test/prefix",
            api_key="unit-test-secret-never-real",
            timeout=11.0,
            layer_a_after=False,
            allow_remote=True,
        ),
    )
    assert ollama == "ollama rewrite"
    assert openai == "openai rewrite"

    ollama_call, openai_call = calls
    assert ollama_call[0:2] == ("http://localhost:11434/api", "/api/chat")
    assert ollama_call[3] is None
    assert ollama_call[4] == 7.0
    assert openai_call[0:2] == ("https://example.test/prefix", "/v1/chat/completions")
    assert openai_call[3] == {"Authorization": "Bearer unit-test-secret-never-real"}
    assert openai_call[4] == 11.0


def test_live_rewrite_uses_transport_validation_before_network(monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("invalid configuration reached transport opener")

    monkeypatch.setattr(layer_b_http._OPENER, "open", fail_network)
    common = {
        "backend": "openai-compatible",
        "model": "test",
        "api_key": "unit-test-secret-never-real",
        "strength": "paraphrase",
        "layer_a_after": True,
    }
    for endpoint in (
        "file:///tmp/socket",
        "http://user:pass@example.test",
        "http://[bad",
        "not a URL",
    ):
        plan = RewritePlan(base_url=endpoint, timeout=1.0, **common)
        with pytest.raises(LayerBHTTPError, match="endpoint") as raised:
            rewrite("hello", plan)
        assert "unit-test-secret-never-real" not in str(raised.value)


def test_boolean_environment_parser_is_strict(monkeypatch):
    for value in ("true", "1", "YES", "on"):
        monkeypatch.setenv("TEST_FLAG", value)
        assert read_bool_env("TEST_FLAG") is True
    for value in ("false", "0", "NO", "off"):
        monkeypatch.setenv("TEST_FLAG", value)
        assert read_bool_env("TEST_FLAG") is False
    monkeypatch.setenv("TEST_FLAG", "sometimes")
    try:
        read_bool_env("TEST_FLAG")
    except ValueError as error:
        assert "true or false" in str(error)
    else:
        raise AssertionError("expected invalid environment flag rejection")


def test_model_adapters_reject_malformed_nested_responses(monkeypatch):
    malformed = [
        ("ollama", {"message": "text", "debug": "unit-test-secret-never-real"}),
        (
            "openai-compatible",
            {"choices": [None], "debug": "unit-test-secret-never-real"},
        ),
        (
            "openai-compatible",
            {"choices": [{"message": "text"}], "debug": "unit-test-secret-never-real"},
        ),
    ]
    for backend, response in malformed:
        monkeypatch.setattr(
            layer_b_http, "request_json", lambda *_args, value=response, **_kwargs: value
        )
        plan = RewritePlan(
            backend=backend,
            model="model",
            base_url="http://localhost",
            timeout=1.0,
            layer_a_after=False,
        )
        with pytest.raises(RuntimeError, match="invalid") as raised:
            rewrite("source", plan)
        assert "unit-test-secret-never-real" not in str(raised.value)


def test_cli_preserves_missing_model_exit_message(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("preserve", encoding="utf-8")
    env = os.environ.copy()
    env.pop("WATERMARKS_REWRITE_MODEL", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "rewrite_text.py"),
            str(source),
            "--backend",
            "openai-compatible",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.strip() == "error: --model required for ollama/openai-compatible backends"


def test_cli_rejects_non_positive_timeout_without_traceback(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("preserve", encoding="utf-8")
    env = os.environ.copy()
    for name in ("WATERMARKS_REWRITE_BACKEND", "WATERMARKS_REWRITE_MODEL"):
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "rewrite_text.py"),
            str(source),
            "--timeout",
            "0",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr.strip() == "timeout must be a finite positive number"


def test_cli_rejects_output_alias(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("preserve", encoding="utf-8")
    env = os.environ.copy()
    for name in ("WATERMARKS_REWRITE_BACKEND", "WATERMARKS_REWRITE_MODEL"):
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "rewrite_text.py"),
            str(source),
            "-o",
            str(source),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert source.read_text(encoding="utf-8") == "preserve"


def test_structural_and_backtranslate_prompts():
    for strength in ("structural", "backtranslate"):
        p = build_prompt(strength, "ABC 123", lang="German", original_lang="English")
        assert "ABC 123" in p


# ---------------------------------------------------------------------------
# Ported from THEIRS: humanize/code prompts, divergence/selection, hardening
# ---------------------------------------------------------------------------


def _rewrite_plan(**overrides) -> RewritePlan:
    values = {
        "backend": "print-prompt",
        "model": None,
        "base_url": None,
        "api_key": None,
        "strength": "paraphrase",
        "lang": "French",
        "original_lang": "English",
        "timeout": 5.0,
        "layer_a_after": True,
        "temperature": 0.9,
        "candidates": 1,
    }
    values.update(overrides)
    return RewritePlan(**values)


def test_build_prompt_humanize_and_code_contain_text():
    for strength, keyword in (("humanize", "human wrote it"), ("code", "comments")):
        p = build_prompt(strength, "ABC 123", lang="French", original_lang="English")
        assert "ABC 123" in p
        assert keyword in p


def test_build_prompt_unknown_strength_raises():
    with pytest.raises(ValueError):
        build_prompt("nope", "ABC", lang="French", original_lang="English")


def test_print_prompt_backend_reports_temperature():
    out, info = rewrite("Sample prose about water marks.", _rewrite_plan())
    assert info["mode"] == "print-prompt"
    assert "Sample prose" in out
    assert info["backend"] == "print-prompt"
    assert info["temperature"] == 0.9


def test_print_prompt_ignores_candidates():
    out, info = rewrite("Sample prose about water marks.", _rewrite_plan(candidates=2))
    assert info["mode"] == "print-prompt"
    assert isinstance(out, str)
    assert "Sample prose" in out


def test_lexical_divergence_identical_is_zero():
    assert _lexical_divergence("the cat sat", "the cat sat") == 0.0


def test_lexical_divergence_fully_different_higher_than_similar():
    similar = _lexical_divergence("the cat sat on the mat", "the dog sat on the mat")
    different = _lexical_divergence("the cat sat on the mat", "alpha beta gamma delta")
    assert different > similar


def test_lexical_divergence_empty_inputs():
    assert _lexical_divergence("", "") == 0.0
    assert _lexical_divergence("", "text") == 1.0
    assert _lexical_divergence("text", "") == 1.0


def test_select_candidate_prefers_more_divergent():
    original = "the cat sat on the mat"
    best, scores = _select_candidate(
        original,
        ["the cat sat on the mat", "the dog sat on the mat", "alpha beta gamma delta"],
    )
    assert best == "alpha beta gamma delta"
    assert len(scores) == 3


def test_candidates_select_most_divergent(monkeypatch):
    texts = iter(["the cat sat on the mat", "alpha beta gamma delta"])
    monkeypatch.setattr(rewrite_text, "_call_ollama", lambda *a, **k: next(texts))
    out, info = rewrite(
        "the cat sat on the mat",
        _rewrite_plan(
            backend="ollama",
            model="m",
            base_url="http://127.0.0.1:11434",
            timeout=5.0,
            layer_a_after=False,
            candidates=2,
        ),
    )
    assert out == "alpha beta gamma delta"
    selected = [entry["selected"] for entry in info["candidate_scores"]]
    assert selected == [False, True]


def test_check_remote_loopback_allowed_without_opt_in():
    _check_remote("http://127.0.0.1:11434", allow_remote=False)
    _check_remote("http://localhost:11434", allow_remote=False)
    _check_remote("http://[::1]:11434", allow_remote=False)


def test_check_remote_denies_non_loopback_without_opt_in():
    with pytest.raises(SystemExit):
        _check_remote("http://example.com:11434", allow_remote=False)


def test_check_remote_allows_non_loopback_with_opt_in(capsys):
    _check_remote("http://example.com:11434", allow_remote=True)
    err = capsys.readouterr().err
    assert "content will leave this machine" in err


def test_check_remote_denies_non_http_scheme():
    with pytest.raises(SystemExit):
        _check_remote("file:///etc/passwd", allow_remote=True)


def test_flag_env(monkeypatch):
    assert not _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    assert _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "true")
    assert _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "0")
    assert not _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")


def _http_plan(base_url: str, **overrides) -> RewritePlan:
    values = {
        "backend": "openai-compatible",
        "model": "m",
        "base_url": base_url,
        "api_key": "sk-test-key-123",
        "strength": "paraphrase",
        "lang": "French",
        "original_lang": "English",
        "timeout": 5.0,
        "layer_a_after": False,
        "temperature": 0.9,
        "candidates": 1,
    }
    values.update(overrides)
    return RewritePlan(**values)


def test_openai_compatible_sends_reasoning_effort_when_set():
    captured = {}

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            captured["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices": [{"message": {"content": "rewritten"}}]}')

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result, _ = rewrite(
            "hello",
            _http_plan(
                f"http://127.0.0.1:{server.server_address[1]}",
                reasoning_effort="none",
            ),
        )
        assert result == "rewritten"
        assert captured["body"]["reasoning_effort"] == "none"

        captured.clear()
        rewrite(
            "hello",
            _http_plan(
                f"http://127.0.0.1:{server.server_address[1]}",
                reasoning_effort=None,
            ),
        )
        assert "reasoning_effort" not in captured["body"]
    finally:
        server.shutdown()


def test_openai_compatible_omits_reasoning_effort_when_off(monkeypatch):
    """'off' disables the parameter entirely; enabled values are still sent."""
    captured: dict = {}

    def fake_request_json(base_url, route, payload, **kwargs):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "rewritten"}}]}

    monkeypatch.setattr(rewrite_text.layer_b_http, "request_json", fake_request_json)

    rewrite_text._call_openai_compatible(
        "http://127.0.0.1:9", "m", "hello", "key", 5.0, reasoning_effort="off"
    )
    assert "reasoning_effort" not in captured["payload"]

    rewrite_text._call_openai_compatible(
        "http://127.0.0.1:9", "m", "hello", "key", 5.0, reasoning_effort="low"
    )
    assert captured["payload"]["reasoning_effort"] == "low"

    rewrite_text._call_openai_compatible(
        "http://127.0.0.1:9", "m", "hello", "key", 5.0, reasoning_effort=None
    )
    assert "reasoning_effort" not in captured["payload"]


def test_rewrite_denies_remote_host_without_opt_in():
    with pytest.raises(LayerBHTTPError, match="endpoint"):
        rewrite("secret text", _http_plan("http://example.com:11434"))


def test_rewrite_blocks_redirect_and_never_sends_key():
    """A 302 from the loopback endpoint must not re-send the API key to an
    unvalidated redirect target. OURS' shared transport refuses cross-origin
    redirects (host or port change) with LayerBHTTPError, so no request ever
    reaches the redirect target — the key cannot leak."""
    state: dict = {"collector_port": None}
    captured: dict = {}

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{state['collector_port']}/collect",
            )
            self.end_headers()

        def log_message(self, format, *args):
            pass

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            captured["auth"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices": [{"message": {"content": "rewritten"}}]}')

        def log_message(self, format, *args):
            pass

    collector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    redirector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    state["collector_port"] = collector.server_address[1]
    threading.Thread(target=collector.serve_forever, daemon=True).start()
    threading.Thread(target=redirector.serve_forever, daemon=True).start()
    try:
        with pytest.raises(LayerBHTTPError, match="cross-origin redirect"):
            rewrite(
                "secret text",
                _http_plan(f"http://127.0.0.1:{redirector.server_address[1]}"),
            )
        assert captured == {}, "redirect target received a request (key leak?)"
    finally:
        collector.shutdown()
        redirector.shutdown()
