"""Tests for Layer B rewrite_text hook (offline / print-prompt)."""

from __future__ import annotations

import os
import subprocess
import sys
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
from rewrite_text import RewritePlan, build_prompt, rewrite


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
    )

    assert result.returncode == 1
    assert result.stderr.strip() == "error: --model required for ollama/openai-compatible backends"


def test_cli_rejects_non_positive_timeout_without_traceback(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("preserve", encoding="utf-8")
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
    )

    assert result.returncode == 1
    assert result.stderr.strip() == "timeout must be a finite positive number"


def test_cli_rejects_output_alias(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("preserve", encoding="utf-8")
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
    )
    assert result.returncode == 2
    assert source.read_text(encoding="utf-8") == "preserve"


def test_structural_and_backtranslate_prompts():
    for strength in ("structural", "backtranslate"):
        p = build_prompt(strength, "ABC 123", lang="German", original_lang="English")
        assert "ABC 123" in p
