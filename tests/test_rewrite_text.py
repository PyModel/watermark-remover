"""Tests for Layer B rewrite_text hook (offline / print-prompt)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
from common import read_bool_env
from rewrite_text import build_prompt, call_openai_compatible, rewrite


def test_build_prompt_paraphrase_contains_text():
    p = build_prompt("paraphrase", "Hello world facts 42.", lang="French", original_lang="English")
    assert "Hello world facts 42." in p
    assert "Rewrite" in p or "rewrite" in p.lower()


def test_print_prompt_backend():
    out, info = rewrite(
        "Sample prose about water marks.",
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
        strength="paraphrase",
        lang="French",
        original_lang="English",
        timeout=5.0,
        layer_a_after=True,
    )
    assert info["mode"] == "print-prompt"
    assert "Sample prose" in out
    assert info["backend"] == "print-prompt"


def test_openai_adapter_adds_disable_thinking_only_when_requested(monkeypatch):
    payloads = []

    def fake_http(_url, payload, _headers, _timeout):
        payloads.append(payload)
        return {"choices": [{"message": {"content": "rewritten"}}]}

    monkeypatch.setattr(rewrite_text, "_http_json", fake_http)
    common = ("http://localhost:8080", "model", "prompt", None, 1.0)
    assert call_openai_compatible(*common) == "rewritten"
    assert "chat_template_kwargs" not in payloads[-1]
    assert call_openai_compatible(*common, disable_thinking=True) == "rewritten"
    assert payloads[-1]["chat_template_kwargs"] == {"enable_thinking": False}


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


def test_http_adapter_bounds_and_validates_json_response(monkeypatch):
    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit):
            return self.body[:limit]

    monkeypatch.setattr("common.DEFAULT_HTTP_JSON_LIMIT", 8)
    monkeypatch.setattr(
        rewrite_text.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"x" * 9),
    )
    try:
        rewrite_text._http_json("http://localhost", {}, {}, 1.0)
    except RuntimeError as error:
        assert "safety limit" in str(error)
    else:
        raise AssertionError("expected oversized response rejection")

    monkeypatch.setattr("common.DEFAULT_HTTP_JSON_LIMIT", 64)
    monkeypatch.setattr(
        rewrite_text.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"[]"),
    )
    try:
        rewrite_text._http_json("http://localhost", {}, {}, 1.0)
    except RuntimeError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("expected non-object response rejection")


def test_model_adapters_reject_malformed_nested_responses(monkeypatch):
    malformed = [
        ({"message": "text"}, rewrite_text.call_ollama),
        ({"choices": [None]}, rewrite_text.call_openai_compatible),
        ({"choices": [{"message": "text"}]}, rewrite_text.call_openai_compatible),
    ]
    for response, adapter in malformed:
        monkeypatch.setattr(
            rewrite_text, "_http_json", lambda *_args, value=response, **_kwargs: value
        )
        try:
            if adapter is rewrite_text.call_ollama:
                adapter("http://localhost", "model", "prompt", 1.0)
            else:
                adapter("http://localhost", "model", "prompt", None, 1.0)
        except RuntimeError as error:
            assert "invalid" in str(error)
        else:
            raise AssertionError("expected malformed response rejection")


def test_rewrite_rejects_invalid_timeout_and_endpoint():
    common = {
        "text": "hello",
        "backend": "openai-compatible",
        "model": "test",
        "api_key": None,
        "strength": "paraphrase",
        "lang": "French",
        "original_lang": "English",
        "layer_a_after": True,
    }
    for timeout in (0.0, -1.0, float("nan")):
        try:
            rewrite(base_url="http://localhost:8080", timeout=timeout, **common)
        except ValueError as error:
            assert "timeout" in str(error)
        else:
            raise AssertionError("expected timeout rejection")
    for field, value in (("disable_thinking", "yes"),):
        try:
            rewrite(base_url="http://localhost:8080", timeout=1.0, **common, **{field: value})
        except TypeError as error:
            assert field in str(error)
        else:
            raise AssertionError(f"expected {field} type rejection")
    try:
        rewrite(base_url="file:///tmp/socket", timeout=1.0, **common)
    except ValueError as error:
        assert "http(s)" in str(error)
    else:
        raise AssertionError("expected endpoint rejection")


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
