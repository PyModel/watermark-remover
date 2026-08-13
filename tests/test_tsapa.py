"""Offline tests for TSAPA evolutionary Layer B."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import layer_b_http
from layer_b_http import LayerBHTTPError
from rewrite_text import RewritePlan, rewrite
from tsapa import (
    Candidate,
    TSAPAAdapterError,
    chunk_text,
    cosine,
    crossover,
    http_embed,
    http_pll,
    lexical_diversity,
    ngram_diversity,
    non_dominated_sort,
    select_knee_point,
    shingle_similarity,
    tsapa,
)

TEXT = (
    "The project removes metadata while preserving the image pixels. "
    "Every operation reports its actions and residual risks. "
    "No method can guarantee that every vendor detector will fail."
)


def fake_llm(prompt: str) -> str:
    body = prompt.rsplit("---\n", 1)[-1].strip()
    replacements = {
        "The project": "This utility",
        "removes": "strips",
        "while preserving": "without changing",
        "Every operation": "Each processing step",
        "reports": "documents",
        "No method": "No technique",
        "guarantee": "ensure",
        "will fail": "returns a miss",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    if "more concisely" in prompt:
        body = body.replace("the image pixels", "pixels")
    if "casual register" in prompt:
        body = body.replace("This utility", "The tool")
    if "expanded detail" in prompt:
        body = body.replace("residual risks", "any residual risks that remain")
    return body


def live_tsapa_plan(**overrides) -> RewritePlan:
    values = {
        "backend": "openai-compatible",
        "model": "test-model",
        "base_url": "http://127.0.0.1:9999",
        "strength": "tsapa",
        "timeout": 1.0,
        "layer_a_after": True,
        "generations": 0,
        "population": 2,
    }
    values.update(overrides)
    return RewritePlan(**values)


def test_http_scorers_keep_routes_auth_timeout_and_response_parsing(monkeypatch):
    calls = []

    def fake_request(endpoint, route, payload, *, headers=None, timeout):
        calls.append((endpoint, route, payload, headers, timeout))
        if route == "/v1/completions":
            return {"choices": [{"logprobs": {"token_logprobs": [-1.0, None, -3.0]}}]}
        return {"data": [{"embedding": [1, 2.5]}]}

    monkeypatch.setattr(layer_b_http, "request_json", fake_request)

    assert (
        http_pll(
            "https://example.test/prefix",
            TEXT,
            model="pll-model",
            api_key="unit-test-secret-never-real",
            timeout=5.0,
        )
        == -2.0
    )
    assert http_embed(
        "https://example.test/prefix",
        TEXT,
        model="embed-model",
        api_key="unit-test-secret-never-real",
        timeout=9.0,
    ) == [1.0, 2.5]

    pll_call, embed_call = calls
    assert pll_call[0:2] == ("https://example.test/prefix", "/v1/completions")
    assert pll_call[3:] == ({"Authorization": "Bearer unit-test-secret-never-real"}, 5.0)
    assert embed_call[0:2] == ("https://example.test/prefix", "/v1/embeddings")
    assert embed_call[3:] == ({"Authorization": "Bearer unit-test-secret-never-real"}, 9.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_http_pll_rejects_non_finite_token_logprobs(monkeypatch, value):
    def fake_request(*_args, **_kwargs):
        return {"choices": [{"logprobs": {"token_logprobs": [-1.0, value]}}]}

    monkeypatch.setattr(layer_b_http, "request_json", fake_request)

    with pytest.raises(TSAPAAdapterError, match="non-finite"):
        http_pll("http://localhost:8000", TEXT)


def test_http_scorers_share_transport_validation(monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("invalid configuration reached transport opener")

    monkeypatch.setattr(layer_b_http._OPENER, "open", fail_network)
    for scorer in (http_pll, http_embed):
        with pytest.raises(LayerBHTTPError, match="endpoint"):
            scorer("file:///tmp/provider.sock", TEXT, timeout=1.0)
        for timeout in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(LayerBHTTPError, match="timeout"):
                scorer("http://localhost:8000", TEXT, timeout=timeout)


def test_http_scorers_validate_provider_shapes(monkeypatch):
    monkeypatch.setattr(
        layer_b_http,
        "request_json",
        lambda *_args, **_kwargs: {"choices": [None]},
    )
    with pytest.raises(RuntimeError, match="choices"):
        http_pll("http://localhost", TEXT, timeout=1.0)

    monkeypatch.setattr(
        layer_b_http,
        "request_json",
        lambda *_args, **_kwargs: {"data": [None]},
    )
    with pytest.raises(RuntimeError, match="data"):
        http_embed("http://localhost", TEXT, timeout=1.0)


def test_cosine_rejects_dimension_mismatch():
    try:
        cosine([1.0, 2.0], [1.0])
    except ValueError as error:
        assert "dimensions differ" in str(error)
    else:
        raise AssertionError("expected embedding dimension rejection")


def test_metrics():
    assert ngram_diversity("one two three four five", 2) == 1.0
    assert ngram_diversity("one two one two one two", 2) < 1.0
    assert lexical_diversity(TEXT, TEXT) == 0.0
    assert lexical_diversity("alpha beta gamma delta", TEXT) == 1.0
    assert shingle_similarity(TEXT, TEXT) == 1.0


def test_chunk_text():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, max_chars=35)
    assert len(chunks) >= 2
    assert "First sentence." in chunks[0]


def test_non_dominated_sort_and_knee():
    left = Candidate("left", f_atk=0.9, f_fid=0.2)
    middle = Candidate("middle", f_atk=0.6, f_fid=0.6)
    right = Candidate("right", f_atk=0.2, f_fid=0.9)
    weak = Candidate("weak", f_atk=0.1, f_fid=0.1)
    fronts = non_dominated_sort([left, middle, right, weak])
    assert {c.text for c in fronts[0]} == {"left", "middle", "right"}
    assert fronts[1] == [weak]
    assert select_knee_point(fronts[0]).text == "middle"


def test_crossover_uses_parent_sentences():
    import random

    a = Candidate("Alpha one. Alpha two. Alpha three.")
    b = Candidate("Beta one. Beta two. Beta three.")
    child = crossover(a, b, random.Random(7))
    assert child.text
    assert "Alpha" in child.text or "Beta" in child.text


def test_tsapa_degrades_gracefully_with_one_usable_candidate():
    calls = 0

    def mostly_empty_llm(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "one valid rewrite" if calls == 1 else ""

    result = tsapa(TEXT, llm=mostly_empty_llm, generations=2, population=4, seed=1)
    assert result["text"] == "one valid rewrite"
    assert result["stats"][0]["skipped_evolution"] is True
    assert result["stats"][0]["usable_candidates"] == 1


def test_tsapa_offline_end_to_end():
    result = tsapa(
        TEXT,
        llm=fake_llm,
        generations=2,
        population=6,
        seed=42,
    )
    assert result["text"] != TEXT
    assert result["chunks"] == 1
    assert result["generations"] == 2
    assert result["population"] == 6
    assert result["stats"][0]["front_size"] >= 1
    assert "best-effort" in result["note"]


def test_tsapa_validates_evolution_parameters():
    for kwargs in (
        {"population": 1},
        {"generations": -1},
        {"chunk_chars": 0},
        {"weights": (float("nan"), 0.2, 0.8)},
    ):
        try:
            tsapa(TEXT, llm=fake_llm, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_tsapa_print_prompt_backend():
    out, info = rewrite(
        TEXT,
        RewritePlan.prompt("tsapa", generations=7, population=20),
    )
    assert "7 generations" in out
    assert "20 diverse" in out
    assert "Pareto" in out
    assert TEXT in out
    assert info["mode"] == "print-prompt"


def test_live_tsapa_reports_adapter_fallbacks(monkeypatch):
    def fake_request(_endpoint, route, payload, *, headers=None, timeout):
        if route == "/v1/chat/completions":
            return {
                "choices": [{"message": {"content": fake_llm(payload["messages"][0]["content"])}}]
            }
        if route == "/v1/completions":
            return {"choices": [{}]}
        if route == "/v1/embeddings":
            return {"data": [None]}
        raise AssertionError(f"unexpected route: {route}")

    monkeypatch.setattr(layer_b_http, "request_json", fake_request)
    _out, info = rewrite(
        TEXT,
        live_tsapa_plan(api_key="unit-test-secret-never-real"),
    )
    fallbacks = info["tsapa"]["adapter_fallbacks"]
    assert fallbacks["pll"] >= 2
    assert fallbacks["embedding"] >= 1


def test_live_tsapa_routes_generation_and_scorers_through_one_transport(monkeypatch):
    routes = []

    def fake_request(_endpoint, route, payload, *, headers=None, timeout):
        routes.append(route)
        if route == "/v1/chat/completions":
            return {
                "choices": [{"message": {"content": fake_llm(payload["messages"][0]["content"])}}]
            }
        if route == "/v1/completions":
            return {"choices": [{"logprobs": {"token_logprobs": [-1.0]}}]}
        if route == "/v1/embeddings":
            return {"data": [{"embedding": [1.0, 0.0]}]}
        raise AssertionError(f"unexpected route: {route}")

    monkeypatch.setattr(layer_b_http, "request_json", fake_request)
    _out, info = rewrite(
        TEXT,
        live_tsapa_plan(
            base_url="http://127.0.0.1:9999/prefix",
            api_key="unit-test-secret-never-real",
            timeout=4.0,
        ),
    )

    assert {"/v1/chat/completions", "/v1/completions", "/v1/embeddings"} <= set(routes)
    assert info["tsapa"]["adapter_fallbacks"] == {"pll": 0, "embedding": 0}


def test_live_tsapa_reports_inconsistent_embedding_dimensions_as_fallback(monkeypatch):
    embedding_calls = 0

    def fake_request(_endpoint, route, payload, *, headers=None, timeout):
        nonlocal embedding_calls
        if route == "/v1/chat/completions":
            return {
                "choices": [{"message": {"content": fake_llm(payload["messages"][0]["content"])}}]
            }
        if route == "/v1/completions":
            return {"choices": [{"logprobs": {"token_logprobs": [-1.0]}}]}
        if route == "/v1/embeddings":
            embedding_calls += 1
            dimensions = 2 if embedding_calls == 1 else 3
            return {"data": [{"embedding": [1.0] * dimensions}]}
        raise AssertionError(f"unexpected route: {route}")

    monkeypatch.setattr(layer_b_http, "request_json", fake_request)
    _out, info = rewrite(TEXT, live_tsapa_plan())

    assert info["tsapa"]["adapter_fallbacks"]["embedding"] >= 1


def test_live_tsapa_falls_back_only_for_expected_adapter_failures(monkeypatch):
    def expected_failures(_endpoint, route, payload, *, headers=None, timeout):
        if route == "/v1/chat/completions":
            return {
                "choices": [{"message": {"content": fake_llm(payload["messages"][0]["content"])}}]
            }
        if route in ("/v1/completions", "/v1/embeddings"):
            raise LayerBHTTPError("offline")
        raise AssertionError(f"unexpected route: {route}")

    monkeypatch.setattr(layer_b_http, "request_json", expected_failures)
    _out, info = rewrite(TEXT, live_tsapa_plan())
    assert info["tsapa"]["adapter_fallbacks"]["pll"] >= 2
    assert info["tsapa"]["adapter_fallbacks"]["embedding"] >= 1

    def pll_programming_defect(_endpoint, route, payload, *, headers=None, timeout):
        if route == "/v1/chat/completions":
            return {
                "choices": [{"message": {"content": fake_llm(payload["messages"][0]["content"])}}]
            }
        if route == "/v1/completions":
            raise RuntimeError("programming defect")
        return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(layer_b_http, "request_json", pll_programming_defect)
    with pytest.raises(RuntimeError, match="programming defect"):
        rewrite(TEXT, live_tsapa_plan())

    def embedding_programming_defect(_endpoint, route, payload, *, headers=None, timeout):
        if route == "/v1/chat/completions":
            return {
                "choices": [{"message": {"content": fake_llm(payload["messages"][0]["content"])}}]
            }
        if route == "/v1/completions":
            return {"choices": [{"logprobs": {"token_logprobs": [-1.0]}}]}
        if route == "/v1/embeddings":
            raise RuntimeError("embedding defect")
        raise AssertionError(f"unexpected route: {route}")

    monkeypatch.setattr(layer_b_http, "request_json", embedding_programming_defect)
    with pytest.raises(RuntimeError, match="embedding defect"):
        rewrite(TEXT, live_tsapa_plan())


def test_clean_file_tsapa_requires_live_backend(tmp_path: Path):
    f = tmp_path / "draft.txt"
    f.write_text(TEXT, encoding="utf-8")
    env = os.environ.copy()
    env.pop("WATERMARKS_REWRITE_BACKEND", None)
    env.pop("WATERMARKS_REWRITE_MODEL", None)
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_file.py"), str(f), "--tsapa"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 1
    assert "requires a live backend" in r.stderr


def test_tsapa_cli_print_prompt(tmp_path: Path):
    f = tmp_path / "draft.txt"
    f.write_text(TEXT, encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "rewrite_text.py"),
            str(f),
            "--strength",
            "tsapa",
            "--generations",
            "3",
            "--population",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "3 generations" in r.stdout
    assert "8 diverse" in r.stdout
