"""Offline tests for TSAPA evolutionary Layer B."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
from rewrite_text import rewrite
from tsapa import (
    Candidate,
    chunk_text,
    cosine,
    crossover,
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
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
        strength="tsapa",
        lang="French",
        original_lang="English",
        timeout=30,
        layer_a_after=True,
        generations=7,
        population=20,
    )
    assert "7 generations" in out
    assert "20 diverse" in out
    assert "Pareto" in out
    assert TEXT in out
    assert info["mode"] == "print-prompt"


def test_live_tsapa_reports_adapter_fallbacks(monkeypatch):
    import tsapa as tsapa_module

    monkeypatch.setattr(
        rewrite_text,
        "call_openai_compatible",
        lambda _base, _model, prompt, _key, _timeout: fake_llm(prompt),
    )
    monkeypatch.setattr(
        tsapa_module,
        "http_pll",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no logprobs")),
    )
    monkeypatch.setattr(
        tsapa_module,
        "http_embed",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no embeddings")),
    )
    _out, info = rewrite(
        TEXT,
        backend="openai-compatible",
        model="test-model",
        base_url="http://127.0.0.1:9999",
        api_key="test-key",
        strength="tsapa",
        lang="French",
        original_lang="English",
        timeout=1,
        layer_a_after=True,
        generations=0,
        population=2,
    )
    fallbacks = info["tsapa"]["adapter_fallbacks"]
    assert fallbacks["pll"] >= 2
    assert fallbacks["embedding"] >= 1


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
