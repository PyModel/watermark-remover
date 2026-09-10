"""Tests for zero-LLM statistical & stylometric AI-text detector."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "remove-ai-marks" / "scripts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS_DIR))

from score_stylometry import (
    compute_burstiness,
    compute_mattr,
    extract_sentences,
    extract_words,
    scan_ai_phrases,
    score_text_stylometry,
)


def test_sentence_and_word_extraction():
    sample = (
        "Hello world! This is a test. Here is a code block:\n```python\nprint('hi')\n```\nDone."
    )
    sentences = extract_sentences(sample)
    assert len(sentences) == 4
    assert sentences[0] == "Hello world!"
    assert sentences[1] == "This is a test."
    assert sentences[2] == "Here is a code block:"
    assert sentences[3] == "Done."

    words = extract_words("Hello, World! Testing 1-2-3.")
    assert "hello" in words
    assert "world" in words
    assert "testing" in words


def test_burstiness_variance():
    # Empty / single sentence edge cases
    assert compute_burstiness([]) == (0.0, 0.0, 0.0)
    assert compute_burstiness(["Only one sentence here."])[1] == 0.0

    # Uniform sentences (every sentence is 5 words) -> CV should be 0.0
    uniform = [
        "One two three four five.",
        "Six seven eight nine ten.",
        "Alpha beta gamma delta epsilon.",
    ]
    mean, std, cv = compute_burstiness(uniform)
    assert mean == 5.0
    assert std == 0.0
    assert cv == 0.0

    # Bursty sentences (highly varied lengths: 2 words vs 20 words) -> CV should be high (>0.7)
    bursty = [
        "Too short.",
        "This is an extraordinarily long and verbose sentence crafted intentionally to introduce high variance into the token distribution model.",
        "Short again.",
    ]
    _, _, cv_bursty = compute_burstiness(bursty)
    assert cv_bursty > 0.70


def test_mattr_lexical_diversity():
    words = ["apple"] * 100
    # Completely repetitive words -> MATTR should be 1/50 = 0.02
    mattr_rep = compute_mattr(words, window_size=50)
    assert mattr_rep < 0.05

    # Distinct vocabulary
    distinct = [f"word_{i}" for i in range(100)]
    mattr_dist = compute_mattr(distinct, window_size=50)
    assert mattr_dist == 1.0


def test_ai_phrase_scanner():
    text = (
        "In today's fast-paced digital world, we must delve into this problem. "
        "It is crucial to note that this is a testament to progress."
    )
    matches = scan_ai_phrases(text)
    matched_labels = [m.phrase for m in matches]
    assert "in today's fast-paced world/landscape" in matched_labels
    assert "delve into" in matched_labels
    assert "it is important/crucial to note" in matched_labels
    assert "a testament to" in matched_labels


def test_short_text_floor():
    short_text = "This is a very short text with only nine words."
    report = score_text_stylometry(short_text)
    assert report.status == "insufficient_length"
    assert report.score == 0.0
    assert report.confidence_level == "CLEAN"
    assert any("uncalibrated" in n for n in report.notes)


def test_ai_vs_human_discrimination():
    ai_sample = (FIXTURES_DIR / "stylometry_ai_sample.txt").read_text(encoding="utf-8")
    human_sample = (FIXTURES_DIR / "stylometry_human_sample.txt").read_text(encoding="utf-8")

    ai_report = score_text_stylometry(ai_sample, path="ai_sample.txt")
    human_report = score_text_stylometry(human_sample, path="human_sample.txt")

    assert ai_report.status == "ok"
    assert human_report.status == "ok"

    # AI sample should have high score and HIGH/MEDIUM confidence
    assert ai_report.score >= 0.70
    assert ai_report.confidence_level in ("HIGH", "MEDIUM")
    assert len(ai_report.matched_markers) >= 3

    # Human sample should have low score and CLEAN/LOW confidence
    assert human_report.score < 0.35
    assert human_report.confidence_level in ("CLEAN", "LOW")


def test_oversized_input_truncated_before_scanning():
    # Regression: oversized inputs (>MAX_SCAN_CHARS) must be truncated before the
    # regex scans run, with a truncation note, and complete quickly.
    snippet = "This is a sample sentence with enough words for stylometry analysis. "
    oversized = snippet * 40000  # ~2.6M chars > MAX_SCAN_CHARS (2,000,000)
    assert len(oversized) > 2_000_000

    start = time.monotonic()
    report = score_text_stylometry(oversized, path="oversized.txt")
    elapsed = time.monotonic() - start

    assert any("truncat" in n.lower() for n in report.notes)
    assert report.status == "ok"
    assert 0 < report.word_count < 2_000_000
    assert elapsed < 30


def test_under_cap_input_has_no_truncation_note():
    # Inputs within the cap must behave byte-identically: no truncation note,
    # and the established score is preserved.
    sample = (FIXTURES_DIR / "stylometry_ai_sample.txt").read_text(encoding="utf-8")
    assert len(sample) <= 2_000_000
    report = score_text_stylometry(sample, path="ai_sample.txt")
    assert not any("truncat" in n.lower() for n in report.notes)
    assert report.score >= 0.70


def test_score_stylometry_cli():
    ai_path = FIXTURES_DIR / "stylometry_ai_sample.txt"
    human_path = FIXTURES_DIR / "stylometry_human_sample.txt"

    script = SCRIPTS_DIR / "score_stylometry.py"

    # AI text should trigger exit code 1 (score >= 0.65)
    res_ai = subprocess.run(
        [sys.executable, str(script), str(ai_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_ai.returncode == 1
    data_ai = json.loads(res_ai.stdout)
    assert data_ai["score"] >= 0.65
    assert data_ai["status"] == "ok"

    # Human text should exit 0 (score < 0.65)
    res_human = subprocess.run(
        [sys.executable, str(script), str(human_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_human.returncode == 0
    data_human = json.loads(res_human.stdout)
    assert data_human["score"] < 0.65

    # Custom threshold override
    res_thresh = subprocess.run(
        [sys.executable, str(script), str(ai_path), "--threshold", "0.99"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_thresh.returncode == 0
