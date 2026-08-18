"""Tests for text_detectors.py (vendor/research text-watermark detectors).

Adapted from THEIRS test_text_detectors.py to monkeypatch through the OURS
seams (layer_b_http.request_json, external_command.run_command) instead of
urllib.request.urlopen and subprocess.run directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import text_detectors

# ---------------------------------------------------------------------------
# Fake opener adapters
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Mimic the urllib response that layer_b_http.request_json expects."""

    def __init__(self, data: dict | bytes):
        if isinstance(data, dict):
            self._data = json.dumps(data).encode("utf-8")
        else:
            self._data = data
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self._data


class _FakeOpener:
    """Stand-in for urllib.request.build_opener() output."""

    def __init__(self, response: _FakeResponse | Exception):
        self._response = response

    def open(self, request, timeout=None):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _gemini_success(verdict: str | None = None, score: float | None = None) -> dict:
    candidate: dict = {}
    if verdict is not None:
        candidate["content"] = {"parts": [{"text": verdict}]}
    if score is not None:
        candidate["attributionMetadata"] = {"syntheticText": {"score": score}}
    return {"candidates": [candidate]}


class _FakeCommandResult:
    """Mimic external_command.run_command() return shape."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "WATERMARKS_GEMINI_API_KEY",
        "WATERMARKS_GEMINI_MODEL",
        "WATERMARKS_GEMINI_MAX_CHARS",
        "WATERMARKS_MARKLLM_SCHEME",
        "MARKLLM_DIR",
        "WATERMARKS_MARKLLM_TIMEOUT",
        "WATERMARKS_MARKLLM_RLIMIT_AS",
    ):
        monkeypatch.delenv(key, raising=False)


# --- Gemini ----------------------------------------------------------------


def test_gemini_unconfigured():
    report = text_detectors.GeminiSynthIDTextDetector().detect("hello")
    assert report["available"] is False
    assert "WATERMARKS_GEMINI_API_KEY" in report["error"]
    assert text_detectors.GeminiSynthIDTextDetector().available() is False


def test_gemini_verdict_watermarked(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors,
        "request_json",
        lambda *a, **kw: _gemini_success(verdict="Likely AI-generated"),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["verdict"] == "Likely AI-generated"


def test_gemini_verdict_unlikely(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors,
        "request_json",
        lambda *a, **kw: _gemini_success(verdict="Unlikely AI-generated"),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["is_watermarked"] is False


def test_gemini_numeric_score(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors,
        "request_json",
        lambda *a, **kw: _gemini_success(score=0.87),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["is_watermarked"] is True
    assert report["score"] == 0.87


def test_gemini_http_error(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")

    def boom(*a, **kw):
        raise text_detectors.LayerBHTTPError("Layer B HTTP request failed with HTTP 401")

    monkeypatch.setattr(text_detectors, "request_json", boom)
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is False
    assert "HTTP 401" in report["error"]


def test_gemini_retries_once_on_429(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise text_detectors.LayerBHTTPError("Layer B HTTP request failed with HTTP 429")
        return _gemini_success(verdict="Likely AI-generated")

    monkeypatch.setattr(text_detectors, "request_json", flaky)
    monkeypatch.setattr(text_detectors.time, "sleep", lambda _: None)
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert calls["n"] == 2
    assert report["is_watermarked"] is True


def test_gemini_oversize_skips(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_GEMINI_MAX_CHARS", "10")
    report = text_detectors.GeminiSynthIDTextDetector().detect("x" * 100)
    assert report["available"] is True
    assert report["skipped"] is True
    assert report["is_watermarked"] is None


def test_gemini_malformed_max_chars_env_does_not_crash(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_GEMINI_MAX_CHARS", "not-a-number")
    monkeypatch.setattr(
        text_detectors,
        "request_json",
        lambda *a, **kw: _gemini_success(verdict="Likely AI-generated"),
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("short text")
    assert report["available"] is True
    assert report["is_watermarked"] is True


def test_gemini_no_candidates(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        text_detectors,
        "request_json",
        lambda *a, **kw: {"candidates": []},
    )
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is False
    assert "no candidates" in report["error"]


# --- MarkLLM ---------------------------------------------------------------


def test_markllm_unconfigured():
    assert text_detectors.MarkLLMTextDetector().available() is False
    report = text_detectors.MarkLLMTextDetector().detect("hello")
    assert report["available"] is False
    assert "MARKLLM_DIR" in report["error"]


def test_markllm_success(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    payload = {"is_watermarked": True, "score": 4.2, "threshold": 4.0}
    monkeypatch.setattr(
        text_detectors,
        "run_command",
        lambda *a, **k: _FakeCommandResult(0, stdout=json.dumps(payload)),
    )
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert "research harness" in report["note"]


def test_markllm_unavailable_exit3(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    monkeypatch.setattr(
        text_detectors,
        "run_command",
        lambda *a, **k: _FakeCommandResult(3, stderr="missing deps"),
    )
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    assert report["available"] is False
    assert "missing deps" in report["error"]


def test_markllm_scheme_env(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    monkeypatch.setenv("WATERMARKS_MARKLLM_SCHEME", "synthid")
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    det.detect("hello")
    argv = seen["argv"]
    assert "--scheme" in argv
    assert argv[argv.index("--scheme") + 1] == "synthid"


def test_markllm_prefers_checkout_venv(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    if os.name == "nt":
        venv_python = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = upstream / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    assert det.available() is True
    det.detect("hello")
    assert seen["argv"][0] == str(venv_python)


def test_markllm_falls_back_to_sys_executable(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    det.detect("hello")
    assert seen["argv"][0] == sys.executable


def test_markllm_ctor_overrides_passed_to_adapter(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    (upstream / "watermark").mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    det = text_detectors.MarkLLMTextDetector(
        scheme="synthid",
        upstream_dir=str(upstream),
        model="opt-1.3b",
        timeout=5,
    )
    det.detect("hello")
    argv = seen["argv"]
    assert argv[argv.index("--scheme") + 1] == "synthid"
    assert argv[argv.index("--model") + 1] == "opt-1.3b"
    assert argv[argv.index("--upstream-dir") + 1] == str(upstream.resolve())


def test_markllm_ctor_available_with_override(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    assert det.available() is True


def test_run_all_text_detectors_can_exclude_markllm(monkeypatch):
    def boom():
        pytest.fail("must not construct MarkLLM when excluded")

    monkeypatch.setattr(text_detectors, "MarkLLMTextDetector", boom)
    reports = text_detectors.run_all_text_detectors("hello", include_markllm=False)
    assert len(reports) == 2
    assert {r["detector"] for r in reports} == {"gemini-synthid-text", "claude-text"}


def test_run_all_text_detectors_injects_markllm_instance(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream), scheme="synthid")
    seen: list = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    text_detectors.run_all_text_detectors("hello", markllm=det)
    markllm_cmd = next(c for c in seen if "--scheme" in c)
    assert markllm_cmd[markllm_cmd.index("--scheme") + 1] == "synthid"


# --- Claude placeholder ----------------------------------------------------


def test_claude_placeholder():
    det = text_detectors.ClaudeTextDetector()
    assert det.available() is False
    report = det.detect("hello")
    assert report["available"] is False
    assert "WATERMARKS_CLAUDE_API_KEY" in report["error"]


# --- Registry --------------------------------------------------------------


def test_detector_status_keys():
    status = text_detectors.detector_status()
    assert set(status) == {"gemini-synthid-text", "markllm", "claude-text"}


def test_run_all_text_detectors_length():
    reports = text_detectors.run_all_text_detectors("hello")
    assert len(reports) == 3
    assert all("detector" in r for r in reports)


def test_run_text_detectors_filters_unavailable():
    reports = text_detectors.run_text_detectors("hello")
    assert reports == []
