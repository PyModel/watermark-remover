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
# Fake test helpers
# ---------------------------------------------------------------------------


def _gemini_success(verdict: str | None = None, score: float | None = None) -> dict:
    candidate: dict = {}
    if verdict is not None:
        candidate["content"] = {"parts": [{"text": verdict}]}
    if score is not None:
        candidate["attributionMetadata"] = {"syntheticText": {"score": score}}
    return {"candidates": [candidate]}


class _FakeCommandResult:
    """Mimic external_command.CommandResult: raw bytes plus decoded text views."""

    def __init__(self, returncode: int, stdout: str | bytes = "", stderr: str | bytes = ""):
        self.returncode = returncode
        self.stdout = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
        self.stderr = stderr.encode("utf-8") if isinstance(stderr, str) else stderr

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


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


def test_gemini_disabled_with_api_key(monkeypatch):
    """When configured, the detector still reports unavailable (unsupported task type)."""
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert report["available"] is False
    assert "DETECT_TEXT_WATERMARK" in report["error"]
    assert report["detector"] == "gemini-synthid-text"
    assert report["vendor"] == "google"


def test_gemini_available_is_false_until_endpoint_is_supported(monkeypatch):
    """available() must never advertise a detect() that cannot run.

    The DETECT_TEXT_WATERMARK task type is not supported by the Gemini
    generateContent API yet, so available() stays False even when a key is
    configured — /capabilities and run_text_detectors() must not surface it.
    """
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    assert text_detectors.GeminiSynthIDTextDetector().available() is False
    assert text_detectors.detector_status()["gemini-synthid-text"] is False
    # A disabled detector must never be selected by the usable-only runner.
    assert text_detectors.run_text_detectors("some text") == []


def test_gemini_http_error(monkeypatch):
    """Disabled detector never reaches request_json, so no HTTP call is made."""
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise text_detectors.LayerBHTTPError("Layer B HTTP request failed with HTTP 401")

    monkeypatch.setattr(text_detectors, "request_json", boom)
    report = text_detectors.GeminiSynthIDTextDetector().detect("some text")
    assert calls["n"] == 0, "disabled detector must not call request_json"
    assert report["available"] is False
    assert "DETECT_TEXT_WATERMARK" in report["error"]


def test_gemini_oversize_skips(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_GEMINI_MAX_CHARS", "10")
    report = text_detectors.GeminiSynthIDTextDetector().detect("x" * 100)
    assert report["available"] is True
    assert report["skipped"] is True
    assert report["is_watermarked"] is None


def test_gemini_malformed_max_chars_env_does_not_crash(monkeypatch):
    """Malformed WATERMARKS_GEMINI_MAX_CHARS falls back to default; detector still disabled."""
    monkeypatch.setenv("WATERMARKS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("WATERMARKS_GEMINI_MAX_CHARS", "not-a-number")
    report = text_detectors.GeminiSynthIDTextDetector().detect("short text")
    assert report["available"] is False
    assert "DETECT_TEXT_WATERMARK" in report["error"]


# --- parse_gemini_detect_response (standalone unit tests) ------------------


def test_parse_gemini_verdict_watermarked():
    data = _gemini_success(verdict="Likely AI-generated")
    parsed = text_detectors.parse_gemini_detect_response(data)
    assert parsed["is_watermarked"] is True
    assert parsed["verdict"] == "Likely AI-generated"


def test_parse_gemini_verdict_unlikely():
    data = _gemini_success(verdict="Unlikely AI-generated")
    parsed = text_detectors.parse_gemini_detect_response(data)
    assert parsed["is_watermarked"] is False


def test_parse_gemini_numeric_score():
    data = _gemini_success(score=0.87)
    parsed = text_detectors.parse_gemini_detect_response(data)
    assert parsed["is_watermarked"] is True
    assert parsed["score"] == 0.87


def test_parse_gemini_no_candidates():
    with pytest.raises(text_detectors.DetectorError, match="no candidates"):
        text_detectors.parse_gemini_detect_response({"candidates": []})


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
    # Verify the report merge preserves caller-facing metadata from report.
    assert report["detector"] == "markllm"
    assert report["vendor"] == "open-llm"
    assert report["scheme"] == "kgw"


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


def test_markllm_reports_are_json_safe_with_bytes_output(monkeypatch, tmp_path):
    """CommandResult bytes must be decoded before touching the report.

    json.dumps() on a report whose error field is raw bytes raises TypeError;
    the adapter must use CommandResult.stdout_text/stderr_text.
    """
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    monkeypatch.setattr(
        text_detectors,
        "run_command",
        lambda *a, **k: _FakeCommandResult(3, stderr=b"missing deps"),
    )
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    report = det.detect("hello")
    assert report["available"] is False
    assert report["error"] == "missing deps"
    assert isinstance(report["error"], str)
    json.dumps(report)  # must not raise on a bytes stderr


def test_markllm_passes_rlimit_as_to_child_at_subprocess_boundary(monkeypatch, tmp_path):
    """WATERMARKS_MARKLLM_RLIMIT_AS must reach the child via --rlimit-as."""
    if os.name == "nt":
        pytest.skip("rlimit is POSIX-only; the adapter omits --rlimit-as on Windows")
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    monkeypatch.setenv("WATERMARKS_MARKLLM_RLIMIT_AS", "0x10000000")
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    det.detect("hello")
    argv = seen["argv"]
    assert "--rlimit-as" in argv
    assert argv[argv.index("--rlimit-as") + 1] == "268435456"


def test_markllm_omits_rlimit_as_when_unset(monkeypatch, tmp_path):
    upstream = tmp_path / "MarkLLM"
    upstream.mkdir()
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["argv"] = cmd
        return _FakeCommandResult(0, stdout="{}")

    monkeypatch.setattr(text_detectors, "run_command", fake_run)
    det = text_detectors.MarkLLMTextDetector(upstream_dir=str(upstream))
    det.detect("hello")
    assert "--rlimit-as" not in seen["argv"]


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
