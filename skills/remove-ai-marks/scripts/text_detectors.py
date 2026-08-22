#!/usr/bin/env python3
"""Vendor and research text-watermark detectors behind one interface.

Detects statistical (Layer B) text watermarks using vendor-provided or
research detectors. Every detector implements the same small protocol:

    name: str                 stable identifier (surfaced in /capabilities)
    available() -> bool       configured and usable right now
    detect(text) -> dict      JSON-safe report; never raises

Reports follow the fail-soft contract: a detector that is unconfigured,
times out, or errors returns {"available": False, "error": ...} and can
never block cleaning. Every report also carries "configured": whether the
detector was set up to run at all, so an aggregator can tell a detector that
never ran from one that ran and failed. A configured detector that failed is
unresolved evidence, never a clean result.

Detectors:

- gemini-synthid-text — Google's official SynthID-text detector, called
  through the Gemini API (taskType DETECT_TEXT_WATERMARK). Activated by
  WATERMARKS_GEMINI_API_KEY. User text is sent to Google only when the
  operator sets that key.
- markllm — optional research harness (KGW / SynthID schemes) via
  detect_text_watermark.py, activated by MARKLLM_DIR. Same-config-only
  detection; not a vendor oracle.
- claude-text — placeholder for Anthropic's announced text-watermark
  detection API. Reports unavailable until a public endpoint exists; the
  interface it must implement is already defined here.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from common import (
    emit_json,
    read_text_input,
)
from external_command import (
    ExternalCommandTimeout,
    run_command,
)
from layer_b_http import LayerBHTTPError, request_json

SCRIPTS_DIR = Path(__file__).resolve().parent

GEMINI_DETECT_URL = "https://generativelanguage.googleapis.com"
GEMINI_DETECT_ROUTE_TEMPLATE = "/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TIMEOUT = 30.0
DEFAULT_GEMINI_MAX_CHARS = 1_000_000
DEFAULT_MARKLLM_SCHEME = "kgw"
DEFAULT_MARKLLM_TIMEOUT = 600.0
DEFAULT_MARKLLM_OUTPUT_LIMIT = 1_000_000

_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


class DetectorError(RuntimeError):
    """A detector call failed (network, HTTP error, timeout)."""


class TextDetector(Protocol):
    name: str

    def available(self) -> bool: ...

    def detect(self, text: str) -> dict[str, Any]: ...


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Gemini (Google's official SynthID-text detector)
# ---------------------------------------------------------------------------

_WATERMARKED_VERDICTS = ("watermarked", "ai-generated", "ai generated", "likely ai")


def _verdict_is_watermarked(verdict: str | None) -> bool | None:
    """Map the detector model's free-text verdict to a boolean, or None."""
    if not verdict:
        return None
    low = verdict.strip().lower()
    if re.search(r"\b(?:unlikely|no|not)\b", low):
        return False
    return any(marker in low for marker in _WATERMARKED_VERDICTS)


def _extract_numeric_score(candidate: dict[str, Any], top: dict[str, Any]) -> float | None:
    """Pull a numeric watermark score from any of the known response shapes."""
    for container in (candidate, top):
        for key in (
            "syntheticTextScore",
            "synthetic_text_score",
            "watermarkScore",
            "watermark_score",
            "score",
        ):
            value = container.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    attribution = candidate.get("attributionMetadata") or {}
    if isinstance(attribution, dict):
        for key in ("syntheticTextScore", "synthetic_text_score", "score"):
            value = attribution.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        st = attribution.get("syntheticText")
        if isinstance(st, dict):
            for key in ("score", "confidence"):
                value = st.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def parse_gemini_detect_response(data: dict[str, Any]) -> dict[str, Any]:
    """Parse a generateContent response from a DETECT_TEXT_WATERMARK call.

    The endpoint can answer with either a free-text verdict
    ("Likely AI-generated") or a structured score; both shapes are handled
    defensively so upstream schema changes degrade to an error report
    instead of a crash.
    """
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    if not isinstance(candidate, dict):
        candidate = {}

    if not candidate:
        feedback = data.get("promptFeedback") or {}
        block = feedback.get("blockReason")
        if block:
            raise DetectorError(f"Gemini blocked the request: {block}")
        raise DetectorError("Gemini returned no candidates")

    verdict: str | None = None
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    if parts and isinstance(parts[0], dict):
        verdict = parts[0].get("text")

    score = _extract_numeric_score(candidate, data)
    is_watermarked = _verdict_is_watermarked(verdict)
    if is_watermarked is None and score is not None:
        is_watermarked = score >= 0.5

    if verdict is None and score is None:
        raise DetectorError("unexpected Gemini response (no verdict or score)")

    raw = {
        key: candidate[key]
        for key in ("attributionMetadata", "finishReason", "index")
        if candidate.get(key) is not None
    }
    return {
        "is_watermarked": is_watermarked,
        "score": score,
        "verdict": verdict,
        "raw": raw,
    }


def _call_gemini(
    endpoint: str,
    route: str,
    body: dict[str, Any],
    api_key: str,
    timeout: float,
    opener: Any = None,
) -> dict[str, Any]:
    """POST *body* to *route* under *endpoint*, retrying once on transient failures.

    Uses the shared layer_b_http.request_json seam. Distinguishes retryable
    HTTP codes (429, 500, 502, 503, 504) by inspecting the error message
    text layer_b_http emits; the bound is intentionally narrow so we never
    retry on 4xx (except 429) or on programming errors.
    """
    if urlparse(endpoint).scheme not in ("http", "https"):
        raise DetectorError(f"refusing non-http(s) Gemini endpoint: {endpoint}")
    headers = {"x-goog-api-key": api_key}
    last_error: LayerBHTTPError | None = None
    for attempt in range(2):
        try:
            return request_json(
                endpoint,
                route,
                body,
                headers=headers,
                timeout=timeout,
                opener=opener,
            )
        except LayerBHTTPError as error:
            last_error = error
            retryable = _is_retryable_layer_b_error(error)
            if not retryable or attempt == 1:
                raise DetectorError(str(error)) from error
        time.sleep(1.0)
    if last_error is not None:
        raise DetectorError(str(last_error))
    raise DetectorError("Gemini API call failed")


def _is_retryable_layer_b_error(error: LayerBHTTPError) -> bool:
    msg = str(error)
    return any(f"HTTP {code}" in msg for code in _RETRYABLE_HTTP_CODES)


class GeminiSynthIDTextDetector:
    """Google's official SynthID-text detector via the Gemini API."""

    name = "gemini-synthid-text"
    vendor = "google"

    def available(self) -> bool:
        # The DETECT_TEXT_WATERMARK task type is not supported by the Gemini
        # generateContent API yet, so this detector cannot run regardless of
        # configuration. Never advertise a detector detect() cannot execute;
        # flip this once a supported watermark-detection endpoint exists.
        return False

    def detect(self, text: str) -> dict[str, Any]:
        api_key = os.environ.get("WATERMARKS_GEMINI_API_KEY", "").strip()
        if not api_key:
            return {
                "detector": self.name,
                "vendor": self.vendor,
                "available": False,
                "configured": False,
                "error": "WATERMARKS_GEMINI_API_KEY not set",
            }

        max_chars = _env_int("WATERMARKS_GEMINI_MAX_CHARS", DEFAULT_GEMINI_MAX_CHARS)
        if len(text) > max_chars:
            return {
                "detector": self.name,
                "vendor": self.vendor,
                "available": True,
                "configured": True,
                "skipped": True,
                "reason": f"text longer than {max_chars} chars",
                "is_watermarked": None,
            }

        return {
            "detector": self.name,
            "vendor": self.vendor,
            "available": False,
            "configured": False,
            "error": (
                "DETECT_TEXT_WATERMARK task type is not supported by the "
                "Gemini generateContent API; this detector is disabled until "
                "a supported watermark-detection endpoint is available."
            ),
        }


# ---------------------------------------------------------------------------
# MarkLLM (open-source research harness: KGW / SynthID schemes)
# ---------------------------------------------------------------------------


def _venv_python(upstream: Path) -> Path | None:
    """Prefer the MarkLLM checkout's venv interpreter, if it exists."""
    if os.name == "nt":
        candidate = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = upstream / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _markllm_rlimit_as() -> int | None:
    """Optional RLIMIT_AS guard for the MarkLLM child; None means "no limit".

    torch/CUDA usually needs large address spaces, so this is opt-in via
    WATERMARKS_MARKLLM_RLIMIT_AS (byte count, hex/octal allowed). POSIX only.
    """
    raw = os.environ.get("WATERMARKS_MARKLLM_RLIMIT_AS")
    if not raw or os.name != "posix":
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


class MarkLLMTextDetector:
    """Same-config-only research detection via detect_text_watermark.py.

    Constructor overrides (scheme, upstream_dir, model, timeout) take
    precedence over the environment, so callers such as rewrite_text.py can
    keep CLI flags driving the harness. When the MarkLLM checkout has a
    venv, its interpreter runs the child process; otherwise the current
    interpreter is used (the service image bundles the harness deps).
    """

    name = "markllm"

    def __init__(
        self,
        *,
        scheme: str | None = None,
        upstream_dir: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._scheme = scheme
        self._upstream_dir = upstream_dir
        self._model = model
        self._timeout = timeout

    def available(self) -> bool:
        upstream = self._upstream_dir or os.environ.get("MARKLLM_DIR", "").strip()
        return bool(upstream)

    def detect(self, text: str) -> dict[str, Any]:
        upstream = self._upstream_dir or os.environ.get("MARKLLM_DIR", "").strip()
        scheme = (
            self._scheme
            or os.environ.get("WATERMARKS_MARKLLM_SCHEME", "")
            or DEFAULT_MARKLLM_SCHEME
        )
        report: dict[str, Any] = {
            "detector": self.name,
            "scheme": scheme,
            "vendor": "open-llm",
            "available": False,
            "configured": bool(upstream),
        }
        if not upstream:
            report["error"] = "MARKLLM_DIR not set"
            return report

        script = SCRIPTS_DIR / "detect_text_watermark.py"
        timeout = (
            self._timeout
            if self._timeout is not None
            else _env_float("WATERMARKS_MARKLLM_TIMEOUT", DEFAULT_MARKLLM_TIMEOUT)
        )
        venv_python = _venv_python(Path(upstream).expanduser().resolve())
        python = str(venv_python) if venv_python is not None else sys.executable

        # Persist text to a temp file for the child process.
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", encoding="utf-8", delete=False
            ) as f:
                tmp = f.name
                f.write(text)

            cmd = [python, str(script), "detect", tmp, "--scheme", scheme, "--json"]
            if self._model:
                cmd += ["--model", self._model]
            if self._upstream_dir:
                cmd += ["--upstream-dir", str(Path(upstream).expanduser().resolve())]
            rlimit_as = _markllm_rlimit_as()
            if rlimit_as is not None:
                cmd += ["--rlimit-as", str(rlimit_as)]

            try:
                result = run_command(
                    cmd,
                    timeout=timeout,
                    output_limit=DEFAULT_MARKLLM_OUTPUT_LIMIT,
                )
            except ExternalCommandTimeout:
                report["error"] = "MarkLLM detection timed out"
                return report
            # Use the decoded text views: CommandResult.stderr/stdout are raw
            # bytes, and a bytes error field would make the report un-serializable.
            if result.returncode == 3:
                report["error"] = result.stderr_text.strip()[:400] or "MarkLLM unavailable"
                return report
            if result.returncode != 0:
                report["error"] = (
                    result.stderr_text.strip()[:400] or f"MarkLLM exit {result.returncode}"
                )
                return report
            try:
                payload = json.loads(result.stdout_text or "{}")
            except json.JSONDecodeError as e:
                report["error"] = f"bad MarkLLM JSON: {e}"
                return report
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    Path(tmp).unlink()

        if not isinstance(payload, dict):
            report["error"] = "bad MarkLLM response"
            return report
        report["available"] = True
        report.update(payload)
        report["detector"] = self.name
        report["note"] = (
            "MarkLLM is a research harness: detection is only valid against the "
            "same scheme config and keys used at generation; not a vendor detector."
        )
        return report


# ---------------------------------------------------------------------------
# Claude (Anthropic) — announced detector API, not yet public
# ---------------------------------------------------------------------------


class ClaudeTextDetector:
    """Placeholder for Anthropic's announced text-watermark detection API.

    Anthropic has announced a watermark detection API for Claude-generated
    text; no public endpoint exists yet. When it ships, set
    WATERMARKS_CLAUDE_API_KEY, flip available() to check it, and fill in
    detect() against the documented endpoint.
    """

    name = "claude-text"
    vendor = "anthropic"

    def available(self) -> bool:
        return False

    def detect(self, text: str) -> dict[str, Any]:
        return {
            "detector": self.name,
            "vendor": self.vendor,
            "available": False,
            "configured": False,
            "error": (
                "Anthropic has announced a text-watermark detection API for "
                "Claude; no public endpoint is available yet. When it ships, "
                "set WATERMARKS_CLAUDE_API_KEY and implement ClaudeTextDetector."
            ),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def all_detectors(
    markllm: MarkLLMTextDetector | None = None, *, include_markllm: bool = True
) -> list[TextDetector]:
    detectors: list[TextDetector] = [GeminiSynthIDTextDetector()]
    if include_markllm:
        detectors.append(markllm or MarkLLMTextDetector())
    detectors.append(ClaudeTextDetector())
    return detectors


def detector_status() -> dict[str, bool]:
    """Configured/usable status per detector (for /capabilities)."""
    return {d.name: d.available() for d in all_detectors()}


def run_all_text_detectors(
    text: str,
    *,
    markllm: MarkLLMTextDetector | None = None,
    include_markllm: bool = True,
) -> list[dict[str, Any]]:
    """Run every detector (including unavailable ones, with reasons).

    markllm injects a caller-parameterized MarkLLM detector (e.g. one
    driven by rewrite_text.py CLI flags); pass include_markllm=False to
    exclude the MarkLLM harness entirely.
    """
    return [d.detect(text) for d in all_detectors(markllm, include_markllm=include_markllm)]


def run_text_detectors(
    text: str,
    *,
    markllm: MarkLLMTextDetector | None = None,
    include_markllm: bool = True,
) -> list[dict[str, Any]]:
    """Run only the detectors that are configured and usable."""
    return [
        d.detect(text)
        for d in all_detectors(markllm, include_markllm=include_markllm)
        if d.available()
    ]


def main() -> int:
    """CLI: list detector status (default) or run all detectors on a path."""
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", nargs="?", default="-", help="Text file to detect ('-' for stdin)")
    p.add_argument(
        "--scheme",
        default=None,
        help="MarkLLM scheme (kgw|ringid|...) for the MarkLLM detector",
    )
    p.add_argument("--upstream-dir", default=None, help="MarkLLM checkout directory")
    p.add_argument("--model", default=None, help="MarkLLM model name")
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    p.add_argument(
        "--status",
        action="store_true",
        help="Print detector status (configured/usable) and exit 0",
    )

    args = p.parse_args()

    if args.status:
        status = detector_status()
        if args.json:
            emit_json(status)
        else:
            for name, ok in status.items():
                mark = "available" if ok else "unavailable"
                print(f"{name}: {mark}")
        return 0

    text = read_text_input(args.path)
    if text is None:
        return 2

    markllm = MarkLLMTextDetector(
        scheme=args.scheme,
        upstream_dir=args.upstream_dir,
        model=args.model,
    )
    results = run_all_text_detectors(text, markllm=markllm)
    if args.json:
        emit_json({"path": args.path, "results": results})
    else:
        for r in results:
            detector = r.get("detector", "?")
            available = r.get("available", False)
            skipped = r.get("skipped", False)
            err = r.get("error")
            verdict = r.get("verdict")
            score = r.get("score")
            is_wm = r.get("is_watermarked")
            label = "available" if available else "unavailable"
            if skipped:
                label = f"skipped ({r.get('reason')})"
            line = f"{detector}: {label}"
            if verdict is not None:
                line += f" verdict={verdict!r}"
            if score is not None:
                line += f" score={score:.3f}"
            if is_wm is not None:
                line += f" is_watermarked={is_wm}"
            if err:
                line += f" error={err[:200]}"
            print(line)

    # Exit 0 if any available detector ran; 3 if all unavailable.
    if any(r.get("available") and not r.get("skipped") for r in results):
        return 0
    return 3


if __name__ == "__main__":
    sys.exit(main())
