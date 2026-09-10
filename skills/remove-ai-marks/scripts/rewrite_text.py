#!/usr/bin/env python3
"""Layer B optional rewrite hook for statistical (token-sampling) watermarks.

Backends:
  print-prompt       — emit prompt only (default; CI-safe, no model)
  ollama             — POST to Ollama /api/chat
  openai-compatible  — POST to OpenAI-style /v1/chat/completions

Env (optional):
  WATERMARKS_REWRITE_BACKEND
  WATERMARKS_REWRITE_BASE_URL
  WATERMARKS_REWRITE_MODEL
  WATERMARKS_REWRITE_API_KEY      (env-only; never pass keys on argv)
  WATERMARKS_REWRITE_DISABLE_THINKING  — true/false; OpenAI-compatible extension
  WATERMARKS_REWRITE_ALLOW_REMOTE     — set to 1 to allow non-loopback endpoints
  WATERMARKS_REWRITE_REASONING_EFFORT — none/low/medium/high/off for OpenAI-compatible

Security notes:
  - Only http(s) endpoints are accepted; redirects are refused outright so an
    Authorization header (API key) can never be re-sent to an unvalidated host.
  - Non-loopback endpoints are denied unless WATERMARKS_REWRITE_ALLOW_REMOTE=1
    (or --allow-remote) is set explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layer_b_http
from common import (
    cleaned_path,
    eprint,
    read_bool_env,
    read_flag_env,
    read_text_input,
    validate_output_path,
    write_text_output,
)
from layer_b_http import LayerBHTTPError
from text_unicode import clean_text

PROMPTS = {
    "paraphrase": (
        "Rewrite the following text so that every sentence uses different wording and "
        "structure while preserving all facts, numbers, names, and technical identifiers. "
        "Do not add or remove claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "humanize": (
        "Rewrite the following text so it reads as if a human wrote it from scratch. "
        "Vary sentence rhythm and length, replace formulaic AI-style transitions and "
        "filler with concrete natural phrasing, and use plain, varied wording. Preserve "
        "all facts, numbers, names, and technical identifiers. Do not add or remove "
        "claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "code": (
        "Rewrite the natural-language parts of this code — comments, docstrings, and "
        "string literals — using different wording. Rename local variables, function "
        "parameters, and private helper names to semantically equivalent names. Preserve "
        "program behavior, public API names, and all values that affect output. Output "
        "only the rewritten code.\n\n---\n{TEXT}"
    ),
}


TSAPA_PACK = (
    "Run an evolutionary paraphrase attack on the text below (TSAPA-style, ACL 2026).\n"
    "1. Split the text into chunks of ~1200 characters.\n"
    "2. For each chunk, generate {POP} diverse paraphrase candidates (vary register, "
    "conciseness, and sentence structure).\n"
    "3. Iterate for {GEN} generations:\n"
    "   - Score every candidate on attack fitness (fluency/perplexity + n-gram diversity "
    "+ lexical diversity vs. the original) AND fidelity (semantic similarity).\n"
    "   - Keep the Pareto front (maximize both objectives); prefer uncrowded candidates.\n"
    "   - Crossover: swap sentences between candidate pairs.\n"
    "   - Mutate: rewrite ONLY the most machine-like (lowest-perplexity) sentences.\n"
    "4. Pick the knee point (closest to ideal on both objectives) per chunk.\n"
    "Output only the final rewritten text.\n\n---\n{TEXT}"
)

REWRITE_BACKENDS = ("print-prompt", "ollama", "openai-compatible")
LIVE_REWRITE_BACKENDS = ("ollama", "openai-compatible")
REWRITE_STRENGTHS = ("paraphrase", "backtranslate", "structural", "humanize", "code", "tsapa")
REASONING_EFFORTS = ("none", "low", "medium", "high", "off")
#: Hosts that may receive document content without an explicit opt-in.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
#: Private alias kept for in-module call sites predating the export.
_LOOPBACK_HOSTS = LOOPBACK_HOSTS


class RewriteConfigurationError(ValueError):
    """Invalid Layer B mode or required provider configuration."""


@dataclass(frozen=True, slots=True)
class RewritePlan:
    """Immutable policy for one Layer B rewrite."""

    backend: str = "print-prompt"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    strength: str = "paraphrase"
    lang: str = "French"
    original_lang: str = "English"
    timeout: float = 120.0
    layer_a_after: bool = True
    generations: int = 5
    population: int = 12
    disable_thinking: bool = False
    temperature: float = 0.9
    candidates: int = 1
    reasoning_effort: str | None = None
    allow_remote: bool = False
    markllm_scheme: str | None = None
    markllm_dir: str | None = None
    markllm_model: str | None = None
    markllm_timeout: float = 180.0

    def __post_init__(self) -> None:
        if self.backend not in REWRITE_BACKENDS:
            raise RewriteConfigurationError(f"unknown backend: {self.backend}")
        if self.strength not in REWRITE_STRENGTHS:
            raise RewriteConfigurationError(f"unknown strength: {self.strength}")
        if not isinstance(self.disable_thinking, bool):
            raise TypeError("disable_thinking must be a bool")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a finite positive number")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise RewriteConfigurationError("timeout must be a finite positive number")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a finite positive number")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise RewriteConfigurationError("temperature must be a finite positive number")
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int):
            raise TypeError("candidates must be an integer")
        if self.candidates < 1:
            raise RewriteConfigurationError("candidates must be >= 1")
        if self.reasoning_effort not in (None, *REASONING_EFFORTS):
            raise RewriteConfigurationError(f"unknown reasoning_effort: {self.reasoning_effort}")
        if not isinstance(self.allow_remote, bool):
            raise TypeError("allow_remote must be a bool")
        if isinstance(self.markllm_timeout, bool) or not isinstance(
            self.markllm_timeout, (int, float)
        ):
            raise TypeError("markllm_timeout must be a finite positive number")
        if not math.isfinite(self.markllm_timeout) or self.markllm_timeout <= 0:
            raise RewriteConfigurationError("markllm_timeout must be a finite positive number")
        if self.strength == "tsapa":
            if not isinstance(self.generations, int) or isinstance(self.generations, bool):
                raise TypeError("generations must be an integer")
            if self.generations < 0:
                raise RewriteConfigurationError("generations must be >= 0")
            if not isinstance(self.population, int) or isinstance(self.population, bool):
                raise TypeError("population must be an integer")
            if self.population < 2:
                raise RewriteConfigurationError("population must be >= 2")
        if self.backend in LIVE_REWRITE_BACKENDS:
            if not self.model:
                raise RewriteConfigurationError(
                    "error: --model required for ollama/openai-compatible backends"
                )
            if not self.base_url:
                raise RewriteConfigurationError(
                    "error: --base-url required for ollama/openai-compatible backends"
                )

    @classmethod
    def prompt(
        cls,
        strength: str,
        *,
        lang: str = "French",
        original_lang: str = "English",
        generations: int = 5,
        population: int = 12,
    ) -> RewritePlan:
        return cls(
            strength=strength,
            lang=lang,
            original_lang=original_lang,
            layer_a_after=False,
            generations=generations,
            population=population,
        )

    @classmethod
    def live_from_environment(
        cls,
        strength: str,
        *,
        label: str | None = None,
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        generations: int = 5,
        population: int = 12,
        lang: str | None = None,
        original_lang: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        candidates: int | None = None,
        reasoning_effort: str | None = None,
        disable_thinking: bool | None = None,
        allow_remote: bool | None = None,
    ) -> RewritePlan:
        """Resolve a live Layer B plan from explicit values, then the environment.

        Precedence is explicit argument > ``WATERMARKS_REWRITE_*`` > dataclass
        default, so a caller that collects settings interactively (the TUI) and
        a caller that only reads the environment (``wm --tsapa``) build the same
        object through the same checks.  *label* names the surface in the error
        message so the CLI can still say ``--tsapa`` where that is what failed.
        """
        origin = label or f"--rewrite {strength}"
        resolved_backend = backend or os.environ.get("WATERMARKS_REWRITE_BACKEND", "print-prompt")
        if resolved_backend not in LIVE_REWRITE_BACKENDS:
            raise RewriteConfigurationError(
                f"{origin} requires a live backend; set "
                "WATERMARKS_REWRITE_BACKEND=ollama|openai-compatible"
            )
        resolved_model = model or os.environ.get("WATERMARKS_REWRITE_MODEL")
        if not resolved_model:
            raise RewriteConfigurationError(
                f"{origin} requires a model: pass --rewrite-model, set the TUI's model "
                "field, or export WATERMARKS_REWRITE_MODEL"
            )
        defaults = cls()
        if disable_thinking is None:
            disable_thinking = read_bool_env("WATERMARKS_REWRITE_DISABLE_THINKING")
        if allow_remote is None:
            allow_remote = read_flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
        return cls(
            backend=resolved_backend,
            model=resolved_model,
            base_url=base_url
            or os.environ.get("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
            api_key=api_key or os.environ.get("WATERMARKS_REWRITE_API_KEY"),
            strength=strength,
            lang=lang if lang is not None else defaults.lang,
            original_lang=(original_lang if original_lang is not None else defaults.original_lang),
            timeout=timeout if timeout is not None else defaults.timeout,
            temperature=temperature if temperature is not None else defaults.temperature,
            candidates=candidates if candidates is not None else defaults.candidates,
            reasoning_effort=(
                reasoning_effort
                if reasoning_effort is not None
                else os.environ.get("WATERMARKS_REWRITE_REASONING_EFFORT") or None
            ),
            generations=generations,
            population=population,
            disable_thinking=disable_thinking,
            allow_remote=allow_remote,
        )

    @classmethod
    def live_tsapa_from_environment(
        cls,
        *,
        generations: int,
        population: int,
    ) -> RewritePlan:
        return cls.live_from_environment(
            "tsapa",
            label="--tsapa",
            generations=generations,
            population=population,
        )


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(tokens))


def _lexical_divergence(original: str, candidate: str) -> float:
    """Bigram Jaccard distance: 0.0 identical, 1.0 fully different."""
    a = _tokens(original)
    b = _tokens(candidate)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    ba = _bigrams(a)
    bb = _bigrams(b)
    union = ba | bb
    if not union:
        return 0.0
    return 1.0 - len(ba & bb) / len(union)


def _select_candidate(original: str, candidates: list[str]) -> tuple[str, list[float]]:
    """Pick the most lexically diverged rewrite, gently guarding extreme length drift."""
    scores: list[float] = []
    for cand in candidates:
        score = _lexical_divergence(original, cand)
        if original:
            ratio = len(cand) / len(original)
            if ratio > 2.0 or ratio < 0.5:
                score -= 0.15
        scores.append(score)
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], scores


def _check_remote(base_url: str, allow_remote: bool) -> None:
    """Enforce the rewrite-endpoint allowlist.

    Default-deny: only loopback endpoints are accepted. Anything else requires
    an explicit opt-in (--allow-remote / WATERMARKS_REWRITE_ALLOW_REMOTE=1),
    and non-http(s) schemes (e.g. file://) are always refused.
    """
    u = urlparse(base_url)
    if u.scheme not in ("http", "https"):
        raise SystemExit(
            f"error: rewrite base URL must be http(s), got scheme '{u.scheme}': {base_url}"
        )
    host = u.hostname or ""
    if host in _LOOPBACK_HOSTS:
        return
    if not allow_remote:
        raise SystemExit(
            "error: rewrite base URL host is not loopback "
            f"('{host}'); refusing to send content off-machine. "
            "Set WATERMARKS_REWRITE_ALLOW_REMOTE=1 or pass --allow-remote to override."
        )
    eprint(
        f"warning: rewrite base URL host is '{host}' (not localhost); "
        "content will leave this machine"
    )


def _enforce_endpoint(base_url: str, allow_remote: bool) -> None:
    """Enforce the loopback allowlist inside rewrite, normalized to LayerBHTTPError."""
    if not base_url:
        return
    try:
        scheme = urlparse(base_url).scheme
    except ValueError:
        return
    if scheme not in ("http", "https"):
        return
    try:
        _check_remote(base_url, allow_remote)
    except SystemExit as error:
        raise LayerBHTTPError(f"{error} (endpoint: {base_url})") from None


def remote_warning(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    try:
        host = urlparse(base_url).hostname
    except ValueError:
        return None
    if host and host not in LOOPBACK_HOSTS:
        return (
            f"warning: rewrite base URL host is '{host}' (not localhost); "
            "content will leave this machine"
        )
    return None


def build_prompt(strength: str, text: str, *, lang: str, original_lang: str) -> str:
    if strength == "paraphrase":
        return PROMPTS["paraphrase"].format(TEXT=text)
    if strength == "humanize":
        return PROMPTS["humanize"].format(TEXT=text)
    if strength == "code":
        return PROMPTS["code"].format(TEXT=text)
    if strength == "backtranslate":
        # single combined instruction for print-prompt / one-shot backends
        return (
            f"Translate the text to {lang}, then translate that result back to "
            f"{original_lang}. Preserve all facts, numbers, and names. "
            f"Output only the final {original_lang} text.\n\n---\n{text}"
        )
    if strength == "structural":
        return (
            "First extract a bullet outline of all claims (no full sentences). "
            "Then write a complete document from that outline in a clear professional style "
            "without omitting any bullet. Output only the final document.\n\n---\n"
            f"{text}"
        )
    raise ValueError(f"unknown strength: {strength}")


#: Called with each new text fragment as it arrives. Used by front ends that
#: want to show generation progress; the accumulated result is unchanged.
TokenSink = Callable[[str], None]


def _emit(on_token: TokenSink, fragment: str, parts: list[str]) -> None:
    """Record a fragment and hand it to the sink, which must never break the read.

    A front end's callback runs on whatever thread is draining the socket; if it
    raises, the generation would be lost along with it.
    """
    parts.append(fragment)
    with contextlib.suppress(Exception):
        on_token(fragment)


def _no_stream_content(backend: str) -> str:
    """Message for a stream that produced nothing.

    Worth naming streaming explicitly: the same endpoint may answer a plain
    request perfectly well, so "empty content" alone sends the operator looking
    at the wrong thing.
    """
    return (
        f"{backend} returned no content over a streamed request; the endpoint may not "
        "support streaming responses"
    )


def _stream_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
    on_token: TokenSink,
) -> str:
    parts: list[str] = []
    for event in layer_b_http.stream_json_lines(
        base_url,
        "/api/chat",
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    ):
        message = event.get("message")
        if isinstance(message, dict):
            fragment = message.get("content")
            if isinstance(fragment, str) and fragment:
                _emit(on_token, fragment, parts)
        if event.get("done") is True:
            break
    out = "".join(parts).strip()
    if not out:
        raise RuntimeError(_no_stream_content("ollama"))
    return out


def _stream_openai_compatible(
    base_url: str,
    route: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
    on_token: TokenSink,
) -> str:
    parts: list[str] = []
    for event in layer_b_http.stream_json_lines(
        base_url,
        route,
        {**payload, "stream": True},
        headers=headers,
        timeout=timeout,
    ):
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        # ``delta`` is a streamed chunk; ``message`` is what a server that
        # ignored ``stream: true`` sends back instead.  Accepting both means a
        # backend without SSE support degrades to one big fragment rather than
        # failing only when a front end asks to watch.
        chunk = choices[0].get("delta")
        if not isinstance(chunk, dict):
            chunk = choices[0].get("message")
        if isinstance(chunk, dict):
            fragment = chunk.get("content")
            if isinstance(fragment, str) and fragment:
                _emit(on_token, fragment, parts)
    out = "".join(parts).strip()
    if not out:
        raise RuntimeError(_no_stream_content("openai-compatible"))
    return out


def _call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
    *,
    on_token: TokenSink | None = None,
) -> str:
    if on_token is not None:
        return _stream_ollama(base_url, model, prompt, timeout, temperature, on_token)
    data = layer_b_http.request_json(
        base_url,
        "/api/chat",
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    message = data.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("ollama invalid message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("ollama empty response")
    return content.strip()


def _call_openai_compatible(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    *,
    temperature: float = 0.9,
    reasoning_effort: str | None = None,
    disable_thinking: bool = False,
    on_token: TokenSink | None = None,
) -> str:
    route = "/v1/chat/completions"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    # "off" is a documented sentinel that omits the parameter entirely
    # (generic OpenAI-compatible servers may reject it); every other value,
    # including "none", is a real reasoning-effort request and is sent.
    if reasoning_effort and reasoning_effort != "off":
        payload["reasoning_effort"] = reasoning_effort
    if disable_thinking:
        # Supported by Qwen/Transformers-compatible servers; opt-in so generic
        # OpenAI-compatible endpoints never receive an unknown extension.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if on_token is not None:
        return _stream_openai_compatible(base_url, route, payload, headers, timeout, on_token)
    data = layer_b_http.request_json(base_url, route, payload, headers=headers, timeout=timeout)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("openai-compatible invalid choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("openai-compatible invalid message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("openai-compatible empty content")
    return content.strip()


def rewrite(text: str, plan: RewritePlan, *, on_token: TokenSink | None = None) -> tuple[str, dict]:
    """Run the planned Layer B rewrite and return ``(text, info)``.

    *on_token*, when given, receives generated fragments as they arrive so a
    front end can show progress.  It is a presentation callback only: it never
    influences the result, and it is not honoured for ``tsapa``, whose calls are
    an internal evolutionary population rather than one visible generation.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(plan, RewritePlan):
        raise TypeError("plan must be a RewritePlan")

    backend = plan.backend
    model = plan.model
    base_url = plan.base_url
    api_key = plan.api_key
    strength = plan.strength
    lang = plan.lang
    original_lang = plan.original_lang
    timeout = plan.timeout
    layer_a_after = plan.layer_a_after
    generations = plan.generations
    population = plan.population
    disable_thinking = plan.disable_thinking
    temperature = plan.temperature
    candidates = plan.candidates
    reasoning_effort = plan.reasoning_effort
    allow_remote = plan.allow_remote

    info: dict = {
        "backend": backend,
        "strength": strength,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "input_chars": len(text),
    }
    if backend == "openai-compatible":
        info["disable_thinking"] = disable_thinking
    if reasoning_effort:
        info["reasoning_effort"] = reasoning_effort

    if strength == "tsapa":
        return _rewrite_tsapa(
            text,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            layer_a_after=layer_a_after,
            generations=generations,
            population=population,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            disable_thinking=disable_thinking,
            allow_remote=allow_remote,
            info=info,
        )

    prompt = build_prompt(strength, text, lang=lang, original_lang=original_lang)
    info["prompt_chars"] = len(prompt)

    if backend == "print-prompt":
        info["mode"] = "print-prompt"
        if candidates > 1:
            eprint("note: --candidates ignored in print-prompt mode")
        return prompt, info

    _enforce_endpoint(base_url, allow_remote)

    n = max(1, candidates)
    outs: list[str] = []
    for _ in range(n):
        if backend == "ollama":
            outs.append(
                _call_ollama(base_url, model, prompt, timeout, temperature, on_token=on_token)
            )
        elif backend == "openai-compatible":
            outs.append(
                _call_openai_compatible(
                    base_url,
                    model,
                    prompt,
                    api_key,
                    timeout,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    disable_thinking=disable_thinking,
                    on_token=on_token,
                )
            )
        else:
            raise RewriteConfigurationError(f"unknown backend: {backend}")

    if len(outs) == 1:
        out = outs[0]
    else:
        info["candidates"] = n
        out, scores = _select_candidate(text, outs)
        selected_idx = max(range(len(outs)), key=lambda i: scores[i])
        info["candidate_scores"] = []
        for i, cand in enumerate(outs):
            info["candidate_scores"].append(
                {
                    "lexical_divergence": _lexical_divergence(text, cand),
                    "selection_score": scores[i],
                    "selected": i == selected_idx,
                }
            )

    if layer_a_after:
        out, stats = clean_text(out)
        info["layer_a_after"] = stats

    info["output_chars"] = len(out)
    info["mode"] = "rewritten"
    info["note"] = (
        "Layer B is best-effort against statistical token-sampling watermarks; "
        "cannot certify removal against a vendor detector."
    )
    return out, info


@dataclass(frozen=True, slots=True)
class RewriteCandidate:
    """One generated rewrite, with the score the auto-selector would give it."""

    index: int
    text: str
    lexical_divergence: float
    selection_score: float
    selected: bool


def generate_candidates(
    text: str,
    plan: RewritePlan,
    *,
    on_token: TokenSink | None = None,
) -> list[RewriteCandidate]:
    """Generate every candidate and score them, without discarding the losers.

    ``rewrite`` returns only the winner — correct for a pipeline, useless for a
    human picker.  This runs the same prompt through the same backend the same
    number of times and hands back all of them, scored identically, so a caller
    can present the choice.  Candidate *text* is deliberately not routed through
    ``rewrite``'s ``info`` dict: that dict lands in the JSON payload and the
    audit file, and document bodies do not belong there.

    Not supported for ``tsapa``, whose candidates are an internal evolutionary
    population rather than N independent generations.

    When *on_token* is given, each backend call streams and the sink receives
    fragments as they arrive, so a front end can show generation in progress
    instead of a spinner.
    """
    if not isinstance(plan, RewritePlan):
        raise TypeError("plan must be a RewritePlan")
    if plan.strength == "tsapa":
        raise RewriteConfigurationError(
            "candidate selection is not available for tsapa; its population is "
            "internal to the evolutionary search"
        )
    if plan.backend not in LIVE_REWRITE_BACKENDS:
        raise RewriteConfigurationError(
            f"candidate generation requires a live backend, got {plan.backend}"
        )

    prompt = build_prompt(plan.strength, text, lang=plan.lang, original_lang=plan.original_lang)
    _enforce_endpoint(plan.base_url, plan.allow_remote)

    outs: list[str] = []
    for _ in range(max(1, plan.candidates)):
        if plan.backend == "ollama":
            outs.append(
                _call_ollama(
                    plan.base_url,
                    plan.model,
                    prompt,
                    plan.timeout,
                    plan.temperature,
                    on_token=on_token,
                )
            )
        else:
            outs.append(
                _call_openai_compatible(
                    plan.base_url,
                    plan.model,
                    prompt,
                    plan.api_key,
                    plan.timeout,
                    temperature=plan.temperature,
                    reasoning_effort=plan.reasoning_effort,
                    disable_thinking=plan.disable_thinking,
                    on_token=on_token,
                )
            )

    _, scores = _select_candidate(text, outs)
    best = max(range(len(outs)), key=lambda i: scores[i])
    finished = [clean_text(candidate)[0] if plan.layer_a_after else candidate for candidate in outs]
    return [
        RewriteCandidate(
            index=i,
            text=finished[i],
            lexical_divergence=_lexical_divergence(text, outs[i]),
            selection_score=scores[i],
            selected=i == best,
        )
        for i in range(len(outs))
    ]


def _rewrite_tsapa(
    text: str,
    *,
    backend: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    layer_a_after: bool,
    generations: int,
    population: int,
    temperature: float,
    reasoning_effort: str | None,
    disable_thinking: bool,
    allow_remote: bool,
    info: dict,
) -> tuple[str, dict]:
    from tsapa import TSAPAAdapterError, heuristic_pll, http_embed, http_pll, tsapa

    if backend == "print-prompt":
        prompt = (
            TSAPA_PACK.replace("{GEN}", str(generations))
            .replace("{POP}", str(population))
            .replace("{TEXT}", text)
        )
        info.update(
            mode="print-prompt",
            prompt_chars=len(prompt),
            note="TSAPA operator pack for agent execution (no model called).",
        )
        return prompt, info

    if backend not in ("ollama", "openai-compatible"):
        raise RewriteConfigurationError(f"unknown backend: {backend}")
    _enforce_endpoint(base_url, allow_remote)

    if backend == "ollama":

        def llm(prompt: str) -> str:
            return _call_ollama(base_url, model, prompt, timeout, temperature)
    elif backend == "openai-compatible":

        def llm(prompt: str) -> str:
            return _call_openai_compatible(
                base_url,
                model,
                prompt,
                api_key,
                timeout,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                disable_thinking=disable_thinking,
            )

    pll_model = os.environ.get("WATERMARKS_PLL_MODEL", model or "")
    embed_model = os.environ.get("WATERMARKS_EMBED_MODEL", model or "")
    fallbacks = {"pll": 0, "embedding": 0}
    embedding_dimensions: int | None = None

    def pll(t: str) -> float:
        try:
            return http_pll(base_url, t, model=pll_model, api_key=api_key, timeout=timeout)
        except (layer_b_http.LayerBHTTPError, TSAPAAdapterError):
            fallbacks["pll"] += 1
            return heuristic_pll(t)  # labeled fallback

    def embed(t: str) -> list[float]:
        nonlocal embedding_dimensions
        try:
            values = http_embed(base_url, t, model=embed_model, api_key=api_key, timeout=timeout)
            if embedding_dimensions is None:
                embedding_dimensions = len(values)
            elif len(values) != embedding_dimensions:
                raise TSAPAAdapterError("embedding dimensions changed between responses")
            return values
        except (layer_b_http.LayerBHTTPError, TSAPAAdapterError):
            fallbacks["embedding"] += 1
            raise

    result = tsapa(
        text,
        llm=llm,
        pll=pll,
        embed=embed,
        generations=generations,
        population=population,
    )
    out = result["text"]
    if layer_a_after:
        out, stats = clean_text(out)
        info["layer_a_after"] = stats
    info.update(
        mode="rewritten",
        output_chars=len(out),
        tsapa={
            **{k: result[k] for k in ("chunks", "generations", "population", "stats")},
            "adapter_fallbacks": fallbacks,
            "pll_model": pll_model,
            "embedding_model": embed_model,
        },
        note=result["note"],
    )
    return out, info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout or *.rewritten.*)")
    p.add_argument(
        "--backend",
        choices=("print-prompt", "ollama", "openai-compatible"),
        default=_env("WATERMARKS_REWRITE_BACKEND", "print-prompt"),
    )
    p.add_argument("--model", default=_env("WATERMARKS_REWRITE_MODEL"))
    p.add_argument(
        "--base-url",
        default=_env("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        default=None,
        help="Allow non-loopback rewrite endpoints (default: deny; "
        "WATERMARKS_REWRITE_ALLOW_REMOTE=1 has the same effect)",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "off"),
        default=_env("WATERMARKS_REWRITE_REASONING_EFFORT", "none"),
        help="OpenAI-compatible reasoning_effort; 'none' skips chain-of-thought "
        "(reasoning models otherwise burn minutes on a rewrite). 'off' omits "
        "the parameter entirely.",
    )
    # NOTE: no --api-key flag on purpose — keys on argv are visible in `ps`
    # and shell history. Set WATERMARKS_REWRITE_API_KEY instead.
    p.add_argument(
        "--strength",
        choices=("paraphrase", "backtranslate", "structural", "humanize", "code", "tsapa"),
        default="paraphrase",
        help="tsapa: evolutionary multi-objective attack (ACL 2026 class)",
    )
    p.add_argument("--generations", type=int, default=5, help="TSAPA generations")
    p.add_argument("--population", type=int, default=12, help="TSAPA population per chunk")
    p.add_argument("--lang", default="French", help="Pivot language for backtranslate")
    p.add_argument("--original-lang", default="English")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for the rewrite backend",
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Number of rewrite candidates to generate and score",
    )
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument(
        "--disable-thinking",
        action="store_true",
        default=None,
        help="OpenAI-compatible: request direct answers from thinking-capable models",
    )
    thinking.add_argument(
        "--allow-thinking",
        action="store_false",
        dest="disable_thinking",
        help="OpenAI-compatible: allow the model's default reasoning mode",
    )
    p.add_argument(
        "--no-layer-a-after",
        action="store_true",
        help="Skip Layer A scrub on model output",
    )
    p.add_argument("--json-stats", action="store_true", help="Stats JSON on stderr")
    args = p.parse_args()

    text = read_text_input(args.path)
    try:
        disable_thinking = (
            read_bool_env("WATERMARKS_REWRITE_DISABLE_THINKING")
            if args.disable_thinking is None
            else args.disable_thinking
        )
        allow_remote = (
            read_flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
            if args.allow_remote is None
            else args.allow_remote
        )
        plan = RewritePlan(
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=os.environ.get("WATERMARKS_REWRITE_API_KEY"),
            strength=args.strength,
            lang=args.lang,
            original_lang=args.original_lang,
            timeout=args.timeout,
            layer_a_after=not args.no_layer_a_after,
            generations=args.generations,
            population=args.population,
            disable_thinking=disable_thinking,
            temperature=args.temperature,
            candidates=args.candidates,
            reasoning_effort=args.reasoning_effort,
            allow_remote=allow_remote,
        )
        if warning := remote_warning(plan.base_url):
            eprint(warning)
        result, info = rewrite(text, plan)
    except RewriteConfigurationError as error:
        eprint(str(error))
        return 1
    except (RuntimeError, ValueError) as error:
        eprint(f"rewrite failed: {error}")
        return 1

    out = args.output
    if out is None and args.path not in (None, "-") and args.backend != "print-prompt":
        out = str(cleaned_path(Path(args.path), suffix=".rewritten"))
    elif out is None and args.backend == "print-prompt":
        out = "-"

    if args.path not in (None, "-") and out not in (None, "-"):
        try:
            validate_output_path(Path(args.path), Path(out))
        except ValueError as error:
            eprint(f"error: {error}")
            return 2
    write_text_output(result, out)
    if args.json_stats:
        eprint(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"backend={info['backend']} strength={info['strength']} "
            f"mode={info.get('mode')} chars {info['input_chars']}->{info.get('output_chars', len(result))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
