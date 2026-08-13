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
  WATERMARKS_REWRITE_API_KEY
  WATERMARKS_REWRITE_DISABLE_THINKING  — true/false; OpenAI-compatible extension
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    cleaned_path,
    eprint,
    read_bool_env,
    read_json_object_bounded,
    read_text_input,
    validate_output_path,
    write_text_output,
)
from text_unicode import clean_text

PROMPTS = {
    "paraphrase": (
        "Rewrite the following text so that every sentence uses different wording and "
        "structure while preserving all facts, numbers, names, and technical identifiers. "
        "Do not add or remove claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "backtranslate_out": (
        "Translate the following text to {LANG}. Output only the translation.\n\n---\n{TEXT}"
    ),
    "backtranslate_back": (
        "Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural "
        "phrasing. Output only the translation.\n\n---\n{TEXT}"
    ),
    "structural_outline": (
        "Extract a bullet outline of all claims and structure from the text "
        "(no full sentences). Output only the outline.\n\n---\n{TEXT}"
    ),
    "structural_write": (
        "Write a complete document from this outline in a clear professional style. "
        "Do not omit any bullet. Output only the document.\n\n---\n{TEXT}"
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


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _warn_remote(base_url: str) -> None:
    host = urlparse(base_url).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        eprint(
            f"warning: rewrite base URL host is '{host}' (not localhost); "
            "content will leave this machine"
        )


def build_prompt(strength: str, text: str, *, lang: str, original_lang: str) -> str:
    if strength == "paraphrase":
        return PROMPTS["paraphrase"].format(TEXT=text)
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


def _http_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_ollama(base_url: str, model: str, prompt: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    data = _http_json(
        url,
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        {},
        timeout,
    )
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"ollama empty response: {data!r}"[:500])
    return str(content).strip()


def call_openai_compatible(
    base_url: str, model: str, prompt: str, api_key: str | None, timeout: float
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _http_json(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        headers,
        timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openai-compatible empty choices: {data!r}"[:500])
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError(f"openai-compatible empty content: {data!r}"[:500])
    return str(content).strip()


def rewrite(
    text: str,
    *,
    backend: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    strength: str,
    lang: str,
    original_lang: str,
    timeout: float,
    layer_a_after: bool,
    generations: int = 5,
    population: int = 12,
) -> tuple[str, dict]:
    info: dict = {
        "backend": backend,
        "strength": strength,
        "model": model,
        "base_url": base_url,
        "input_chars": len(text),
    }

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
            info=info,
        )

    prompt = build_prompt(strength, text, lang=lang, original_lang=original_lang)
    info["prompt_chars"] = len(prompt)

    if backend == "print-prompt":
        info["mode"] = "print-prompt"
        return prompt, info

    if not model:
        raise SystemExit("error: --model required for ollama/openai-compatible backends")
    if not base_url:
        raise SystemExit("error: --base-url required for ollama/openai-compatible backends")

    _warn_remote(base_url)

    if backend == "ollama":
        out = call_ollama(base_url, model, prompt, timeout)
    elif backend == "openai-compatible":
        out = call_openai_compatible(base_url, model, prompt, api_key, timeout)
    else:
        raise SystemExit(f"unknown backend: {backend}")

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
    info: dict,
) -> tuple[str, dict]:
    from tsapa import heuristic_pll, http_embed, http_pll, tsapa

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

    if not model or not base_url:
        raise SystemExit("error: --model and --base-url required for tsapa with live backends")
    _warn_remote(base_url)

    if backend == "ollama":

        def llm(prompt: str) -> str:
            return call_ollama(base_url, model, prompt, timeout)
    elif backend == "openai-compatible":

        def llm(prompt: str) -> str:
            return call_openai_compatible(base_url, model, prompt, api_key, timeout)
    else:
        raise SystemExit(f"unknown backend: {backend}")

    pll_model = os.environ.get("WATERMARKS_PLL_MODEL", model or "")
    embed_model = os.environ.get("WATERMARKS_EMBED_MODEL", model or "")
    fallbacks = {"pll": 0, "embedding": 0}

    def pll(t: str) -> float:
        try:
            return http_pll(base_url, t, model=pll_model, api_key=api_key, timeout=timeout)
        except Exception:
            fallbacks["pll"] += 1
            return heuristic_pll(t)  # labeled fallback

    def embed(t: str) -> list[float]:
        try:
            return http_embed(base_url, t, model=embed_model, api_key=api_key, timeout=timeout)
        except Exception:
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
    p.add_argument("--api-key", default=_env("WATERMARKS_REWRITE_API_KEY"))
    p.add_argument(
        "--strength",
        choices=("paraphrase", "backtranslate", "structural", "tsapa"),
        default="paraphrase",
        help="tsapa: evolutionary multi-objective attack (ACL 2026 class)",
    )
    p.add_argument("--generations", type=int, default=5, help="TSAPA generations")
    p.add_argument("--population", type=int, default=12, help="TSAPA population per chunk")
    p.add_argument("--lang", default="French", help="Pivot language for backtranslate")
    p.add_argument("--original-lang", default="English")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--no-layer-a-after",
        action="store_true",
        help="Skip Layer A scrub on model output",
    )
    p.add_argument("--json-stats", action="store_true", help="Stats JSON on stderr")
    args = p.parse_args()

    text = read_text_input(args.path)
    try:
        result, info = rewrite(
            text,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            strength=args.strength,
            lang=args.lang,
            original_lang=args.original_lang,
            timeout=args.timeout,
            layer_a_after=not args.no_layer_a_after,
            generations=args.generations,
            population=args.population,
        )
    except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
        eprint(f"rewrite failed: {e}")
        return 1

    out = args.output
    if out is None and args.path not in (None, "-") and args.backend != "print-prompt":
        out = str(cleaned_path(Path(args.path), suffix=".rewritten"))
    elif out is None and args.backend == "print-prompt":
        out = "-"

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
