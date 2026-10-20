# Text

Hidden-character cleaning (Layer A), model rewrites (Layer B), character perturbation, and text watermark detection.

### Layer A, deterministic Unicode hygiene

```bash
python3 "$SCRIPTS/inspect_text.py" draft.txt --json
python3 "$SCRIPTS/clean_text.py" draft.txt -o draft.cleaned.txt --stats
```

Layer A removes or normalizes configured hidden carriers. Some Unicode is meaningful, so preservation rules are context-aware rather than blanket deletion: ZWJ and variation selectors affect emoji and orthographies, and bidi controls affect display order.

### Layer B, rewrite attacks

```bash
# No model call: emit an execution prompt
python3 "$SCRIPTS/rewrite_text.py" draft.txt \
  --backend print-prompt --strength paraphrase

# TSAPA-style operator pack, still no model call
python3 "$SCRIPTS/rewrite_text.py" draft.txt \
  --backend print-prompt --strength tsapa --generations 5 --population 12

# Execute against a local OpenAI-compatible endpoint
WATERMARKS_REWRITE_BACKEND=openai-compatible \
WATERMARKS_REWRITE_BASE_URL=http://127.0.0.1:8080 \
WATERMARKS_REWRITE_MODEL=my-local-model \
  python3 "$SCRIPTS/rewrite_text.py" draft.txt \
    --strength tsapa --generations 5 --population 12

# Qwen/Transformers-compatible servers: suppress reasoning preambles
WATERMARKS_REWRITE_DISABLE_THINKING=true \
  python3 "$SCRIPTS/rewrite_text.py" draft.txt \
    --backend openai-compatible --base-url http://127.0.0.1:8080 \
    --model my-thinking-model --strength paraphrase
```

The TSAPA-style engine is a real multi-objective evolutionary loop:

1. Generate a diverse candidate population.
2. Score attack fitness: PLL, n-gram diversity, lexical diversity.
3. Score fidelity: embedding cosine similarity, with a labeled shingle-Jaccard fallback.
4. Apply NSGA-II non-dominated sorting and crowding distance.
5. Cross over at sentence boundaries, mutate the lowest-PLL sentence.
6. Select the Pareto knee point.

A logprobs-capable `/v1/completions` endpoint supplies PLL, and `/v1/embeddings` supplies semantic similarity. Either can fail independently and degrade to an explicitly labeled standard-library proxy. `--disable-thinking` (or `WATERMARKS_REWRITE_DISABLE_THINKING=true`) opts into the Qwen/Transformers `chat_template_kwargs.enable_thinking=false` extension; it is never sent by default to generic OpenAI-compatible servers.

Layer B is also reachable from `wm` directly, so a rewrite and the Layer A
re-scrub happen in one pass over one file or a whole tree:

```bash
wm draft.txt -o draft.cleaned.txt \
  --rewrite humanize \
  --rewrite-backend openai-compatible \
  --rewrite-base-url http://127.0.0.1:8000 \
  --rewrite-model my-local-model \
  --rewrite-candidates 3 --rewrite-timeout 120

wm draft.txt -o draft.cleaned.txt --tsapa    # alias for --rewrite tsapa
```

Strengths: `paraphrase`, `backtranslate`, `structural`, `humanize`, `code`,
`tsapa`. Settings resolve explicit flag > environment
(`WATERMARKS_REWRITE_*`) > default, so an existing environment still works with
a bare `--rewrite`. Non-loopback endpoints are refused unless
`--rewrite-allow-remote` is passed, and the warning names what leaves the
machine. Text-body options are deliberate no-ops on non-text assets so mixed
batches work; when that happens the run reports exactly which transforms it
skipped rather than letting you assume the body was rewritten.

**Cost:** Layer B replaces the original wording and can flatten voice or precision. Prefer a non-origin model so the rewrite does not re-stamp the same scheme.

### Character-level perturbations

```bash
python3 "$SCRIPTS/perturb_text.py" draft.txt \
  --mode zero-width --strength 0.10 --seed 42
```

This is an opt-in anti-watermark transform inspired by 2026 character-perturbation research. It intentionally adds artifacts that Layer A removes. `zero-width` and `space-swap` are Layer-A reversible, `confusable` and `case` are not. It is not a hygiene pass and carries no detector guarantee.

## Text watermark detection and stylometry

Optional, stdlib-first, fail-soft:

- `score_stylometry.py` — zero-LLM stylometry (burstiness, MATTR, AI-phrase density) with confidence bands and `--explain`.
- `text_detectors.py` / `detect_text_watermark.py` — a MarkLLM research harness (same-scheme-config only, not a vendor oracle). Both vendor detectors are reserved but unavailable, and `/capabilities` always reports them `false`: Gemini's official SynthID-text detector is wired for `WATERMARKS_GEMINI_API_KEY` (env only) but the Gemini API exposes no `DETECT_TEXT_WATERMARK` task type, and Claude detection waits on a public API.
- MarkLLM detection reports a document-level `verdict` (`DETECTED`, `NOT_DETECTED`, `INCONCLUSIVE`, `UNSUPPORTED`, `ERROR`) alongside the raw `detector_verdict`. Only a below-threshold score with a confirmed key/provenance match and enough scored tokens reads as `NOT_DETECTED`; unknown provenance, a near-threshold score, or a short sample produces `INCONCLUSIVE`. An unsupported scheme produces `UNSUPPORTED`, and a detector result that cannot be scored produces `ERROR`. Provenance comes from the `<output>.wm.json` sidecar the `watermark` subcommand writes, or from an operator `--key-id` assertion.
- `inspect_text.py --stylometry` and `rewrite_text.py` MarkLLM before/after hooks.

```bash
python3 "$SCRIPTS/score_stylometry.py" draft.txt --threshold 0.65 --explain
python3 "$SCRIPTS/detect_text_watermark.py" detect draft.txt --scheme kgw --json
```
