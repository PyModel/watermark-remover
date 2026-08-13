"""TSAPA-style evolutionary paraphrase attack (ACL 2026, "The Mark Fades").

Training-free, black-box Layer B: multi-objective genetic optimization over
paraphrase candidates.

    attack fitness  f_atk = w1 * PLL + w2 * ngram_diversity + w3 * lexical_diversity
    fidelity        f_fid = semantic_similarity(candidate, original)

Selection is NSGA-II over attack fitness and fidelity; mutation rewrites the
lowest-PLL sentence and final selection uses the Pareto knee point. Rewrite
prompts request factual preservation, but no automatic factual-validity judge is
claimed.

Pluggable backends (all optional, honest fallbacks):
  llm(prompt) -> str        candidate generation / mutation / synthesis
  pll(text) -> float        pseudo-log-likelihood. HTTP /v1/completions logprobs
                            when configured, else a heuristic proxy (labeled).
  embed(text) -> [float]    for semantic similarity. HTTP /v1/embeddings when
                            configured, else word-shingle Jaccard proxy.

Everything except the two HTTP scorers is pure stdlib and offline-testable.
"""

from __future__ import annotations

import json
import math
import random
import re
import urllib.request
from dataclasses import dataclass

from common import read_json_object_bounded

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "no",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
    ]
)


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Split on paragraph, then sentence boundaries, packing to ~max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paras:
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(s for s in _SENT_RE.split(para) if s.strip())
    chunks: list[str] = []
    cur = ""
    for unit in units:
        if cur and len(cur) + len(unit) + 2 > max_chars:
            chunks.append(cur)
            cur = unit
        else:
            cur = f"{cur}\n\n{unit}" if cur else unit
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ---------------------------------------------------------------------------
# Stdlib metrics
# ---------------------------------------------------------------------------


def ngram_diversity(text: str, n: int = 3) -> float:
    """Distinct n-grams / total n-grams (1.0 = all unique)."""
    tokens = _words(text)
    total = max(len(tokens) - n + 1, 1)
    return len(_ngrams(tokens, n)) / total


def lexical_diversity(candidate: str, original: str, n: int = 3) -> float:
    """1 - Self-BLEU-lite: fraction of candidate n-grams already in the original."""
    cand = _ngrams(_words(candidate), n)
    if not cand:
        return 0.0
    orig = _ngrams(_words(original), n)
    return 1.0 - len(cand & orig) / len(cand)


def shingle_similarity(a: str, b: str, n: int = 3) -> float:
    """Word-shingle Jaccard similarity — stdlib semantic-similarity proxy."""
    sa, sb = _ngrams(_words(a), n), _ngrams(_words(b), n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimensions differ")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# PLL backends
# ---------------------------------------------------------------------------


def heuristic_pll(text: str) -> float:
    """Stdlib pseudo-LL proxy: common short words score high, rare long words low.

    NOT a real masked-LM PLL — a labeled fallback for offline use. Use
    http_pll() against a logprobs-capable server for the real signal.
    """
    words = _words(text)
    if not words:
        return 0.0
    total = 0.0
    for w in words:
        if w in STOPWORDS:
            p = 0.9
        elif len(w) <= 4:
            p = 0.6
        elif len(w) <= 7:
            p = 0.4
        else:
            p = 0.2
        total += math.log(p)
    return total / len(words)


def http_pll(
    base_url: str,
    text: str,
    *,
    model: str = "",
    api_key: str | None = None,
    timeout: float = 60.0,
) -> float:
    """Token logprobs via OpenAI-compatible /v1/completions (echo + logprobs).

    Works with llama.cpp / MLX-style servers. Raises on failure — callers
    should catch and fall back to heuristic_pll.
    """
    payload: dict = {
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0,
    }
    if model:
        payload["model"] = model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = read_json_object_bounded(resp, label="PLL response")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("invalid choices in PLL response")
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError("invalid logprobs in PLL response")
    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        raise RuntimeError("invalid token_logprobs in PLL response")
    toks = [value for value in token_logprobs if isinstance(value, (int, float))]
    if not toks:
        raise RuntimeError("no token_logprobs in response")
    return sum(toks) / len(toks)


def http_embed(
    base_url: str,
    text: str,
    *,
    model: str = "",
    api_key: str | None = None,
    timeout: float = 60.0,
) -> list[float]:
    """OpenAI-compatible /v1/embeddings. Raises on failure — fall back to shingles."""
    payload: dict = {"input": text[:8000]}
    if model:
        payload["model"] = model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/embeddings",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = read_json_object_bounded(resp, label="embedding response")
    items = data.get("data")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        raise RuntimeError("invalid data in embedding response")
    embedding = items[0].get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("no embedding in response")
    try:
        values = [float(value) for value in embedding]
    except (TypeError, ValueError) as error:
        raise RuntimeError("embedding contains a non-numeric value") from error
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("embedding contains a non-finite value")
    return values


# ---------------------------------------------------------------------------
# Evolution core
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    text: str
    f_atk: float = 0.0
    f_fid: float = 0.0
    rank: int = 0
    crowding: float = 0.0


def evaluate(
    cand: Candidate,
    original: str,
    *,
    pll,
    embed,
    weights: tuple[float, float, float],
    orig_emb: list[float] | None,
) -> None:
    w1, w2, w3 = weights
    raw_pll = pll(cand.text)
    # normalize PLL proxy to ~[0, 1]: heuristic/logprob values are <= 0
    pll_norm = min(1.0, max(0.0, 1.0 + raw_pll / 5.0))
    cand.f_atk = (
        w1 * pll_norm
        + w2 * ngram_diversity(cand.text)
        + w3 * lexical_diversity(cand.text, original)
    )
    if embed is not None and orig_emb is not None:
        try:
            cand.f_fid = max(0.0, min(1.0, cosine(embed(cand.text), orig_emb)))
            return
        except Exception:
            pass
    cand.f_fid = shingle_similarity(cand.text, original)


def _dominates(a: Candidate, b: Candidate) -> bool:
    return (a.f_atk >= b.f_atk and a.f_fid >= b.f_fid) and (a.f_atk > b.f_atk or a.f_fid > b.f_fid)


def non_dominated_sort(pop: list[Candidate]) -> list[list[Candidate]]:
    fronts: list[list[Candidate]] = [[]]
    dom_count = {id(c): 0 for c in pop}
    dominated: dict[int, list[Candidate]] = {id(c): [] for c in pop}
    for c in pop:
        for o in pop:
            if c is o:
                continue
            if _dominates(c, o):
                dominated[id(c)].append(o)
            elif _dominates(o, c):
                dom_count[id(c)] += 1
        if dom_count[id(c)] == 0:
            c.rank = 0
            fronts[0].append(c)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: list[Candidate] = []
        for c in fronts[i]:
            for d in dominated[id(c)]:
                dom_count[id(d)] -= 1
                if dom_count[id(d)] == 0:
                    d.rank = i + 1
                    nxt.append(d)
        if nxt:
            fronts.append(nxt)
        i += 1
    return [f for f in fronts if f]


def assign_crowding(front: list[Candidate]) -> None:
    if len(front) <= 2:
        for c in front:
            c.crowding = float("inf")
        return
    for c in front:
        c.crowding = 0.0
    for attr in ("f_atk", "f_fid"):
        ordered = sorted(front, key=lambda c: getattr(c, attr))
        ordered[0].crowding = ordered[-1].crowding = float("inf")
        lo, hi = getattr(ordered[0], attr), getattr(ordered[-1], attr)
        span = (hi - lo) or 1.0
        for i in range(1, len(ordered) - 1):
            ordered[i].crowding += (
                getattr(ordered[i + 1], attr) - getattr(ordered[i - 1], attr)
            ) / span


def nsga2_reduce(pop: list[Candidate], size: int) -> list[Candidate]:
    fronts = non_dominated_sort(pop)
    for front in fronts:
        assign_crowding(front)
    out: list[Candidate] = []
    for front in fronts:
        if len(out) + len(front) <= size:
            out.extend(front)
        else:
            front.sort(key=lambda c: c.crowding, reverse=True)
            out.extend(front[: size - len(out)])
            break
    return out


def crossover(a: Candidate, b: Candidate, rng: random.Random) -> Candidate:
    """Sentence-level multi-point crossover."""
    sa, sb = _SENT_RE.split(a.text), _SENT_RE.split(b.text)
    if len(sa) < 2 or len(sb) < 2:
        child = a.text if rng.random() < 0.5 else b.text
        return Candidate(text=child)
    cuts = sorted(
        rng.sample(range(1, min(len(sa), len(sb)) + 1), k=min(2, min(len(sa), len(sb)) - 1))
    )
    child: list[str] = []
    src_a = rng.random() < 0.5
    prev = 0
    for cut in [*cuts, max(len(sa), len(sb))]:
        seg = (sa if src_a else sb)[prev:cut]
        child.extend(seg)
        src_a = not src_a
        prev = cut
    text = " ".join(child).strip()
    return Candidate(text=text or a.text)


def pll_guided_mutation(cand: Candidate, *, llm, pll, rng: random.Random) -> Candidate:
    """Rewrite only the most suspicious (lowest-PLL) sentence via the LLM."""
    sentences = [s for s in _SENT_RE.split(cand.text) if s.strip()]
    if not sentences:
        return Candidate(text=llm(f"Rewrite with completely different wording:\n\n{cand.text}"))
    scored = sorted(sentences, key=pll)
    target = scored[0] if rng.random() < 0.8 else rng.choice(scored[: max(2, len(scored) // 4)])
    replacement = llm(
        "Rewrite the following sentence with completely different wording and "
        "structure while preserving its meaning exactly. Output only the sentence.\n\n"
        f"---\n{target}"
    ).strip()
    if not replacement:
        return cand
    return Candidate(text=cand.text.replace(target, replacement, 1))


def select_knee_point(front: list[Candidate]) -> Candidate:
    """Min Euclidean distance to the utopia point (1, 1) on the normalized front."""
    if len(front) == 1:
        return front[0]
    atk = [c.f_atk for c in front]
    fid = [c.f_fid for c in front]
    span_a = (max(atk) - min(atk)) or 1.0
    span_f = (max(fid) - min(fid)) or 1.0

    def dist(c: Candidate) -> float:
        na = (c.f_atk - min(atk)) / span_a
        nf = (c.f_fid - min(fid)) / span_f
        return math.hypot(1.0 - na, 1.0 - nf)

    return min(front, key=dist)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

STYLE_VARIANTS = (
    "in a formal register",
    "in a casual register",
    "more concisely",
    "with expanded detail",
    "with a completely different sentence structure",
    "as if for a different audience",
)


def tsapa(
    text: str,
    *,
    llm,
    pll=heuristic_pll,
    embed=None,
    generations: int = 5,
    population: int = 12,
    chunk_chars: int = 1200,
    weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int | None = None,
) -> dict:
    """Run the evolutionary attack. llm(prompt)->str is required.

    Returns dict with rewritten text and per-chunk stats.
    """
    if not text.strip():
        raise ValueError("text must not be empty")
    if generations < 0:
        raise ValueError("generations must be >= 0")
    if population < 2:
        raise ValueError("population must be >= 2")
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must contain three finite non-negative values")
    if sum(weights) <= 0:
        raise ValueError("weights must have a positive sum")

    rng = random.Random(seed)
    chunks = chunk_text(text, max_chars=chunk_chars)
    out_chunks: list[str] = []
    stats: list[dict] = []

    for chunk in chunks:
        orig_emb = None
        if embed is not None:
            try:
                orig_emb = embed(chunk)
            except Exception:
                orig_emb = None

        # 1. initialize population with diverse paraphrases
        pop: list[Candidate] = []
        for i in range(population):
            style = STYLE_VARIANTS[i % len(STYLE_VARIANTS)]
            candidate = llm(
                f"Rewrite the following text {style}, preserving all facts, numbers, "
                f"and names. Output only the rewritten text.\n\n---\n{chunk}"
            ).strip()
            if candidate:
                pop.append(Candidate(text=candidate))
        if len(pop) < 2:
            out_chunks.append(pop[0].text if pop else chunk)
            stats.append(
                {
                    "chunk_chars": len(chunk),
                    "skipped_evolution": True,
                    "usable_candidates": len(pop),
                }
            )
            continue
        for c in pop:
            evaluate(c, chunk, pll=pll, embed=embed, weights=weights, orig_emb=orig_emb)
        pop = nsga2_reduce(pop, population)

        # 2. generations
        for _ in range(generations):
            offspring: list[Candidate] = []
            while len(offspring) < population:
                a, b = rng.sample(pop, 2)
                if rng.random() < 0.75:
                    child = crossover(a, b, rng)
                    synthesized = llm(
                        "Synthesize one coherent rewrite using the two candidate phrasings below. "
                        "Preserve EVERY fact, number, name, and technical identifier from the "
                        "original; do not add claims. Output only the rewrite.\n\n"
                        f"Candidate A:\n{a.text}\n\nCandidate B:\n{b.text}\n\n"
                        f"Original source of truth:\n---\n{chunk}"
                    ).strip()
                    if synthesized:
                        child = Candidate(text=synthesized)
                else:
                    child = Candidate(text=a.text)
                if rng.random() < 0.5:
                    child = pll_guided_mutation(child, llm=llm, pll=pll, rng=rng)
                offspring.append(child)
            for c in offspring:
                evaluate(c, chunk, pll=pll, embed=embed, weights=weights, orig_emb=orig_emb)
            pop = nsga2_reduce(pop + offspring, population)

        # 3. knee point of final Pareto front
        front = non_dominated_sort(pop)[0]
        best = select_knee_point(front)
        out_chunks.append(best.text)
        stats.append(
            {
                "chunk_chars": len(chunk),
                "front_size": len(front),
                "f_atk": round(best.f_atk, 4),
                "f_fid": round(best.f_fid, 4),
            }
        )

    return {
        "text": "\n\n".join(out_chunks),
        "chunks": len(chunks),
        "generations": generations,
        "population": population,
        "weights": weights,
        "pll_backend": getattr(pll, "__name__", type(pll).__name__),
        "embed_backend": (
            getattr(embed, "__name__", type(embed).__name__)
            if embed is not None
            else "shingle-jaccard"
        ),
        "stats": stats,
        "note": (
            "TSAPA-style evolutionary rewrite (best-effort). ASR figures in the "
            "literature are paper-reported; no guarantee against a vendor detector."
        ),
    }
