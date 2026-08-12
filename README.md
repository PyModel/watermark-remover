```
_ _ _ ____ ___ ____ ____ _  _ ____ ____ _  _    ____ ____ _  _ ____ _  _ ____ ____
| | | |__|  |  |___ |__/ |\/| |__| |__/ |_/  __ |__/ |___ |\/| |  | |  | |___ |__/
|_|_| |  |  |  |___ |  \ |  | |  | |  \ | \_    |  \ |___ |  | |__|  \/  |___ |  \
```

# watermark-remover

<!-- logo: figlet -d .figlet -f cybermedium -w 120 "watermark-remover" -->

[![CI](https://github.com/Pythoughts-labs/watermark-remover/actions/workflows/ci.yml/badge.svg)](https://github.com/Pythoughts-labs/watermark-remover/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Pythoughts-labs/watermark-remover)](https://github.com/Pythoughts-labs/watermark-remover/releases)

Agent skill + stdlib Python scripts to strip **multi-vendor AI provenance marks** from text and files — for privacy and hygiene on content **you own**.

| Layer | Target | How |
| --- | --- | --- |
| **A** | Invisible Unicode, exotic spaces, bidi, tag chars | Deterministic Python scripts |
| **B** | Statistical (token-sampling) text watermarks | Agent rewrite + optional `rewrite_text.py` hook |
| **Files** | C2PA / EXIF / XMP / doc props | PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, Markdown |

Vendors / ecosystems (class-level): **Claude**, **Gemini / SynthID-Text**, **OpenAI** provenance surfaces, **open-LLM** Kirchenbauer-style marks.

**Latest release:** [v0.1.0](https://github.com/Pythoughts-labs/watermark-remover/releases/tag/v0.1.0)

Skill path: [`skills/remove-ai-marks/`](skills/remove-ai-marks/)

## Install (agent skill)

```bash
# Grok Build / project-local
mkdir -p .grok/skills
ln -sfn "$(pwd)/skills/remove-ai-marks" .grok/skills/remove-ai-marks

# User-global Grok
mkdir -p ~/.grok/skills
ln -sfn "$(pwd)/skills/remove-ai-marks" ~/.grok/skills/remove-ai-marks
```

Invoke with `/remove-ai-marks` or ask to “strip AI watermarks / C2PA / Claude marks / SynthID-class text.”

Optional system tools (auto-used when present):

| Tool | Role |
| --- | --- |
| [`c2patool`](https://github.com/contentauth/c2pa-rs/tree/main/cli) | Inspect C2PA manifests |
| [`exiftool`](https://exiftool.org/) | Residual metadata strip (esp. **PDF**) |

Core scripts need **Python 3.10+** stdlib only. Layer B model calls are optional.

## Quick use (scripts)

```bash
SCRIPTS=skills/remove-ai-marks/scripts

# Unified inspect / clean
python3 "$SCRIPTS/inspect_file.py" draft.md
python3 "$SCRIPTS/clean_file.py" draft.md -o draft.cleaned.md
python3 "$SCRIPTS/clean_file.py" photo.png -o photo.cleaned.png
python3 "$SCRIPTS/clean_file.py" notes.docx -o notes.cleaned.docx

# Text Layer A
python3 "$SCRIPTS/inspect_text.py" draft.md
python3 "$SCRIPTS/clean_text.py" draft.md -o draft.cleaned.md --stats

# Layer B rewrite hook (default: print prompt only — no model required)
python3 "$SCRIPTS/rewrite_text.py" draft.md --backend print-prompt --strength paraphrase
# Optional local Ollama:
# WATERMARKS_REWRITE_BACKEND=ollama WATERMARKS_REWRITE_MODEL=llama3.2 \
#   python3 "$SCRIPTS/rewrite_text.py" draft.md -o draft.rewritten.md

# Images
python3 "$SCRIPTS/inspect_image.py" shot.png
python3 "$SCRIPTS/clean_image.py" shot.png -o shot.cleaned.png
```

## Optional SynthID pixel scoring

`inspect_image.py` and `clean_image.py` can report a pixel-domain SynthID
confidence score when an external checkout of
[`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID)
is available. The scorer is **not bundled**: it is loaded at runtime from your
checkout, and its code remains under the upstream project's non-commercial
Research License.

### Option 1: one-command bootstrap (no Docker)

```bash
SCRIPTS=skills/remove-ai-marks/scripts

# Clones upstream, creates a venv, and installs scorer-only dependencies.
"$SCRIPTS/setup_synthid.sh"

# Score an image (default checkout: ~/reverse-SynthID).
REVERSE_SYNTHID_DIR=~/reverse-SynthID \
~/reverse-SynthID/.venv/bin/python "$SCRIPTS/score_synthid.py" shot.png

# Or surface the score from inspect / clean (same venv Python).
REVERSE_SYNTHID_DIR=~/reverse-SynthID \
~/reverse-SynthID/.venv/bin/python "$SCRIPTS/inspect_image.py" shot.png
```

`setup_synthid.sh` accepts `--dir PATH`, `--ref REF`, and `--full` (install the
full upstream `requirements.txt`, which adds `torch`/`diffusers` for the
upstream VAE bypass this project does not use).

### Option 2: local Docker build

```bash
make docker-synthid-build
docker run --rm -v "$(pwd):/data" watermark-remover-synthid-scorer /data/shot.png
```

The image is built locally from the upstream source at build time. It is not
published, so it does not redistribute the upstream code.

V4 scoring uses `artifacts/spectral_codebook_v4.npz` from the upstream checkout
(~220 MB). This is **detection/scoring only** — it does not remove pixel
watermarks.

## Coverage matrix

| Channel | Claude | Gemini/SynthID | OpenAI | Open-LLM |
| --- | --- | --- | --- | --- |
| Unicode / edit-based text | Layer A | Layer A | Layer A | Layer A |
| Statistical sampling text | Layer B best-effort | Layer B best-effort | Layer B if present | Layer B best-effort |
| C2PA / file metadata | Yes (listed formats) | Yes when present | Yes when present | Yes when present |
| Pixel image marks | Out of scope | Optional SynthID score (external); removal out of scope | Out of scope | Out of scope |
| Training backdoors | Out of scope | Out of scope | Out of scope | Out of scope |

Details: [`skills/remove-ai-marks/references/vendor-notes.md`](skills/remove-ai-marks/references/vendor-notes.md), [`mark-classes.md`](skills/remove-ai-marks/references/mark-classes.md).

---

## How text marking works (short)

Modern LLM watermarks often hide a signal in **which tokens are chosen** (generative / sampling bias), not only in invisible characters. Edit-based schemes inject Unicode or synonym rules. File schemes attach **C2PA** or generator metadata.

- **Layer A** removes edit-based Unicode carriers (testable).
- **Layer B** attacks sampling watermarks via heavy rewrite (best-effort; literature-standard attacks such as paraphrase / back-translation).
- **File cleaners** strip C2PA/XMP/props from supported containers.

Until vendors ship public detectors and keys, **no tool can honestly certify** “this fails the official check.” Reports must separate verifiable vs best-effort work.

Prefer a **non-origin** model for Layer B (do not rewrite Claude text with Claude if you are trying to avoid re-stamping).

---

## Disclaimer: what removing a text watermark costs

Text watermarks live in **the wording itself**: the signal is spread across token choices, so nearly every sentence carries a little of it. Two consequences follow, and they are why Layer B is honestly described as *best-effort* rather than a magic eraser.

1. **Removal means rewording, not restructuring.** Shuffling paragraphs, changing headings, or light touch-ups barely move the signal. Stripping a statistical mark requires rewriting a substantial fraction of the text — sentence by sentence, not section by section.

2. **Rewording degrades the copy.** Any rewrite replaces the original word choices with the rewriting model's, which flattens tone, voice, and precision. On production copy (SEO, marketing, client work) that degradation is real and often visible to the people who care most about the writing. It is like taking text from a top-tier model and asking a less capable model to rewrite it from scratch: the result cannot exceed the rewrite model's ceiling.

Which leads to the honest full-circle question:

> If the plan is to rewrite the text with a cheaper model anyway, why pay for a premium model in the first place? Generating directly with the cheaper model is simpler, cheaper, and produces the same — or better — end result.

Layer B makes sense when you specifically want the premium model's **thinking and drafting** and accept a rewrite pass to satisfy a hygiene or privacy requirement — not as a cheap route to mark-free text.

**When to skip Layer B:**

- **Quality matters more than hygiene:** use the lossless path — Layer A Unicode scrub plus the file metadata cleaners — and keep the original prose.
- **Rewriting anyway:** use a **non-origin** model (rewriting with the origin model can re-stamp the text), and remember residual risk remains — no tool can certify a vendor detector will fail.

---

## File formats

| Format | Inspect | Clean |
| --- | --- | --- |
| PNG / JPEG | C2PA chunks / APP11, AI XMP hints | Drop metadata segments |
| SVG | `<metadata>`, XMP | Strip blocks |
| PDF | Byte/XMP + optional tools | **exiftool** preferred; degraded without it |
| DOCX | docProps / customXml | Scrub props, drop customXml |
| ODT | meta.xml | Drop generator / AI-ish meta |
| HTML | meta, JSON-LD, data-ai* | Strip tags/attrs |
| Markdown | YAML frontmatter AI keys | Drop keys + Layer A body |

Pixel-domain watermark **removal** and **C2PA soft binding** (in-content watermark that can re-link a remote Content Credentials manifest after metadata is stripped) remain **out of scope**. Stripping hard-bound C2PA does **not** clear those channels. An optional local SynthID scorer is available for detection only (see above).

### Residual risk after a clean

This tool reports **verifiable** removals (Unicode counts, metadata actions) and **best-effort** Layer B rewrites. It cannot certify that vendor detectors will fail.

To check residual signals yourself (optional, external):

| Channel | What we remove | What may remain | External check (examples) |
| --- | --- | --- | --- |
| Hard-bound C2PA / EXIF / XMP | Yes | Soft-bound / pixel marks | [c2patool](https://github.com/contentauth/c2pa-rs/tree/main/cli), [Content Credentials verify](https://contentcredentials.org/verify) |
| SynthID-class media | No (optional local score only) | Pixel/audio/video watermark | Provider tools (e.g. [Google SynthID](https://deepmind.google/science/synthid/) / Vertex detector where offered); optional local [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) scorer |
| Statistical text | Best-effort rewrite | Strong marks after light edit | No public universal detector; vendor tools when available |

Industry two-layer context (C2PA + imperceptible watermark): [Institute of AI PM guide](https://www.institutepm.com/knowledge-hub/ai-content-provenance-watermarking).

---

## Removal options (summary)

| Option | Removes | Notes |
| --- | --- | --- |
| Unicode scrub (Layer A) | ZWSP, bidi, tags, exotic spaces, … | Safe default for text |
| Rewrite (Layer B) | Statistical token marks (best-effort) | Always offered by skill; costs style — see [Disclaimer](#disclaimer-what-removing-a-text-watermark-costs) |
| Container/metadata strip | File provenance | See format table |
| Open-weight local models | Avoid re-stamping with origin model | Operational alternative |

Matrix: [`skills/remove-ai-marks/references/removal-matrix.md`](skills/remove-ai-marks/references/removal-matrix.md).

## Ethics

See [`skills/remove-ai-marks/references/ethics.md`](skills/remove-ai-marks/references/ethics.md). For privacy and research on **your** content — not academic fraud or false “human-written” claims.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest          # or: make test
make smoke                          # quick CLI smoke on fixtures
```

## License

MIT — see [LICENSE](LICENSE).

## References

- [How Claude marks AI-generated content](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content) (Anthropic)
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4) (SynthID-Text, Nature 2024)
- Google AI for Developers, [*SynthID safeguards*](https://ai.google.dev/responsible/docs/safeguards/synthid) (Gemini API docs)
- [C2PA](https://c2pa.org/) / [c2patool](https://github.com/contentauth/c2pa-rs/tree/main/cli)
- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://arxiv.org/abs/2301.10226)
- [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text) (research reference; not used for detection here)
- [aloshdenny/reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) (research reference)
- Institute of AI PM, [*AI Content Provenance and Watermarking: The PM's Guide to C2PA and SynthID*](https://www.institutepm.com/knowledge-hub/ai-content-provenance-watermarking) (two-layer industry model: C2PA + imperceptible watermark / soft binding; SB 942 / EU AI Act Art. 50 context)
