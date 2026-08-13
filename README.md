# watermark-remover

**Inspect and clean AI provenance signals from content you own.** Stdlib-first Python tools cover hidden Unicode, statistical text-watermark attacks, visible image marks, and C2PA / EXIF / XMP metadata.

[![CI](https://github.com/Pythoughts-labs/watermark-remover/actions/workflows/ci.yml/badge.svg)](https://github.com/Pythoughts-labs/watermark-remover/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Pythoughts-labs/watermark-remover)](https://github.com/Pythoughts-labs/watermark-remover/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Honesty contract:** deterministic cleaners report what they removed. Rewrite, inpainting, and detector-evasion methods are always labeled **best-effort**. This project never certifies that a vendor detector will fail.

---

## What ships

| Module | Target | Method | Result class |
| --- | --- | --- | --- |
| **Layer A** | Hidden Unicode, bidi controls, tags, exotic spaces | Context-aware deterministic scrub | Verifiable |
| **Layer B** | Token-distribution text watermarks | Paraphrase, back-translation, structural rewrite, or TSAPA-style evolutionary optimization | Best-effort |
| **Layer V** | Visible image logos / overlays | Mask → hole fill → morphological dilation → inpaint → restore | Mask removal verifiable; fidelity best-effort |
| **Layer M** | C2PA, EXIF, XMP, document properties | Format-aware metadata rewrite | Verifiable per format |
| **Soft-binding inspection** | Remote-manifest / in-content binding risk | Detection and warning only | Detection only |
| **SynthID scoring** | Pixel-domain SynthID-class signal | Optional external `reverse-SynthID` adapter | Maintainer-scoped score only |

The core remains **Python 3.10+ stdlib only**. Heavy or licensed edges sit behind explicit adapters:

- `pypdf` — optional structural PDF rewrite
- `exiftool`, `c2patool` — optional system tools
- local Ollama / OpenAI-compatible endpoint — optional Layer B execution
- external detector / LaMa / MI-GAN / diffusion command — optional visible-mark backend
- `gradio` — optional demo UI

---

## Quick start

```bash
git clone https://github.com/Pythoughts-labs/watermark-remover.git
cd watermark-remover
SCRIPTS=skills/remove-ai-marks/scripts

# Inspect before changing anything
python3 "$SCRIPTS/inspect_file.py" draft.md
python3 "$SCRIPTS/inspect_file.py" image.png --soft-binding

# Unified clean
python3 "$SCRIPTS/clean_file.py" draft.md -o draft.cleaned.md
python3 "$SCRIPTS/clean_file.py" image.png -o image.cleaned.png
```

### Install as an agent skill

```bash
# Project-local (Grok Build)
mkdir -p .grok/skills
ln -sfn "$(pwd)/skills/remove-ai-marks" .grok/skills/remove-ai-marks

# User-global
mkdir -p ~/.grok/skills
ln -sfn "$(pwd)/skills/remove-ai-marks" ~/.grok/skills/remove-ai-marks
```

Invoke with `/remove-ai-marks`, or ask to inspect/clean C2PA, hidden Unicode, visible marks, or statistical text marks.

---

## Batch mode

Directory mode preserves relative paths, skips its own `.cleaned.*`, `.mask.*`, and `.bak` artifacts, rejects output collisions, and returns non-zero when any file fails or retains a requested risk signal.

```bash
# All supported files below a directory
python3 "$SCRIPTS/clean_file.py" ./inputs -o ./cleaned --recursive

# Glob + extension allow-list
python3 "$SCRIPTS/clean_file.py" ./inputs -o ./cleaned \
  --recursive --glob "*.png" --extensions png

# Inspect a tree
python3 "$SCRIPTS/inspect_file.py" ./inputs --recursive --glob "*.md" --json
```

---

## Text

### Layer A — deterministic Unicode hygiene

```bash
python3 "$SCRIPTS/inspect_text.py" draft.txt --json
python3 "$SCRIPTS/clean_text.py" draft.txt -o draft.cleaned.txt --stats
```

Layer A removes or normalizes configured hidden carriers. Some Unicode is meaningful—ZWJ and variation selectors can affect emoji and orthographies, while bidi controls affect display order—so preservation rules are context-aware rather than blanket deletion.

### Layer B — rewrite attacks

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
```

The TSAPA-style engine is a real multi-objective evolutionary loop:

1. Generate a diverse candidate population.
2. Score attack fitness: PLL + n-gram diversity + lexical diversity.
3. Score fidelity: embedding cosine similarity, with a labeled shingle-Jaccard fallback.
4. Use NSGA-II non-dominated sorting and crowding distance.
5. Crossover at sentence boundaries; mutate the lowest-PLL sentence.
6. Select the Pareto knee point.

A logprobs-capable `/v1/completions` endpoint supplies PLL; `/v1/embeddings` supplies semantic similarity. Either can fail independently and degrade to an explicitly labeled stdlib proxy.

**Cost:** Layer B replaces the original wording and can flatten voice or precision. Prefer a non-origin model so the rewrite does not simply re-stamp the same scheme.

### Character-level perturbations

```bash
python3 "$SCRIPTS/perturb_text.py" draft.txt \
  --mode zero-width --strength 0.10 --seed 42
```

This is an opt-in anti-watermark transform inspired by 2026 character-perturbation research. It intentionally adds artifacts that Layer A removes. `zero-width` and `space-swap` are Layer-A reversible; `confusable` and `case` are not. It is **not** a hygiene pass and carries no detector guarantee.

---

## Visible image marks — MorphoMod-inspired pipeline

`morphomod.py` never guesses a watermark region. Supply one of:

- `--mask mask.pgm|mask.png` (white = remove)
- `--box X,Y,W,H`
- `--detect-command '...'` with `{input}`, `{mask}`, `{prompt}` placeholders

```bash
# Stdlib fallback: useful for simple backgrounds
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --box 900,920,120,60 --dilation 3 --backend simple

# Production adapter: external detector + inpainter
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --detect-command 'detector --input "{input}" --output "{mask}"' \
  --backend external \
  --command 'inpainter --image "{input}" --mask "{mask}" --output "{output}"'

# Combined visible pass, then metadata clean
python3 "$SCRIPTS/clean_file.py" input.png -o output.png \
  --visible-mask mask.pgm --dilate 3 --visible-backend simple
```

The stdlib dilation is non-cascading and O(width × height). PNG decoding supports non-interlaced 8-bit grayscale, RGB, and RGBA. The simple backend performs nearest-boundary wavefront fill; it is deliberately not marketed as LaMa-quality. JPEG visible cleaning requires an external backend; that backend owns JPEG compositing.

Paper-reported MorphoMod improvements are **not** this implementation's measured results. The report includes initial/refined mask pixels and requires manual fidelity review.

---

## Metadata and provenance

```bash
# PNG/JPEG/HEIC/HEIF/AVIF
python3 "$SCRIPTS/clean_image.py" photo.heic -o photo.cleaned.heic

# Any supported format
python3 "$SCRIPTS/clean_file.py" document.pdf -o document.cleaned.pdf

# Detect soft-binding / remote-manifest risk
python3 "$SCRIPTS/inspect_soft_binding.py" image.png --json
```

C2PA facts used by the parser:

- Manifest Stores use JUMBF/BMFF structures, claims/assertions, and COSE signatures.
- JPEG embeds through APP11/JUMBF.
- PNG uses the private ancillary `caBX` chunk—not generic `tEXt`.
- HEIF/AVIF are ISO-BMFF. Cleaning neutralizes matching boxes/items **in place**, preserving offsets.

Soft-binding **removal remains out of scope**. The inspector warns when an in-content binding or remote manifest may re-link provenance after hard-bound metadata is stripped.

### PDF quality

PDF cleaning uses this fallback order:

1. `exiftool`
2. optional `pypdf` full-document clone (outlines/forms/attachments/catalog retained; docinfo/XMP removed)
3. byte-exact unchanged copy with an explicit residual warning

```bash
python3 -m pip install pypdf
```

Encrypted PDFs are never regex-edited.

---

## Optional SynthID pixel scoring

The external [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID) checkout is not bundled and remains under its upstream non-commercial Research License.

```bash
SCRIPTS=skills/remove-ai-marks/scripts
"$SCRIPTS/setup_synthid.sh"

REVERSE_SYNTHID_DIR=~/reverse-SynthID \
  ~/reverse-SynthID/.venv/bin/python "$SCRIPTS/score_synthid.py" shot.png
```

Or build the scorer locally:

```bash
make docker-synthid-build
docker run --rm -v "$(pwd):/data" watermark-remover-synthid-scorer /data/shot.png
```

This is **scoring only**. The repository's carrier model and success figures are maintainer-reported reverse-engineering claims, not public Google architecture or an independent guarantee.

---

## Format support

| Format | Inspect / clean behavior |
| --- | --- |
| PNG | `caBX`, text/XMP/EXIF chunks; optional visible pipeline |
| JPEG | APP11/JUMBF and APP metadata; external visible backend supported |
| HEIC / HEIF / AVIF | ISO-BMFF brands, JUMBF/C2PA boxes, Exif/XMP item extents; offset-preserving neutralization |
| SVG | `<metadata>`, XMP, provenance-like comments |
| PDF | XMP/docinfo via exiftool or full-document pypdf clone; otherwise unchanged copy |
| DOCX | docProps cleaned; customXml inspected and preserved to avoid application-data loss |
| ODT | meta.xml and generator-like metadata |
| HTML | meta tags, provenance JSON-LD, `data-ai*` attributes |
| Markdown | AI-like YAML frontmatter + Layer A body |
| Text / code | Layer A; optional Layer B or character perturbation |

## Coverage and limits

| Channel | What this project does | What can remain |
| --- | --- | --- |
| Hidden/edit-based text | Deterministic Layer A scrub | Meaningful Unicode preserved by policy |
| Statistical text | Best-effort rewrite / TSAPA-style optimization | Strong or updated detector signal |
| Visible images | User/external mask + dilation + inpaint | Missed regions; inpaint artifacts |
| Hard-bound metadata | Format-aware stripping | Soft bindings, remote manifests, pixel marks |
| Pixel SynthID-class | Optional external score | Pixel/audio/video watermark remains |
| Training backdoors | Nothing | Out of scope |

No public universal text detector exists, and a detector miss does not prove every signal trace is absent. Stronger attacks trade fidelity for lower detectability. Provider implementations evolve.

## Ethics

For privacy, hygiene, accessibility, and research on **content you own**. Not for academic fraud, evading lawful disclosure requirements, or claiming that output is "proven human-written." See [`skills/remove-ai-marks/references/ethics.md`](skills/remove-ai-marks/references/ethics.md).

---

## Optional demo

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo.py
```

The demo calls the same modules as the CLI; it is not a separate implementation.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt
make check
```

## Documentation

- [`DESIGN.md`](DESIGN.md) — architecture, seams, guarantee classes, roadmap
- [`skills/remove-ai-marks/SKILL.md`](skills/remove-ai-marks/SKILL.md) — agent workflow
- [`references/mark-classes.md`](skills/remove-ai-marks/references/mark-classes.md)
- [`references/removal-matrix.md`](skills/remove-ai-marks/references/removal-matrix.md)
- [`references/vendor-notes.md`](skills/remove-ai-marks/references/vendor-notes.md)

## License

MIT — see [LICENSE](LICENSE).

## Primary references

- Anthropic, [*How Claude marks AI-generated content*](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4) (SynthID-Text, Nature 2024)
- Zhao et al., [*Invisible Image Watermarks Are Provably Removable Using Generative AI*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/10272bfd0371ef960ec557ed6c866058-Abstract-Conference.html) (NeurIPS 2024)
- [UnMarker](https://arxiv.org/abs/2405.08363) (IEEE S&P 2025)
- [CtrlRegen](https://arxiv.org/abs/2410.05470) (ICLR 2025)
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html)
- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://arxiv.org/abs/2301.10226)
