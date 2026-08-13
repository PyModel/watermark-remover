# watermark-remover — Design

Architecture informed by a primary-source evidence review completed in August 2026. The corrected facts and claims discipline are recorded here so the shipped design does not depend on local, gitignored research notes.

## 1. Product contract

1. **Verifiable and best-effort results never mix.** Unicode counts and removed metadata structures are verifiable. Rewrite/inpainting/detector attacks are probabilistic and labeled as such.
2. **No “watermark-free” certification.** A detector miss is not proof that every signal trace is absent.
3. **Inspect before clean.** Mutation interfaces have corresponding inspection/reporting paths.
4. **Stdlib core, explicit adapters.** Core behavior works on Python 3.10+ without third-party packages. Network/model/GPU/system tools sit behind visible seams.
5. **No destructive guessing.** Visible cleaning requires a user mask, explicit box, or detector adapter. The implementation never guesses “the center must be the watermark.”
6. **Content you own.** See `skills/remove-ai-marks/references/ethics.md`.

## 2. Deep modules and seams

Each module exposes one small interface; implementation complexity stays local.

| Module | Interface | Hidden implementation / invariant |
| --- | --- | --- |
| Batch discovery | `batch_inputs.collect_inputs(...)` | Glob/recursive discovery, generated-file skipping, deduplication, relative output paths |
| Layer A | `text_unicode.clean_text(text, ...)` | Unicode classification, context-aware semantic preservation, normalization, stats |
| Layer B | `rewrite_text.rewrite(...)` | Backend routing; delegates TSAPA optimization to `tsapa.tsapa(...)` |
| TSAPA engine | `tsapa.tsapa(text, llm=..., pll=..., embed=...)` | Chunking, fitness, NSGA-II, crowding, crossover, PLL-guided mutation, knee selection |
| Visible marks | `morphomod.remove_visible(path, dest, ...)` | Mask I/O, hole fill, O(n) dilation, PNG codec, inpaint adapter, restore |
| Raster metadata | `image_meta.clean_image(path, dest, ...)` | PNG/JPEG parsing; delegates ISO-BMFF work to `heif_meta.neutralize_heif(...)` |
| Containers | `container_meta.clean_container(path, dest)` | SVG/PDF/DOCX/ODT/HTML/Markdown format logic |
| Soft-binding risk | `inspect_soft_binding.inspect_soft_binding(path)` | Byte scan plus optional C2PA reader; detection only |

### Real adapter seams

A seam exists only where at least two adapters are real:

- **LLM rewrite:** Ollama and OpenAI-compatible HTTP; offline tests inject a fake callable.
- **PLL:** OpenAI-compatible token logprobs and a labeled heuristic fallback.
- **Semantic similarity:** `/v1/embeddings` and shingle-Jaccard fallback.
- **Visible inpainting:** stdlib nearest-boundary fallback and an external command adapter (LaMa/MI-GAN/diffusion tools).
- **Visible localization:** user mask/box and an external detector command.
- **PDF:** exiftool, full-document pypdf clone, and unchanged-copy fallback.
- **SynthID scoring:** unavailable/no-op default and external `reverse-SynthID` checkout.

The CLI modules orchestrate these interfaces; they do not contain alternate implementations.

## 3. Channel × layer matrix

| Channel | Layer | Mechanism | Guarantee class | Status |
| --- | --- | --- | --- | --- |
| Text | **A** | Hidden Unicode/space cleanup with semantic preservation | Verifiable | Shipped |
| Text | **B** | Paraphrase/back-translation/structural/TSAPA-style rewrite | Best-effort | Shipped |
| Text | **C** | Seeded character perturbation | Best-effort; can reduce text integrity | Shipped, opt-in |
| Files | **M** | C2PA/EXIF/XMP/doc properties | Verifiable per parser | Shipped |
| Images | **V** | Explicit mask → refine → dilate → inpaint → restore | Region action verifiable; fidelity best-effort | Shipped |
| Images | **P** | Optional external SynthID score | Maintainer/version scoped | Scoring only |
| Files | **S** | Soft-binding/remote-manifest inspection | Detection only | Shipped |
| Video | **V/M/P** | Temporal inpaint, metadata, regeneration | Best-effort | Roadmap |
| — | Training backdoors / soft-binding removal | — | Out of scope | — |

## 4. Correctness invariants

### Text

- ZWSP/tag-character/orphan controls remain removable.
- Contextual ZWJ/ZWNJ, variation selectors, invisible math operators, and balanced bidi controls are preserved by default because removal can change rendering or meaning.
- `--strip-semantic-format` is an explicit aggressive mode.
- Character perturbation runs last because Layer A intentionally reverses its safe modes.
- TSAPA requested through `clean_file.py` requires a live configured backend; it never writes an operator prompt into the output file or silently falls back.

### Binary formats

- PNG/JPEG critical pixel data is never removed because a payload contains marker-like bytes.
- HEIF/AVIF cleaning preserves file length and item offsets: matching JUMBF boxes become `free`; targeted item bytes are overwritten at equal length.
- Camera/editor EXIF is preserved when `--keep-non-ai-metadata` is selected.
- PDFs are never byte-deleted without rebuilding cross-reference/object offsets; absent a structural cleaner, input is copied unchanged and reported residual.
- Encrypted PDFs are never regex-edited.
- pypdf clones the complete document graph, writes to memory first, then publishes a complete parseable result.
- DOCX `customXml` is preserved because it can back content controls/business data; residual provenance is reported rather than silently deleting application data.

### Visible images

- Dilation reads the original mask—not a mask being mutated in place—so it cannot cascade into a frame-wide flood.
- A completely masked image is rejected: no surrounding context exists for inpainting.
- The stdlib codec accepts only non-interlaced 8-bit gray/RGB/RGBA PNG and fails closed otherwise.
- External commands run with `shell=False`, have timeouts, and must produce their declared output.
- Original pixels outside the refined mask are restored exactly for PNG.

### Batch / CLI

- Directory outputs preserve relative paths.
- Multiple roots are namespaced; output collisions fail.
- `.cleaned.*`, `.mask.*`, and `.bak` artifacts are not reprocessed.
- A directory is batch mode even if a glob returns one file.
- Exit `0` means all requested operations completed without retained requested risk; `1` means processing error or residual signal; `2` means usage/input selection error.

## 5. Layer B — TSAPA-style implementation

```
chunk → diverse population → evaluate → NSGA-II reduce
      → sentence crossover → PLL-guided mutation → repeat
      → Pareto knee point
```

Attack fitness:

```
f_atk = w1·PLL + w2·ngram_diversity + w3·lexical_diversity
```

Fidelity is embedding cosine similarity when available, otherwise a labeled shingle-Jaccard proxy. The HTTP PLL adapter consumes token logprobs from `/v1/completions`; failures fall back to `heuristic_pll`. Literature ASR/BERTScore figures describe the paper's experiments—not this run.

## 6. Layer V — MorphoMod-inspired implementation

```
explicit/external localization
  → fill enclosed mask holes
  → square-kernel dilation (d=3 default)
  → stdlib or external inpaint
  → restore original outside mask
  → metadata clean (when invoked through clean_file.py)
```

The stdlib backend is intentionally modest: nearest-boundary wavefront fill works on simple backgrounds. Production-quality texture synthesis belongs in an external LaMa/MI-GAN/diffusion adapter. No model weights or research-licensed code are vendored.

## 7. Corrected research facts

- C2PA Manifest Stores use JUMBF/BMFF structures, claims/assertions, and COSE signatures; assertions may contain JSON/JSON-LD, CBOR, or embedded content.
- JPEG embeds through APP11/JUMBF; PNG uses private ancillary `caBX`, not generic `tEXt`.
- Inserting bytes is easy; creating a valid, trusted, correctly bound Content Credential is not.
- Hard-bound metadata stripping does not remove pixel marks, fingerprints, remote manifests, or soft bindings.
- The NeurIPS 2024 paper is *Invisible Image Watermarks Are Provably Removable Using Generative AI*; *Erasing the Invisible* is a separate competition.
- Deep Image Prior removal is 2025; CtrlRegen is ICLR 2025; NFPA is *The Future Unmarked* (NeurIPS 2025).
- `reverse-SynthID` carrier models and figures are maintainer-reported reverse engineering, not published Google internals or independent proof.
- Unicode sanitization is distinct from token-distribution watermarking such as SynthID-Text.
- `C2PAremover` is Go/WASM, not a Python package; `noai-watermark` does not document visible-mark removal.

## 8. Shipped formats

| Format | Behavior |
| --- | --- |
| PNG | C2PA/text/XMP/EXIF chunks; stdlib visible pipeline |
| JPEG | APP/JUMBF metadata; external visible backend |
| HEIC/HEIF/AVIF | ISO-BMFF brands, JUMBF boxes, Exif/XMP item extents |
| SVG/PDF/DOCX/ODT | Format-aware metadata rewrite |
| HTML/Markdown | Provenance metadata + Layer A text |
| Text/code | Layer A, Layer B, character perturbation |

## 9. Roadmap

1. WebP metadata support.
2. Video metadata without transcoding.
3. Video visible removal with optical-flow temporal consistency.
4. Optional diffusion regeneration adapters, only with version-scoped detector evidence.
5. Ground-truth-aware visible-removal metrics when paired originals exist.

## 10. Out of scope

Pixel-watermark removal guarantees, C2PA soft-binding removal, training backdoors, secret-key detector emulation, “proven human-written” claims, and any universal detector-failure certification. Detection/warning for soft binding is shipped; removal is not.
