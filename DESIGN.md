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
| Asset routing | `asset_kind.classify_asset(path, forced_kind=...)` | Immutable extension catalog, override/extension/content precedence, 4 KiB sniff, text fallback |
| Batch discovery | `batch_inputs.select_inputs(...) -> InputSelection` | Source validation, glob/recursive discovery, generated-file skipping, deduplication, excluded output roots, batch determination |
| Single-asset cleaning | `clean_asset.clean_asset(path, dest, plan) -> CleanResult` | Immutable per-kind plans, semantic residual state, backup/write ownership; no presentation or exit-code mapping |
| External commands | `external_command.run_command(argv, *, timeout=..., output_limit=...) -> CommandResult` | Shell-free argv execution, bounded output tails, one end-to-end deadline, process-group cleanup |
| PNG traversal | `png_chunks.iter_png_chunks(data)` | Shared memoryview-backed chunk payloads, strict IHDR/IEND order, length bounds, and CRC validation |
| Layer A | `text_unicode.clean_text(text, ...)` | Unicode classification, context-aware semantic preservation, normalization, stats |
| Layer B | `rewrite_text.RewritePlan`; `rewrite_text.rewrite(text, plan)` | Immutable rewrite policy, backend routing, private provider adapters, TSAPA delegation |
| Layer B transport | `layer_b_http.request_json(endpoint, route, payload, ...)` | Endpoint/route/timeout validation, path-prefix-safe joining, same-origin redirects, bounded JSON-object decoding, safe transport errors |
| TSAPA engine | `tsapa.tsapa(text, llm=..., pll=..., embed=...)` | Chunking, fitness, NSGA-II, crowding, crossover, PLL-guided mutation, knee selection |
| Visible marks | `morphomod.remove_visible(path, dest, plan)` | Immutable `VisiblePlan`; mask I/O, hole fill, O(n) dilation, PNG codec, inpaint adapter, restore |
| Raster metadata | `image_meta.clean_image(path, dest, ...)` | PNG/JPEG parsing; delegates ISO-BMFF work to `heif_meta.neutralize_heif(...)` |
| Containers | `container_meta.clean_container(path, dest)` | SVG/PDF/DOCX/ODT/HTML/Markdown format logic |
| Soft-binding risk | `inspect_soft_binding.inspect_soft_binding(path)` | Byte scan plus optional C2PA reader; detection only |

### Real adapter seams

A seam exists only where at least two adapters are real:

- **LLM rewrite:** Ollama and OpenAI-compatible HTTP; offline tests inject a fake callable. Thinking suppression is an explicit Qwen/Transformers-compatible option, never an assumed extension.
- **PLL:** OpenAI-compatible token logprobs and a labeled heuristic fallback.
- **Semantic similarity:** `/v1/embeddings` and shingle-Jaccard fallback.
- **Visible inpainting:** stdlib texture-patch (default), nearest-boundary uniform-background fallback, and an external command adapter (LaMa/MI-GAN/diffusion tools).
- **Visible localization:** user mask/box and an external detector command.
- **PDF:** exiftool, qpdf structural rewrite (when present), full-document pypdf clone, and unchanged-copy fallback.
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

### Asset routing

- `clean_file.py`, `inspect_file.py`, and `demo.py` obtain the processing family through `asset_kind.classify_asset(...)`.
- Explicit override wins; otherwise a known extension wins over conflicting bytes. Unknown suffixes use a 4 KiB prefix for image/container detection, then fall back to text.
- ZIP-based container detection opens the file-backed archive because its central directory is at the end; classification never loads the whole file.
- Source eligibility, presentation, cleaning, and reporting remain caller-owned.

### Text

- ZWSP/tag-character/orphan controls remain removable.
- Contextual ZWJ/ZWNJ, variation selectors, invisible math operators, and balanced bidi controls are preserved by default because removal can change rendering or meaning.
- `--strip-semantic-format` is an explicit aggressive mode.
- Character perturbation runs last because Layer A intentionally reverses its safe modes.
- TSAPA requested through `clean_file.py` requires a live configured backend; it never writes an operator prompt into the output file or silently falls back.

### Binary formats

- PNG/JPEG critical pixel data is never removed because a payload contains marker-like bytes.
- HEIF/AVIF cleaning preserves file length and item offsets: matching JUMBF/C2PA UUID boxes become `free`; targeted item bytes are overwritten at equal length. Unsupported external/idat metadata extents fail closed.
- Camera/editor EXIF is preserved when `--keep-non-ai-metadata` is selected.
- PDFs are never byte-deleted without rebuilding cross-reference/object offsets; absent a structural cleaner, input is copied unchanged and reported residual.
- Encrypted PDFs are never regex-edited.
- pypdf clones the complete document graph, writes to memory first, then publishes a complete parseable result.
- DOCX `customXml` parts are dropped and dangling relationships / Content-Type overrides are pruned: leftover customXml can re-carry provenance data, and pruning keeps the package valid.

### Visible images

- Dilation reads the original mask—not a mask being mutated in place—so it cannot cascade into a frame-wide flood.
- A completely masked image is rejected: no surrounding context exists for inpainting.
- The stdlib codec accepts only non-interlaced 8-bit gray/RGB/RGBA PNG and fails closed otherwise.
- External commands run with `shell=False`, have timeouts, and must produce their declared output within the encoded-size limit.
- Original pixels outside the refined mask are restored exactly for PNG.

### Batch / CLI

- Directory outputs preserve relative paths; every input/output/mask alias and collision is rejected before the first write.
- Directory roots are namespaced when multiple roots are provided; explicit-file basename collisions fail before any write.
- `.cleaned.*`, `.mask.*`, and `.bak` artifacts are not reprocessed; in-place backups use exclusive no-follow creation.
- A directory is batch mode even if a glob returns one file.
- Exit `0` means all requested operations completed without retained requested risk; `1` means processing error or residual signal; `2` means usage/input selection error.

### Confidence levels and audit reporting

- Finding confidence is one of `confirmed`, `probable`, `informational`,
  `likely_false_positive` — shared by the inspectors and the audit suite.
- SARIF 2.1.0 export (`wm-audit-dir`, `wm-audit-site`): rules
  `AI-WATERMARK-C2PA` (error), `AI-WATERMARK-METADATA` (warning),
  `AI-WATERMARK-UNICODE-LAYER-A` (warning), `AI-STYLES-HIGH-PROBABILITY` (note);
  URIs are relative via `%SRCROOT%`; driver name `watermark-remover`.
- Audit exit codes: `0` no actionable findings, `1` actionable findings,
  `2` usage/refusal error, `3` partial scan (inconclusive — some inputs could
  not be scanned).

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

`RewritePlan.prompt(...)` creates an offline-only plan for the demo;
`RewritePlan.live_tsapa_from_environment(...)` resolves the unified cleaner's live
configuration. The standalone CLI builds the same frozen plan from its flags.
`rewrite(text, plan)` is the sole execution interface.

Generation, PLL, and embedding requests share the Layer B transport seam. Provider
adapters still own request payloads, authentication policy, and response fields;
TSAPA owns fallback labels and degradation counts.

## 6. Layer V — MorphoMod-inspired implementation

```
explicit/external localization
  → fill enclosed mask holes
  → square-kernel dilation (d=3 default)
  → stdlib or external inpaint
  → restore original outside mask
  → metadata clean (when invoked through clean_file.py)
```

The default stdlib backend selects a nearby texture patch by boundary error and fully replaces the dilated mask; optional module-level feathering is explicit. Dogfood tests cover flat and textured backgrounds. Nearest-boundary wavefront fill remains available as `simple` for uniform backgrounds. Production-grade semantic reconstruction still belongs in an external LaMa/MI-GAN/diffusion adapter. No model weights or research-licensed code are vendored.

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
| WebP/BMP/GIF/TIFF | Format-aware metadata rewrite (RIFF chunks, trailing bytes, app extensions, IFD patching) |
| HEIC/HEIF/AVIF | ISO-BMFF brands, JUMBF boxes, top-level XMP uuid, Exif/XMP item extents |
| SVG/PDF/DOCX/XLSX/PPTX/EPUB/ODT | Format-aware metadata rewrite (+ Layer A on inline text) |
| HTML/Markdown | Provenance metadata + Layer A text |
| Text/code | Layer A, Layer B, character perturbation |

## 9. Roadmap

1. Video metadata without transcoding.
2. Video visible removal with optical-flow temporal consistency.
3. Spectral subtraction via reverse-SynthID V4 remains research-only; shipped pixel work is DCT band suppression (--remove-synthid) plus external regeneration (--remove-pixel ctrlregen|diffusion), all best-effort.
4. Ground-truth-aware visible-removal metrics when paired originals exist.
5. Audio watermark detection/removal research.

Shipped since the 0.4.0 port: the HTTP service (wm-serve + OpenAPI), the audit suite (wm-audit-dir, wm-audit-site, SARIF export), zero-LLM stylometry, vendor/harness text detectors, and the MarkLLM/CtrlRegen/MarkDiffusion heavy-backend adapters.

## 10. Out of scope

Pixel-watermark removal guarantees, C2PA soft-binding removal, training backdoors, secret-key detector emulation, “proven human-written” claims, and any universal detector-failure certification. Detection/warning for soft binding is shipped; removal is not.
