# How it works

Implementation notes, the libraries involved, and the repository layout.

## Interesting techniques

**Context-aware Unicode scrubbing.** [`text_unicode.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/text_unicode.py) classifies every hidden carrier rather than deleting a blocklist. Zero-width joiners hold emoji sequences together and variation selectors change glyph form, so those survive. Tag characters and bidi overrides do not. Bidi handling follows the same directional model browsers expose through [`unicode-bidi`](https://developer.mozilla.org/en-US/docs/Web/CSS/unicode-bidi), and cleaned text is normalized to NFC, the form described under [`String.prototype.normalize()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/normalize).

**NSGA-II inside a text rewriter.** [`tsapa.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/tsapa.py) is a genuine multi-objective loop, not a prompt template: population generation, non-dominated sorting, crowding distance, crossover at sentence boundaries, mutation targeted at the weakest sentence, and Pareto knee selection. Attack strength and semantic fidelity are optimized as two separate objectives so neither silently wins.

**Scoring that names its own fallback.** Pseudo-log-likelihood comes from an OpenAI-compatible endpoint with logprobs, and cosine similarity from an embeddings endpoint. Either can fail on its own and drop to a standard-library proxy, which is reported as a proxy rather than passed off as the real measurement.

**Non-cascading dilation in pure Python.** [`morphomod.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/morphomod.py) dilates a mask in a single pass against the original buffer, so growth does not compound across iterations. It runs in O(width × height) with no array library.

**Patch-based inpainting with no model.** The default visible backend searches nearby patches by boundary error, picks the best match, and fully replaces the refined mask so watermark pixels cannot bleed through. It is not LaMa quality and is not sold as such, but it needs nothing installed.

**In-place ISO-BMFF neutralization.** [`heif_meta.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/heif_meta.py) overwrites JUMBF and C2PA UUID boxes and direct-file Exif and XMP extents without moving anything, so every byte offset in the file stays valid. Layouts it cannot prove safe fail closed instead of being rewritten.

**PNG chunk surgery.** [`image_meta.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/image_meta.py) walks the chunk stream, removes the private ancillary `caBX` chunk that C2PA actually uses in PNG, and recomputes CRCs with `zlib.crc32`.

**Markup-aware container cleaning.** [`container_meta.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/container_meta.py) targets SVG [`<metadata>`](https://developer.mozilla.org/en-US/docs/Web/SVG/Element/metadata) elements, HTML [`<meta>`](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta) tags, JSON-LD provenance blocks, and [`data-*` attributes](https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/data-*) matching `data-ai*`. DOCX `customXml` parts are dropped and the dangling relationships and Content-Type overrides are pruned, because customXml can re-carry provenance data; this is a provenance tool, so pruning keeps the package valid instead of leaving orphaned parts behind.

**Fallback chains that degrade loudly.** PDF cleaning tries `exiftool`, then a full-document `pypdf` clone, then a byte-exact copy with an explicit residual warning. The third case still returns a file, and still tells you nothing was removed.

**The banner is drawn, not exported.** [`assets/banner.svg`](https://github.com/PyModel/watermark-remover/blob/main/assets/banner.svg) uses two overlapping [`<clipPath>`](https://developer.mozilla.org/en-US/docs/Web/SVG/Element/clipPath) regions over one duplicated block of text, so the same codepoints render dim on the left and lit on the right. That puts the scrub line in the middle with no gradient mask and no raster asset.

## Technologies and libraries

Nothing below is required to run the core.

- [pypdf](https://github.com/py-pdf/pypdf) for a structural PDF clone that keeps outlines, forms, and attachments while dropping docinfo and XMP.
- [ExifTool](https://exiftool.org/), [qpdf](https://qpdf.sourceforge.io/), and [c2patool](https://github.com/contentauth/c2patool) as system binaries, used when present.
- [Gradio](https://www.gradio.app/) for [`demo.py`](https://github.com/PyModel/watermark-remover/blob/main/demo.py), which calls the same modules as the CLI rather than reimplementing them.
- [Ollama](https://ollama.com/) or any OpenAI-compatible endpoint for Layer B execution. Tests inject a fake callable instead.
- [LaMa](https://github.com/advimman/lama) and [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) as external inpainting commands behind the `external` backend.
- [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) for optional pixel scoring. Not bundled, and non-commercial upstream.
- [pytest](https://docs.pytest.org/) and [Ruff](https://docs.astral.sh/ruff/) for the gate behind `make check`.
- The [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html) and JUMBF (ISO/IEC 19566-5) define the box structures the parsers walk.

The banner loads no web font. It requests a system monospace stack in order: `ui-monospace`, [SF Mono](https://developer.apple.com/fonts/), Menlo, [Consolas](https://learn.microsoft.com/en-us/typography/font-list/consolas), [DejaVu Sans Mono](https://dejavu-fonts.github.io/), then generic `monospace`. GitHub proxies README images, so an external font request would be stripped anyway.

## Project structure

```
watermark-remover/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── assets/
├── docs/
│   └── windows-autostart.md
├── integrations/
│   └── cursor/
├── research/                       # gitignored reference material
├── skills/
│   ├── clean-user-facing-text/     # lightweight Cursor skill (vendored engine)
│   └── remove-ai-marks/
│       ├── references/
│       ├── scripts/
│       ├── tui/                    # wm-tui frontend (Bun + pi-tui)
│       └── web/                    # web UI (React; dist/ committed for wm-serve, pages/ builds the browser site)
├── tests/
│   └── fixtures/
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── DESIGN.md
├── Dockerfile                      # core HTTP service image
├── Dockerfile.ctrlregen
├── Dockerfile.markdiffusion
├── Dockerfile.markllm
├── Dockerfile.synthid
├── LICENSE
├── Makefile
├── README.md
├── SECURITY.md
├── compose-check.sh
├── compose.yaml
├── demo.py
├── install-skill.sh
├── install_skill.py
├── pyproject.toml
├── pytest.ini
├── requirements-demo.txt
└── requirements-test.txt
```

[`skills/remove-ai-marks/`](https://github.com/PyModel/watermark-remover/tree/main/skills/remove-ai-marks) is the whole product. It is laid out as an agent skill so it can be symlinked into `.grok/skills` or `~/.grok/skills` and invoked directly, but [`scripts/`](https://github.com/PyModel/watermark-remover/tree/main/skills/remove-ai-marks/scripts) is a set of ordinary CLIs that work on their own. [`references/`](https://github.com/PyModel/watermark-remover/tree/main/skills/remove-ai-marks/references) holds the source notes the parsers were built from, including [`ethics.md`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/references/ethics.md) and the vendor behavior notes.

`research/` is intentionally gitignored local evidence and dogfood material; the durable claims discipline is captured in [`DESIGN.md`](https://github.com/PyModel/watermark-remover/blob/main/DESIGN.md). [`assets/`](https://github.com/PyModel/watermark-remover/tree/main/assets) holds repository images. [`tests/fixtures/`](https://github.com/PyModel/watermark-remover/tree/main/tests/fixtures) holds the small binary files the format parsers are tested against.
