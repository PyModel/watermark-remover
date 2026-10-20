![watermark-remover](https://raw.githubusercontent.com/PyModel/watermark-remover/main/assets/banner.svg)

<p align="center">
  <a href="https://pypi.org/project/watermark-remover/"><img alt="PyPI" src="https://img.shields.io/pypi/v/watermark-remover?logo=pypi&logoColor=white&label=pypi&color=e8a33d&labelColor=0f1110"></a>
  <a href="https://pypi.org/project/watermark-remover/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-30363d?logo=python&logoColor=white&labelColor=0f1110"></a>
  <a href="https://github.com/PyModel/watermark-remover/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/PyModel/watermark-remover/ci.yml?logo=github&logoColor=white&label=ci&labelColor=0f1110"></a>
  <a href="https://github.com/PyModel/watermark-remover/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-30363d?labelColor=0f1110"></a>
</p>

watermark-remover finds and removes hidden marks from files you own: invisible Unicode in text, AI and provenance metadata in images, documents, and video, and visible logos when you tell it where they are. It writes a new file and leaves the original alone unless you pass `--in-place`.

Every result says how far you can trust it. Deterministic cleaners report exactly what they removed. Model rewrites, inpainting, and detector evasion are labeled best-effort, and nothing here promises that a vendor's detector will miss the result.

## Install

```bash
python3 -m pip install watermark-remover
```

The core needs Python 3.10 or later and nothing else. Heavier features are optional extras, listed under [Optional extras](#optional-extras).

You can also skip the install and use the browser version at [watermark.pymodel.com](https://watermark.pymodel.com). Files stay on your device.

## First run

Clean a Markdown file that has a zero-width space, a word joiner, a zero-width joiner, and a no-break space hidden in it:

```bash
wm draft.md -o draft.cleaned.md
```

```text
wrote draft.cleaned.md format=markdown
  - no AI frontmatter keys or embedded data URIs removed
  - layer A text: removed=3 replaced=1
```

Three hidden characters were removed and the no-break space became a normal space. Add `--json` for a machine-readable report, or `--audit` to write a JSON record of what was removed.

Exit codes: 0 means nothing risky is left, 1 means a file failed or still carries a signal you asked to remove, 2 is a usage error, and 3 means an optional backend the clean needed is not installed, so that stage was skipped.

## Common tasks

Clean an image's metadata (C2PA, EXIF, XMP):

```bash
wm photo.jpg -o photo.cleaned.jpg
```

Clean a whole folder, keeping the folder layout:

```bash
wm ./inputs -o ./cleaned --recursive
wm ./inputs -o ./cleaned --recursive --glob "*.png" --extensions png
wm ./inputs -o ./cleaned --recursive --mode all --json
```

`--mode` picks what a batch does: `metadata` (the default), `visible` (known visible marks only), or `all`. With `--json`, a batch prints a summary with each file's outcome and the counts.

Remove a visible logo by pointing at it, or let the mark registry find known vendor marks (needs the `[pixel]` extra and a PNG output):

```bash
wm input.png -o output.png --visible-box 900,920,120,60
wm input.png -o output.png --mark auto
```

Rewrite text through your own local model, then re-scrub it:

```bash
wm draft.txt -o draft.cleaned.txt \
  --rewrite humanize --rewrite-backend openai-compatible \
  --rewrite-base-url http://127.0.0.1:8000 --rewrite-model YOUR_MODEL_ID
```

Endpoints that are not on your machine are refused unless you pass `--rewrite-allow-remote`. For Ollama, use `--rewrite-backend ollama` and port 11434.

Strip provenance from a video (needs the `[video]` extra, Python 3.11 or later, and a system `ffmpeg` and `ffprobe`):

```bash
python3 -m pip install 'watermark-remover[video]'
wm clip.mp4 -o clip.cleaned.mp4
wm clip.mp4 -o clip.cleaned.mp4 --mark sora
```

Visible marks on video are opt-in. `--mark` takes one of sora, veo, seedance, doubao, dola, hailuo, vidu, or kling, or `auto` to check them all; a mark is filled only on frames where it recurs. `--visible-box` fills a region you name on every frame.

Audit a folder or a website without changing anything:

```bash
wm-audit-dir ./inputs --json
wm-audit-site --sitemap https://example.com/sitemap.xml --json
```

Run the HTTP service and web UI on `http://127.0.0.1:8765/`:

```bash
wm-serve
```

## Ways to run it

| Command | What it does |
| --- | --- |
| `wm` | Cleans files and folders |
| `wm-tui` | Terminal UI for inspecting and cleaning |
| `wm-serve` | HTTP API and web UI |
| `wm-audit-dir` | Audits a directory, with JSON or SARIF output |
| `wm-audit-site` | Audits a website from its sitemap |

### Agent skill

The same cleaners ship as an agent skill. Clone the repository and link the whole skill folder into your harness's skills directory:

```bash
git clone https://github.com/PyModel/watermark-remover.git
ln -s "$(pwd)/watermark-remover/skills/remove-ai-marks" ~/.claude/skills/remove-ai-marks
```

Use your harness's own directory (Pi `~/.pi/agent/skills`, Codex `~/.agents/skills`, Cursor `.cursor/skills`). Then invoke `/remove-ai-marks`, or ask the agent to read `skills/remove-ai-marks/SKILL.md` and follow it.

### Terminal UI

```bash
wm-tui ./drafts --recursive --glob "*.md"
```

`wm-tui` renders with [@earendil-works/pi-tui](https://github.com/earendil-works/pi/tree/main/packages/tui) and needs [Bun](https://bun.sh) 1.4 or later on the `PATH`. On first launch it runs `bun install --frozen-lockfile --production`, which installs three runtime packages for the frontend. Move to a file, press Space to mark it, Ctrl+E to inspect, and Ctrl+R to clean only the marked file. Set `PI_TUI_WRITE_LOG` to a file or directory to capture the raw terminal output when reporting a rendering bug. Keys, presets, and setup are in [docs/terminal-ui.md](https://github.com/PyModel/watermark-remover/blob/main/docs/terminal-ui.md).

### From a clone

The scripts in `skills/remove-ai-marks/scripts/` are plain CLIs and run without installing the package:

```bash
python3 skills/remove-ai-marks/scripts/inspect_file.py draft.md
python3 skills/remove-ai-marks/scripts/clean_file.py draft.md -o draft.cleaned.md
```

`inspect_file.py` reports what it finds and writes nothing.

## What it removes

| Target | Method | Result |
| --- | --- | --- |
| Hidden Unicode, bidi controls, tag characters, unusual spaces | Context-aware deterministic scrub | Verifiable |
| C2PA, EXIF, XMP, document properties | Format-aware metadata rewrite | Verifiable per format |
| Video container provenance | Container metadata strip and remux | Verifiable per container |
| Visible logos and overlays | Supplied box, mask, or detector, or the opt-in mark registry, then fill | Mask removal verifiable, fill quality best-effort |
| Statistical text watermarks | Paraphrase, back-translation, or evolutionary rewrite | Best-effort |
| SynthID-class pixel signals | Optional external scorer, opt-in DCT suppression | Score, best-effort removal |
| Soft bindings and remote manifests | Detection and warning | Detection only |

### Format support

| Format | Inspect and clean behavior |
| --- | --- |
| PNG | `caBX`, text, XMP, EXIF, and `tIME` chunks, plus the optional visible pipeline |
| JPEG | APP11/JUMBF and APP metadata; visible cleaning through an external backend |
| WebP | `C2PA` chunk always stripped; `ICCP`, `EXIF`, and `XMP ` per mode |
| BMP, GIF | Trailing metadata and comment or XMP extensions |
| TIFF / BigTIFF | In-place IFD patching: XMP, IPTC, Exif, GPS, MakerNote, Photoshop, XP, SubIFD, and Interoperability tags dropped; structural tags kept |
| HEIC, HEIF, AVIF | JUMBF and C2PA UUID boxes, XMP `uuid` boxes, direct Exif and XMP extents; unsupported layouts are refused |
| MP4, MOV, M4V, WebM, MKV, AVI, FLV | Container provenance markers read and stripped |
| SVG | `<metadata>`, XMP, provenance-like comments |
| PDF | XMP and docinfo through exiftool, then qpdf, then pypdf, otherwise an unchanged copy with a warning |
| DOCX / XLSX / PPTX | docProps cleaned, `docProps/thumbnail` previews dropped, embedded media and customXml removed, hidden Unicode cleaned in body text |
| EPUB, ODT | Package metadata scrubbed, hidden Unicode cleaned in body text |
| HTML | meta tags, JSON-LD provenance keys, provenance-phrase comment blocks, `data-ai*` attributes |
| Markdown | AI-like YAML frontmatter, hidden Unicode in the body |
| Text and code | Hidden Unicode, optional model rewrite or character perturbation |

Format details are in [docs/metadata.md](https://github.com/PyModel/watermark-remover/blob/main/docs/metadata.md).

## Optional extras

None of these is needed for text and metadata cleaning.

| Extra | Adds |
| --- | --- |
| `[visible]`, `[pixel]` | NumPy, Pillow, and OpenCV for visible-mark removal; `[pixel]` also powers the mark registry |
| `[migan]`, `[lama]` | ONNX inpainting models, downloaded on first use |
| `[video]` | PyAV for video probing and remuxing (Python 3.11 or later) |
| `[provenance]` | `c2pa-python` |
| `[quality]` | scikit-image |
| `[ai]` | PyTorch and torchvision |
| `[detect]`, `[source-classify]` | Invisible-watermark and source detectors |
| `[classify]`, `[classify-onnx]`, `[trustmark]` | Image classifiers and TrustMark; installed only when named |
| `[all]` | `[visible]`, `[quality]`, `[ai]`, `[provenance]`, `[detect]`, and `[source-classify]` |

Detectors only report evidence. A missing signal means unknown, not clean.

## Limits

| Channel | What can remain |
| --- | --- |
| Hidden text | Unicode that changes meaning, such as emoji joiners, is kept on purpose |
| Statistical text | A strong or updated detector can still find the signal |
| Visible marks | Missed regions and fill artifacts; check the output |
| Metadata | Soft bindings, remote manifests, and pixel-level marks |
| Pixel watermarks | The watermark itself; evasion is never certified |
| Training backdoors | Out of scope |

No public detector covers every text watermark, so a detector miss does not prove a file is clean.

## Use it responsibly

This tool is for privacy, hygiene, accessibility, and research on content you own or may modify. Do not use it to claim others' work, bypass copyright protections, or hide AI use where you are required to disclose it. Removing a mark does not give you rights to the content or change your legal obligations. See [docs/responsible-use.md](https://github.com/PyModel/watermark-remover/blob/main/docs/responsible-use.md) and [ethics.md](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/references/ethics.md).

## More documentation

- [Text cleaning, rewriting, and detection](https://github.com/PyModel/watermark-remover/blob/main/docs/text.md)
- [Images: visible marks, degradation, SynthID scoring](https://github.com/PyModel/watermark-remover/blob/main/docs/images.md)
- [Metadata and provenance](https://github.com/PyModel/watermark-remover/blob/main/docs/metadata.md)
- [HTTP service, browser app, and Docker](https://github.com/PyModel/watermark-remover/blob/main/docs/http-service.md)
- [Terminal UI](https://github.com/PyModel/watermark-remover/blob/main/docs/terminal-ui.md)
- [How it works, libraries, and repository layout](https://github.com/PyModel/watermark-remover/blob/main/docs/how-it-works.md)
- [Development and the demo app](https://github.com/PyModel/watermark-remover/blob/main/docs/development.md)
- [Agent skill instructions](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/SKILL.md) and [design notes](https://github.com/PyModel/watermark-remover/blob/main/DESIGN.md)
- [Release notes](https://github.com/PyModel/watermark-remover/tree/main/docs/release-notes/), [contributing](https://github.com/PyModel/watermark-remover/blob/main/CONTRIBUTING.md), [security](https://github.com/PyModel/watermark-remover/blob/main/SECURITY.md)

## License

MIT. See [LICENSE](https://github.com/PyModel/watermark-remover/blob/main/LICENSE).
