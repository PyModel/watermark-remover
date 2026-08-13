---
name: remove-ai-marks
description: >
  Inspect and clean multi-vendor AI provenance signals: hidden Unicode (Layer A),
  statistical text watermarks via rewrite/TSAPA (Layer B), visible image marks via
  mask+dilation+inpainting, and C2PA/EXIF/XMP/container metadata on PNG/JPEG/HEIC/
  HEIF/AVIF/SVG/PDF/DOCX/ODT/HTML/Markdown. Detects (does not remove) C2PA soft-
  binding risk. Use for watermark/C2PA/Content Credentials/AI metadata/hidden
  Unicode requests or /remove-ai-marks.
---

# Remove AI marks

Use only on content the user owns. Never describe output as “watermark-free,” “undetectable,” or “proven human-written.” Separate **verifiable** actions from **best-effort** attacks.

Read when needed:

- `references/mark-classes.md`
- `references/vendor-notes.md`
- `references/removal-matrix.md`
- `references/ethics.md`
- `references/how-claude-marks.md`

Resolve scripts from this skill directory:

```bash
SCRIPTS="<skill_dir>/scripts"
```

## Workflow

### 1. Classify

| Input | Pipeline |
| --- | --- |
| Pasted text / `.txt` / code | Layer A; offer Layer B for prose |
| Markdown / HTML | container metadata + Layer A; offer Layer B for prose |
| PNG / JPEG | image metadata; visible pipeline only when a mask/localizer is available |
| HEIC / HEIF / AVIF | ISO-BMFF metadata neutralization |
| SVG / PDF / DOCX / ODT | container cleaner |
| Directory / mixed files | unified batch CLI |

### 2. Inspect first

```bash
python3 "$SCRIPTS/inspect_file.py" INPUT --json
python3 "$SCRIPTS/inspect_text.py" INPUT --json
python3 "$SCRIPTS/inspect_image.py" IMAGE --json
python3 "$SCRIPTS/inspect_soft_binding.py" IMAGE --json
```

Summarize suspicious code points, metadata structures, optional-tool findings, and soft-binding risk. `inspect_soft_binding.py` is detection only.

### 3. Deterministic clean

```bash
python3 "$SCRIPTS/clean_file.py" INPUT -o OUTPUT
python3 "$SCRIPTS/inspect_file.py" OUTPUT --json
```

Batch:

```bash
python3 "$SCRIPTS/clean_file.py" ./inputs -o ./cleaned --recursive --glob "*.png"
```

Layer A defaults to semantic preservation: contextual ZWJ/ZWNJ, variation selectors, invisible math operators, and balanced bidi controls remain. Only use `--strip-semantic-format` after warning that rendering or meaning can change.

```bash
python3 "$SCRIPTS/clean_text.py" INPUT -o OUTPUT --stats
# aggressive: --strip-semantic-format --aggressive-homoglyphs --nfkc
```

Optional tools are auto-detected. PDF order: exiftool → full-document pypdf clone → byte-exact unchanged copy with residual warning. Encrypted PDFs are never regex-edited. DOCX customXml is inspected but preserved because it may contain application data.

### 4. Visible image marks (only when requested)

Never guess a region. Require `--mask`, `--box`, or `--detect-command`.

```bash
# Simple-background stdlib fallback
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --box X,Y,W,H --dilation 3 --backend simple

# External localizer / inpainter
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --detect-command 'detector --input "{input}" --output "{mask}"' \
  --backend external \
  --command 'inpainter --image "{input}" --mask "{mask}" --output "{output}"'
```

The report must say “MorphoMod-inspired”; paper metrics are not this run’s metrics. PNG original pixels outside the refined mask are restored exactly. JPEG visible work requires an external backend.

### 5. Layer B — always offer for natural-language prose

Default prompt-only path:

```bash
python3 "$SCRIPTS/rewrite_text.py" INPUT --backend print-prompt --strength paraphrase
```

TSAPA-style multi-objective path:

```bash
# Operator pack, no model call
python3 "$SCRIPTS/rewrite_text.py" INPUT \
  --backend print-prompt --strength tsapa --generations 5 --population 12

# Live local endpoint
WATERMARKS_REWRITE_BACKEND=openai-compatible \
WATERMARKS_REWRITE_BASE_URL=http://127.0.0.1:8080 \
WATERMARKS_REWRITE_MODEL=my-model \
  python3 "$SCRIPTS/rewrite_text.py" INPUT \
    --strength tsapa --generations 5 --population 12
```

Prefer a rewrite model different from the suspected origin. Preserve facts, numbers, names, and technical identifiers. Report style/precision degradation risk. Code files should use formatter + Layer A unless the user explicitly approves semantic rewriting.

`clean_file.py --tsapa` refuses to run without a live backend; it never writes a prompt into the user’s output or silently falls back.

### 6. Character perturbation — explicit opt-in only

```bash
python3 "$SCRIPTS/perturb_text.py" INPUT --mode zero-width --strength 0.1 --seed 42
```

This deliberately adds anti-watermark noise after Layer A. `zero-width` and `space-swap` are Layer-A reversible. `confusable` and `case` are not and can harm search/copy/accessibility. Do not present this as hygiene.

### 7. Optional SynthID score

When `REVERSE_SYNTHID_DIR` is configured, image inspect/clean can invoke the external scorer. It is not bundled, not an official Google detector, and does not remove pixel watermarks.

```bash
"$SCRIPTS/setup_synthid.sh"
REVERSE_SYNTHID_DIR=~/reverse-SynthID \
  ~/reverse-SynthID/.venv/bin/python "$SCRIPTS/score_synthid.py" IMAGE
```

### 8. Report

Always include:

- verifiable Layer A / metadata actions and counts;
- best-effort Layer B / visible actions, with no detector guarantee;
- residual warnings (soft binding, remote manifest, pixel/audio/video signal);
- output paths and whether a `.bak` was created;
- ethics: owned content, no fraudulent authorship/compliance claim.

## Hard limits

- Layer A does not remove token-distribution watermarks.
- Layer B cannot be gold-verified without vendor detectors/keys.
- Visible inpainting can damage texture or miss regions.
- Soft-binding removal, pixel/audio/video watermark removal, training backdoors, and secret-key detector emulation are out of scope.
- Hard-bound C2PA stripping does not clear in-content binding or remote-manifest recovery.
