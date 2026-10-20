# Images

Visible marks, image degradation, SynthID-class scoring, and the heavy external backends.

## Visible image marks

[`morphomod.py`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/scripts/morphomod.py) never guesses a watermark region. Supply one of:

- `--mask mask.pgm|mask.png` (white means remove)
- `--box X,Y,W,H`
- `--detect-command '...'` with `{input}`, `{mask}`, `{prompt}` placeholders

```bash
# Stdlib texture backend (default in clean_file): nearby-patch search
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --box 900,920,120,60 --dilation 3 --backend texture

# Uniform-background fallback
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --box 900,920,120,60 --dilation 3 --backend simple

# Production adapter: external detector and inpainter
python3 "$SCRIPTS/morphomod.py" input.png -o output.png \
  --detect-command 'detector --input "{input}" --output "{mask}"' \
  --backend external \
  --command 'inpainter --image "{input}" --mask "{mask}" --output "{output}"'

# Combined visible pass, then metadata clean
python3 "$SCRIPTS/clean_file.py" input.png -o output.png \
  --visible-mask mask.pgm --dilate 3 --visible-backend texture
```

The stdlib dilation is non-cascading and runs in O(width × height). PNG decoding supports non-interlaced 8-bit grayscale, RGB, and RGBA. The default `texture` backend searches nearby patches by boundary error and fully replaces the refined mask; optional feathering is available at the module interface. `simple` is retained for uniform backgrounds only. Neither is marketed as LaMa quality. JPEG visible cleaning requires an external backend, and that backend owns JPEG compositing.

Paper-reported MorphoMod improvements are not this implementation's measured results. The report includes initial and refined mask pixels and requires manual fidelity review.

### Automatic mark detection (opt-in)

`--mark auto` (or a named key such as `--mark doubao`) replaces the manual mask
with a registry of known vendor marks: each registered mark is localized,
filled through one shared inpaint backend, and re-checked by the same detector.

```bash
wm input.png -o output.png --mark auto
wm input.png -o output.png --mark auto --sensitivity strict
wm input.png -o output.png --mark jimeng --visible-backend lama
```

- Opt-in only: without `--mark`, the explicit-mask behavior is unchanged.
- Detection and removal are best-effort. Every removal is re-validated by the
  same detector that found the mark — a self-consistency check, not an
  independent oracle — and the report states `cleaned`, `partial`,
  `unvalidated`, `error`, or `no_mark` per pass. Nothing certifies a mark is
  gone; inspect the output.
- `--sensitivity auto` (default) relaxes a mark's detection gate only when the
  file's own metadata names that product (TC260 producer codes, Samsung
  genAIType metadata, or a declared platform token in generator fields);
  `strict` never relaxes.
- Fills resolve through the backend stack. The `migan` or `lama` extra
  (`pip install 'watermark-remover[migan]'` — onnxruntime + huggingface-hub)
  enables the learned ONNX models; their weights are pinned to published
  commits and downloaded on first use. cv2 is the no-extra floor, and
  `--visible-backend` selects a specific tier. `--visible-backend external` is
  not applicable to registry marks (the report warns and names the backend that
  actually ran).
- The mark pass runs whole-image (auto-tiling applies only to explicit-mask
  plans, and the report says when tiling was skipped for this reason), and the
  registered engines own their dilation: `--dilate` is not applied there, the
  report states `engine-controlled (0 user-applied)`.
- `--humanize` is a separate, default-off post-stage (film grain, chromatic
  aberration, unsharp); it never runs unless asked.

### Image degradation (opt-in Layer V extension)

`--degrade` and `--morpho` apply frequency-domain or morphological degradation to
PNG images after metadata cleaning:

```bash
# Frequency-domain strategies: freq-dct, blur, median, jpeg, rotate, two-stage
python3 "$SCRIPTS/clean_file.py" photo.png -o out.png \
  --degrade freq-dct --degrade-strength 0.5 --degrade-seed 7

# Morphological strategies: grid, diagonal, noise, quantize
python3 "$SCRIPTS/clean_file.py" photo.png -o out.png --morpho grid --degrade-seed 7
```

Semantics:

- Degradation is an intentional transform, not residual risk: a degraded image
  exits 0 and the report records the strategy under `degrade` instead of
  printing a residual warning.
- `--degrade-strength` (0–1) only configures `freq-dct`; the other strategies
  use conservative built-in defaults. `--degrade-seed` makes seeded strategies
  reproducible.
- DCT-based strategies (`freq-dct`, `jpeg`, `two-stage`) refuse images above
  their pixel caps (65,536 / 262,144) instead of running for hours in pure
  Python.
- The RGBA alpha channel is never modified, output files keep the destination's
  existing permissions, and degradation requires PNG output.

## Optional SynthID pixel scoring

The external [`aloshdenny/reverse-SynthID`](https://github.com/aloshdenny/reverse-SynthID) checkout is not bundled and stays under its upstream non-commercial research license.

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

This is scoring only. That repository's carrier model and success figures are maintainer-reported reverse-engineering claims, not public Google architecture and not an independent guarantee.

Best-effort pixel-domain removal is opt-in on image cleans: `--remove-synthid` (seed-independent DCT mid-band suppression, PNG only) and the CtrlRegen or diffusion regeneration backends (`remove_pixel` in the HTTP API and the library; not a `wm` flag). All are labeled best-effort; none certifies a vendor-detector miss.

### Heavy backends (external checkouts, never bundled)

| Backend | Purpose | License posture |
| --- | --- | --- |
| CtrlRegen (`noai-watermark`) | Regeneration removal (`remove_pixel: ctrlregen`) | No LICENSE upstream — local-only image, never published |
| reverse-SynthID | SynthID-class pixel scoring (`score_synthid.py` / sidecar) | Non-commercial Research License — local-only image, never published |
| MarkLLM | Text scheme verification harness (`detect_text_watermark.py`) | Apache-2.0 — publishable |
| MarkDiffusion | Image scheme harness + DiffusionPurification (`markdiffusion_harness.py`) | Apache-2.0 — publishable |

Bootstrap any of them with `setup_ctrlregen.sh`, `setup_synthid.sh`, `setup_markllm.sh`, `setup_markdiffusion.sh` (or the `make bootstrap-*` targets). CtrlRegen/SynthID images build locally only; MarkLLM/MarkDiffusion images publish to GHCR on `v*` tags.
