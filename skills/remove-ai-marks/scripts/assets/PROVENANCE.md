# Silhouette asset provenance

Every PNG in this directory is **renderer-synthetic**: produced by the in-tree
generator scripts from installed system fonts or primitive geometry, carrying
zero pixels from any captured image. The renderers are the reproducible source
of truth — the in-tree determinism test (`tests/test_silhouette_assets.py`)
regenerates every asset into a temp directory and byte-compares it against the
committed file. The system fonts render the glyphs but are **never shipped**.

| Asset | Generator | Upstream source | License | Engines (wave) | Notes |
| --- | --- | --- | --- | --- | --- |
| generic_ai_label_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | generic_ai_label (1) | Brand-less TC260 fallback; STHeiti Medium. |
| jimeng_label_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | jimeng, jimeng_pill (1) | Text-form label; STHeiti Medium; upstream ships it byte-identical to generic_ai_label's render. |
| doubao_outline_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | doubao (1) | Synthetic outline; Hiragino Sans GB. Upstream's primary doubao_alpha.png is a capture solve and is excluded (see below). |
| qwen_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | qwen (1) | STHeiti Medium; qwen_symbol_alpha.png is a capture solve and is excluded. |
| kling_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | kling (1) | CJK one-line stamp; STHeiti Medium. |
| kling_latin_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | kling (1) | Latin variant; Helvetica Neue. |
| baidu_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | baidu (1) | STHeiti Medium. |
| yuanbao_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | yuanbao (1) | Two-line stacked slanted stamp; Hiragino Sans GB + cv2 dilate/shear. Regenerated with this tree's OpenCV 5: pixel-max diff 4/255 vs the upstream OpenCV-4 commit on ~3% of edge pixels (deterministic in-tree; the determinism test pins this tree's rendering). |
| yuanbao_compact_alpha.png | render_vendor_silhouettes.py (`centered_lines`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | yuanbao (1) | Compact centered variant; pixel-identical to upstream's commit. |
| yuanbao_compact_gray_alpha.png | render_vendor_silhouettes.py (`centered_lines`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | yuanbao (1) | Gray-frontend twin of the compact asset (identical pixels, distinct file). |
| samsung_ko_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | samsung (2) | Korean label; Apple SD Gothic Neo; pixel-identical to upstream's commit. Upstream's registered Italian samsung_alpha.png is a capture solve and is excluded. |
| samsung_en_alpha.png | render_vendor_silhouettes.py (`draw_samsung_en`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | samsung (2) | English label + four-point sparkle; Arial; pixel-identical to upstream's candidate commit. |
| liblib_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | liblib (2) | Arial; liblib_pill reuses jimeng_pill.png. |
| microsoft_alpha.png | render_vendor_silhouettes.py (`draw_msbadge`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | microsoft (2) | Drawn pill with dark-text holes + sparkle cutout; Arial Bold. |
| microsoft_text_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | microsoft (2) | Arial. |
| runninghub_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | runninghub (2) | Faint mid-gray top-left mark; STHeiti Medium. |
| dola_alpha.png | render_vendor_silhouettes.py (`render`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | dola (2) | Arial Bold; upstream's video-form Hershey templates are out of scope here (no video engine ships). |
| workbuddy_alpha.png | render_vendor_silhouettes.py (`centered_lines`) | scripts/render_overlay_variants.py | Apache-2.0 (ported portions) | workbuddy (2) | Two-line block; Hiragino Sans GB. |
| wan_alpha.png | render_vendor_silhouettes.py (`draw_wan`) | scripts/render_vendor_silhouettes.py | Apache-2.0 (ported portions) | wan (3, provisional) | Hexagon-logo approximation + Avenir Next wordmark. |
| openart_alpha.png | render_vendor_silhouettes.py (`draw_openart`) | scripts/build_openart_alpha.py | Apache-2.0 (ported portions) | openart (3, provisional) | Procedural reconstruction of the public OpenArt mark (bowtie icon + Hershey wordmark, peak alpha 173). Upstream pinned OpenCV 4.x for Hershey glyph parity; this tree regenerates with the pixel extra's OpenCV 5 and ships that rendering. |
| jimeng_pill.png | render_pill_silhouette.py (`render_silhouette`) | scripts/render_pill_silhouette.py | Apache-2.0 (ported portions) | jimeng_pill, liblib_pill (1) | 320x160 rounded-pill outline + AI生成 (STHeiti Medium); top-left pill class. |

## Watermarked synthetic-carrier fixtures (`watermarked_fixtures/`)

Four 512x512 PNGs produced by the DWT-DCT oracle (the `invisible-watermark`
library, PyPI 0.2.0 — a **dev-only fixture-generation tool** that is never a
project dependency, enforced by `tests/test_detect_wiring.py`) embedding known
fixed bit patterns into a deterministic synthetic carrier: a seeded
gradient-plus-noise field (`numpy.random.default_rng(20211012)`, gradient 90x +
140y + uniform 0–60 noise, replicated to BGR), so the pixels are
renderer-synthetic with zero captured content. These files are pinned by
**decode-to-expected-bits**, NOT by byte regeneration: the determinism test
decodes each committed PNG and requires the exact pattern below, so a
re-encode with a different PNG encoder would not break the pin — a changed
decode would.

- `plain.png` — the un-watermarked control; the detector must report no
  finding. SHA256 `7b87acbc6d568b51489c0e3ef98a25c6ddcf9fb09a134ddc2f0b537672839e51`.
- `sdxl_48bit.png` — 48-bit payload `0b101100111110110010010000011110111011000110011110`
  (diffusers `StableDiffusionXLWatermarker` `WATERMARK_MESSAGE`). SHA256
  `f5ddabae97bb48b4a5b0010547548e2405743b0c92f0d4ec21e3fe8e5d524e03`.
- `flux2_48bit.png` — 48-bit payload `0b001010101111111010000111100111001111010100101110`
  (Black Forest Labs FLUX.2 `watermark.py`). SHA256
  `4a981314c6e7b577a8cfa24f4e867fef622b3c7af6d68c542c55bf5c24bc74e9`.
- `sd_v1_136bit.png` — the 136-bit pattern of the ASCII bytes
  `StableDiffusionV1` (the invisible-watermark `"StableDiffusionV1"` default).
  SHA256 `062f6fd18b365d28ed9cf6f526335568cdc2c327ababc2953a7d48267128e713`.

Each embed was verified at generation time three ways (oracle pre-PNG decode,
in-tree `dwt_dct` parity on the float array, post-PNG re-decode through the
in-tree decoder). Oracle version at generation: `invisible-watermark 0.2.0`
with OpenCV 5.0.0 / PyWavelets 1.10.0 / numpy 2.x, Python 3.13. Regeneration
recipe: seeded carrier as above, `WatermarkEncoder.set_by_bits` (48-bit
schemes) or `set_by_bytes(b"StableDiffusionV1")`, `encode(..., method="dwtDct")`,
PNG compression 9. The commit is the provenance record; the SHA256s above are
integrity pins, not regeneration targets.

## Upstream attribution

Upstream source: `wiltodelta/remove-ai-watermarks` (Apache-2.0), ported from the
snapshot `remove-ai-watermarks-main` v0.44.0 (2026-10) — the upstream tree was
obtained without git metadata, so the snapshot version (its `pyproject.toml`
version field) is the revision reference; per-file provenance headers name the
exact upstream script per asset in the table above.

## Learned fill-model pins

The ONNX fill backends download their weights from Hugging Face on first use,
and the download pins an explicit commit (never a moving ref), recorded in
`region_eraser.py` (`_LAMA_REVISION` / `_MIGAN_REVISION`) and here:

- `Carve/LaMa-ONNX` / `lama_fp32.onnx` — pinned revision
  `c3c0c9e468934d62e79c329e35d82dd09ff8c444`, Apache-2.0.
- `andraniksargsyan/migan` / `migan.onnx` — pinned revision
  `406830d0fa60666da0071c342ad2fbc8f30c5c64`, MIT.

The upstream reference (v0.44.0 snapshot) does not pin revisions; these are the
repositories' current commit SHAs at the time of pinning (read from the
Hugging Face models API), pinned so an upstream force-push cannot silently
change the model this tree runs. To re-pin, update the constants in
`region_eraser.py`, this list, and the pinning test.

## Source-classifier weights (pinned download, never bundled)

The abstaining source classifiers (`source_classify.py`, `source_classify_v2.py`)
load their frozen weights from Hugging Face **at first use only**; the npz
files are never bundled in this tree (the forbidden-fragment guard stands), and
without them the classifiers abstain to `unknown` — absence is never a clean
verdict. Every download is pinned to an immutable revision SHA and verified
against a SHA-256 before the loader accepts the file; the pins live as
constants in the modules and are asserted by `tests/test_source_classify.py`
and `tests/test_source_classify_v2.py` (which also monkeypatch the hub client
to prove the pinned call happens — no test touches the network).

Both models are trained upstream on private corpora that never enter this
checkout (upstream `docs/release-and-distribution.md` records the publish
workflows: hash-verified derivative artifacts only, published behind manual
model-freeze releases). Neither classifier detects or decodes SynthID; their
claims are evidence-only, and a label is a routing hint, never a verdict.
Upstream publishes each artifact with a model card recording evaluation and
robustness limits; the upstream reference snapshot carries them at
`docs/source-classify-hf/` and `docs/image-source-hf/`. To re-pin, update the
constants in the two modules, this list, and the pinning tests.

- `source-pipeline-mlp.npz` (E2: fused 124-d + spectral features →
  source/export-pipeline MLP, 892→128→64→3, calibrated per-provider margin
  thresholds) — [`wiltodelta/raiw-source-classify`](https://huggingface.co/wiltodelta/raiw-source-classify)
  @ revision `4b8b408d03dac20231d46217977fd164414b6b9e`, SHA-256
  `6430b0738869ff07b6845b7c6149c2a83bc3f3785878082886c241c6f7ca0c4a`,
  offline override `RAIW_SOURCE_CLASSIFY_WEIGHTS`.
- `openai-google-source-v1.npz` (E3: phase-free spectral template filters +
  spatial residual templates → OpenAI/Google/unknown with conflict
  abstention) —
  [`wiltodelta/openai-google-image-source-classifier`](https://huggingface.co/wiltodelta/openai-google-image-source-classifier)
  @ revision `4563fb97e5572030ad8d6acbb6f20eaab7fc3009`, SHA-256
  `0e94e565c181416e4f63a2bfc06cd5179f4201befda7481ac510f6a168c612f4`,
  offline override `RAIW_IMAGE_SOURCE_WEIGHTS`.

## Photo-classifier weights (pinned download, never bundled)

The metadata-free photo classifier (`classify.py` and its CLIP-L-ft embedder
module) downloads its frozen weights from Hugging Face **at first use only**;
neither the model files nor the receipt-gate npz are bundled in this tree (the
forbidden-fragment guard stands). Without them the explicit classify verb
reports `unavailable` — labeled, never a silent pass and never a clean verdict
(absence of the classifier is unknown, never clean). The download is pinned to
an immutable revision SHA via `snapshot_download`; the repo/revision/file pins
are constants in `classify.py`, asserted by `tests/test_photo_classify.py`
(which also monkeypatches the hub client to prove the pinned call happens — no
test touches the network). The receipt-document gate ships inside the SAME
snapshot (`receipt-gate.npz`, threshold inside the artifact): head and
threshold always come from one file, and a snapshot missing the gate is a
partial install that refuses rather than silently substituting a code-side
threshold. The gate's training corpus (CORD-v2, CC BY 4.0 upstream) never
enters this checkout. To re-pin, update the constants in `classify.py`, this
list, and the pinning tests.

- [`wiltodelta/raiw-photo-classify`](https://huggingface.co/wiltodelta/raiw-photo-classify)
  @ revision `de668f0d8a53412923478cc21382a56890748310` — the torch snapshot:
  `clip-l-ft.pt` (frozen CLIP-L-ft, Model 1 embedder), `detector.pt` (MLP
  detector head), `probe-weights-clip-l-ft.npz` (ridge head + threshold),
  `provider.pt` (124-d provider heads), `receipt-gate.npz` (document-domain
  veto), `operating-point.json`, and the vision-only FP32 export
  `clip-l-ft-vision-fp32.onnx` for the `[classify-onnx]` backend. Offline
  override `RAIW_CLASSIFY_WEIGHTS`.

## Excluded capture-derived assets (must never enter this tree)

`tests/test_silhouette_assets.py::test_forbidden_capture_derived_assets_absent_from_tree`
greps the whole tree and fails if any of these names appears:

- `gemini_bg_*.png` — gemini sparkle-on-black capture crops (the gemini engine
  is deferred with the slot reporting pending-provenance);
- `doubao_alpha.png`, `jimeng_alpha.png`, `samsung_alpha.png` (Italian),
  `qwen_symbol_alpha.png` — upstream `scripts/visible_alpha_solve.py` outputs,
  recovered from controlled captures of real watermark carriers;
- `receipt-gate-*.npz` — captured receipt data;
- the `visible_alpha_solve` generator itself.

Upstream's unregistered candidate silhouettes (catlogo, hailuo, notebooklm,
mindvideo, higgsfield, capcut, zsky, chromastudio, digenai, gendo, jianying,
xinghui, qingyan, gemini_text) are also not ported: no engine in this repo
registers them. The video-form Hershey templates are synthetic but are not
shipped (no video engine); they can be regenerated from the upstream recipe if
a video ticket ever claims one.
