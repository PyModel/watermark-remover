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
