# Removal matrix

| Target | Method | Script / action | Side effects | Verifiable today? |
| --- | --- | --- | --- | --- |
| Hidden Unicode / exotic spaces / orphan controls | Context-aware strip / normalize | `inspect_text.py`, `clean_text.py`, `clean_file.py` | Default preserves semantic controls; aggressive mode can alter rendering | Yes (codepoint report) |
| Statistical text watermark | Paraphrase / back-translate / structural / TSAPA-style evolution | `rewrite_text.py` | Meaning, style, precision, and model-call cost | No without vendor key/detector |
| Character-level text perturbation | Seeded ZWSP/space/confusable/case changes | `perturb_text.py` | Deliberately reduces text hygiene; some modes irreversible | Transform yes; detector effect no |
| Visible image mark | Explicit/external mask → hole fill → dilate → inpaint → restore | `morphomod.py` | Texture artifacts; mask miss/overreach | Region action yes; fidelity best-effort |
| C2PA / metadata on PNG/JPEG | Drop `caBX`, APP/JUMBF, EXIF/XMP/text metadata | `clean_image.py` | Loses selected/all metadata | Yes (re-inspect) |
| HEIC/HEIF/AVIF provenance | Offset-preserving ISO-BMFF neutralization | `clean_image.py` | Selected/all Exif/XMP item content may be cleared | Yes (re-inspect) |
| SVG metadata / XMP | Drop `<metadata>`, xmpmeta | `clean_file.py` | Loses SVG metadata | Yes |
| PDF XMP / info | exiftool → full-document pypdf clone → unchanged copy | `clean_file.py` | Loses metadata when structural cleaner succeeds; otherwise explicit residual | Partial to yes |
| DOCX props / customXml | Scrub known docProps; inspect/preserve customXml | `clean_file.py` | Custom XML residual may remain to avoid business-data loss | Partial |
| ODT `meta:generator` | Scrub `meta.xml` | `clean_file.py` | Loses generator tag | Yes |
| HTML generator / JSON-LD provenance | Strip tags/attributes | `clean_file.py` | Loses matching metadata | Yes |
| Markdown AI frontmatter keys | Drop keys + Layer A body | `clean_file.py` | Loses matching YAML keys | Yes |
| C2PA soft binding / remote manifest | Detect and warn | `inspect_soft_binding.py` | No mutation | Detection only; removal out of scope |
| Pixel/audio/video watermarks | Optional external image score only | `score_synthid.py` | External-license/version limits | No universal verification |
| Data-driven model backdoors | — | Out of scope | — | — |

## Default pipeline

1. Inspect with unified or channel-specific inspector.
2. Run deterministic Layer A / metadata cleaning.
3. For visible marks, require a mask/box/localizer; never guess.
4. Offer Layer B for prose; prefer a non-origin model.
5. Run Layer A again after rewrite.
6. Character perturbation is separate, explicit, and last.
7. Report verifiable actions, best-effort actions, output paths, and residual risk.

## Code vs prose

- **Prose / Markdown / HTML body:** Layer A; offer Layer B.
- **Code:** Layer A + formatter; semantic rewrite only with explicit approval.

## Layer B strengths

| Strength | When |
| --- | --- |
| `paraphrase` | Default, least orchestration |
| `backtranslate` | Stronger token reshuffle |
| `structural` | High drift |
| `tsapa` | Multi-objective evolutionary search; highest cost; requires live backend to execute |
