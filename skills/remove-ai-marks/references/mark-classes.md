# Mark classes

## 1. Edit-based text (Unicode / rules)

Invisible or near-invisible characters, exotic spaces, bidi controls, tag characters, and rule-based substitutions.

| Inspect kind | Examples |
| --- | --- |
| `zwj_family` | ZWSP, ZWNJ, ZWJ, WJ, BOM |
| `bidi` | LRE/RLO/LRI/… |
| `tag_chars` | U+E0001–U+E007F |
| `variation_selector` | VS1–VS256 |
| `space` | NBSP, em space, ideographic space |
| `confusable` | Cyrillic/fullwidth Latin (aggressive) |

**Cleaning:** Layer A is deterministic and reports every action. Contextual ZWJ/ZWNJ, variation selectors, invisible math operators, and balanced bidi controls are preserved by default because removing them can change rendering or meaning. `--strip-semantic-format` is explicitly aggressive.

## 2. Generative / statistical text (token sampling)

Next-token sampling is biased toward a pseudo-random set/score (Kirchenbauer, SynthID-Text/tournament sampling, etc.). The signal lives in wording, not metadata.

**Attack:** Layer B rewrite—paraphrase, back-translation, structural regeneration, or TSAPA-style evolutionary optimization. Best-effort; no gold certification without the vendor detector/key.

Character perturbation is a separate opt-in attack. It is not Unicode hygiene.

## 3. Data-driven / backdoor

A model is trained/fine-tuned so trigger prompts produce marked behavior. **Out of scope** (model-side).

## 4. File provenance metadata (C2PA / EXIF / XMP / props)

C2PA Manifest Stores use JUMBF/BMFF structures with claims, assertions, and COSE signatures. Assertions may contain JSON/JSON-LD, CBOR, or embedded content.

| Layer | Mechanism | Survives metadata strip? | This project |
| --- | --- | --- | --- |
| Hard-bound C2PA | Signed manifest in the file | No | Clean + re-inspect |
| Soft binding | In-content signal that can resolve a remote manifest | Yes | Detect/warn only; removal out of scope |
| Standalone SynthID-class | Pixel/waveform/token watermark | Media: often; text: scheme-dependent | Media score only; text Layer B |

| Format | Support |
| --- | --- |
| PNG | `caBX`, EXIF/text/XMP chunks |
| JPEG | APP11/JUMBF and selected/all APP metadata |
| HEIC/HEIF/AVIF | ISO-BMFF JUMBF + Exif/XMP item extents, offsets preserved |
| SVG | Metadata/XMP blocks |
| PDF | exiftool → full-document pypdf clone → unchanged-copy warning |
| DOCX/ODT | Known properties cleaned; DOCX customXml preserved and reported |
| HTML/Markdown | Provenance metadata + Layer A text |

Stripping hard-bound metadata does not clear soft binding, fingerprints, or pixel watermarks.

## 5. Visible image marks

Logos, text stamps, or overlays live in rendered pixels.

**Pipeline:** explicit/external localization → mask hole fill → morphological dilation → inpaint → restore original outside the mask. `morphomod.py` requires a mask, box, or detector adapter and never guesses a location. Region removal is observable; visual fidelity is best-effort.

## 6. Pixel-domain image/audio/video watermarks

Invisible media marks (including SynthID-class schemes) live in the signal. This project provides an optional external image score, not pixel/audio/video removal or a vendor-failure guarantee.
