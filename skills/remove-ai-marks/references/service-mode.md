# Service mode (thin client)

The default `SKILL.md` workflow runs the local CLIs under
`skills/remove-ai-marks/scripts/`. As an alternative, the whole cleaning
pipeline is also exposed over HTTP by `server.py` (console script `wm-serve`),
so an agent host needs no Python, venv, or system tools. This document
describes that thin-client pattern.

## Service access

Base URL comes from `WATERMARKS_SERVICE_URL`, default `http://127.0.0.1:8765`:

```bash
WM="${WATERMARKS_SERVICE_URL:-http://127.0.0.1:8765}"
```

Start it either with Docker/compose (`docker compose up -d`, or the published
GHCR image `ghcr.io/pythoughts-labs/watermark-remover`) or locally
(`make serve` / `wm-serve`). **Always check it first**, and stop with a clear
message if it is unreachable:

```bash
curl -sf "$WM/health"
# {"ok": true, "version": "..."}
```

If `WATERMARKS_SERVER_API_KEY` is set on the service, every request needs
`-H "Authorization: Bearer $WATERMARKS_SERVICE_API_KEY"`.

## Capabilities

```bash
curl -s "$WM/capabilities"
```

Reports which optional tools are available server-side (`c2patool`, `exiftool`,
`qpdf`), scorers present (`scorers.stylometry`, `scorers.synthid`,
`scorers.synthid_http`), vendor text-watermark detectors
(`text_detectors.gemini-synthid-text`, `text_detectors.markllm`,
`text_detectors.claude-text`), and which heavy backends are configured
(`pixel_backends.ctrlregen`, `pixel_backends.diffusion`, `harnesses.markllm`).
**Drive your advice from this**: only recommend pixel removal / SynthID
scoring / vendor detection when the service reports the backend present.

## HTTP API (curl)

Payloads are JSON with the file as **base64**. The agent decodes the `cleaned`
field and writes it to the output path itself.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"ok": true, "version": ...}` |
| GET | `/capabilities` | — | optional tools / backends present |
| GET | `/openapi.json` | — | dynamically generated OpenAPI 3.0.3 spec |
| POST | `/inspect` | `{"file": "<base64>", "name": "notes.md"}` | `{"ok", "kind", "suspicious", "report"}` |
| POST | `/detect` | `{"file": "<base64>", "name": "notes.txt"}` | `{"ok", "kind", "detections": [...]}` |
| POST | `/clean` | `{"file": "<base64>", "name": "notes.md", "options": {...}}` | `{"ok", "kind", "cleaned": "<base64>", "report"}` |

`/clean` and `/inspect` route by the uploaded `name` extension plus the bytes;
unrecognized formats answer `kind: "unknown"` (`/inspect`) or 400 (`/clean`).
When writing a temp file for pasted text, keep a known extension (`.txt` /
`.md`) in the `name` you send.

The machine-readable contract lives at `$WM/openapi.json` — plug it into any
OpenAPI tooling instead of hand-rolling clients.

`options` accepted by `/clean`: `nfkc`, `aggressive_homoglyphs` (text),
`keep_non_ai_metadata`, `strip_all_metadata`, `remove_pixel` (`ctrlregen` |
`diffusion`) (images), `also_layer_a_text` (containers), `detect_before` /
`detect_after` (text and images), plus our extras `remove_synthid` and
`wmct_marker` (images, PNG output).

**Inspect first** (decide, don't guess):

```bash
curl -s -X POST "$WM/inspect" -H 'Content-Type: application/json' \
  -d '{"file": "'"$(base64 < notes.md | tr -d '\n')"'", "name": "notes.md"}'
```

**Clean** (text / image / container are auto-detected by name + bytes):

```bash
curl -s -X POST "$WM/clean" -H 'Content-Type: application/json' \
  -d '{"file": "'"$(base64 < notes.md | tr -d '\n')"'", "name": "notes.md"}'
```

Decode the returned `cleaned` base64 into the output file (`*.cleaned.*` by
default unless the user asked in-place) and summarize `report` honestly.

(On Windows agents, build base64 with
`[Convert]::ToBase64String([IO.File]::ReadAllBytes("notes.md"))`.)

## Watermark detection before/after (when configured)

When `/capabilities` reports a vendor detector (`text_detectors.gemini-synthid-text`)
or an image scorer (`scorers.synthid_http` / `scorers.synthid`), measure the
result by detecting before and after cleaning — either `POST /detect` or fold
it into the clean with `{"options": {"detect_before": true, "detect_after": true}}`
(returns `text_detectors.before/after` for text or `synthid_before/after` for
images). Vendor detection sends text to the configured provider (Gemini) —
only use it with user consent, and report the vendor's verdict honestly
(Gemini = Google's official SynthID-text detector; MarkLLM is same-config-only
research; Claude's detector is not public yet).

## Aggregate audits (directories / websites)

The service image also ships the audit CLIs. Run them as one-shot containers:

```bash
docker run --rm -v "$(pwd)/src:/data:ro" watermark-remover \
  wm-audit-dir /data --json
docker run --rm watermark-remover wm-audit-site --sitemap https://example.com/sitemap.xml --json
```

Or against a local checkout: `python3 skills/remove-ai-marks/scripts/audit_dir.py DIR --json`.

Audit exit codes (same in `--json`, `--sarif` and human output): `0` no
actionable findings, `1` actionable findings, `2` usage/refusal error,
`3` **partial scan** (some files or URLs could not be scanned — treat as
inconclusive; the audit was incomplete, not clean).

## Limitations

- Layer A does **not** remove token-sampling watermarks; Layer B is best-effort.
- Pixel removal (`remove_pixel`) and the MarkLLM/MarkDiffusion harnesses are
  same-config-only research tools, not vendor-detector oracles.
- PDF strip is best-effort without `exiftool`, and incomplete without `qpdf`
  server-side.
- The reverse-SynthID scorer is external, best-effort, and under a
  non-commercial Research License; it is not an official Google detector.
- **C2PA soft binding** (remote-manifest re-linking) is out of scope —
  stripping hard-bound C2PA does not clear it. Data-driven / backdoor model
  marks are out of scope.

## Service not reachable?

If `$WM/health` fails: tell the user the service is down and how to start it
(`docker compose up -d`, `make serve`, or the published GHCR image). Unlike
this repo's primary skill workflow, the thin-client pattern has no local
fallback — the cleaning machinery lives in the service.
