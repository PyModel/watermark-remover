# wm-tui bridge protocol

`wm-tui` is two processes:

- **The frontend** is TypeScript on Bun with OpenTUI and Solid, in this
  directory. It owns the terminal.
- **The bridge** is Python, in `scripts/tui_bridge.py`. It owns every decision
  about a clean.

The frontend spawns the bridge and talks to it in JSON Lines over the bridge's
stdin and stdout. The bridge is the only place a `CleanRequest` is built.
Everything else in the frontend is presentation.

```
wm-tui (python, scripts/tui.py)  ── validates argv, finds bun, execs ──▶  bun src/index.tsx
                                                                            │ spawn
                                                                            ▼
                                        $WM_TUI_PYTHON $WM_TUI_BRIDGE  (scripts/tui_bridge.py)
```

## Environment the launcher sets

| var | meaning |
| --- | --- |
| `WM_TUI_PYTHON` | absolute path of the interpreter `wm-tui` itself runs on; the frontend spawns the bridge with it, never `python` from PATH |
| `WM_TUI_BRIDGE` | absolute path of `tui_bridge.py` |
| `WM_TUI_ARGV` | JSON list: the raw `wm-tui` arguments, already validated by the launcher |
| `WM_TUI_LOG` | file the frontend appends the bridge's stderr to |
| `WM_TUI_CWD` | the user's working directory; the launcher runs Bun from the frontend directory, so the bridge `chdir`s back here at start and relative paths mean what the user typed |

## Framing

One JSON object per line, UTF-8, `\n` terminated, in both directions.

- request, frontend → bridge: `{"id": 7, "method": "clean", "params": {...}}`
- response, bridge → frontend: `{"id": 7, "result": {...}}`, or
  `{"id": 7, "error": {"code": "needs_confirm", "message": "...", "data": {...}}}`
- event, bridge → frontend, zero or more before the response to the same id:
  `{"id": 7, "event": "file_done", "data": {...}}`
- unsolicited event, bridge → frontend, sent once at start:
  `{"event": "ready", "data": {"version": "0.5.0"}}`

Error codes are `bad_request`, `invalid_options`, `needs_confirm`, `busy`,
`cancelled` and `internal`. `invalid_options` covers anything argparse or
`CleanRequest` refuses; its `message` is argparse's own text.

**stdout discipline.** At startup the bridge `os.dup`s fd 1 for protocol
frames only, then points fd 1 and `sys.stdout` at stderr. The pipeline prints,
and a stray `print` must never corrupt a frame. The frontend writes the
bridge's stderr to `WM_TUI_LOG` and never inherits it.

Requests are handled on worker threads, so `detect_endpoints` or `cancel`
stays responsive during a clean. Only one `clean` or `inspect` runs at a time.
A second one returns `busy`.

## State: the one input shape

Every method that plans work takes the same `state` object. The bridge composes
the `CleanRequest` from it through `clean_file`'s own argparse parser, so any
CLI flag is valid here and the command preview is exact.

```json
{
  "paths": ["drafts/", "notes.md"],
  "preset": "hidden",
  "flags": ["--recursive", "--glob", "*.md"],
  "endpoint": {"backend": "ollama", "base_url": "http://127.0.0.1:11434", "model": "qwen3:14b"}
}
```

`endpoint` has the same three keys everywhere: in `state`, in `hello.settings`,
and in `save_settings`. Reasoning effort is set with
`--rewrite-reasoning-effort`, the same way as any other flag, so there is
only one way to set it.

The bridge composes the request in four steps:

1. It parses `[*preset.flags, *flags, "--", *paths]` with
   `clean_file._build_parser()`, where a later flag wins. The `--` keeps a file
   named `-x.md` a path.
2. It builds the request with `CleanRequest.from_args`.
3. It applies the `endpoint` fields, but only when the request has a
   `rewrite_strength`. This keeps a preview command free of endpoint flags
   that do nothing. `rewrite_allow_remote` is true only for a `clean` whose
   `confirmed` list includes `"remote"`; a typed `--rewrite-allow-remote` and
   `WATERMARKS_REWRITE_ALLOW_REMOTE` are both overridden. It is never
   persisted. A confirmed remote clean adds `--rewrite-allow-remote` to its
   `command` and history entry.
4. It forces `json=True` and `quiet=True`.

The bridge never handles the API key. The pipeline reads
`WATERMARKS_REWRITE_API_KEY` at run time, the same way it does for the CLI.

The key never appears in any frame, command string, history entry or settings
file. As a last line of defence the frame writer also redacts the key's value
(when it is at least 8 characters) from every `result`, `error` and `data`
payload. The composition function has unit tests.

## Methods

### `hello` `{}`

Returns:

```json
{"version": "0.5.0",
 "presets": [{"key": "hidden", "label": "Hidden marks", "description": "...",
              "layer": "A", "result_class": "Verifiable", "flags": [],
              "requires_endpoint": false}],
 "settings": {"preset": "hidden", "endpoint": {"backend": null, "base_url": null, "model": null}},
 "settings_path": "/Users/x/.config/watermark-remover/tui.json",
 "onboard": true,
 "initial": {"paths": ["tests/fixtures"], "flags": ["--recursive"]},
 "backends": ["ollama", "openai-compatible"]}
```

`onboard` is true exactly when no settings file exists. `--setup` forces it
and `--no-setup` suppresses it.

### `plan` `{state}`

This is the cheap preview. The frontend calls it, debounced, on every change.
It never writes.

Returns:

```json
{"ok": true, "error": null,
 "command": "wm drafts/ notes.md --recursive --glob '*.md'",
 "files": [{"path": "/abs/drafts/a.md", "display": "drafts/a.md", "kind": "text", "size": 1234}],
 "files_total": 1,
 "discover_error": null,
 "layer": "A", "result_class": "Verifiable",
 "output": "writes NAME.cleaned.EXT next to each file; originals are never touched",
 "confirm": [{"kind": "remote", "message": "..."}],
 "estimate_seconds": 0,
 "warnings": [],
 "preflight_error": null}
```

- **`ok` and `error`.** `ok: false` with `error` set means argparse or
  `CleanRequest` refused the flags. The flags are still echoed in `command`
  as far as possible. A refused plan selects no files, and its `layer`,
  `result_class` and `output` are null.
- **`files` and `files_total`.** Selection comes from
  `batch_inputs.select_inputs`, the same way `clean_file.main` does it.
  `files` is capped at 2000 entries; `files_total` is the full count. A
  file's `size` is null when it cannot be read.
- **`warnings`.** Non-fatal problems with the options, such as a base URL that
  is malformed or not http(s).
- **`preflight_error`.** What `clean_request.plan_work` would refuse for the
  whole selection, such as an output collision or a binary file forced to
  text.
- **`layer` and `result_class`.** The weakest layer the selected files will
  actually get: B if a text file is rewritten, else V if pixels change or text
  is perturbed, else A. Text transforms skip files that are not plain text
  (Markdown, Office, images), so a rewrite over Markdown alone stays A and
  `warnings` says so, suggesting `--as text`. A file's own `file_done.layer` can
  be finer (`M` for container metadata, `perturb`, `synthid`); its
  `result_class` is the label to show.
- **`confirm`.** Every gate a clean would stop at before the first write:
  - `remote`: Layer B endpoint not on loopback, and at least one text file
    to send;
  - `in_place`;
  - `semantic`: `--strip-semantic-format`;
  - `cost`: a Layer B batch estimated over 300 s, counting text files only.

  `estimate_seconds` counts text files only as well.

### `inspect` `{state, soft?: bool}`

Returns `{"files": [Finding], "cancelled": false}` for the selected files, and emits
`{"event": "inspected", "data": Finding}` as each one finishes.

```json
{"path": "/abs/a.md", "display": "a.md", "kind": "text", "suspicious": true,
 "counts": {"hidden": 4, "metadata": 1},
 "lines": ["3 zero-width carriers (U+200B)", "1 bidi control (U+202E)", "Frontmatter key: generator"],
 "reveal": [{"line": 1, "text": "Hello◆ world", "marks": [[5, 6, "ZWSP"]]}],
 "error": null}
```

`lines` are short, human sentences built bridge-side from `inspect_asset`,
at most 12 per file. `counts.metadata` and `lines` cover the report's
`findings` only. Its `notes` (HEIF brands, a CMS generator, marker-free Exif)
say nothing about AI provenance and are left out.

`reveal` shows the hidden characters in place. It holds at most 6 excerpts,
one per affected line, and is empty for anything that is not read as text:

```json
"reveal": [{"line": 1, "text": "Hello◆ world, this is◆ a test.",
            "marks": [[5, 6, "ZWSP"], [21, 22, "RLO"]]}]
```

- In `text`, each invisible or format codepoint is replaced by `◆`, which takes
  one column. An excerpt is at most 100 columns and is centred on its first
  mark. `…` marks the edge where it was cut.
- Each entry in `marks` is `[start, end, short_name]`. The columns count into
  `text`, and `end` is exclusive.
- Offsets come from `inspect_asset`'s `layer_a_hits[*].sample_offsets`.
- `suspicious` is true when the raw report says so, or when
  `counts.hidden > 0`.

### `clean` `{state, confirmed?: ["remote", ...]}`

If `plan` reports a `confirm` kind that is not in `confirmed`, the bridge
returns error `needs_confirm` with `data.confirm` set, and nothing is written.
Otherwise it runs `plan_work`, then `run_clean_item` per file, then
re-inspects each output. Events:

- `{"event": "file_start", "data": {"index": 0, "total": 3, "display": "a.md"}}`
- `{"event": "token", "data": {"index": 0, "text": "..."}}`
  - This is the Layer B stream, coalesced to at most one frame per 50 ms.
- `{"event": "file_done", "data": {"index": 0, "display": "a.md", "output": "/abs/a.cleaned.md", "layer": "A", "result_class": "Verifiable", "exit_code": 0, "before": {"hidden": 4}, "after": {"hidden": 0}, "lines": ["Removed 4 hidden characters.", "Dropped frontmatter key generator."], "diff": "unified diff, text only, <= 200 lines", "error": null}}`
  - `lines` has one sentence per step that changed something, visible
    steps included. The bridge builds each one from the payload's
    `action_details` code and params, never by parsing the `actions` text.
  - With `--dry-run` (images with `--visible-mask`, `--visible-box` or
    `--detect-command` only) nothing is written, `output` is
    null, and `lines` starts with "Dry run: would write PATH.".
  - A file whose `run_clean_item` raises becomes a `file_done` with `error`
    set. The batch continues.
  - `diff` starts with the two `---`/`+++` file headers. Each invisible
    character is written as `<U+200B>`, so the diff is safe to print and the
    frontend can mark it.

A failed selection or a `plan_work` refusal returns `invalid_options`, and
nothing is written.

The response is `{"total": 3, "errors": 0, "cancelled": false, "command": "wm ...", "history": HistoryEntry}`.
`errors` counts every file with a nonzero exit code, as the CLI does, so a file
that still carries C2PA or AI signals after the clean counts as an error.

### `cancel` `{}`

Stops after the current file. Returns `{"ok": true}`.

### `detect_endpoints` `{}`

Returns `{"endpoints": [{"label": "Ollama", "backend": "ollama", "base_url": "http://127.0.0.1:11434", "reachable": true, "models": ["qwen3:14b"], "error": null}]}`.

The candidates are the `WATERMARKS_REWRITE_*` endpoint first, then Ollama
:11434, LM Studio :1234, llama.cpp :8080 and OpenAI-compatible :8000. All
probes run in parallel through `layer_b_discovery.probe_backend` with
`allow_remote=False`, so detection is loopback only.

### `checks` `{}`

Returns `{"checks": [{"name": "Python", "state": "OK", "good": true, "detail": "...", "fix": "", "optional": false}]}`. `state` is display text; `good` is what to test.

### `save_settings` `{settings: {preset, endpoint: {backend, base_url, model}}}`

Any other key is `bad_request`, and that includes anything named like a
key, token or secret.

The write is atomic and returns `{"path": "..."}`. It merges: a missing
top-level key keeps its saved value, and `endpoint: null` clears it. When a
saved file is read, an unknown preset or backend is dropped. On disk, the file keeps
the key names the old `tui.json` used (`preset`, `rewrite_backend`,
`rewrite_base_url`, `rewrite_model`), so an existing file still loads. When
an old file is read, its `rewrite_reasoning_effort` and
`rewrite_allow_remote` keys are ignored and are not written back.

### `history` `{}`

Returns `{"entries": [{"time": "14:02:11", "command": "wm ...", "summary": "3 files, 0 errors. Verifiable."}]}`.

This covers the current session only, newest first.

### `shutdown` `{}`

Returns `{"ok": true}`, then the bridge exits 0. EOF on stdin also exits.
Either way the bridge cancels any running work and waits up to 5 s for
in-flight requests to answer first.

## Invariants (enforced by tests)

- The bridge builds requests only through `clean_file._build_parser` and
  `CleanRequest.from_args`. It never constructs a `CleanPlan`: plans come from
  `plan_work`, and runs from `run_clean_item`.
- No `urllib` import in `tui_bridge.py`, `tui_core.py` or `tui.py`. HTTP goes
  through `layer_b_discovery` and `rewrite_text` only.
- Detection is loopback only.
- No API key in any output frame, including under a hostile
  `WATERMARKS_REWRITE_API_KEY`.
- Results are labelled by layer, never by outcome.
