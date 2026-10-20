# Terminal UI

`wm-tui` is an interactive front end for `wm`. It needs Bun 1.4 or later.

```bash
wm-tui                     # the current directory
wm-tui ./drafts --recursive --glob "*.md"
```

`wm-tui` is a single-screen terminal UI in the style of opencode and pi. It
renders on [@earendil-works/pi-tui](https://github.com/earendil-works/pi/tree/main/packages/tui)
(alternate-screen fullscreen layout, differential rendering), so it needs
[Bun](https://bun.sh) 1.4 or later on the `PATH`. The Python install stays
dependency-free: on first launch `wm-tui` runs
`bun install --frozen-lockfile --production` for its frontend — three runtime
packages, not the previous hundreds — and it prints the install hint if Bun is
missing. A read-only install is copied into the user cache before that install.
macOS, Linux and Windows are supported.

The screen has four parts:

- **The file list** on the left: `→` is the focused row, `[x]` is the one file marked for cleaning. Moving the arrow does not change the mark.
- **The selected file** on the right: what was found, what was removed, and a
  before/after diff.
- **One prompt** at the bottom:
  - a path adds files;
  - a `--flag` adds a `wm` option (any flag `wm` accepts, checked by the
    CLI's own parser; type a bare flag again to turn it off);
  - a `/command` runs a command.
- **The footer** prompts you to mark a file, then shows the `wm …` command for that marked file.

| Key | Does |
| --- | --- |
| `ctrl+e` | inspect, which writes nothing |
| `ctrl+r` | clean **only the marked file**; writes `NAME.cleaned.EXT` beside it, preserving the original |
| `tab` | next preset: Hidden marks, Hidden marks aggressive, Deep clean (LLM rewrite), Images |
| `ctrl+p` | every command: model, history, doctor, setup, help |
| `up` / `down` | previous / next file when the prompt is empty; cursor/completion while typing |
| `space` | mark the focused file; marking another replaces the old mark; Space again unmarks it |
| `ctrl+k` / `ctrl+j` | alternative previous / next file shortcuts |
| `alt+up` / `alt+down` | prompt history |
| `enter` | submit the typed path, flag or command; never automatically clean |
| `esc` | stop after the current file |

**Typical workflow:** ↑/↓ to a file → Space to mark it → Ctrl+E to inspect → Ctrl+R to clean. Ctrl+E inspects the discovered list; Ctrl+R never cleans the whole directory. With no mark, cleaning is refused.

**Your local model:** type `/model` to choose a model reported by a local server, or select **Enter a server**. Use ←/→ to choose Ollama/OpenAI-compatible, Tab to edit the local URL and model ID (Ctrl+U clears a field), then Enter to save. Each user supplies their own server and model; there is no fixed model, Pi authentication, or cloud-model catalog. A local OpenAI-compatible URL ending in `/v1` is normalized to its server root automatically. Select **Deep clean (LLM rewrite)** with Tab or `/preset rewrite` to use it. Markdown/documents normally use container cleaning and skip rewriting; `--as text` explicitly opts into text-body rewriting.

**First run** opens a three-step setup:

1. What the result classes mean.
2. A scan for a local model server. The scan only contacts `127.0.0.1`: it
   checks Ollama, LM Studio, llama.cpp and OpenAI-compatible servers for their model lists, and
   never sends a document.
3. The default preset.

`esc` skips the setup, and it does not come back unless you run `/setup` or
`wm-tui --setup`. `--no-setup` never shows it.

The choices are saved to `~/.config/watermark-remover/tui.json`. The API key is
never in that file: it is read from `WATERMARKS_REWRITE_API_KEY` at run time
and never displayed.

Results are labelled by layer, never by outcome: Verifiable, Best-effort or
Detection-only. A clean that needs confirmation stops and asks before the
first write. That covers a remote endpoint, `--in-place`,
`--strip-semantic-format`, and a long LLM run.

The UI is two processes:

- A Bun frontend (`skills/remove-ai-marks/tui/`, entry `src/index.ts`) that only
  renders, on pi-tui. First launch installs three runtime packages with
  `bun install --frozen-lockfile --production`.
- A Python bridge (`scripts/tui_bridge.py`) that owns every decision. It builds
  the request through `wm`'s own parser, `plan_work` and `run_clean_item`, so
  it hits every refusal the CLI makes.

[`tui/PROTOCOL.md`](../skills/remove-ai-marks/tui/PROTOCOL.md) documents the protocol between them.

Set `PI_TUI_WRITE_LOG` to a file path, or to an existing directory (a
timestamped log is created there), to capture the raw ANSI stream the frontend
writes. That is the rendering-bug log. It is not `WM_TUI_LOG`, which is the
bridge stderr log.
