# ADR-0001: wm-tui renders on @earendil-works/pi-tui

Status: accepted (2026-10-04, to-auto run 2) · Supersedes: the OpenTUI/Solid frontend choice (unrecorded, 2026-08)

## Context

The wm-tui frontend (skills/remove-ai-marks/tui) renders on @opentui/core + @opentui/solid 0.5.12 with
SolidJS. That stack pulls an 84 MB / ~500-package `bun install` on every user's first run, sits on a 0.x
API line, and couples the test harness to renderer internals (`testRender`/`captureCharFrame`). The prompt
input has no history, undo, or paste handling. `@earendil-works/pi-tui` 1.0.2 (Pi's own terminal UI
library) provides the whole needed surface — differential renderer with CSI 2026 sync, Editor with
undo/kill-ring/history/bracketed-paste markers + CombinedAutocompleteProvider (slash commands and paths),
overlay stack, SelectList, ScrollView layouts on TuiAltScreen — with 3 packages / 3.2 MB and a 1.0 semver
contract. Verified locally under Bun 1.4.2 including the darwin-arm64 native clipboard prebuild.

## Decision

1. The frontend renders on `@earendil-works/pi-tui` (pinned exact) as its only runtime dependency;
   @opentui/*, solid-js, the Solid JSX transform, and bunfig preload are deleted at cutover, with no
   compatibility flag.
2. Renderer is `TuiAltScreen` (fullscreen, app-owned scrolling, transcript printed to scrollback on
   quit). Native mid-session scrollback is given up; pageup/pagedown become viewport keys (selection
   paging by 10 is removed with them).
3. The prompt is a pi-tui `Editor` (multi-line-capable; paste markers expanded on submit) wired to
   `CombinedAutocompleteProvider` over the existing `COMMANDS` registry; the app calls `addToHistory`
   after each successful submit.
4. Dialogs are one `showOverlay` each over the built-in overlay stack; dismissal uses the handle.
5. Bridge mark offsets are defined as Unicode code-point indices (PROTOCOL.md) and converted to UTF-16
   at the one protocol boundary in the frontend (`protocol.ts`).
6. Tests assert on `component.render(width)` output and drive a `FakeTerminal` implementing the full
   `Terminal` interface; no test touches a concrete renderer.

## Consequences

- First-run install drops from ~500 packages/84 MB to 3 packages/3.2 MB (launcher passes
  `--production` so devDependencies never install for users).
- The old 30-test bun suite is replaced 1:1 by behavior tests at the new seams; the SVG snapshot
  harness is dropped (it never gated anything).
- Frontend rewrite ~1.3k lines TSX -> plain TS components; SolidJS knowledge no longer needed.
- Rejected alternative: stay on OpenTUI and bump 0.5.12 -> 0.5.14 — smaller diff, but keeps the
  weight, the 0.x churn, and forfeits Editor/autocomplete; loses on install size and semver
  stability, the deciding properties.
