/**
 * Clipboard, OSC 8 path links, and the terminal title/progress for a running clean.
 *
 * PI_TUI_WRITE_LOG captures the raw ANSI stream pi-tui's ProcessTerminal writes
 * to stdout. Set it to a file path, or to an existing directory (a timestamped
 * `tui-<date>-<pid>.log` is created there). Use it when a rendering bug needs
 * the exact bytes the terminal received. It is not WM_TUI_LOG, which is the
 * Python-side fatal log named on the fatal screen. t06's README pass must
 * surface PI_TUI_WRITE_LOG; this header is the protocol note until then.
 */
import { isAbsolute, resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { getNativeClipboard, hyperlink, type NativeClipboard, type Terminal } from "@earendil-works/pi-tui"
import type { FileDone } from "./protocol"
import type { Run } from "./store"

export type CopyTone = "info" | "error" | "success"

export interface CopyDeps {
  /**
   * Omit to load `getNativeClipboard()`. `undefined` from that helper means
   * unavailable. Pass `null` to force the unavailable path.
   */
  clipboard?: NativeClipboard | null
  toast: (text: string, tone?: CopyTone) => void
  terminal?: Terminal
}

/** OSC 8 link. Absolute paths become file:// URLs; relative paths resolve first. */
export function fileHyperlink(label: string, path: string): string {
  const absolute = isAbsolute(path) ? path : resolve(path)
  return hyperlink(label, pathToFileURL(absolute).href)
}

function loadClipboard(deps: CopyDeps): { clipboard?: NativeClipboard; failed: boolean } {
  if (deps.clipboard === null) return { failed: false }
  if (deps.clipboard) return { clipboard: deps.clipboard, failed: false }
  try {
    return { clipboard: getNativeClipboard(), failed: false }
  } catch (error) {
    deps.toast(`Clipboard unavailable: ${error instanceof Error ? error.message : error}`, "error")
    return { failed: true }
  }
}

/**
 * Place `text` on the native clipboard. When the helper is missing or has no
 * `setText`, write OSC 52 (the library's terminal fallback) and toast — a bare
 * OSC 52 write is not a confirmed copy.
 */
export async function copyText(text: string, deps: CopyDeps): Promise<void> {
  const loaded = loadClipboard(deps)
  if (loaded.failed) return
  const clipboard = loaded.clipboard
  if (clipboard?.setText) {
    try {
      await clipboard.setText(text)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      deps.toast(`Copy failed: ${message}`, "error")
      return
    }
    deps.toast("Copied.", "success")
    return
  }
  if (deps.terminal) deps.terminal.write(`\x1b]52;c;${Buffer.from(text).toString("base64")}\x07`)
  deps.toast("Clipboard unavailable.", "error")
}

/** Copy the cleaned file's output path. No path → toast, nothing written. */
export async function copyOutputPath(done: FileDone, deps: CopyDeps): Promise<void> {
  if (!done.output) {
    deps.toast("No output path to copy.", "error")
    return
  }
  await copyText(done.output, deps)
}

/** Copy the unified diff. No diff → toast, nothing written. */
export async function copyDiffText(done: FileDone, deps: CopyDeps): Promise<void> {
  if (!done.diff) {
    deps.toast("No diff to copy.", "error")
    return
  }
  await copyText(done.diff, deps)
}

export function cleanChrome(run: Run | null): { title: string; progress: boolean } {
  if (run?.kind === "clean") {
    const at = Math.min(run.index + 1, run.total)
    const display = run.display.replace(/[\u0000-\u001f\u007f]/g, "")
    return { title: `Cleaning ${at} of ${run.total}: ${display}`, progress: true }
  }
  return { title: "wm", progress: false }
}

/** Title and progress for a clean. Call again with `run === null` to clear. */
export function applyCleanChrome(
  terminal: Terminal,
  run: Run | null,
  seen: { title?: string; progress?: boolean } = {},
): void {
  const next = cleanChrome(run)
  if (seen.title !== next.title) {
    seen.title = next.title
    terminal.setTitle(next.title)
  }
  if (seen.progress !== next.progress) {
    seen.progress = next.progress
    terminal.setProgress(next.progress)
  }
}
