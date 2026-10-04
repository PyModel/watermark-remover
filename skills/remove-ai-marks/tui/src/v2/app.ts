import { isViewportTUI, matchesKey, type Component, type Editor, type NativeClipboard, type OverlayHandle, type TUI } from "@earendil-works/pi-tui"
import { runCommand } from "./commands"
import { buildLayout, SPINNER } from "./components/layout"
import { createEditor } from "./components/prompt"
import { syncDialogOverlays } from "./dialogs"
import { copyDiffText, copyOutputPath, type CopyDeps } from "./ops"
import { parsePrompt } from "./prompt"
import type { BridgeLike, FileDone } from "./protocol"
import { createApp, type AppController, type Dialog } from "./store"

export interface MountedApp {
  ctl: AppController
  editor: Editor
  root: Component
  dispose(): void
}

export interface MountOptions {
  /** `null` forces the unavailable clipboard. Omit to load the native helper. */
  clipboard?: NativeClipboard | null
}

/**
 * Key routing sits on addInputListener, which runs before the focused editor.
 * ctrl+j is also the editor's newline default, so selection must consume it.
 * up/down are not consumed: the editor recalls history only on the first visual
 * line (or when already browsing). Selection is ctrl+k / ctrl+j. pageup/pagedown
 * belong to the alt-screen viewport and are not selection keys.
 * ctrl+o copies the selected file's output path; f2 copies its diff. Neither is an
 * editor or viewport binding, so both are consumed here.
 */
export function mountApp(tui: TUI, bridge: BridgeLike, onQuit: () => void, options: MountOptions = {}): MountedApp {
  if (!isViewportTUI(tui)) throw new Error("wm-tui requires a fullscreen viewport")
  const handles = new Map<Dialog, OverlayHandle>()
  let alive = true
  let current: AppController | undefined
  const ctl = createApp(bridge, onQuit, () => {
    if (!alive || !current) return
    syncDialogOverlays(tui, current, handles, {
      copy: (text) => copyOsc52(tui, text),
      rows: () => tui.terminal.rows,
    })
    tui.requestRender()
  })
  current = ctl
  const editor = createEditor(tui)
  editor.onSubmit = (text) => submitPrompt(ctl, editor, text)
  let frame = 0
  const timer = setInterval(() => {
    if (!ctl.app.run) return
    frame = (frame + 1) % SPINNER.length
    tui.requestRender()
  }, 80)
  timer.unref?.()
  const root = buildLayout(ctl, editor, () => frame, tui.terminal)
  tui.setLayoutRoot(root)
  tui.setFocus(editor)
  const unsubscribe = tui.addInputListener((data) => routeKeys(data, ctl, editor, tui, options.clipboard))
  return {
    ctl,
    editor,
    root,
    dispose() {
      alive = false
      clearInterval(timer)
      unsubscribe()
      for (const handle of handles.values()) handle.hide()
      handles.clear()
    },
  }
}

function copyOsc52(tui: TUI, text: string): boolean {
  try {
    tui.terminal.write(`\x1b]52;c;${Buffer.from(text, "utf8").toString("base64")}\x07`)
    return true
  } catch {
    return false
  }
}

function copyDeps(ctl: AppController, tui: TUI, clipboard?: NativeClipboard | null): CopyDeps {
  return {
    ...(clipboard !== undefined ? { clipboard } : {}),
    terminal: tui.terminal,
    toast: (text, tone) => ctl.toast(text, tone ?? "info"),
  }
}

function copySelected(ctl: AppController, kind: "output" | "diff", deps: CopyDeps): Promise<void> {
  const file = ctl.app.plan?.files[ctl.app.selected]
  const done: FileDone | undefined = file ? ctl.app.results[file.path] : undefined
  if (!done) {
    ctl.toast(kind === "output" ? "No output path to copy." : "No diff to copy.", "error")
    return Promise.resolve()
  }
  return kind === "output" ? copyOutputPath(done, deps) : copyDiffText(done, deps)
}

function submitPrompt(ctl: AppController, editor: Editor, text: string): void {
  const parsed = parsePrompt(text)
  if (parsed.kind === "empty") return
  // submitValue does not record history. The app does, after a successful submit.
  editor.addToHistory(text)
  if (parsed.kind === "paths") ctl.addPaths(parsed.tokens)
  else if (parsed.kind === "flags") ctl.applyFlags(parsed.tokens)
  else runCommand(ctl, parsed.name, parsed.arg)
}

function routeKeys(data: string, ctl: AppController, editor: Editor, tui: TUI, clipboard?: NativeClipboard | null) {
  if (matchesKey(data, "ctrl+c") || matchesKey(data, "ctrl+d")) {
    ctl.quit()
    return { consume: true }
  }
  // Overlay handleInput runs after this listener. Do not consume: arrows,
  // enter, escape, and type-to-filter belong to the focused dialog.
  if (ctl.app.dialogs.length > 0 || tui.hasOverlay()) return
  if (matchesKey(data, "ctrl+j")) {
    ctl.select(ctl.app.selected + 1)
    return { consume: true }
  }
  if (matchesKey(data, "ctrl+k")) {
    ctl.select(ctl.app.selected - 1)
    return { consume: true }
  }
  if (matchesKey(data, "ctrl+p")) {
    ctl.openDialog({ type: "palette" })
    return { consume: true }
  }
  if (matchesKey(data, "ctrl+r")) {
    void ctl.clean()
    return { consume: true }
  }
  if (matchesKey(data, "ctrl+e")) {
    void ctl.inspect()
    return { consume: true }
  }
  // ctrl+o / f2 are not editor or viewport bindings. Consume them here, before the editor.
  if (matchesKey(data, "ctrl+o")) {
    void copySelected(ctl, "output", copyDeps(ctl, tui, clipboard))
    return { consume: true }
  }
  if (matchesKey(data, "f2")) {
    void copySelected(ctl, "diff", copyDeps(ctl, tui, clipboard))
    return { consume: true }
  }
  if (matchesKey(data, "f1")) {
    ctl.openDialog({ type: "help" })
    return { consume: true }
  }
  if (matchesKey(data, "shift+tab")) {
    ctl.cyclePreset(-1)
    return { consume: true }
  }
  if (matchesKey(data, "tab")) {
    ctl.cyclePreset(1)
    return { consume: true }
  }
  if (matchesKey(data, "escape") && !editor.isShowingAutocomplete()) {
    if (ctl.app.run) {
      void ctl.cancel()
      return { consume: true }
    }
    if (editor.getText()) {
      editor.setText("")
      return { consume: true }
    }
  }
}
