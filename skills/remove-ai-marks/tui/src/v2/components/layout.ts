import { HStack, ScrollView, truncateToWidth, visibleWidth, VStack, type Component } from "@earendil-works/pi-tui"
import type { FileEntry } from "../protocol"
import { type AppController, endpointLabel } from "../store"
import { paint, theme } from "../theme"
import { PromptStatus } from "./prompt"

export const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

/** The one thing an empty screen tells you to do. Plain text so render(width) is assertable. */
export function renderEmpty(width: number): string[] {
  const lines = [
    "Find what is hidden in a file, then remove it.",
    "",
    "Type a file or folder below and press enter.",
    "",
    "ctrl+e    See what is hidden, without writing",
    "ctrl+r    Clean, leaving the originals untouched",
    "tab       Choose what to remove",
    "ctrl+p    Every command",
  ]
  return lines.map((line) => truncateToWidth(line, Math.max(1, width), ""))
}

function fit(text: string, width: number): string {
  return truncateToWidth(text, Math.max(1, width), "")
}

function fileCount(total: number, shown: number): string {
  const noun = `${total} file${total === 1 ? "" : "s"}`
  return shown < total ? `${noun}, first ${shown} shown` : noun
}

function status(ctl: AppController, file: FileEntry, frame: number): { glyph: string; text: string } {
  const app = ctl.app
  const done = app.results[file.path]
  const run = app.run
  const index = app.plan?.files.findIndex((f) => f.path === file.path) ?? -1
  if (run && run.kind === "clean" && index === run.index && !done) {
    return { glyph: SPINNER[frame] ?? SPINNER[0]!, text: "cleaning" }
  }
  if (done) {
    if (done.error) return { glyph: "✗", text: "error" }
    const left = Object.values(done.after).reduce((a, b) => a + b, 0)
    return left ? { glyph: "◆", text: `${left} left` } : { glyph: "✓", text: "cleaned" }
  }
  const finding = app.findings[file.path]
  if (finding) {
    if (finding.error) return { glyph: "✗", text: "error" }
    const found = Object.values(finding.counts).reduce((a, b) => a + b, 0)
    return finding.suspicious || found
      ? { glyph: "◆", text: found ? `${found} mark${found === 1 ? "" : "s"}` : "suspicious" }
      : { glyph: "○", text: "no marks" }
  }
  return { glyph: "·", text: "" }
}

class Header implements Component {
  constructor(private ctl: AppController) {}
  invalidate(): void {}
  render(width: number): string[] {
    const left = paint("wm", theme.primary, true) + paint("  watermark remover", theme.muted)
    const endpoint = endpointLabel(this.ctl.app.state.endpoint)
    const right = endpoint ? paint("● ", theme.primary) + paint(endpoint, theme.muted) : paint("No model", theme.faint)
    const gap = Math.max(1, width - visibleWidth(left) - visibleWidth(right))
    return [fit(left + " ".repeat(gap) + right, width)]
  }
}

class EmptyView implements Component {
  invalidate(): void {}
  render(width: number): string[] {
    return renderEmpty(width).map((line, index) => paint(line, index === 2 ? theme.faint : theme.text))
  }
}

class FatalView implements Component {
  constructor(private ctl: AppController) {}
  invalidate(): void {}
  render(width: number): string[] {
    const lines = [
      paint("The Python side stopped.", theme.error, true),
      paint(this.ctl.app.fatal ?? "", theme.muted),
      "",
      paint(`Details are in ${process.env.WM_TUI_LOG ?? "the log"}. ctrl+c quits.`, theme.faint),
    ]
    return lines.map((line) => fit(line, width))
  }
}

class FileListView implements Component {
  constructor(
    private ctl: AppController,
    private frame: () => number,
  ) {}
  invalidate(): void {}
  render(width: number): string[] {
    const app = this.ctl.app
    const files = app.plan?.files ?? []
    const lines = [paint(fileCount(app.plan?.files_total ?? 0, files.length), theme.muted)]
    if (app.plan?.discover_error) lines.push(paint(app.plan.discover_error, theme.error))
    for (const file of files) {
      const selected = files[app.selected]?.path === file.path
      const row = status(this.ctl, file, this.frame())
      const mark = selected ? "→ " : "  "
      const text = `${mark}${row.glyph} ${file.display}${row.text ? "  " + row.text : ""}`
      lines.push(paint(text, selected ? theme.text : theme.muted))
    }
    return lines.map((line) => fit(line, width))
  }
}

class DetailView implements Component {
  constructor(private ctl: AppController) {}
  invalidate(): void {}
  render(width: number): string[] {
    const file = this.ctl.app.plan?.files[this.ctl.app.selected]
    if (!file) return [fit("Select a file.", width)]
    const done = this.ctl.app.results[file.path]
    const finding = this.ctl.app.findings[file.path]
    const lines = [file.display, file.kind]
    if (done) lines.push(done.error ? "error" : "cleaned")
    else if (finding) {
      const found = Object.values(finding.counts).reduce((sum, count) => sum + count, 0)
      lines.push(found ? `${found} marks` : "no marks")
    } else lines.push("Not inspected yet. Ctrl+E shows what is hidden.")
    return lines.map((line) => fit(line, width))
  }
}

class Footer implements Component {
  constructor(
    private ctl: AppController,
    private frame: () => number,
  ) {}
  invalidate(): void {}
  render(width: number): string[] {
    const app = this.ctl.app
    let left = ""
    let color: string = theme.faint
    if (app.run) {
      const verb = app.run.kind === "clean" ? "Cleaning" : "Inspecting"
      const at = Math.min(app.run.index + 1, app.run.total)
      const spinner = `${SPINNER[this.frame()] ?? SPINNER[0]} ${verb} ${at} of ${app.run.total}: ${app.run.display}`
      left = app.toast ? `${app.toast.text}  ${spinner}` : `${spinner}    esc stops`
      color = theme.primary
    } else if (app.toast) {
      left = app.toast.text
      color = app.toast.tone === "error" ? theme.error : app.toast.tone === "success" ? theme.success : theme.text
    } else if (app.plan && !app.plan.ok) {
      left = app.plan.error ?? "These options are not valid."
      color = theme.error
    } else if (app.plan?.preflight_error) {
      left = app.plan.preflight_error
      color = theme.error
    } else if (app.plan?.warnings.length) {
      left = app.plan.warnings[0]!
      color = theme.warning
    } else if (app.plan?.files.length) {
      left = `$ ${app.plan.command}`
      color = theme.faint
    }
    const hints =
      width >= 100
        ? "ctrl+r clean   ctrl+e inspect   tab preset   ctrl+p commands"
        : "ctrl+r clean   ctrl+p commands"
    const room = Math.max(0, width - visibleWidth(hints) - 2)
    return [fit(paint(truncateToWidth(left, room, ""), color) + "  " + paint(hints, theme.muted), width)]
  }
}

function scroll(child: Component, primary = false): ScrollView {
  return new ScrollView(child, { follow: "none", primary, scrollbar: "hidden" })
}

/**
 * Fullscreen alt-screen root: header, list+detail scroll panes, prompt status,
 * editor, footer. Fatal replaces everything under the header.
 */
export function buildLayout(ctl: AppController, editor: Component, frame: () => number): Component {
  const fatal = scroll(new FatalView(ctl))
  const empty = scroll(new EmptyView(), true)
  const list = scroll(new FileListView(ctl, frame), true)
  const detail = scroll(new DetailView(ctl))
  const panes = new HStack(
    [
      { component: list, grow: 2, minSize: 24 },
      { component: detail, grow: 3, minSize: 16, visible: (viewport) => viewport.width - 4 >= 70 },
    ],
    { gap: 2 },
  )
  const alive = () => ctl.app.fatal == null
  const hasFiles = () => (ctl.app.plan?.files.length ?? 0) > 0 || Boolean(ctl.app.plan?.discover_error)
  return new VStack([
    { component: new Header(ctl), basis: 1, shrink: 0 },
    { component: fatal, grow: 1, visible: () => ctl.app.fatal != null },
    { component: empty, grow: 1, visible: () => alive() && !hasFiles() },
    { component: panes, grow: 1, visible: () => alive() && hasFiles() },
    { component: new PromptStatus(ctl), basis: 1, shrink: 0, visible: alive },
    { component: editor, shrink: 0, visible: alive },
    { component: new Footer(ctl, frame), basis: 1, shrink: 0, visible: alive },
  ])
}

