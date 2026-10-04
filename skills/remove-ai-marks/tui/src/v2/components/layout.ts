import { HStack, ScrollView, truncateToWidth, visibleWidth, VStack, type Component, type Terminal } from "@earendil-works/pi-tui"
import { clearTerminalChrome } from "../ops"
import type { FileEntry } from "../protocol"
import { type AppController, endpointLabel } from "../store"
import { paint, theme } from "../theme"
import { renderWordmark } from "./brand"
import { DetailView } from "./detail"
import { Footer } from "./footer"
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
    "↑ ↓       Move through files; Space marks one",
    "ctrl+r    Clean the marked file; originals stay untouched",
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
  if (run && run.kind === "clean" && run.path === file.path) {
    return { glyph: SPINNER[frame] ?? SPINNER[0]!, text: "cleaning" }
  }
  if (done) return { glyph: done.error ? "✗" : "✓", text: done.result_class }
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
  private chrome: { title?: string; progress?: boolean } = {}
  constructor(
    private ctl: AppController,
    private terminal: Terminal,
  ) {}
  invalidate(): void {}
  render(width: number): string[] {
    if (this.ctl.app.fatal) clearTerminalChrome(this.terminal, this.chrome)
    const banner = renderWordmark(width, this.terminal.rows)
    const left = paint("wm", theme.primary, true) + paint("  WATERMARK REMOVER", theme.text)
    const endpoint = endpointLabel(this.ctl.app.state.endpoint)
    const model = endpoint ? paint("● ", theme.primary) + paint(endpoint, theme.muted) : paint("No model", theme.faint)
    const available = Math.max(0, width - visibleWidth(left) - 2)
    const right = available > 0 ? truncateToWidth(model, available, "…") : ""
    const gap = Math.max(1, width - visibleWidth(left) - visibleWidth(right))
    const status = fit(left + " ".repeat(gap) + right, width)
    return banner.length ? [...banner, "", status, ""] : [status]
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
    const count = fileCount(app.plan?.files_total ?? 0, files.length)
    const lines = [paint(`${count} · ${app.state.clean_path ? "1 marked" : "Space marks one"}`, theme.muted)]
    if (app.plan?.discover_error) lines.push(paint(app.plan.discover_error, theme.error))
    for (const file of files) {
      const selected = files[app.selected]?.path === file.path
      const row = status(this.ctl, file, this.frame())
      const mark = selected ? "→ " : "  "
      const check = app.state.clean_path === file.path ? "[x]" : "[ ]"
      const text = `${mark}${check} ${row.glyph} ${file.display}${row.text ? "  " + row.text : ""}`
      lines.push(paint(text, selected ? theme.text : theme.muted))
    }
    return lines.map((line) => fit(line, width))
  }
}

class SelectedFileScrollView extends ScrollView {
  private selection = ""
  private selectionRow = -1

  constructor(child: Component, private ctl: AppController) {
    super(child, { follow: "none", primary: true, scrollbar: "hidden" })
  }

  override updateLayout(contentHeight: number, viewportHeight: number, requestRender: () => void): void {
    const resized = this.viewportHeight !== viewportHeight
    super.updateLayout(contentHeight, viewportHeight, requestRender)
    const app = this.ctl.app
    const path = app.plan?.files[app.selected]?.path ?? ""
    const row = app.selected + 1 + (app.plan?.discover_error ? 1 : 0)
    // Manual page/wheel scrolling must remain independent until selection
    // changes (or the viewport shrinks).
    if (path && viewportHeight > 0 && (resized || path !== this.selection || row !== this.selectionRow)) {
      if (row < this.scrollTop) this.scrollTo(row)
      else if (row >= this.scrollTop + viewportHeight) this.scrollTo(row - viewportHeight + 1)
    }
    this.selection = path
    this.selectionRow = row
  }
}

function scroll(child: Component, primary = false): ScrollView {
  return new ScrollView(child, { follow: "none", primary, scrollbar: "hidden" })
}

/**
 * Fullscreen alt-screen root: header, list+detail scroll panes, prompt status,
 * editor, footer. Fatal replaces everything under the header. Detail and footer
 * are the operability components, not local stubs.
 */
export function buildLayout(ctl: AppController, editor: Component, frame: () => number, terminal: Terminal): Component {
  const fatal = scroll(new FatalView(ctl))
  const empty = scroll(new EmptyView(), true)
  const list = new SelectedFileScrollView(new FileListView(ctl, frame), ctl)
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
    { component: new Header(ctl, terminal), basis: "auto", shrink: 0 },
    { component: fatal, grow: 1, visible: () => ctl.app.fatal != null },
    { component: empty, grow: 1, visible: () => alive() && !hasFiles() },
    { component: panes, grow: 1, visible: () => alive() && hasFiles() },
    { component: new PromptStatus(ctl), basis: 1, shrink: 0, visible: alive },
    { component: editor, shrink: 0, visible: alive },
    { component: new Footer(ctl, terminal, frame), basis: 1, shrink: 0, visible: alive },
  ])
}
