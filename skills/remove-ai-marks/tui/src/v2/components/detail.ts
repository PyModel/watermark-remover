import { sliceByColumn, truncateToWidth, visibleWidth, type Component } from "@earendil-works/pi-tui"
import { fileHyperlink } from "../ops"
import { outputDisplay, shortPath } from "../prompt"
import { parseFinding, type FileDone, type Reveal } from "../protocol"
import type { AppController } from "../store"
import { paint, resultClassColor, theme } from "../theme"

const GUTTER = 6
const ESCAPED = /<U\+[0-9A-F]{4,6}>/g

/**
 * One pass over before/after counts. The detail pane shows this value once;
 * it does not reduce the same maps again for a second label.
 */
export function markTotals(
  before: Record<string, number>,
  after: Record<string, number>,
): { before: number; after: number; removed: number } {
  const keys = new Set([...Object.keys(before), ...Object.keys(after)])
  let beforeTotal = 0
  let afterTotal = 0
  for (const key of keys) {
    const beforeCount = before[key]
    const afterCount = after[key]
    if (typeof beforeCount === "number" && Number.isFinite(beforeCount)) beforeTotal += beforeCount
    if (typeof afterCount === "number" && Number.isFinite(afterCount)) afterTotal += afterCount
  }
  return { before: beforeTotal, after: afterTotal, removed: beforeTotal - afterTotal }
}

function fit(line: string, width: number): string {
  return truncateToWidth(line, Math.max(1, width), "")
}

function onBoundary(text: string, index: number): boolean {
  if (index < 0 || index > text.length) return false
  if (index === text.length) return true
  const unit = text.charCodeAt(index)
  return unit < 0xdc00 || unit > 0xdfff
}

function highlight(text: string, marks: Reveal["marks"]): string {
  const ordered = [...marks].sort((a, b) => a[0] - b[0])
  let out = ""
  let at = 0
  for (const [start, end] of ordered) {
    if (start < at || end > text.length || end <= start) continue
    if (!onBoundary(text, start) || !onBoundary(text, end)) continue
    if (start > at) out += paint(text.slice(at, start), theme.text)
    out += paint(text.slice(start, end), theme.glow, true)
    at = end
  }
  if (at < text.length) out += paint(text.slice(at), theme.text)
  return out
}

function labelAt(text: string, marks: Reveal["marks"]): string {
  let row = ""
  const late: string[] = []
  for (const [start, , name] of marks) {
    if (start < 0 || start > text.length) continue
    const col = visibleWidth(text.slice(0, start))
    const used = visibleWidth(row)
    if (col >= used + (row ? 1 : 0)) row = row + " ".repeat(col - used) + name
    else late.push(name)
  }
  return late.length ? `${row}  ${late.join(" ")}` : row
}

function clipReveal(text: string, marks: Reveal["marks"], columns: number): { text: string; marks: Reveal["marks"] } {
  if (visibleWidth(text) <= columns) return { text, marks }
  const clipped = sliceByColumn(text, 0, columns, true)
  const kept = marks.filter(([start, end]) => start >= 0 && end <= clipped.length && start < end)
  return { text: clipped, marks: kept }
}

function revealLines(reveal: Reveal, width: number): string[] {
  const gutter = String(reveal.line).padStart(GUTTER - 2).padEnd(GUTTER)
  const budget = Math.max(1, width - GUTTER)
  const clipped = clipReveal(reveal.text, reveal.marks, budget)
  const text = fit(gutter + highlight(clipped.text, clipped.marks), width)
  const labels = fit(gutter + paint(labelAt(clipped.text, clipped.marks), theme.muted), width)
  return [text, labels]
}

function diffLines(diff: string, width: number): string[] {
  const out: string[] = []
  for (const line of diff.split("\n")) {
    if (!line || line.startsWith("---") || line.startsWith("+++") || line.startsWith("@@")) continue
    const color = line.startsWith("+") ? theme.success : line.startsWith("-") ? theme.error : theme.faint
    let painted = ""
    let last = 0
    for (const match of line.matchAll(ESCAPED)) {
      const index = match.index ?? 0
      if (index > last) painted += paint(line.slice(last, index), color)
      painted += paint("◆", theme.glow, true)
      last = index + match[0].length
    }
    if (last < line.length) painted += paint(line.slice(last), color)
    out.push(fit(painted, width))
  }
  return out
}

function classLine(done: FileDone): string {
  const totals = markTotals(done.before, done.after)
  return (
    paint(done.result_class, resultClassColor(done.result_class), true) +
    paint(`  layer ${done.layer}`, theme.faint) +
    paint(`  ${totals.removed} marks removed`, theme.muted)
  )
}

/** Detail pane. Marks are converted by `parseFinding` before any slice. */
export class DetailView implements Component {
  constructor(private ctl: AppController) {}
  invalidate(): void {}
  render(width: number): string[] {
    const file = this.ctl.app.plan?.files[this.ctl.app.selected]
    if (!file) return [fit("Select a file.", width)]
    const done = this.ctl.app.results[file.path]
    const finding = this.ctl.app.findings[file.path]
    const parsed = finding ? parseFinding(finding) : undefined
    const lines = [fileHyperlink(truncateToWidth(file.display, Math.max(1, width), "…"), file.path), fit(paint(file.kind, theme.faint), width)]
    if (done) {
      lines.push(fit(classLine(done), width))
      if (done.output) {
        const shown = shortPath(outputDisplay(file, done.output), Math.max(1, width - 9))
        lines.push(fit(paint("Saved to ", theme.muted) + fileHyperlink(shown, done.output), width))
      }
      if (done.error) lines.push(fit(paint(done.error, theme.error), width))
      for (const line of done.lines.slice(0, 8)) lines.push(fit(paint(line, theme.text), width))
      if (done.diff) lines.push(...diffLines(done.diff, width))
      return lines
    }
    const stream = this.ctl.app.streams[file.path]
    if (stream && this.ctl.app.run) {
      lines.push(fit(paint("Rewriting…", theme.primary), width))
      for (const line of stream.split("\n").slice(-4)) lines.push(fit(paint(line, theme.muted), width))
      return lines
    }
    if (!parsed) return [...lines, fit(paint("Not inspected yet. Ctrl+E shows what is hidden.", theme.faint), width)]
    if (parsed.error) lines.push(fit(paint(parsed.error, theme.error), width))
    const hidden = parsed.counts.hidden ?? 0
    if (!parsed.error && !hidden && parsed.lines.length === 0) {
      lines.push(fit(paint("Nothing hidden found.", theme.success), width))
      return lines
    }
    if (hidden) lines.push(fit(paint(`${hidden} hidden character${hidden === 1 ? "" : "s"}`, theme.muted), width))
    for (const excerpt of parsed.reveal.slice(0, 6)) lines.push(...revealLines(excerpt, width))
    for (const line of parsed.lines) lines.push(fit(paint(line, theme.muted), width))
    return lines
  }
}
