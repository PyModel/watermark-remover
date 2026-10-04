import {
  matchesKey,
  SelectList,
  truncateToWidth,
  visibleWidth,
  wrapTextWithAnsi,
  type Component,
  type SelectItem,
} from "@earendil-works/pi-tui"
import { fuzzyScore } from "../prompt"
import { editorTheme, paint, theme } from "../theme"

export interface DialogItem {
  id: string
  title: string
  /** Shown under the highlighted row only. */
  detail?: string
  /** Right annotation. SelectList has no such field; it rides in description. */
  right?: string
  disabled?: boolean
}

export interface ListDialogOptions {
  title: string
  items: () => DialogItem[]
  filter?: boolean
  placeholder?: string
  initial?: string
  steps?: [number, number]
  footer?: string
  empty?: string
  /** History rows are pre-cut; do not cap the primary column at 32. */
  wide?: boolean
  maxVisible?: number
  onSelect: (item: DialogItem) => void
  onClose: () => void
}

/**
 * Shared list chassis. SelectList.setFilter matches `value` by prefix only, so
 * fuzzy rank is applied here and the matched title is what goes in `value`.
 * Disabled rows stay visible; enter and click refuse them.
 */
export class ListDialog implements Component {
  private list: SelectList | null = null
  private signature = ""
  private query = ""
  private builtQuery = ""
  private index = 0
  private primed = false
  private lastWidth = 64

  constructor(private readonly options: ListDialogOptions) {}

  invalidate(): void {
    this.list = null
    this.signature = ""
  }

  handleInput(data: string): void {
    const visible = this.sync(this.lastWidth)
    if (matchesKey(data, "escape")) {
      this.options.onClose()
      return
    }
    if (matchesKey(data, "up") || matchesKey(data, "ctrl+p")) {
      this.move(visible, -1)
      return
    }
    if (matchesKey(data, "down") || matchesKey(data, "ctrl+n")) {
      this.move(visible, 1)
      return
    }
    if (matchesKey(data, "pageUp")) {
      this.move(visible, -this.visibleRows())
      return
    }
    if (matchesKey(data, "pageDown")) {
      this.move(visible, this.visibleRows())
      return
    }
    if (data === "\r") {
      this.choose(visible)
      return
    }
    if (!this.options.filter) return
    if (matchesKey(data, "backspace") || data === "\x7f" || data === "\x08") {
      this.query = [...this.query].slice(0, -1).join("")
      return
    }
    if (data.length === 1 && data >= " " && data <= "~") this.query += data
  }

  render(width: number): string[] {
    this.lastWidth = Math.max(1, width)
    const visible = this.sync(this.lastWidth)
    const lines = [titleLine(this.options.title, this.options.steps, this.lastWidth)]
    if (this.options.filter) {
      const query = this.query || this.options.placeholder || "Type to filter"
      lines.push(paint(truncateToWidth(query, this.lastWidth, ""), this.query ? theme.text : theme.faint))
    }
    if (!visible.length) {
      lines.push(paint(this.options.empty ?? "Nothing matches.", theme.muted))
    } else if (this.list) {
      lines.push(...this.list.render(this.lastWidth))
      const current = visible[this.index]
      if (current?.detail) lines.push(...wrap(current.detail, this.lastWidth).map((line) => paint(line, theme.muted)))
    }
    if (this.options.footer) lines.push(...wrap(this.options.footer, this.lastWidth).map((line) => paint(line, theme.faint)))
    return lines.map((line) => fit(line, this.lastWidth))
  }

  private visibleRows(): number {
    return this.options.maxVisible ?? 12
  }

  private sync(width: number): DialogItem[] {
    const visible = rankItems(this.options.items(), this.query)
    const signature = `${width}|${this.query}|${visible.map((item) => `${item.id}|${item.title}|${item.right ?? ""}|${item.disabled ? 1 : 0}`).join("\n")}`
    if (!this.list || signature !== this.signature) {
      const queryChanged = this.query !== this.builtQuery
      this.list = new SelectList(
        visible.map(toSelectItem),
        this.visibleRows(),
        editorTheme.selectList,
        this.options.wide ? { minPrimaryColumnWidth: 8, maxPrimaryColumnWidth: Math.max(8, width - 2) } : undefined,
      )
      if (!this.primed && !this.query && this.options.initial) {
        const at = visible.findIndex((item) => item.id === this.options.initial)
        this.index = at >= 0 ? at : 0
      } else if (queryChanged) {
        this.index = 0
      } else {
        this.index = Math.min(this.index, Math.max(0, visible.length - 1))
      }
      this.primed = true
      this.list.setSelectedIndex(this.index)
      this.signature = signature
      this.builtQuery = this.query
    }
    return visible
  }

  private move(visible: DialogItem[], step: number): void {
    if (!visible.length) return
    this.index = (this.index + step + visible.length) % visible.length
    this.list?.setSelectedIndex(this.index)
    this.list?.invalidate()
  }

  private choose(visible: DialogItem[]): void {
    const item = visible[this.index]
    if (!item || item.disabled) return
    this.options.onSelect(item)
  }
}

export function rankItems(items: DialogItem[], query: string): DialogItem[] {
  const q = query.trim()
  if (!q) return items
  return items
    .map((item) => ({ item, score: fuzzyScore(q, item.title) }))
    .filter((entry): entry is { item: DialogItem; score: number } => entry.score !== null)
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.item)
}

function toSelectItem(item: DialogItem): SelectItem {
  return {
    value: item.title,
    label: item.disabled ? paint(item.title, theme.faint) : item.title,
    description: item.right,
  }
}

export function stepDots(steps: [number, number]): string {
  return "●".repeat(steps[0]) + "○".repeat(Math.max(0, steps[1] - steps[0]))
}

function titleLine(title: string, steps: [number, number] | undefined, width: number): string {
  const right = (steps ? `${stepDots(steps)}   ` : "") + "esc"
  const gap = Math.max(1, width - visibleWidth(title) - visibleWidth(right))
  return paint(title, theme.text, true) + " ".repeat(gap) + paint(right, steps ? theme.primary : theme.faint)
}

export function wrap(text: string, width: number): string[] {
  const limit = Math.max(1, width)
  return text.split("\n").flatMap((line) => (line ? wrapTextWithAnsi(line, limit) : [""]))
}

function fit(line: string, width: number): string {
  return truncateToWidth(line, Math.max(1, width), "")
}

/**
 * Cut a history command so the start, an ASCII ellipsis, and the time share
 * one row. SelectList's own clamp uses an empty ellipsis, so the cut happens
 * before the row is handed over.
 */
export function historyLabel(command: string, time: string, width: number): string {
  const budget = Math.max(4, width - 4)
  const suffix = `  ${time}`
  const room = Math.max(1, budget - suffix.length)
  if (command.length <= room) return command + suffix
  const ellipsis = "..."
  const end = command.endsWith("--nfkc") ? "--nfkc" : command.slice(-Math.min(12, command.length))
  const startRoom = room - ellipsis.length - end.length
  if (startRoom < 1) return truncateToWidth(command, room, "...") + suffix
  return command.slice(0, startRoom) + ellipsis + end + suffix
}
