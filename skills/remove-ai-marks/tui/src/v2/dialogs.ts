import {
  matchesKey,
  truncateToWidth,
  visibleWidth,
  type Component,
  type OverlayHandle,
  type TUI,
} from "@earendil-works/pi-tui"
import { COMMANDS, runCommand } from "./commands"
import type { Check, DetectedEndpoint } from "./protocol"
import { endpointLabel, NO_ENDPOINT, type AppController, type Dialog } from "./store"
import { paint, theme } from "./theme"
import { historyLabel, ListDialog, wrap, type DialogItem } from "./ui/list-dialog"

export interface DialogHost {
  copy(text: string): boolean
  rows(): number
}

interface DialogView extends Component {
  shown?: () => void
}

/**
 * One showOverlay per store dialog. Dismiss with the handle, never hideOverlay:
 * focus() can put a later dialog visually in front of an earlier push.
 */
export function syncDialogOverlays(
  tui: TUI,
  ctl: AppController,
  handles: Map<Dialog, OverlayHandle>,
  host: DialogHost,
): void {
  const live = new Set(ctl.app.dialogs)
  for (const [dialog, handle] of handles) {
    if (live.has(dialog)) continue
    handle.hide()
    handles.delete(dialog)
  }
  for (const dialog of ctl.app.dialogs) {
    if (handles.has(dialog)) continue
    const view = new Scrim(buildDialog(dialog, ctl, host), dialogWidth(dialog.type), host)
    const handle = tui.showOverlay(view, { width: "100%", anchor: "center" })
    handles.set(dialog, handle)
    view.shown?.()
  }
  const top = ctl.app.dialogs.at(-1)
  const handle = top ? handles.get(top) : undefined
  if (handle && !handle.isFocused()) handle.focus()
}

function dialogWidth(type: Dialog["type"]): number {
  switch (type) {
    case "palette":
      return 64
    case "preset":
    case "welcome":
      return 70
    case "model":
    case "help":
    case "confirm":
      return 72
    case "doctor":
      return 80
    case "history":
      return 96
  }
}

function buildDialog(dialog: Dialog, ctl: AppController, host: DialogHost): DialogView {
  switch (dialog.type) {
    case "palette":
      return palette(ctl)
    case "preset":
      return preset(ctl, Boolean(dialog.onboarding))
    case "model":
      return new ModelDialog(ctl, Boolean(dialog.onboarding))
    case "welcome":
      return new WelcomeDialog(ctl)
    case "help":
      return new HelpDialog(ctl)
    case "doctor":
      return new DoctorDialog(ctl)
    case "history":
      return new HistoryDialog(ctl, host)
    case "confirm":
      return new ConfirmDialog(ctl, dialog.items, dialog.resolve)
    default: {
      const never: never = dialog
      return never
    }
  }
}

function palette(ctl: AppController): ListDialog {
  return new ListDialog({
    title: "Commands",
    filter: true,
    placeholder: "search",
    maxVisible: 16,
    items: () => COMMANDS.map((command) => ({ id: command.name, title: command.title, right: command.key ?? `/${command.name}` })),
    onSelect: (item) => {
      ctl.closeDialog()
      runCommand(ctl, item.id)
    },
    onClose: () => ctl.closeDialog(),
  })
}

function preset(ctl: AppController, onboarding: boolean): ListDialog {
  return new ListDialog({
    title: "What to remove",
    steps: onboarding ? [3, 3] : undefined,
    initial: ctl.app.state.preset,
    footer: "Labels are promises: Verifiable is counted before and after; Best-effort has no detector guarantee.",
    items: () =>
      ctl.presets().map((item) => ({
        id: item.key,
        title: item.label,
        right: item.result_class,
        detail: item.flags.length ? `${item.description}\nadds ${item.flags.join(" ")}` : item.description,
      })),
    onSelect: (item) => {
      ctl.setPreset(item.id)
      ctl.closeDialog()
      if (onboarding) void ctl.finishOnboarding(false)
    },
    onClose: () => {
      ctl.closeDialog()
      if (onboarding) void ctl.finishOnboarding(true)
    },
  })
}

/** Model escape skips the whole chain. finishOnboarding does not drop preset. */
function skipOnboarding(ctl: AppController): void {
  for (;;) {
    const top = ctl.app.dialogs.at(-1)
    if (!top || (top.type !== "welcome" && top.type !== "model" && top.type !== "preset")) break
    ctl.closeDialog()
  }
  void ctl.finishOnboarding(true)
}

function advanceFromModel(ctl: AppController, onboarding: boolean): void {
  ctl.closeDialog()
  if (onboarding && ctl.app.dialogs.at(-1)?.type !== "preset") ctl.openDialog({ type: "preset", onboarding: true })
}

class Scrim implements DialogView {
  constructor(
    private readonly inner: DialogView,
    private readonly panelWidth: number,
    private readonly host: DialogHost,
  ) {}

  shown(): void {
    this.inner.shown?.()
  }

  invalidate(): void {
    this.inner.invalidate()
  }

  handleInput(data: string): void {
    this.inner.handleInput?.(data)
  }

  render(width: number): string[] {
    const panel = Math.max(30, Math.min(this.panelWidth, width))
    const body = this.inner.render(panel).map((line) => place(line, panel, width))
    const height = Math.max(body.length, this.host.rows())
    const top = Math.max(0, Math.floor((height - body.length) / 2))
    const blank = " ".repeat(Math.max(1, width))
    const lines = Array.from({ length: height }, () => blank)
    for (let i = 0; i < body.length && top + i < lines.length; i++) lines[top + i] = body[i]!
    return lines
  }
}

function place(line: string, panel: number, width: number): string {
  const inner = truncateToWidth(line, panel, "")
  const left = Math.max(0, Math.floor((width - panel) / 2))
  const placed = " ".repeat(left) + inner
  return placed + " ".repeat(Math.max(0, width - visibleWidth(placed)))
}

class ModelDialog implements DialogView {
  private manual = false
  private picker: ListDialog | null = null
  private readonly servers: ListDialog
  private readonly form: ManualForm

  constructor(
    private readonly ctl: AppController,
    private readonly onboarding: boolean,
  ) {
    this.form = new ManualForm(
      ctl,
      () => advanceFromModel(ctl, onboarding),
      () => {
        this.manual = false
      },
    )
    this.servers = new ListDialog({
      title: "A model for rewriting",
      steps: onboarding ? [2, 3] : undefined,
      empty: "Scanning local servers…",
      footer:
        "Only the LLM rewrite needs a model. The scan asks this machine's own servers for their model list. It never sends a document and never contacts another machine.",
      items: () => serverItems(ctl, onboarding),
      onSelect: (item) => this.select(item),
      onClose: () => this.close(),
    })
  }

  shown(): void {
    if (!this.ctl.app.endpoints && !this.ctl.app.scanning) void this.ctl.detect()
  }

  invalidate(): void {
    this.servers.invalidate()
    this.picker?.invalidate()
    this.form.invalidate()
  }

  handleInput(data: string): void {
    if (this.manual) {
      this.form.handleInput(data)
      return
    }
    if (this.picker) {
      this.picker.handleInput(data)
      return
    }
    this.servers.handleInput(data)
  }

  render(width: number): string[] {
    if (this.manual) return this.form.render(width)
    if (this.picker) return this.picker.render(width)
    return this.servers.render(width)
  }

  private close(): void {
    if (this.onboarding) skipOnboarding(this.ctl)
    else this.ctl.closeDialog()
  }

  private select(item: DialogItem): void {
    if (item.id === "rescan") {
      void this.ctl.detect()
      return
    }
    if (item.id === "manual") {
      this.manual = true
      return
    }
    if (item.id === "none") {
      this.ctl.setEndpoint({ ...NO_ENDPOINT })
      advanceFromModel(this.ctl, this.onboarding)
      return
    }
    const endpoint = (this.ctl.app.endpoints ?? []).find((found) => `${found.backend}|${found.base_url}` === item.id)
    if (!endpoint || !endpoint.reachable) return
    if (endpoint.models.length > 1) {
      this.picker = new ListDialog({
        title: `Models on ${endpoint.label}`,
        filter: true,
        initial: this.ctl.app.state.endpoint.model ?? undefined,
        items: () => endpoint.models.map((model) => ({ id: model, title: model })),
        onSelect: (choice) => this.choose(endpoint, choice.id),
        onClose: () => {
          this.picker = null
        },
      })
      return
    }
    this.choose(endpoint, endpoint.models[0] ?? null)
  }

  private choose(endpoint: DetectedEndpoint, model: string | null): void {
    this.ctl.setEndpoint({ backend: endpoint.backend, base_url: endpoint.base_url, model })
    this.ctl.toast(`Using ${model ?? endpoint.base_url}.`, "success")
    advanceFromModel(this.ctl, this.onboarding)
  }
}

function serverItems(ctl: AppController, onboarding: boolean): DialogItem[] {
  const found = ctl.app.endpoints ?? []
  const current = endpointLabel(ctl.app.state.endpoint)
  return [
    ...found.map((endpoint) => ({
      id: `${endpoint.backend}|${endpoint.base_url}`,
      title: `${endpoint.label}  ${endpoint.base_url.replace(/^https?:\/\//, "")}`,
      right: endpoint.reachable ? `${endpoint.models.length} model${endpoint.models.length === 1 ? "" : "s"}` : "Not running",
      disabled: !endpoint.reachable,
    })),
    {
      id: "rescan",
      title: ctl.app.scanning ? "Scanning…" : "Scan again",
      right: "This machine only",
      disabled: ctl.app.scanning,
    },
    { id: "manual", title: "Enter a server", right: "Any URL" },
    {
      id: "none",
      title: onboarding ? "Skip, clean hidden marks only" : "No model",
      right: current ? `Now ${current}` : "Rewrite off",
    },
  ]
}

class ManualForm implements Component {
  private field = 0
  private backend: string
  private url: string
  private model: string
  private error: string | null = null

  constructor(
    private readonly ctl: AppController,
    private readonly onDone: () => void,
    private readonly onBack: () => void,
  ) {
    const current = ctl.app.state.endpoint
    const backends = ctl.app.hello?.backends ?? []
    this.backend = current.backend ?? backends[0] ?? ""
    this.url = current.base_url ?? ""
    this.model = current.model ?? ""
  }

  invalidate(): void {}

  handleInput(data: string): void {
    if (matchesKey(data, "escape")) {
      this.onBack()
      return
    }
    if (matchesKey(data, "tab") || matchesKey(data, "down")) {
      this.field = (this.field + 1) % 3
      return
    }
    if (matchesKey(data, "up")) {
      this.field = (this.field + 2) % 3
      return
    }
    if (this.field === 0 && (matchesKey(data, "left") || matchesKey(data, "right"))) {
      this.cycle(matchesKey(data, "left") ? -1 : 1)
      return
    }
    if (data === "\r") {
      if (this.field === 0) this.field = 1
      else if (this.field === 1) this.field = 2
      else this.submit()
      return
    }
    if (this.field === 0) return
    if (matchesKey(data, "backspace") || data === "\x7f" || data === "\x08") {
      if (this.field === 1) this.url = this.url.slice(0, -1)
      else this.model = this.model.slice(0, -1)
      this.error = null
      return
    }
    if (data.length === 1 && data >= " " && data <= "~") {
      if (this.field === 1) this.url += data
      else this.model += data
      this.error = null
    }
  }

  render(width: number): string[] {
    const backends = this.ctl.app.hello?.backends ?? []
    const lines = [
      heading("Enter a server", width),
      label("Backend", this.field === 0),
      backends.map((name) => (name === this.backend ? `● ${name}` : `○ ${name}`)).join("  "),
      label("Base URL", this.field === 1),
      this.url || "http://127.0.0.1:11434",
      label("Model", this.field === 2),
      this.model || "e.g. qwen3:14b",
      "A host other than this machine is saved as typed; each clean asks before any text leaves.",
      "tab next   ← → backend   enter save   esc back",
    ]
    if (this.error) lines.push(this.error)
    return lines.flatMap((line) => wrap(line, width))
  }

  private cycle(step: number): void {
    const list = this.ctl.app.hello?.backends ?? []
    if (!list.length) return
    const at = Math.max(0, list.indexOf(this.backend))
    this.backend = list[(at + step + list.length) % list.length]!
  }

  private submit(): void {
    const url = this.url.trim().replace(/\/+$/, "")
    if (!/^https?:\/\/[^/\s]+/.test(url)) {
      this.error = "base URL needs http:// or https:// and a host"
      return
    }
    this.ctl.setEndpoint({ backend: this.backend, base_url: url, model: this.model.trim() || null })
    this.onDone()
  }
}

class WelcomeDialog implements DialogView {
  constructor(private readonly ctl: AppController) {}
  invalidate(): void {}

  handleInput(data: string): void {
    if (data === "\r") {
      this.ctl.closeDialog()
      this.ctl.openDialog({ type: "preset", onboarding: true })
      this.ctl.openDialog({ type: "model", onboarding: true })
      return
    }
    if (matchesKey(data, "escape")) void this.ctl.finishOnboarding(true)
  }

  render(width: number): string[] {
    return panel(width, "Welcome to wm", [1, 3], [
      "wm finds the watermarks and provenance marks hidden in files you own, removes them, and tells you how sure it is.",
      "Verifiable   Counted before and after the clean",
      "Best-effort  No detector can confirm the result",
      "Removing hidden characters and metadata is Verifiable and needs nothing installed. Next, pick an optional local model for rewriting, then the default clean.",
      "enter continue   esc skip setup",
    ])
  }
}

class HelpDialog implements DialogView {
  constructor(private readonly ctl: AppController) {}
  invalidate(): void {}

  handleInput(data: string): void {
    if (matchesKey(data, "escape") || data === "\r" || matchesKey(data, "f1")) this.ctl.closeDialog()
  }

  render(width: number): string[] {
    const rows = COMMANDS.map((command) => `${command.key ?? `/${command.name}`}  ${command.title}`)
    return panel(width, "Help", undefined, [
      "enter  add the path, --flag or /command typed in the prompt",
      "↑ ↓  move through files",
      "ctrl+p  search every command",
      ...rows,
      "Any flag `wm` accepts works in the prompt: `--nfkc`, `--glob *.md`, `-o out/`. A bare flag typed again turns it off. The command in the footer is the exact `wm` run, so a clean here can be repeated anywhere.",
      `settings  ${this.ctl.app.hello?.settings_path ?? ""}`,
      "API key   WATERMARKS_REWRITE_API_KEY, read at run time, never shown or saved",
    ])
  }
}

class DoctorDialog implements DialogView {
  constructor(private readonly ctl: AppController) {}
  invalidate(): void {}
  shown(): void {
    void this.ctl.loadChecks()
  }

  handleInput(data: string): void {
    if (matchesKey(data, "escape") || data === "\r") this.ctl.closeDialog()
  }

  render(width: number): string[] {
    const checks = this.ctl.app.checks
    const body = checks ? checks.flatMap((check) => checkLines(check)) : ["checking…"]
    return panel(width, "This install", undefined, body)
  }
}

function checkLines(check: Check): string[] {
  const mark = check.good ? "✓" : "·"
  const lines = [`${mark} ${check.name}  ${check.state}.  ${check.detail}`]
  if (check.fix) lines.push(`   ${check.fix}`)
  return lines
}

class HistoryDialog implements DialogView {
  private width = 96
  private readonly list: ListDialog

  constructor(
    private readonly ctl: AppController,
    host: DialogHost,
  ) {
    this.list = new ListDialog({
      title: "History",
      wide: true,
      empty: "Nothing cleaned yet in this session.",
      footer: "Enter copies the command. Some terminals ignore the copy.",
      items: () =>
        this.ctl.app.history.map((entry, index) => ({
          id: String(index),
          title: historyLabel(entry.command, entry.time, this.width),
          detail: entry.summary,
        })),
      onSelect: (item) => {
        const entry = this.ctl.app.history[Number(item.id)]
        if (!entry) return
        const ok = host.copy(entry.command)
        this.ctl.toast(ok ? "Command copied." : "This terminal did not accept the copy.", ok ? "success" : "error")
      },
      onClose: () => this.ctl.closeDialog(),
    })
  }

  shown(): void {
    void this.ctl.loadHistory()
  }

  invalidate(): void {
    this.list.invalidate()
  }

  handleInput(data: string): void {
    this.list.handleInput(data)
  }

  render(width: number): string[] {
    this.width = width
    return this.list.render(width)
  }
}

class ConfirmDialog implements DialogView {
  private answered = false

  constructor(
    private readonly ctl: AppController,
    private readonly items: { message: string }[],
    private readonly resolve: (ok: boolean) => void,
  ) {}

  invalidate(): void {}

  handleInput(data: string): void {
    if (data === "\r" || data === "y" || data === "Y") this.answer(true)
    else if (matchesKey(data, "escape") || data === "n" || data === "N") this.answer(false)
  }

  render(width: number): string[] {
    return panel(width, "Before anything is written", undefined, [
      ...this.items.map((item) => `! ${item.message}`),
      "enter go ahead   esc cancel",
    ])
  }

  private answer(ok: boolean): void {
    if (this.answered) return
    this.answered = true
    this.ctl.closeDialog()
    this.resolve(ok)
  }
}

function heading(title: string, width: number): string {
  const right = "esc"
  const gap = Math.max(1, width - title.length - right.length)
  return paint(title, theme.text, true) + " ".repeat(gap) + paint(right, theme.faint)
}

function label(text: string, active: boolean): string {
  return paint(text, active ? theme.primary : theme.muted)
}

function panel(width: number, title: string, steps: [number, number] | undefined, body: string[]): string[] {
  const right = (steps ? `${"●".repeat(steps[0])}${"○".repeat(Math.max(0, steps[1] - steps[0]))}   ` : "") + "esc"
  const gap = Math.max(1, width - title.length - right.length)
  const head = paint(title, theme.text, true) + " ".repeat(gap) + paint(right, steps ? theme.primary : theme.faint)
  return [head, ...body.flatMap((line) => wrap(line, width))]
}
