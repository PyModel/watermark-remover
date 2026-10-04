import { afterEach, expect, test } from "bun:test"
import { isViewportTUI, stripTerminalSequences, TuiAltScreen } from "@earendil-works/pi-tui"
import { COMMANDS } from "../src/v2/commands"
import { mountApp, type MountedApp } from "../src/v2/app"
import { BridgeError } from "../src/v2/protocol"
import { ListDialog, type DialogItem } from "../src/v2/ui/list-dialog"
import { FakeBridge } from "./fake-bridge"
import { FakeTerminal } from "./fake-terminal"

/**
 * The shared fake throws the pre-v2 BridgeError. The v2 store matches its own
 * class, which is what v2/bridge.ts rejects with. Translate only that error.
 */
class DialogBridge extends FakeBridge {
  override async request<T>(method: string, params: any = {}, onEvent?: (event: any) => void): Promise<T> {
    try {
      return await super.request(method, params, onEvent)
    } catch (error) {
      if (isNeedsConfirm(error)) throw new BridgeError(error.code, error.message, error.confirm)
      throw error
    }
  }
}

function isNeedsConfirm(error: unknown): error is { code: string; message: string; confirm: { kind: string; message: string }[] } {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as { code?: string }).code === "needs_confirm" &&
    "confirm" in error &&
    Array.isArray((error as { confirm?: unknown }).confirm)
  )
}

const key = {
  ctrlP: "\x10",
  ctrlR: "\x12",
  enter: "\r",
  esc: "\x1b",
  up: "\x1b[A",
  down: "\x1b[B",
  right: "\x1b[C",
  tab: "\t",
  f1: "\x1bOP",
  pageDown: "\x1b[6~",
  backspace: "\x7f",
}

let shutdown: (() => void) | undefined

afterEach(() => {
  shutdown?.()
  shutdown = undefined
})

function items(): DialogItem[] {
  return [
    { id: "stop", title: "Stop after this file", right: "esc" },
    { id: "check", title: "Check this install", right: "/doctor" },
    { id: "help", title: "Help", right: "f1" },
    { id: "dead", title: "The hidden one", right: "Not running", disabled: true },
  ]
}

test("list dialog render(width) filters in fuzzy rank, moves, refuses disabled, and escapes", () => {
  const selected: string[] = []
  let closed = 0
  const dialog = new ListDialog({
    title: "Commands",
    steps: [2, 3],
    filter: true,
    items,
    onSelect: (item) => selected.push(item.id),
    onClose: () => {
      closed += 1
    },
  })
  const open = dialog.render(64).join("\n")
  expect(open).toContain("Commands")
  expect(open).toContain("●●○")
  expect(open).toContain("Stop after this file")
  expect(open).toContain("esc")
  dialog.handleInput("h")
  dialog.handleInput("e")
  const filtered = dialog.render(64).join("\n")
  const helpAt = filtered.indexOf("Help")
  const checkAt = filtered.indexOf("Check this install")
  const stopAt = filtered.indexOf("Stop after this file")
  expect(helpAt).toBeGreaterThan(-1)
  expect(checkAt).toBeGreaterThan(helpAt)
  expect(stopAt).toBeGreaterThan(checkAt)
  expect(filtered).toContain("The hidden one")
  dialog.handleInput(key.down)
  dialog.handleInput(key.down)
  const moved = dialog.render(64).join("\n")
  expect(moved.split("\n").find((line) => line.includes("→"))).toContain("The hidden one")
  dialog.handleInput(key.enter)
  expect(selected).toEqual([])
  dialog.handleInput(key.up)
  dialog.handleInput(key.up)
  dialog.handleInput(key.enter)
  expect(selected).toEqual(["help"])
  dialog.handleInput(key.esc)
  expect(closed).toBe(1)
  const detail = new ListDialog({
    title: "What to remove",
    items: () => [{ id: "hidden", title: "Hidden marks", detail: "Zero-width carriers", right: "Verifiable" }],
    onSelect: () => {},
    onClose: () => {},
  })
  expect(detail.render(70).join("\n")).toContain("Zero-width carriers")
})

interface Boot {
  term: FakeTerminal
  tui: TuiAltScreen
  bridge: FakeBridge
  mounted: MountedApp
  screen: () => string
  type: (text: string) => Promise<void>
  press: (data: string) => void
  stop: () => void
  quits: () => number
}

async function boot(columns = 100, rows = 40, configure?: (bridge: FakeBridge) => void): Promise<Boot> {
  const bridge = new DialogBridge()
  configure?.(bridge)
  const term = new FakeTerminal(columns, rows)
  const tui = new TuiAltScreen(term, false)
  if (!isViewportTUI(tui)) throw new Error("expected a viewport TUI")
  let quitCount = 0
  const mounted = mountApp(tui, bridge, () => {
    quitCount += 1
  })
  tui.start()
  await mounted.ctl.start()
  tui.renderNow()
  shutdown = () => {
    mounted.dispose()
    tui.stop({ preserveScreen: true })
  }
  const screen = () => stripTerminalSequences(tui.getScreenLines().join("\n"))
  const type = async (text: string) => {
    for (const ch of text) term.input(ch)
    term.input(key.enter)
    await new Promise((resolve) => setTimeout(resolve, 160))
    tui.renderNow()
  }
  const press = (data: string) => {
    term.input(data)
    tui.renderNow()
  }
  const stop = () => {
    shutdown?.()
    shutdown = undefined
  }
  return { term, tui, bridge, mounted, screen, type, press, stop, quits: () => quitCount }
}

async function until(tui: TuiAltScreen, read: () => string, needle: string) {
  for (let i = 0; i < 30; i++) {
    tui.renderNow()
    if (read().includes(needle)) return
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
  throw new Error(`timed out waiting for ${needle}`)
}

test("ctrl+p opens the palette, filters in rank order, moves, runs a command, and escapes", async () => {
  const { screen, press, mounted } = await boot(100, 40)
  press(key.ctrlP)
  const open = screen()
  expect(open).toContain("Commands")
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["palette"])
  for (const command of COMMANDS) {
    const titleHits = open.split(command.title).length - 1
    expect(titleHits).toBe(1)
    const row = open.split("\n").find((line) => line.includes(command.title))
    expect(row).toContain(command.key ?? `/${command.name}`)
  }
  press("h")
  press("e")
  const filtered = screen()
  expect(filtered.indexOf("Help")).toBeLessThan(filtered.indexOf("Check this install"))
  expect(filtered.indexOf("Check this install")).toBeLessThan(filtered.indexOf("Stop after this file"))
  press(key.down)
  expect(screen().split("\n").find((line) => line.includes("→"))).toContain("Check this install")
  press(key.esc)
  expect(mounted.ctl.app.dialogs).toEqual([])
  expect(screen()).not.toContain("Commands")
  press(key.ctrlP)
  for (const ch of "help") press(ch)
  press(key.enter)
  expect(screen()).toContain("Any flag `wm` accepts")
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["help"])
  press(key.esc)
  expect(screen()).not.toContain("Any flag `wm` accepts")
})

test("preset moves, shows the detail line, selects, and escape cancels", async () => {
  const { screen, type, press, mounted } = await boot()
  await type("/preset")
  expect(screen()).toContain("What to remove")
  expect(screen()).toContain("Zero-width carriers")
  press(key.down)
  expect(screen()).toContain("Adds NFKC")
  press(key.enter)
  expect(mounted.ctl.app.state.preset).toBe("hidden-aggressive")
  expect(mounted.ctl.app.dialogs).toEqual([])
  await type("/preset")
  press(key.esc)
  expect(mounted.ctl.app.dialogs).toEqual([])
  expect(mounted.ctl.app.state.preset).toBe("hidden-aggressive")
})

test("first run stacks model over preset and closing either lands in today's next state", async () => {
  const { screen, press, mounted, bridge, tui } = await boot(100, 30, (b) => {
    b.onboard = true
  })
  expect(screen()).toContain("Welcome to wm")
  expect(screen()).toContain("●○○")
  press(key.esc)
  await until(tui, screen, "Setup skipped")
  expect(mounted.ctl.app.dialogs).toEqual([])
  expect(bridge.calls.some((call) => call.method === "save_settings")).toBe(true)
  mounted.ctl.openDialog({ type: "welcome" })
  tui.renderNow()
  press(key.enter)
  await until(tui, screen, "Ollama")
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["preset", "model"])
  expect(screen()).toContain("A model for rewriting")
  expect(screen()).toContain("●●○")
  press(key.esc)
  await until(tui, screen, "Setup skipped")
  expect(mounted.ctl.app.dialogs).toEqual([])
  mounted.ctl.openDialog({ type: "welcome" })
  tui.renderNow()
  press(key.enter)
  await until(tui, screen, "Ollama")
  press(key.enter)
  expect(screen()).toContain("Models on Ollama")
  press(key.esc)
  expect(screen()).toContain("A model for rewriting")
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["preset", "model"])
  press(key.down)
  press(key.enter)
  expect(screen()).toContain("A model for rewriting")
  expect(mounted.ctl.app.state.endpoint.model).toBeNull()
  for (let i = 0; i < 3; i++) press(key.down)
  press(key.enter)
  expect(screen()).toContain("What to remove")
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["preset"])
  expect(screen()).toContain("●●●")
  press(key.enter)
  await until(tui, screen, "Setup saved")
  expect(mounted.ctl.app.dialogs).toEqual([])
  expect(mounted.ctl.app.state.preset).toBe("hidden")
  for (const save of bridge.calls.filter((call) => call.method === "save_settings")) {
    expect(JSON.stringify(save.params)).not.toMatch(/api_key|token|secret/i)
  }
})

test("onboarding choose walks welcome, model picker, and preset and saves the model", async () => {
  const { screen, press, mounted, bridge, tui } = await boot(100, 30, (b) => {
    b.onboard = true
  })
  press(key.enter)
  await until(tui, screen, "Ollama")
  press(key.enter)
  expect(screen()).toContain("Models on Ollama")
  press("l")
  press("l")
  expect(screen()).toContain("llama3.1:8b")
  expect(screen()).not.toContain("qwen3:14b")
  press(key.backspace)
  press(key.backspace)
  press(key.enter)
  expect(screen()).toContain("What to remove")
  press(key.enter)
  await until(tui, screen, "qwen3:14b")
  expect(mounted.ctl.app.dialogs).toEqual([])
  expect(mounted.ctl.app.state.endpoint.model).toBe("qwen3:14b")
  for (const save of bridge.calls.filter((call) => call.method === "save_settings")) {
    expect(JSON.stringify(save.params)).not.toMatch(/api_key|token|secret/i)
  }
})

test("a server entered by hand saves backend, base URL and model", async () => {
  const { screen, type, press, mounted, bridge, tui } = await boot()
  await type("/model")
  await until(tui, screen, "Ollama")
  for (let i = 0; i < 3; i++) press(key.down)
  press(key.enter)
  expect(screen()).toContain("Enter a server")
  press(key.right)
  press(key.tab)
  for (const ch of "http://127.0.0.1:8080/") press(ch)
  press(key.tab)
  for (const ch of "local-model") press(ch)
  press(key.enter)
  await until(tui, () => JSON.stringify(mounted.ctl.app.state.endpoint), "openai-compatible")
  expect(mounted.ctl.app.state.endpoint).toEqual({
    backend: "openai-compatible",
    base_url: "http://127.0.0.1:8080",
    model: "local-model",
  })
  const saved = bridge.calls.filter((call) => call.method === "save_settings").at(-1)!.params.settings
  expect(Object.keys(saved).sort()).toEqual(["endpoint", "preset"])
  expect(Object.keys(saved.endpoint).sort()).toEqual(["backend", "base_url", "model"])
})

test("a hand-entered URL without a host is refused", async () => {
  const { screen, type, press, mounted, tui } = await boot()
  await type("/model")
  await until(tui, screen, "Ollama")
  for (let i = 0; i < 3; i++) press(key.down)
  press(key.enter)
  press(key.tab)
  for (const ch of "notaurl") press(ch)
  press(key.tab)
  press(key.enter)
  expect(screen()).toContain("base URL needs http:// or https:// and a host")
  expect(mounted.ctl.app.state.endpoint.base_url).toBeNull()
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["model"])
})

test("help lists every command and f1, enter, and escape close it", async () => {
  const { screen, press, mounted } = await boot(120, 60)
  press(key.f1)
  const help = screen()
  expect(help).toContain("Help")
  expect(help).toContain("Any flag `wm` accepts")
  for (const command of COMMANDS) expect(help).toContain(command.title)
  press(key.f1)
  expect(mounted.ctl.app.dialogs).toEqual([])
  press(key.f1)
  press(key.enter)
  expect(mounted.ctl.app.dialogs).toEqual([])
  press(key.f1)
  press(key.esc)
  expect(screen()).not.toContain("Any flag `wm` accepts")
})

test("the install check shows each detail in full", async () => {
  const { screen, type, press, mounted, tui } = await boot(120, 40)
  await type("/doctor")
  await until(tui, screen, "standard library alone.")
  const text = screen().replace(/\s+/g, " ")
  expect(text).toContain("work with the standard library alone.")
  expect(text).toContain('pip install "watermark-remover[visible]"')
  press(key.esc)
  expect(mounted.ctl.app.dialogs).toEqual([])
})

test("a long history command is cut with an ellipsis before its time and enter copies it", async () => {
  const { screen, type, press, term, mounted, tui } = await boot(120, 40)
  await type("/history")
  await until(tui, screen, "20:14:56")
  const row = screen().split("\n").find((line) => line.includes("20:14:56"))!
  expect(row).toContain("wm drafts --rewrite paraphrase")
  expect(row).toContain("...")
  expect(row).toContain("--nfkc  20:14:56")
  press(key.enter)
  expect(mounted.ctl.app.toast?.text).toBe("Command copied.")
  expect(term.writes.some((chunk) => chunk.startsWith("\x1b]52;c;"))).toBe(true)
  press(key.esc)
  expect(mounted.ctl.app.dialogs).toEqual([])
})

test("a needed confirmation lists the gates; decline writes nothing, accept retries those kinds", async () => {
  const gate = [{ kind: "remote", message: "Text will be sent to 10.0.0.5" }]
  const declined = await boot(120, 40, (b) => {
    b.confirm = gate
  })
  await declined.type("drafts")
  declined.press(key.ctrlR)
  await until(declined.tui, declined.screen, "Before anything is written")
  expect(declined.screen()).toContain("10.0.0.5")
  declined.press(key.esc)
  await until(declined.tui, declined.screen, "Nothing was written.")
  expect(declined.bridge.calls.filter((call) => call.method === "clean")).toHaveLength(1)
  expect(declined.mounted.ctl.app.dialogs).toEqual([])
  declined.stop()

  let release: () => void = () => {}
  const accepted = await boot(120, 40, (b) => {
    b.confirm = gate
    b.cleanGate = new Promise((resolve) => {
      release = resolve
    })
  })
  await accepted.type("drafts")
  accepted.press(key.ctrlR)
  await until(accepted.tui, accepted.screen, "10.0.0.5")
  accepted.press("y")
  await until(accepted.tui, () => String(accepted.mounted.ctl.app.run != null), "true")
  expect(accepted.mounted.ctl.app.run).not.toBeNull()
  const retried = accepted.bridge.calls.filter((call) => call.method === "clean")
  expect(retried).toHaveLength(2)
  expect(retried[1]!.params.confirmed).toEqual(["remote"])
  release()
  await until(accepted.tui, () => String(accepted.mounted.ctl.app.run == null), "true")
  expect(accepted.mounted.ctl.app.run).toBeNull()
})

test("resolving the confirm dialog directly keeps the retried clean stoppable", async () => {
  let release: () => void = () => {}
  const { bridge, mounted, type, press, tui } = await boot(120, 40, (b) => {
    b.confirm = [{ kind: "remote", message: "Text will be sent to 10.0.0.5" }]
    b.cleanGate = new Promise((resolve) => {
      release = resolve
    })
  })
  await type("drafts")
  press(key.ctrlR)
  await until(tui, () => mounted.ctl.app.dialogs.at(-1)?.type ?? "", "confirm")
  const dialog = mounted.ctl.app.dialogs.at(-1)
  if (dialog?.type !== "confirm") throw new Error("expected the confirm dialog")
  dialog.resolve(true)
  await until(tui, () => String(mounted.ctl.app.run != null), "true")
  expect(mounted.ctl.app.run).not.toBeNull()
  release()
  await until(tui, () => String(mounted.ctl.app.run == null), "true")
  expect(mounted.ctl.app.run).toBeNull()
  expect(bridge.calls.filter((call) => call.method === "clean")).toHaveLength(2)
})

test("page down moves the palette selection by a page", async () => {
  const { screen, press } = await boot(100, 40)
  press(key.ctrlP)
  press(key.pageDown)
  const marked = screen().split("\n").find((line) => line.includes("→")) ?? ""
  expect(marked).not.toContain("Clean")
})

/** CSI-u enter (kitty flags 7). CR is \r; this is the form Ghostty/kitty send. */
const KITTY_ENTER = "\x1b[13u"

test("kitty CSI-u enter runs the palette selection", async () => {
  const { screen, press, mounted } = await boot(100, 40)
  press(key.ctrlP)
  for (const ch of "help") press(ch)
  press(KITTY_ENTER)
  expect(mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["help"])
  expect(screen()).toContain("Any flag `wm` accepts")
})

/** theme.bg #0f1219 as a truecolor background, independent of styleText. */
const THEME_BG = "\x1b[48;2;15;18;25m"

test("dialog scrim paints the theme background instead of bare spaces", async () => {
  const { tui, press } = await boot(100, 40)
  press(key.ctrlP)
  const raw = tui.getScreenLines()
  const scrim = raw.find((line) => stripTerminalSequences(line).trim() === "")
  expect(scrim).toBeDefined()
  expect(scrim).toContain(THEME_BG)
  const titled = raw.find((line) => stripTerminalSequences(line).includes("Commands"))
  expect(titled).toContain(THEME_BG)
})

test("kitty CSI-u enter continues, closes, confirms, and saves the other dialogs", async () => {
  const welcome = await boot(100, 30, (b) => {
    b.onboard = true
  })
  welcome.press(KITTY_ENTER)
  expect(welcome.mounted.ctl.app.dialogs.map((dialog) => dialog.type)).toEqual(["preset", "model"])
  welcome.stop()

  const help = await boot(120, 60)
  help.press(key.f1)
  help.press(KITTY_ENTER)
  expect(help.mounted.ctl.app.dialogs).toEqual([])
  help.stop()

  const doctor = await boot(120, 40)
  await doctor.type("/doctor")
  await until(doctor.tui, doctor.screen, "standard library alone.")
  doctor.press(KITTY_ENTER)
  expect(doctor.mounted.ctl.app.dialogs).toEqual([])
  doctor.stop()

  let release: () => void = () => {}
  const confirm = await boot(120, 40, (b) => {
    b.confirm = [{ kind: "remote", message: "Text will be sent to 10.0.0.5" }]
    b.cleanGate = new Promise((resolve) => {
      release = resolve
    })
  })
  await confirm.type("drafts")
  confirm.press(key.ctrlR)
  await until(confirm.tui, confirm.screen, "Before anything is written")
  confirm.press(KITTY_ENTER)
  await until(confirm.tui, () => String(confirm.mounted.ctl.app.run != null), "true")
  expect(confirm.bridge.calls.filter((call) => call.method === "clean")).toHaveLength(2)
  release()
  await until(confirm.tui, () => String(confirm.mounted.ctl.app.run == null), "true")
  confirm.stop()

  const manual = await boot()
  await manual.type("/model")
  await until(manual.tui, manual.screen, "Ollama")
  for (let i = 0; i < 3; i++) manual.press(key.down)
  manual.press(key.enter)
  manual.press(key.right)
  manual.press(KITTY_ENTER)
  for (const ch of "http://127.0.0.1:8080") manual.press(ch)
  manual.press(key.tab)
  for (const ch of "local-model") manual.press(ch)
  manual.press(KITTY_ENTER)
  await until(manual.tui, () => JSON.stringify(manual.mounted.ctl.app.state.endpoint), "local-model")
  expect(manual.mounted.ctl.app.state.endpoint).toEqual({
    backend: "openai-compatible",
    base_url: "http://127.0.0.1:8080",
    model: "local-model",
  })
})

test("escaping welcome closes its overlay before the save, and a second enter stacks nothing", async () => {
  const { screen, press, mounted, tui, bridge } = await boot(100, 30, (b) => {
    b.onboard = true
  })
  expect(screen()).toContain("Welcome to wm")
  expect(tui.hasOverlay()).toBe(true)
  press(key.esc)
  // Stay synchronous. The save is already requested; its toast is a microtask.
  expect(bridge.calls.some((call) => call.method === "save_settings")).toBe(true)
  expect(screen()).not.toContain("Setup skipped")
  expect(tui.hasOverlay()).toBe(false)
  expect(tui.getFocusedComponent()).toBe(mounted.editor)
  press(key.enter)
  expect(mounted.ctl.app.dialogs).toEqual([])
  expect(tui.hasOverlay()).toBe(false)
  expect(screen()).not.toContain("A model for rewriting")
})
