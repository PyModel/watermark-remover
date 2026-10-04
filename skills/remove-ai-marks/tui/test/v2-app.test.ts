import { afterEach, expect, test } from "bun:test"
import { isViewportTUI, stripTerminalSequences, TuiAltScreen, type Terminal } from "@earendil-works/pi-tui"
import { mountApp, type MountedApp } from "../src/v2/app"
import { renderEmpty } from "../src/v2/components/layout"
import { FakeBridge } from "./fake-bridge"
import { FakeTerminal } from "./fake-terminal"

let shutdown: (() => void) | undefined

afterEach(() => {
  shutdown?.()
  shutdown = undefined
})

test("fake terminal implements the full interface", async () => {
  const term = new FakeTerminal(80, 24)
  const asTerm: Terminal = term
  expect(asTerm.columns).toBe(80)
  expect(asTerm.rows).toBe(24)
  expect(asTerm.kittyProtocolActive).toBe(false)
  asTerm.setTitle("wm")
  asTerm.setProgress(true)
  asTerm.hideCursor()
  asTerm.showCursor()
  asTerm.clearLine()
  asTerm.clearFromCursor()
  asTerm.clearScreen()
  asTerm.moveBy(1)
  asTerm.write("x")
  await asTerm.drainInput(1, 1)
  asTerm.start(() => {}, () => {})
  asTerm.stop()
  expect(term.titles).toEqual(["wm"])
  expect(term.progress).toEqual([true])
  expect(term.writes).toEqual(["\x1b]0;wm\x07", "\x1b]9;4;3\x07", "x"])
})

test("empty layout render(width) names the one thing to do", () => {
  for (const width of [40, 80, 120]) {
    expect(renderEmpty(width).join("\n")).toContain("Type a file or folder below")
  }
})

interface Boot {
  term: FakeTerminal
  tui: TuiAltScreen
  bridge: FakeBridge
  mounted: MountedApp
  screen: () => string
  type: (text: string) => Promise<void>
  quits: () => number
}

async function boot(columns: number, rows: number, bridge = new FakeBridge()): Promise<Boot> {
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
    term.input("\r")
    await new Promise((resolve) => setTimeout(resolve, 150))
    tui.renderNow()
  }
  return { term, tui, bridge, mounted, screen, type, quits: () => quitCount }
}

for (const size of [
  [80, 24],
  [120, 40],
] as const) {
  test(`empty state at ${size[0]}x${size[1]} shows the one thing to do`, async () => {
    const { screen } = await boot(size[0], size[1])
    const frame = screen()
    expect(frame).toContain("Type a file or folder below")
    expect(frame).toContain("Hidden marks")
    expect(frame).toContain("Verifiable")
    expect(frame).toContain("No model")
  })
}

test("a path refreshes the plan and lists files", async () => {
  const { bridge, screen, type, mounted } = await boot(80, 24)
  await type("drafts")
  expect(mounted.ctl.app.state.paths).toContain("drafts")
  expect(bridge.calls.some((call) => call.method === "plan" && call.params.state.paths.includes("drafts"))).toBe(true)
  const frame = screen()
  expect(frame).toContain("5 files")
  expect(frame).toContain("intro.md")
  expect(frame).toContain("Space marks one")
})

test("a flag typed in the prompt is added, and typed again is removed", async () => {
  const { screen, type, mounted } = await boot(80, 24)
  await type("--nfkc")
  expect(mounted.ctl.app.state.flags).toEqual(["--nfkc"])
  expect(screen()).toContain("--nfkc")
  await type("--nfkc")
  expect(mounted.ctl.app.state.flags).toEqual([])
  expect(screen()).not.toContain("--nfkc")
})

test("an invalid flag shows the parser message and keeps the file list", async () => {
  const { screen, type } = await boot(80, 24)
  await type("drafts")
  await type("--bogus")
  const frame = screen()
  expect(frame).toContain("unrecognized arguments: --bogus")
  expect(frame).toContain("5 files")
})

test("/tmp is a path, /clear clears, /preset routes, /quit quits", async () => {
  const { screen, type, mounted, quits, bridge } = await boot(100, 30)
  await type("/tmp")
  expect(mounted.ctl.app.state.paths).toContain("/tmp")
  await type("/clear")
  expect(mounted.ctl.app.state.paths).toEqual([])
  await type("/preset rewrite")
  expect(mounted.ctl.app.state.preset).toBe("rewrite")
  expect(screen()).toContain("Needs a model")
  expect(screen()).toContain("Best-effort")
  await type("drafts")
  mounted.ctl.markFile()
  await type("/clean")
  expect(bridge.calls.some((call) => call.method === "clean")).toBe(true)
  await type("/quit")
  expect(quits()).toBe(1)
})

test("ctrl+j selects the next file and does not insert a newline", async () => {
  const { term, tui, type, mounted, screen } = await boot(80, 24)
  await type("drafts")
  expect(mounted.editor.getText()).toBe("")
  term.input("\n")
  tui.renderNow()
  expect(mounted.ctl.app.selected).toBe(1)
  expect(mounted.editor.getText()).toBe("")
  expect(screen()).toContain("chapter-01.md")
  term.input("\x0b")
  tui.renderNow()
  expect(mounted.ctl.app.selected).toBe(0)
})

test("tab accepts visible autocomplete without changing the preset", async () => {
  const { term, tui, mounted } = await boot(80, 24)
  for (const ch of "/cle") term.input(ch)
  await new Promise((resolve) => setTimeout(resolve, 200))
  tui.renderNow()
  expect(mounted.editor.isShowingAutocomplete()).toBe(true)
  const preset = mounted.ctl.app.state.preset
  term.input("\t")
  expect(mounted.editor.getText()).toBe("/clean ")
  expect(mounted.ctl.app.state.preset).toBe(preset)
})

test("selection scrolls into view without overriding manual viewport scrolling", async () => {
  const { term, tui, mounted, screen } = await boot(60, 12)
  const plan = mounted.ctl.app.plan!
  mounted.ctl.app.plan = {
    ...plan,
    files: Array.from({ length: 40 }, (_, index) => ({
      path: `/file-${index}`, display: `FILE_${index}`, kind: "text", size: 100,
    })),
    files_total: 40,
  }
  tui.renderNow()
  for (let i = 0; i < 30; i++) term.input("\n")
  tui.renderNow()
  expect(screen()).toContain("→ [ ] · FILE_30")
  term.input("\x1b[5~")
  tui.renderNow()
  expect(mounted.ctl.app.selected).toBe(30)
  expect(screen()).not.toContain("→ [ ] · FILE_30")
  term.input("\x0b")
  tui.renderNow()
  expect(screen()).toContain("→ [ ] · FILE_29")
})

test("pageup scrolls the viewport and does not change selection", async () => {
  const { term, tui, type, mounted } = await boot(80, 24)
  await type("drafts")
  term.input("\x1b[5~")
  tui.renderNow()
  expect(mounted.ctl.app.selected).toBe(0)
})

test("run stays cancellable and shows a spinner until the clean finishes", async () => {
  let release: () => void = () => {}
  const bridge = new FakeBridge()
  bridge.cleanGate = new Promise((resolve) => {
    release = resolve
  })
  const { term, tui, screen, type, mounted } = await boot(80, 24, bridge)
  await type("drafts")
  term.input(" ")
  term.input("\x12")
  tui.renderNow()
  expect(mounted.ctl.app.run).not.toBeNull()
  expect(screen()).toMatch(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏] Cleaning /)
  term.input("\x1b")
  await new Promise((resolve) => setTimeout(resolve, 20))
  tui.renderNow()
  expect(bridge.calls.some((call) => call.method === "cancel")).toBe(true)
  expect(screen()).toContain("Stopping after the current file.")
  expect(mounted.ctl.app.run).not.toBeNull()
  release()
  await new Promise((resolve) => setTimeout(resolve, 30))
  tui.renderNow()
  expect(mounted.ctl.app.run).toBeNull()
})

test("the bridge dying shows an error screen instead of hanging", async () => {
  const { bridge, tui, screen } = await boot(80, 24)
  bridge.die("bridge exited with code 1")
  tui.renderNow()
  const frame = screen()
  expect(frame).toContain("The Python side stopped")
  expect(frame).toContain("exited with code 1")
})

test("each successful submit is recalled by alt+up on an empty prompt", async () => {
  const { term, type, mounted } = await boot(80, 24)
  await type("drafts")
  await type("notes.txt")
  expect(mounted.editor.getText()).toBe("")
  term.input("\x1b[1;3A")
  expect(mounted.editor.getText()).toBe("notes.txt")
  term.input("\x1b[1;3A")
  expect(mounted.editor.getText()).toBe("drafts")
})

test("arrows navigate and Space marks exactly one file, without submitting or cleaning", async () => {
  const { term, tui, type, mounted, bridge, screen } = await boot(100, 30)
  await type("drafts")
  const files = mounted.ctl.app.plan!.files
  term.input("\x1b[B")
  expect(mounted.ctl.app.selected).toBe(1)
  expect(mounted.editor.getText()).toBe("")
  term.input(" ")
  expect(mounted.ctl.app.state.clean_path).toBe(files[1]!.path)
  term.input("\x1b[B")
  expect(mounted.ctl.app.state.clean_path).toBe(files[1]!.path)
  term.input(" ")
  expect(mounted.ctl.app.state.clean_path).toBe(files[2]!.path)
  tui.renderNow()
  expect(screen().split("[x]")).toHaveLength(2)
  term.input("\r")
  expect(bridge.calls.some((call) => call.method === "clean")).toBe(false)
  term.input(" ")
  expect(mounted.ctl.app.state.clean_path).toBeNull()
  term.input("\x1b[A")
  expect(mounted.ctl.app.selected).toBe(1)
})

test("clean refuses without a mark, then sends only the marked target even after navigation", async () => {
  const { term, type, mounted, bridge } = await boot(100, 30)
  await type("drafts")
  await mounted.ctl.clean()
  expect(bridge.calls.filter((call) => call.method === "clean")).toHaveLength(0)
  term.input("\x1b[B")
  term.input(" ")
  const target = mounted.ctl.app.state.clean_path
  term.input("\x1b[B")
  await mounted.ctl.clean()
  expect(bridge.calls.find((call) => call.method === "clean")!.params.state.clean_path).toBe(target)
  expect(Object.keys(mounted.ctl.app.results)).toEqual([target!])
})

test("prompt editing/completion owns arrows and spaces while typing", async () => {
  const { term, type, mounted, tui } = await boot(100, 30)
  await type("drafts")
  mounted.editor.setText("first\nsecond")
  term.input("\x1b[A")
  expect(mounted.ctl.app.selected).toBe(0)
  term.input(" ")
  expect(mounted.editor.getText()).toContain(" ")
  expect(mounted.ctl.app.state.clean_path).toBeNull()
  mounted.editor.setText("")
  for (const ch of "/cle") term.input(ch)
  await new Promise((resolve) => setTimeout(resolve, 200))
  tui.renderNow()
  expect(mounted.editor.isShowingAutocomplete()).toBe(true)
  term.input("\x1b[B")
  expect(mounted.ctl.app.selected).toBe(0)
})

test("refresh clears a disappeared mark and clearing paths never retains a clean target", async () => {
  const { term, type, mounted, bridge } = await boot(100, 30)
  await type("drafts")
  term.input(" ")
  const target = mounted.ctl.app.state.clean_path!
  bridge.removedFiles.add(target)
  mounted.ctl.addPaths(["another-folder"])
  await new Promise((resolve) => setTimeout(resolve, 100))
  expect(mounted.ctl.app.state.clean_path).toBeNull()
  expect(mounted.ctl.app.plan!.files.some((file) => file.path === target)).toBe(false)
  term.input(" ")
  expect(mounted.ctl.app.state.clean_path).not.toBeNull()
  mounted.ctl.clearPaths()
  await new Promise((resolve) => setTimeout(resolve, 100))
  expect(mounted.ctl.app.state.clean_path).toBeNull()
  await mounted.ctl.clean()
  expect(bridge.calls.some((call) => call.method === "clean")).toBe(false)
})
