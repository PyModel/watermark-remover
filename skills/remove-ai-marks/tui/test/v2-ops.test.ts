import { readFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
import { afterEach, expect, test } from "bun:test"
import { hyperlink, isViewportTUI, stripTerminalSequences, TuiAltScreen, type NativeClipboard } from "@earendil-works/pi-tui"
import { mountApp } from "../src/v2/app"
import { Footer } from "../src/v2/components/footer"
import { copyDiffText, copyOutputPath, fileHyperlink, type CopyDeps } from "../src/v2/ops"
import type { FileDone } from "../src/v2/protocol"
import { createApp } from "../src/v2/store"
import { FakeBridge } from "./fake-bridge"
import { FakeTerminal } from "./fake-terminal"

function done(patch: Partial<FileDone> = {}): FileDone {
  return {
    index: 0,
    display: "drafts/intro.md",
    output: "/tmp/drafts/intro.out.md",
    layer: "A",
    result_class: "Verifiable",
    exit_code: 0,
    before: { hidden: 3 },
    after: { hidden: 0 },
    lines: ["Removed 3 hidden characters."],
    diff: "-Hello<U+200B> world\n+Hello world",
    error: null,
    ...patch,
  }
}

function occurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1
}

function memoryClipboard(): NativeClipboard & { texts: string[] } {
  const texts: string[] = []
  return {
    texts,
    getText: async () => texts[0] ?? null,
    setText: async (text: string) => {
      texts.unshift(text)
    },
  }
}

test("copy actions put the output path and the diff on the native clipboard", async () => {
  const clip = memoryClipboard()
  const toasts: string[] = []
  const deps: CopyDeps = { clipboard: clip, toast: (text) => toasts.push(text) }
  const file = done()
  await copyOutputPath(file, deps)
  await copyDiffText(file, deps)
  expect(clip.texts).toEqual([file.diff, file.output])
  expect(toasts).toEqual(["Copied.", "Copied."])
})

test("a helper with no setText degrades to a toast", async () => {
  const toasts: string[] = []
  const clip: NativeClipboard = { getText: async () => null }
  await copyOutputPath(done(), { clipboard: clip, toast: (text) => toasts.push(text) })
  expect(toasts).toEqual(["Clipboard unavailable."])
})

test("a missing clipboard helper degrades to a toast", async () => {
  const term = new FakeTerminal()
  const toasts: { text: string; tone?: string }[] = []
  const deps: CopyDeps = {
    clipboard: null,
    terminal: term,
    toast: (text, tone) => toasts.push({ text, tone }),
  }
  await copyOutputPath(done(), deps)
  expect(toasts.some((toast) => toast.tone === "error" && /unavailable/i.test(toast.text))).toBe(true)
  expect(term.writes.some((chunk) => chunk.startsWith("\x1b]52;c;"))).toBe(false)
  expect(term.titles).toEqual([])
})

test("file paths in the running app are OSC 8 sequences written to the terminal", async () => {
  const { term, type, press } = await bootCopy(null)
  await type("drafts")
  await press(" ")
  await press("\x12")
  const written = term.writes.join("")
  const source = pathToFileURL("/abs/drafts/intro.md").href
  const output = pathToFileURL("/abs/drafts/intro.cleaned.md").href
  expect(written).toContain(`\x1b]8;;${source}`)
  expect(written).toContain(`\x1b]8;;${output}`)
  expect(stripTerminalSequences(written)).toContain("drafts/intro.md")
  expect(fileHyperlink("intro.md", "/abs/drafts/intro.md")).toBe(hyperlink("intro.md", source))
})

test("the terminal title and progress change while a clean runs and clear after", () => {
  const term = new FakeTerminal()
  const ctl = createApp(new FakeBridge(), () => {})
  ctl.app.plan = {
    ok: true,
    error: null,
    command: "wm drafts",
    files: [{ path: "/tmp/drafts/intro.md", display: "drafts/intro.md", kind: "text", size: 12 }],
    files_total: 1,
    discover_error: null,
    layer: "A",
    result_class: "Verifiable",
    output: null,
    confirm: [],
    estimate_seconds: 0,
    warnings: [],
    preflight_error: null,
  }
  ctl.app.run = { kind: "clean", index: 0, total: 2, display: "drafts/intro.md" }
  const footer = new Footer(ctl, term, () => 0)
  footer.render(80)
  const cleaning = term.writes.join("")
  expect(occurrences(cleaning, "\x1b]0;Cleaning 1 of 2: drafts/intro.md\x07")).toBe(1)
  expect(occurrences(cleaning, "\x1b]9;4;3\x07")).toBe(1)
  ctl.app.run = null
  footer.render(80)
  const cleared = term.writes.join("")
  expect(occurrences(cleared, "\x1b]0;wm\x07")).toBe(1)
  expect(occurrences(cleared, "\x1b]9;4;0\x07")).toBe(1)
  ctl.app.run = { kind: "clean", index: 0, total: 1, display: "bad\u0007name" }
  footer.render(80)
  const written = term.writes.join("")
  expect(written).toContain("\x1b]0;Cleaning 1 of 1: badname\x07")
  expect(written).not.toContain("bad\u0007name")
})

test("PI_TUI_WRITE_LOG is documented in the bridge protocol notes", () => {
  const notes = readFileSync(new URL("../PROTOCOL.md", import.meta.url), "utf8")
  const section = notes.slice(notes.indexOf("Frontend operability"))
  expect(section.length).toBeGreaterThan(40)
  expect(section).toContain("PI_TUI_WRITE_LOG")
  expect(section).toContain("WM_TUI_LOG")
  expect(section).toContain("file path")
  expect(section).toContain("directory")
})

let shutdown: (() => void) | undefined

afterEach(() => {
  shutdown?.()
  shutdown = undefined
})

async function bootCopy(clipboard: NativeClipboard | null, columns = 160, rows = 40, bridge = new FakeBridge()) {
  const term = new FakeTerminal(columns, rows)
  const tui = new TuiAltScreen(term, false)
  if (!isViewportTUI(tui)) throw new Error("expected a viewport TUI")
  const mounted = mountApp(tui, bridge, () => {}, { clipboard })
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
  const press = async (data: string) => {
    term.input(data)
    for (let i = 0; i < 8; i++) await Promise.resolve()
    tui.renderNow()
  }
  return { term, tui, mounted, screen, type, press }
}

test("copy keys on the running app copy the output path and the diff without touching the editor", async () => {
  const clip = memoryClipboard()
  const { term, mounted, screen, type, press } = await bootCopy(clip)
  await type("drafts")
  await press(" ")
  await press("\x12")
  term.input("z")
  await press("\x0f")
  expect(mounted.editor.getText()).toBe("z")
  expect(clip.texts[0]).toBe("/abs/drafts/intro.cleaned.md")
  expect(screen()).toContain("Copied.")
  await press("\x1bOQ")
  expect(mounted.editor.getText()).toBe("z")
  expect(clip.texts[0]).toContain("-Hello<U+200B> world")
  expect(screen()).toContain("Copied.")
})

test("the running app writes title and progress sequences, and a dying bridge clears them", async () => {
  let release: () => void = () => {}
  const bridge = new FakeBridge()
  bridge.cleanGate = new Promise((resolve) => {
    release = resolve
  })
  const { term, tui, screen, type } = await bootCopy(null, 120, 30, bridge)
  await type("drafts")
  term.input(" ")
  term.input("\x12")
  tui.renderNow()
  const running = term.writes.join("")
  expect(occurrences(running, "\x1b]0;Cleaning 1 of 1: drafts/intro.md\x07")).toBe(1)
  expect(occurrences(running, "\x1b]9;4;3\x07")).toBe(1)
  bridge.die("bridge exited with code 1")
  tui.renderNow()
  const died = term.writes.join("")
  expect(died.lastIndexOf("\x1b]0;wm\x07")).toBeGreaterThan(died.lastIndexOf("\x1b]0;Cleaning"))
  expect(died.lastIndexOf("\x1b]9;4;0\x07")).toBeGreaterThan(died.lastIndexOf("\x1b]9;4;3\x07"))
  expect(screen()).toContain("The Python side stopped")
  release()
})

test("an unavailable clipboard helper on the running app toasts and writes no OSC 52", async () => {
  const { term, screen, type, press } = await bootCopy(null)
  await type("drafts")
  await press(" ")
  await press("\x12")
  const before = term.writes.length
  await press("\x0f")
  expect(screen()).toContain("Clipboard unavailable.")
  expect(term.writes.slice(before).join("")).not.toContain("\x1b]52;c;")
  const mid = term.writes.length
  await press("\x1bOQ")
  expect(screen()).toContain("Clipboard unavailable.")
  expect(term.writes.slice(mid).join("")).not.toContain("\x1b]52;c;")
})
