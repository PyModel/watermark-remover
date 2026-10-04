import { readFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
import { expect, test } from "bun:test"
import { hyperlink, stripTerminalSequences, type NativeClipboard } from "@earendil-works/pi-tui"
import { DetailView } from "../src/v2/components/detail"
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
  expect(term.writes.some((chunk) => chunk.startsWith("\x1b]52;c;"))).toBe(true)
  expect(term.titles).toEqual([])
})

test("file paths in the detail pane are OSC 8 hyperlinks", () => {
  const ctl = createApp(new FakeBridge(), () => {})
  const file = { path: "/tmp/drafts/intro.md", display: "drafts/intro.md", kind: "text", size: 12 }
  ctl.app.plan = {
    ok: true,
    error: null,
    command: "wm drafts/intro.md",
    files: [file],
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
  const result = done({ display: file.display, output: "/tmp/drafts/intro.out.md" })
  ctl.app.results[file.path] = result
  const rendered = new DetailView(ctl).render(80).join("\n")
  expect(rendered).toContain(fileHyperlink(file.display, file.path))
  expect(rendered).toContain("\x1b]8;;")
  expect(rendered).toContain(pathToFileURL(file.path).href)
  expect(rendered).toContain(pathToFileURL(result.output!).href)
  expect(stripTerminalSequences(rendered)).toContain(file.display)
  expect(fileHyperlink("intro.md", file.path)).toBe(hyperlink("intro.md", pathToFileURL(file.path).href))
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
  expect(term.titles.at(-1)).toBe("Cleaning 1 of 2: drafts/intro.md")
  expect(term.progress.at(-1)).toBe(true)
  ctl.app.run = null
  footer.render(80)
  expect(term.titles.at(-1)).toBe("wm")
  expect(term.progress.at(-1)).toBe(false)
  ctl.app.run = { kind: "clean", index: 0, total: 1, display: "bad\u0007name" }
  footer.render(80)
  expect(term.titles.at(-1)).toBe("Cleaning 1 of 1: badname")
  expect(term.titles.at(-1)).not.toContain("\u0007")
})

test("PI_TUI_WRITE_LOG is documented on the ops module", () => {
  const src = readFileSync(new URL("../src/v2/ops.ts", import.meta.url), "utf8")
  const header = src.slice(0, src.indexOf("export"))
  expect(header).toContain("PI_TUI_WRITE_LOG")
  expect(header).toContain("WM_TUI_LOG")
  expect(header).toContain("t06")
})
