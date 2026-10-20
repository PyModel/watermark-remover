import { afterEach, expect, test } from "bun:test"
import { isViewportTUI, stripTerminalSequences, TuiAltScreen, visibleWidth } from "@earendil-works/pi-tui"
import { mountApp } from "../src/v2/app"
import { renderWordmark, WORDMARK } from "../src/v2/components/brand"
import { FakeBridge } from "./fake-bridge"
import { FakeTerminal } from "./fake-terminal"

let cleanup: (() => void) | undefined
afterEach(() => { cleanup?.(); cleanup = undefined })

async function boot(columns: number, rows: number) {
  const term = new FakeTerminal(columns, rows)
  const tui = new TuiAltScreen(term, false)
  if (!isViewportTUI(tui)) throw new Error("expected viewport")
  const mounted = mountApp(tui, new FakeBridge(), () => {})
  tui.start()
  await mounted.ctl.start()
  cleanup = () => { mounted.dispose(); tui.stop({ preserveScreen: true }) }
  const lines = () => {
    tui.renderNow()
    return tui.getScreenLines().map(stripTerminalSequences)
  }
  return { term, tui, mounted, lines }
}

test("the wordmark uses ASCII only and never clips its letterforms", () => {
  expect(WORDMARK).toHaveLength(3)
  for (const line of WORDMARK) expect(line).toMatch(/^[\x20-\x7e]+$/)
  expect(renderWordmark(120, 40)).toHaveLength(3)
  expect(renderWordmark(60, 40)).toEqual([])
  expect(renderWordmark(120, 24)).toEqual([])
  for (let width = 1; width <= 160; width++) {
    for (const line of renderWordmark(width, 40)) expect(visibleWidth(line)).toBeLessThanOrEqual(width)
  }
})

test("wide mounted header shows the full banner and truthful model status above the files", async () => {
  const { mounted, lines } = await boot(120, 40)
  mounted.ctl.addPaths(["drafts"])
  await new Promise((resolve) => setTimeout(resolve, 100))
  const screen = lines()
  WORDMARK.forEach((line, index) => expect(screen[index]?.trimEnd()).toBe(" " + line))
  expect(screen[4]).toContain("WATERMARK REMOVER")
  expect(screen[4]).toContain("No model")
  expect(screen.join("\n")).toContain("drafts/intro.md")
  expect(screen.join("\n")).toContain("ctrl+r clean")
})

test("short and narrow screens retain a compact readable title and the prompt", async () => {
  const { lines } = await boot(60, 20)
  const screen = lines()
  expect(screen[0]).toContain("WATERMARK REMOVER")
  expect(screen[0]).toContain("No model")
  expect(screen.join("\n")).not.toContain(WORDMARK[0]!)
  expect(screen.join("\n")).toContain("Type a file or folder")
  for (const line of screen) expect(visibleWidth(line)).toBeLessThanOrEqual(60)
})

test("resize reflows the banner without losing the single-file mark or controls", async () => {
  const { term, mounted, lines } = await boot(120, 40)
  mounted.ctl.addPaths(["drafts"])
  await new Promise((resolve) => setTimeout(resolve, 100))
  term.input(" ")
  const target = mounted.ctl.app.state.clean_path
  expect(lines()[0]).toContain(WORDMARK[0]!)
  term.resize(60, 20)
  expect(lines()[0]).toContain("WATERMARK REMOVER")
  expect(lines().join("\n")).toContain("[x]")
  expect(mounted.ctl.app.state.clean_path).toBe(target)
  term.resize(120, 40)
  expect(lines()[0]).toContain(WORDMARK[0]!)
  expect(lines().join("\n")).toContain("ctrl+r clean")
})

test("long local model IDs do not push status off the header", async () => {
  const { mounted, lines } = await boot(90, 30)
  mounted.ctl.setEndpoint({ backend: "openai-compatible", base_url: "http://localhost:9123", model: "my-own-local-model-".repeat(12) }, false)
  const screen = lines()
  expect(screen[4]).toContain("WATERMARK REMOVER")
  expect(screen[4]).toContain("my-own-local-model")
  expect(screen[4]).toContain("…")
  for (const line of screen) expect(visibleWidth(line)).toBeLessThanOrEqual(90)
})
