import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { expect, test } from "bun:test"
import { stripTerminalSequences, visibleWidth } from "@earendil-works/pi-tui"
import { DetailView, markTotals } from "../src/v2/components/detail"
import { convertReveal, parseFinding, type Finding, type Reveal } from "../src/v2/protocol"
import { createApp } from "../src/v2/store"
import { paint, theme } from "../src/v2/theme"
import { FakeBridge } from "./fake-bridge"

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..")
const bridgeTest = readFileSync(resolve(repoRoot, "tests/test_tui_bridge.py"), "utf8")
const fixtureJson = bridgeTest.match(/ASTRAL_REVEAL_FIXTURE_JSON = """\n([\s\S]*?)"""/)?.[1]
if (!fixtureJson) throw new Error("ASTRAL_REVEAL_FIXTURE_JSON missing from tests/test_tui_bridge.py")

const fixture = JSON.parse(fixtureJson) as {
  text: string
  marks: [number, number, string][]
  names: string[]
}

function astralReveal(line = 1): Reveal {
  return {
    line,
    text: fixture.text,
    marks: fixture.marks.map((mark) => [mark[0], mark[1], mark[2]]),
  }
}

function findingWith(reveal: Reveal, path = "/tmp/astral.txt"): Finding {
  return {
    path,
    display: "astral.txt",
    kind: "text",
    suspicious: true,
    counts: { hidden: 1 },
    lines: ["1 zero-width space (U+200B)"],
    reveal: [reveal],
    error: null,
  }
}

function renderFinding(finding: Finding, width = 80): string[] {
  const ctl = createApp(new FakeBridge(), () => {})
  ctl.app.plan = {
    ok: true,
    error: null,
    command: "wm astral.txt",
    files: [{ path: finding.path, display: finding.display, kind: finding.kind, size: 8 }],
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
  ctl.app.selected = 0
  ctl.app.findings[finding.path] = finding
  return new DetailView(ctl).render(width)
}

function plain(lines: string[]): string {
  return lines.map((line) => stripTerminalSequences(line)).join("\n")
}

test("the astral fixture converts code-point marks and the reveal highlights the hidden character", () => {
  expect(fixture.text).toBe("😀x◆y")
  expect(fixture.marks).toEqual([[2, 3, "ZWSP"]])
  expect(fixture.names).toEqual(["ZWSP"])
  // JS UTF-16 indexes the emoji as two units, so the wire slice is not the mark.
  expect(fixture.text.slice(fixture.marks[0]![0], fixture.marks[0]![1])).not.toBe("◆")

  const wire = findingWith(astralReveal())
  const parsed = parseFinding(wire)
  const again = parseFinding(wire)
  const [start, end, name] = parsed.reveal[0]!.marks[0]!
  expect(again.reveal[0]!.marks[0]![0]).toBe(start)
  expect(again.reveal[0]!.text.slice(again.reveal[0]!.marks[0]![0], again.reveal[0]!.marks[0]![1])).toBe("◆")
  expect(name).toBe("ZWSP")
  expect(parsed.reveal[0]!.text).toBe(fixture.text)
  expect(parsed.reveal[0]!.text.slice(start, end)).toBe("◆")
  expect(start).not.toBe(fixture.marks[0]![0])

  const lines = renderFinding(findingWith(astralReveal()))
  const text = plain(lines)
  expect(text).not.toContain("\uFFFD")
  expect(lines.join("\n")).toContain(paint("◆", theme.glow, true))
  expect(lines.join("\n")).not.toContain(paint("x", theme.glow, true))
  const textLine = text.split("\n").find((line) => line.includes("◆"))
  const labelLine = text.split("\n").find((line) => line.includes("ZWSP"))
  expect(textLine).toBeTruthy()
  expect(labelLine).toBeTruthy()
  expect(visibleWidth(textLine!.slice(0, textLine!.indexOf("◆")))).toBe(
    visibleWidth(labelLine!.slice(0, labelLine!.indexOf("ZWSP"))),
  )

  const narrow = plain(renderFinding(findingWith(astralReveal()), 9))
  expect(narrow).toContain("😀")
  expect(narrow).not.toContain("◆")
  expect(narrow).not.toContain("\uFFFD")
})

test("an ellipsis already shifted by the bridge is not shifted again", () => {
  const text = "…😀x◆y"
  const reveal: Reveal = { line: 4, text, marks: [[3, 4, "ZWSP"]] }
  const converted = convertReveal(reveal)
  const [start, end] = converted.marks[0]!
  expect(converted.text).toBe(text)
  expect(text.slice(start, end)).toBe("◆")
  expect(convertReveal(converted).marks[0]![0]).toBe(start)
})

test("a BMP excerpt takes the no-op fast path", () => {
  const reveal: Reveal = { line: 1, text: "Hello◆ world", marks: [[5, 6, "ZWSP"]] }
  const converted = convertReveal(reveal)
  expect(converted).toBe(reveal)
  expect(converted.text).toBe(reveal.text)
  expect(converted.marks).toBe(reveal.marks)

  const finding = findingWith(reveal, "/tmp/bmp.txt")
  finding.display = "bmp.txt"
  expect(parseFinding(finding)).toBe(finding)

  const lines = renderFinding(finding)
  const text = plain(lines)
  expect(text).toContain("◆")
  expect(text).toContain("ZWSP")
  expect(text).not.toContain("\uFFFD")
  const textLine = text.split("\n").find((line) => line.includes("◆"))!
  const labelLine = text.split("\n").find((line) => line.includes("ZWSP"))!
  expect(visibleWidth(textLine.slice(0, textLine.indexOf("◆")))).toBe(
    visibleWidth(labelLine.slice(0, labelLine.indexOf("ZWSP"))),
  )
})

test("a fullwidth character before a mark aligns the label by columns, not code units", () => {
  const reveal: Reveal = { line: 2, text: "Ａ◆", marks: [[1, 2, "ZWSP"]] }
  expect(convertReveal(reveal)).toBe(reveal)
  expect(visibleWidth("Ａ")).toBe(2)
  const finding = findingWith(reveal, "/tmp/wide.txt")
  finding.display = "wide.txt"
  const text = plain(renderFinding(finding))
  const textLine = text.split("\n").find((line) => line.includes("◆"))!
  const labelLine = text.split("\n").find((line) => line.includes("ZWSP"))!
  expect(visibleWidth(textLine.slice(0, textLine.indexOf("◆")))).toBe(
    visibleWidth(labelLine.slice(0, labelLine.indexOf("ZWSP"))),
  )
  expect(visibleWidth(textLine.slice(0, textLine.indexOf("◆")))).toBeGreaterThan(textLine.indexOf("◆"))
})

test("a finished file is labelled by result class and the mark reduction is computed once", () => {
  const ctl = createApp(new FakeBridge(), () => {})
  const file = { path: "/tmp/a.txt", display: "drafts/a.txt", kind: "text", size: 20 }
  ctl.app.plan = {
    ok: true,
    error: null,
    command: "wm drafts/a.txt",
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
  ctl.app.results[file.path] = {
    index: 0,
    display: file.display,
    output: "/tmp/a.out.txt",
    layer: "A",
    result_class: "Verifiable",
    exit_code: 0,
    before: { hidden: 3, metadata: 1 },
    after: { hidden: 0, metadata: 0 },
    lines: ["Removed 3 hidden characters."],
    diff: "--- a\n+++ b\n-a<U+200B>b\n+ab",
    error: null,
  }
  const verifiable = plain(new DetailView(ctl).render(80))
  expect(verifiable).toContain("Verifiable")
  expect(verifiable).not.toContain("cleaned")
  expect(verifiable.split("\n").some((line) => line.trim() === "error")).toBe(false)
  expect(markTotals({ hidden: 3, metadata: 1 }, { hidden: 0, metadata: 0 }).removed).toBe(4)
  expect(verifiable.match(/4 marks removed/g)).toHaveLength(1)
  expect(verifiable).toContain("◆")
  expect(verifiable).not.toContain("<U+200B>")

  ctl.app.results[file.path] = {
    ...ctl.app.results[file.path]!,
    layer: "B",
    result_class: "Best-effort",
    before: { hidden: 2 },
    after: { hidden: 2 },
    diff: null,
    error: "the model refused",
  }
  const effort = plain(new DetailView(ctl).render(80))
  expect(effort).toContain("Best-effort")
  expect(effort).toContain("the model refused")
  expect(effort).not.toContain("cleaned")
  expect(effort.split("\n").some((line) => line.trim() === "error")).toBe(false)
  expect(effort.match(/0 marks removed/g)).toHaveLength(1)
})
