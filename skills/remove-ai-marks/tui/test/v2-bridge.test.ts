import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, expect, test } from "bun:test"
import { isViewportTUI, TuiAltScreen } from "@earendil-works/pi-tui"
import { Bridge } from "../src/v2/bridge"
import { mountApp } from "../src/v2/app"
import { parseFinding, type Finding } from "../src/v2/protocol"
import { FakeTerminal } from "./fake-terminal"

/**
 * Speaks just enough of the bridge protocol to emit one malformed finding,
 * then keep answering. PYTHONUNBUFFERED is set by Bridge, so each line flushes.
 */
const CHILD = `
import json, sys

HELLO = {
  "version": "0.6.0",
  "presets": [{
    "key": "hidden", "label": "Hidden marks", "description": "marks",
    "layer": "A", "result_class": "Verifiable", "flags": [], "requires_endpoint": False,
  }],
  "settings": {"preset": "hidden", "endpoint": {"backend": None, "base_url": None, "model": None}},
  "settings_path": "/tmp/wm/tui.json",
  "onboard": False,
  "initial": {"paths": ["drafts"], "flags": []},
  "backends": ["ollama"],
}
PLAN = {
  "ok": True, "error": None, "command": "wm drafts",
  "files": [{"path": "/tmp/drafts/a.md", "display": "drafts/a.md", "kind": "text", "size": 4}],
  "files_total": 1, "discover_error": None, "layer": "A", "result_class": "Verifiable",
  "output": None, "confirm": [], "estimate_seconds": 0, "warnings": [], "preflight_error": None,
}
CHECKS = {"checks": [{"name": "Python", "state": "OK", "good": True, "detail": "stdlib", "fix": "", "optional": False}]}
BAD = {
  "path": "/tmp/drafts/a.md", "display": "drafts/a.md", "kind": "text",
  "suspicious": True, "counts": {}, "lines": [],
  "reveal": [{"line": 1, "text": None, "marks": []}],
  "error": None,
}

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    msg = json.loads(raw)
    method = msg.get("method")
    ident = msg.get("id")
    if method == "shutdown":
        break
    if method == "inspect":
        send({"id": ident, "event": "inspected", "data": BAD})
    result = {"hello": HELLO, "plan": PLAN, "checks": CHECKS, "inspect": {"files": [], "cancelled": False}}.get(method, {"ok": True})
    send({"id": ident, "result": result})
`

let dir: string | undefined
let bridge: Bridge | undefined

afterEach(() => {
  bridge?.close()
  bridge = undefined
  if (dir) rmSync(dir, { recursive: true, force: true })
  dir = undefined
})

function spawnBridge(): Bridge {
  dir = mkdtempSync(join(tmpdir(), "wm-bridge-"))
  const script = join(dir, "bridge.py")
  writeFileSync(script, CHILD)
  bridge = new Bridge("python3", script, undefined)
  return bridge
}

function badFinding(): Finding {
  return {
    path: "/tmp/drafts/a.md",
    display: "drafts/a.md",
    kind: "text",
    suspicious: true,
    counts: {},
    lines: [],
    reveal: [{ line: 1, text: 1 as unknown as string, marks: [] }],
    error: null,
  }
}

async function within<T>(promise: Promise<T>, ms = 1500): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error("request did not settle")), ms)
  })
  try {
    return await Promise.race([promise, timeout])
  } finally {
    clearTimeout(timer)
  }
}

test("parseFinding rejects a finding whose reveal text is not a string", () => {
  expect(parseFinding(badFinding())).toBeNull()
  const missing = badFinding()
  missing.reveal = [{ line: 1, text: null as unknown as string, marks: [] }]
  expect(parseFinding(missing)).toBeNull()
})

test("a throwing event callback cannot stop the reader; the result and a later request still settle", async () => {
  const child = spawnBridge()
  const first = await within(
    child.request("inspect", {}, () => {
      throw new Error("malformed finding")
    }),
  )
  expect(first).toEqual({ files: [], cancelled: false })
  const later = await within(child.request<{ checks: { name: string }[] }>("checks"))
  expect(later.checks[0]?.name).toBe("Python")
})

test("the app ignores a malformed finding event and a later request still settles", async () => {
  const child = spawnBridge()
  const term = new FakeTerminal(100, 30)
  const tui = new TuiAltScreen(term, false)
  if (!isViewportTUI(tui)) throw new Error("expected a viewport TUI")
  const mounted = mountApp(tui, child, () => {})
  tui.start()
  try {
    await within(mounted.ctl.start(), 2000)
    expect(mounted.ctl.app.fatal).toBeNull()
    expect(mounted.ctl.app.plan?.files).toHaveLength(1)
    await within(mounted.ctl.inspect(), 2000)
    expect(mounted.ctl.app.fatal).toBeNull()
    expect(mounted.ctl.app.run).toBeNull()
    expect(mounted.ctl.app.findings["/tmp/drafts/a.md"]).toBeUndefined()
    expect(mounted.ctl.app.toast?.text).toContain("Inspected")
    await within(mounted.ctl.loadChecks(), 2000)
    expect(mounted.ctl.app.checks?.[0]?.name).toBe("Python")
  } finally {
    mounted.dispose()
    tui.stop({ preserveScreen: true })
  }
})
