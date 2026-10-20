import { afterEach, expect, test } from "bun:test"
import { isViewportTUI, stripTerminalSequences, TuiAltScreen } from "@earendil-works/pi-tui"
import { mountApp } from "../src/v2/app"
import { FakeBridge } from "./fake-bridge"
import { FakeTerminal } from "./fake-terminal"

let close: (() => void) | undefined
afterEach(() => { close?.(); close = undefined })

async function boot() {
  const bridge = new FakeBridge()
  const term = new FakeTerminal(120, 40)
  const tui = new TuiAltScreen(term, false)
  if (!isViewportTUI(tui)) throw new Error("expected viewport")
  const mounted = mountApp(tui, bridge, () => {})
  tui.start()
  await mounted.ctl.start()
  close = () => { mounted.dispose(); tui.stop({ preserveScreen: true }) }
  const press = (key: string) => { term.input(key); tui.renderNow() }
  const type = async (text: string) => {
    for (const ch of text) term.input(ch)
    term.input("\r")
    await new Promise((resolve) => setTimeout(resolve, 160))
    tui.renderNow()
  }
  const screen = () => stripTerminalSequences(tui.getScreenLines().join("\n"))
  const manual = async () => {
    await type("/model")
    for (let i = 0; i < 3; i++) press("\x1b[B")
    press("\r")
  }
  return { bridge, mounted, press, type, screen, manual }
}

test("each user can replace the saved local URL and model, without Pi or model calls", async () => {
  const app = await boot()
  app.mounted.ctl.app.state.endpoint = { backend: "openai-compatible", base_url: "http://localhost:8181", model: "old-model" }
  await app.manual()
  app.press("\t")
  app.press("\x15")
  for (const ch of "http://127.0.0.1:9123/v1") app.press(ch)
  app.press("\r")
  app.press("\x15")
  await app.type("my-own-model:custom")
  const expected = { backend: "openai-compatible", base_url: "http://127.0.0.1:9123", model: "my-own-model:custom" }
  expect(app.mounted.ctl.app.state.endpoint).toEqual(expected)
  expect(app.bridge.calls.filter((call) => call.method === "save_settings").at(-1)?.params.settings.endpoint).toEqual(expected)
  expect(app.bridge.calls.some((call) => call.method === "clean")).toBe(false)
  expect(app.screen()).not.toContain("Pi models")
})

test("manual configuration requires the user's model ID and a credential-free local URL", async () => {
  const app = await boot()
  await app.manual()
  app.press("\t")
  await app.type("http://cloud.example.test:9000")
  await app.type("own-model")
  expect(app.screen()).toContain("Use a local server")
  expect(app.bridge.calls.some((call) => call.method === "save_settings")).toBe(false)
  app.press("\x1b[A")
  app.press("\x15")
  await app.type("http://localhost:9000")
  app.press("\x15")
  app.press("\r")
  expect(app.screen()).toContain("Enter the model ID")
  await app.type("replacement-model")
  expect(app.mounted.ctl.app.state.endpoint.model).toBe("replacement-model")
})

test("a malformed local URL is refused without crashing the dialog", async () => {
  const app = await boot()
  await app.manual()
  app.press("\t")
  await app.type("http://localhost:invalid")
  await app.type("own-model")
  expect(app.screen()).toContain("Enter a valid http:// or https:// local server URL")
  expect(app.bridge.calls.some((call) => call.method === "save_settings")).toBe(false)
})

test("confirmation retries preserve the original marked target and options", async () => {
  const app = await boot()
  app.bridge.confirm = [{ kind: "in_place", message: "1 file will be overwritten." }]
  await app.type("drafts")
  app.press(" ")
  app.press("\x12")
  await new Promise((resolve) => setTimeout(resolve, 20))
  app.press("")
  expect(app.screen()).toContain("Before anything is written")
  const original = app.bridge.calls.find((call) => call.method === "clean")!.params.state
  app.mounted.ctl.app.state.clean_path = "/abs/notes.txt"
  app.mounted.ctl.app.state.flags.push("--strip-semantic-format")
  app.mounted.ctl.app.state.endpoint.model = "different-model"
  app.press("y")
  await new Promise((resolve) => setTimeout(resolve, 100))
  const calls = app.bridge.calls.filter((call) => call.method === "clean")
  expect(calls).toHaveLength(2)
  expect(calls[1]!.params.state).toEqual(original)
  expect(Object.keys(app.mounted.ctl.app.results)).toEqual([original.clean_path])
})
