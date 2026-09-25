import { afterEach, describe, expect, test } from "bun:test"
import { testRender } from "@opentui/solid"
import { App } from "../src/app"
import { COMMANDS } from "../src/commands"
import { createApp } from "../src/store"
import { FakeBridge } from "./fake-bridge"
import { saveFrame } from "./snapshot"

type Setup = Awaited<ReturnType<typeof testRender>>
let setup: Setup | undefined

afterEach(() => {
  setup?.renderer.destroy()
  setup = undefined
})

async function mount(width = 120, height = 40, configure?: (b: FakeBridge) => void) {
  const bridge = new FakeBridge()
  configure?.(bridge)
  let quit = 0
  const ctl = createApp(bridge, () => quit++)
  setup = await testRender(() => <App ctl={ctl} />, { width, height })
  await settle()
  return { bridge, ctl, quits: () => quit }
}

async function settle(ms = 120) {
  for (let i = 0; i < 4; i++) {
    await Bun.sleep(ms / 4)
    await setup!.renderOnce()
  }
}

async function type(text: string) {
  await setup!.mockInput.typeText(text)
  setup!.mockInput.pressEnter()
  await settle()
}

const frame = () => setup!.captureCharFrame()

describe("wm-tui", () => {
  for (const [w, h] of [[80, 24], [120, 40]] as const) {
    test(`empty state at ${w}x${h} explains the one thing to do`, async () => {
      await mount(w, h)
      const text = frame()
      expect(text).toContain("Type a file or folder below")
      expect(text).toContain("Hidden marks")
      expect(text).toContain("Verifiable")
      expect(text).toContain("No model")
      saveFrame(`empty-${w}x${h}`, setup!.captureSpans())
    })

    test(`adding a path lists files and previews the command at ${w}x${h}`, async () => {
      const { bridge } = await mount(w, h)
      await type("drafts")
      const text = frame()
      expect(text).toContain("5 files")
      expect(text).toContain("intro.md")
      expect(text).toContain("$ wm drafts")
      expect(bridge.calls.some((c) => c.method === "plan" && c.params.state.paths.includes("drafts"))).toBe(true)
      saveFrame(`files-${w}x${h}`, setup!.captureSpans())
    })

    test(`inspect then clean shows findings and results at ${w}x${h}`, async () => {
      await mount(w, h)
      await type("drafts")
      setup!.mockInput.pressKey("e", { ctrl: true })
      await settle()
      expect(frame()).toContain("4 marks")
      expect(frame()).toContain("3 hidden characters")
      saveFrame(`inspected-${w}x${h}`, setup!.captureSpans())
      setup!.mockInput.pressKey("r", { ctrl: true })
      await settle()
      const text = frame()
      expect(text).toContain("cleaned")
      expect(text).toContain("error")
      saveFrame(`cleaned-${w}x${h}`, setup!.captureSpans())
    })
  }

  test("a flag typed in the prompt is added, and typed again is removed", async () => {
    const { ctl } = await mount()
    await type("--nfkc")
    expect(ctl.app.state.flags).toEqual(["--nfkc"])
    expect(frame()).toContain("--nfkc")
    await type("--nfkc")
    expect(ctl.app.state.flags).toEqual([])
  })

  test("an invalid flag shows the parser's own message", async () => {
    await mount()
    await type("drafts")
    await type("--bogus")
    expect(frame()).toContain("unrecognized arguments: --bogus")
    expect(frame()).toContain("5 files")
  })

  test("tab cycles the preset and saves it", async () => {
    const { ctl, bridge } = await mount()
    setup!.mockInput.pressTab()
    await settle()
    expect(ctl.app.state.preset).toBe("hidden-aggressive")
    expect(frame()).toContain("Hidden marks, aggressive")
    expect(bridge.calls.some((c) => c.method === "save_settings" && c.params.settings.preset === "hidden-aggressive")).toBe(true)
  })

  test("the rewrite preset says it needs a model", async () => {
    const { ctl } = await mount()
    ctl.setPreset("rewrite")
    await settle()
    expect(frame()).toContain("Needs a model")
    expect(frame()).toContain("Best-effort")
  })

  test("ctrl+p opens the palette, filters, and runs a command", async () => {
    await mount()
    setup!.mockInput.pressKey("p", { ctrl: true })
    await settle()
    expect(frame()).toContain("Commands")
    saveFrame("palette-120x40", setup!.captureSpans())
    await setup!.mockInput.typeText("help")
    await settle()
    setup!.mockInput.pressEnter()
    await settle()
    expect(frame()).toContain("Any flag `wm` accepts")
    setup!.mockInput.pressEscape()
    await settle()
    expect(frame()).not.toContain("Any flag `wm` accepts")
  })

  test("first run walks welcome → model → preset and saves once", async () => {
    const { ctl, bridge } = await mount(100, 30, (b) => (b.onboard = true))
    expect(frame()).toContain("Welcome to wm")
    saveFrame("setup-1-100x30", setup!.captureSpans())
    setup!.mockInput.pressEnter()
    await settle(200)
    expect(frame()).toContain("A model for rewriting")
    expect(frame()).toContain("Ollama")
    saveFrame("setup-2-100x30", setup!.captureSpans())
    setup!.mockInput.pressEnter() // Ollama has two models → pick one
    await settle()
    expect(frame()).toContain("Models on Ollama")
    setup!.mockInput.pressEnter()
    await settle()
    expect(frame()).toContain("What to remove")
    saveFrame("setup-3-100x30", setup!.captureSpans())
    setup!.mockInput.pressEnter()
    await settle()
    expect(ctl.app.dialogs.length).toBe(0)
    expect(ctl.app.state.endpoint.model).toBe("qwen3:14b")
    expect(frame()).toContain("qwen3:14b")
    const saves = bridge.calls.filter((c) => c.method === "save_settings")
    expect(saves.length).toBeGreaterThan(0)
    for (const save of saves) expect(JSON.stringify(save.params)).not.toMatch(/api_key|token|secret/i)
  })

  test("escape on the first step skips setup and remembers it", async () => {
    const { ctl, bridge } = await mount(100, 30, (b) => (b.onboard = true))
    setup!.mockInput.pressEscape()
    await settle()
    expect(ctl.app.dialogs.length).toBe(0)
    expect(bridge.calls.some((c) => c.method === "save_settings")).toBe(true)
    expect(frame()).toContain("Setup skipped")
  })

  test("a needed confirmation stops the clean until accepted", async () => {
    const { bridge } = await mount(120, 40, (b) => (b.confirm = [{ kind: "remote", message: "Text will be sent to 10.0.0.5" }]))
    await type("drafts")
    setup!.mockInput.pressKey("r", { ctrl: true })
    await settle()
    expect(frame()).toContain("Before anything is written")
    expect(frame()).toContain("10.0.0.5")
    saveFrame("confirm-120x40", setup!.captureSpans())
    setup!.mockInput.pressEscape()
    await settle()
    expect(frame()).toContain("Nothing was written")
    expect(bridge.calls.filter((c) => c.method === "clean").length).toBe(1)
  })

  test("the bridge dying shows an error screen instead of hanging", async () => {
    const { bridge } = await mount()
    bridge.die("bridge exited with code 1")
    await settle()
    expect(frame()).toContain("The Python side stopped")
    expect(frame()).toContain("exited with code 1")
  })

  test("ctrl+c quits", async () => {
    const { quits } = await mount()
    setup!.mockInput.pressKey("c", { ctrl: true })
    await settle()
    expect(quits()).toBe(1)
  })

  test("every command is one registry: palette, /name and help all list it", async () => {
    await mount(120, 60)
    setup!.mockInput.pressKey("p", { ctrl: true })
    await settle()
    const palette = frame()
    setup!.mockInput.pressEscape()
    await settle()
    await type("/help")
    const help = frame()
    for (const command of COMMANDS) {
      expect(help).toContain(command.title)
    }
    // The palette shows as many rows as fit; the first screenful is enough to
    // prove it reads the registry rather than a list of its own.
    expect(palette).toContain(COMMANDS[0]!.title)
  })

  test("the install check shows each detail in full", async () => {
    await mount()
    await type("/doctor")
    await settle(200)
    const text = frame().replace(/\s+/g, " ")
    expect(text).toContain("work with the standard library alone.")
    expect(text).toContain('pip install "watermark-remover[visible]"')
    saveFrame("doctor-120x40", setup!.captureSpans())
  })

  test("a long history command is cut with an ellipsis before its time", async () => {
    await mount()
    await type("/history")
    await settle(200)
    const row = frame().split("\n").find((line) => line.includes("20:14:56"))!
    expect(row).toContain("wm drafts --rewrite paraphrase")
    expect(row).toContain("...")
    expect(row).toContain("--nfkc  20:14:56")
    saveFrame("history-120x40", setup!.captureSpans())
  })

  test("an unknown /command says so instead of doing nothing", async () => {
    await mount()
    await type("/frobnicate")
    expect(frame()).toContain("No command /frobnicate")
  })

  test("a server entered by hand saves exactly backend, base URL and model", async () => {
    const { ctl, bridge } = await mount()
    await type("/model")
    await settle(200)
    // Rows: Ollama, LM Studio (disabled), Scan again, Enter by hand, No model.
    for (let i = 0; i < 3; i++) setup!.mockInput.pressArrow("down")
    await settle()
    setup!.mockInput.pressEnter()
    await settle()
    expect(frame()).toContain("Enter a server")
    setup!.mockInput.pressArrow("right") // backend: ollama → openai-compatible
    setup!.mockInput.pressTab()
    await setup!.mockInput.typeText("http://127.0.0.1:8080/")
    setup!.mockInput.pressTab()
    await setup!.mockInput.typeText("local-model")
    setup!.mockInput.pressEnter()
    await settle()
    expect(ctl.app.state.endpoint).toEqual({ backend: "openai-compatible", base_url: "http://127.0.0.1:8080", model: "local-model" })
    const saved = bridge.calls.filter((c) => c.method === "save_settings").at(-1)!.params.settings
    expect(Object.keys(saved).sort()).toEqual(["endpoint", "preset"])
    expect(Object.keys(saved.endpoint).sort()).toEqual(["backend", "base_url", "model"])
  })
})
