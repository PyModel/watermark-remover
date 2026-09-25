import { BridgeError, type BridgeEvent, type BridgeLike, type Hello, type Plan, type State } from "../src/protocol"

export const PRESETS = [
  { key: "hidden", label: "Hidden marks", description: "Zero-width carriers, bidi controls and AI metadata.", layer: "A", result_class: "Verifiable", flags: [], requires_endpoint: false },
  { key: "hidden-aggressive", label: "Hidden marks, aggressive", description: "Adds NFKC and homoglyph folding.", layer: "A", result_class: "Verifiable", flags: ["--nfkc", "--aggressive-homoglyphs"], requires_endpoint: false },
  { key: "rewrite", label: "Deep clean (LLM rewrite)", description: "A local model rephrases the text.", layer: "B", result_class: "Best-effort", flags: ["--rewrite", "paraphrase"], requires_endpoint: true },
  { key: "image", label: "Images: metadata + degrade", description: "Strips C2PA, perturbs the frequency domain.", layer: "V", result_class: "Best-effort", flags: ["--degrade", "freq-dct"], requires_endpoint: false },
]

export const FILES = [
  "drafts/intro.md",
  "drafts/chapter-01.md",
  "drafts/chapter-02-with-a-long-name-that-truncates.md",
  "assets/cover.png",
  "notes.txt",
]

/** A scripted bridge: the UI under test sees protocol-shaped answers only. */
export class FakeBridge implements BridgeLike {
  calls: { method: string; params: any }[] = []
  onboard = false
  confirm: { kind: string; message: string }[] = []
  private exit: ((reason: string) => void)[] = []

  async request<T>(method: string, params: any = {}, onEvent?: (e: BridgeEvent) => void): Promise<T> {
    this.calls.push({ method, params })
    const state: State | undefined = params.state
    const files = () =>
      (state?.paths.length ? FILES : []).map((display) => ({ path: `/abs/${display}`, display, kind: display.endsWith(".png") ? "image" : "text", size: 100 }))
    switch (method) {
      case "hello":
        return {
          version: "0.5.0",
          presets: PRESETS,
          settings: { preset: null, endpoint: { backend: null, base_url: null, model: null } },
          settings_path: "/tmp/wm/tui.json",
          onboard: this.onboard,
          initial: { paths: [], flags: [] },
          backends: ["ollama", "openai-compatible"],
        } satisfies Hello as T
      case "plan": {
        const preset = PRESETS.find((p) => p.key === state!.preset)!
        const bad = state!.flags.includes("--bogus")
        return {
          ok: !bad,
          error: bad ? "unrecognized arguments: --bogus" : null,
          command: ["wm", ...state!.paths, ...preset.flags, ...state!.flags].join(" "),
          files: bad ? [] : files(),
          files_total: bad ? 0 : files().length,
          discover_error: null,
          layer: bad ? null : preset.layer,
          result_class: bad ? null : preset.result_class,
          output: bad ? null : "writes NAME.cleaned.EXT next to each file; originals are never touched",
          confirm: [],
          estimate_seconds: 0,
          warnings: [],
          preflight_error: null,
        } satisfies Plan as T
      }
      case "inspect": {
        const found = files().map((f, i) => ({
          path: f.path,
          display: f.display,
          kind: f.kind,
          suspicious: i % 2 === 0,
          counts: i % 2 === 0 ? { hidden: 3, metadata: 1 } : { hidden: 0, metadata: 0 },
          lines: i % 2 === 0 ? ["3 zero-width carriers (U+200B)", "1 bidi control (U+202E)", "Frontmatter key: generator"] : [],
          reveal:
            i % 2 === 0
              ? [
                  { line: 3, text: "The committee◆ reviewed the◆ proposal on Tuesday.", marks: [[13, 14, "ZWSP"], [27, 28, "ZWSP"]] as [number, number, string][] },
                  { line: 9, text: "…final figures◆ are attached below.", marks: [[14, 15, "RLO"]] as [number, number, string][] },
                ]
              : [],
          error: null,
        }))
        for (const f of found) onEvent?.({ event: "inspected", data: f })
        return { files: found, cancelled: false } as T
      }
      case "clean": {
        if (this.confirm.length && !(params.confirmed ?? []).length) {
          throw new BridgeError("needs_confirm", "confirm", this.confirm)
        }
        const list = files()
        list.forEach((f, index) => {
          onEvent?.({ event: "file_start", data: { index, total: list.length, display: f.display } })
          onEvent?.({
            event: "file_done",
            data: {
              index,
              display: f.display,
              output: f.path.replace(/(\.\w+)$/, ".cleaned$1"),
              layer: "A",
              result_class: "Verifiable",
              exit_code: 0,
              before: { hidden: 3 },
              after: { hidden: 0 },
              lines: ["Removed 3 hidden characters.", "Dropped frontmatter key generator."],
              diff: "--- a\n+++ b\n@@ -1 +1 @@\n-Hello<U+200B> world\n+Hello world",
              error: index === 3 ? "refusing to treat assets/cover.png as text: binary content" : null,
            },
          })
        })
        return {
          total: list.length,
          errors: 1,
          cancelled: false,
          command: "wm drafts",
          history: { time: "14:02:11", command: "wm drafts", summary: "5 files, 1 error. Verifiable." },
        } as T
      }
      case "detect_endpoints":
        return {
          endpoints: [
            { label: "Ollama", backend: "ollama", base_url: "http://127.0.0.1:11434", reachable: true, models: ["qwen3:14b", "llama3.1:8b"], error: null },
            { label: "LM Studio", backend: "openai-compatible", base_url: "http://127.0.0.1:1234", reachable: false, models: [], error: "connection refused" },
          ],
        } as T
      case "checks":
        return {
          checks: [
            { name: "Python", state: "OK", good: true, detail: "Python 3.13.1. Hidden-mark and metadata cleaning work with the standard library alone.", fix: "", optional: false },
            { name: "Extra: visible", state: "Not installed", good: false, detail: "Images: visible-mark cleaning.", fix: 'pip install "watermark-remover[visible]"', optional: true },
          ],
        } as T
      case "history":
        return {
          entries: [
            {
              time: "20:14:56",
              command: "wm drafts --rewrite paraphrase --rewrite-backend openai-compatible --rewrite-model qwen3:14b --nfkc",
              summary: "5 files, 0 errors. Best-effort.",
            },
          ],
        } as T
      case "save_settings":
        return { path: "/tmp/wm/tui.json" } as T
      case "cancel":
      case "shutdown":
        return { ok: true } as T
    }
    throw new BridgeError("bad_request", `unknown method ${method}`)
  }

  onExit(callback: (reason: string) => void) {
    this.exit.push(callback)
  }

  die(reason: string) {
    for (const callback of this.exit) callback(reason)
  }

  close() {}
}
