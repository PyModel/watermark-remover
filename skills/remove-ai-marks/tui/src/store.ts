import { batch } from "solid-js"
import { createStore, produce } from "solid-js/store"
import { expandHome, mergeFlags, removeFlag } from "./prompt"
import {
  BridgeError,
  type BridgeLike,
  type Check,
  type CleanResult,
  type Confirm,
  type DetectedEndpoint,
  type Endpoint,
  type FileDone,
  type Finding,
  type Hello,
  type HistoryEntry,
  type Plan,
  type Preset,
  type Settings,
  type State,
} from "./protocol"

export type Dialog =
  | { type: "palette" }
  | { type: "preset"; onboarding?: boolean }
  | { type: "model"; onboarding?: boolean }
  | { type: "welcome" }
  | { type: "help" }
  | { type: "doctor" }
  | { type: "history" }
  | { type: "confirm"; items: Confirm[]; resolve: (ok: boolean) => void }

export interface Run {
  kind: "clean" | "inspect"
  index: number
  total: number
  display: string
}

export interface Toast {
  text: string
  tone: "info" | "error" | "success"
}

export interface AppState {
  hello: Hello | null
  fatal: string | null
  state: State
  plan: Plan | null
  findings: Record<string, Finding>
  results: Record<string, FileDone>
  streams: Record<string, string>
  selected: number
  run: Run | null
  dialogs: Dialog[]
  toast: Toast | null
  history: HistoryEntry[]
  endpoints: DetectedEndpoint[] | null
  scanning: boolean
  checks: Check[] | null
}

export const NO_ENDPOINT: Endpoint = { backend: null, base_url: null, model: null }

/**
 * Everything the UI knows, and every action it can take.  Views read the
 * store and call these; only this module talks to the bridge.
 */
export function createApp(bridge: BridgeLike, onQuit: () => void) {
  const [app, set] = createStore<AppState>({
    hello: null,
    fatal: null,
    state: { paths: [], preset: "hidden", flags: [], endpoint: { ...NO_ENDPOINT } },
    plan: null,
    findings: {},
    results: {},
    streams: {},
    selected: 0,
    run: null,
    dialogs: [],
    toast: null,
    history: [],
    endpoints: null,
    scanning: false,
    checks: null,
  })

  let toastTimer: ReturnType<typeof setTimeout> | undefined
  let planTimer: ReturnType<typeof setTimeout> | undefined
  let planSeq = 0

  const snapshot = (): State => JSON.parse(JSON.stringify(app.state))

  function toast(text: string, tone: Toast["tone"] = "info") {
    set("toast", { text, tone })
    clearTimeout(toastTimer)
    toastTimer = setTimeout(() => set("toast", null), tone === "error" ? 6000 : 3500)
  }

  function fail(error: unknown) {
    const message = error instanceof Error ? error.message : String(error)
    toast(message, "error")
  }

  bridge.onExit((reason) => set("fatal", reason))

  // -- plan preview --------------------------------------------------------

  async function refreshPlan() {
    const seq = ++planSeq
    try {
      const plan = await bridge.request<Plan>("plan", { state: snapshot() })
      if (seq !== planSeq) return
      // A refused plan selects no files.  Keep the last list on screen so a
      // typo in a flag shows the error without emptying the view.
      const shown = !plan.ok && app.plan ? { ...plan, files: app.plan.files, files_total: app.plan.files_total } : plan
      batch(() => {
        set("plan", shown)
        if (app.selected >= shown.files.length) set("selected", Math.max(0, shown.files.length - 1))
      })
    } catch (error) {
      if (seq === planSeq) fail(error)
    }
  }

  function schedulePlan() {
    clearTimeout(planTimer)
    planTimer = setTimeout(refreshPlan, 60)
  }

  // -- startup -------------------------------------------------------------

  async function start() {
    try {
      const hello = await bridge.request<Hello>("hello")
      const settings = hello.settings
      const preset = hello.presets.find((p) => p.key === settings.preset)?.key ?? hello.presets[0]!.key
      batch(() => {
        set("hello", hello)
        set("state", {
          paths: hello.initial.paths,
          preset,
          flags: hello.initial.flags,
          endpoint: { ...settings.endpoint },
        })
        if (hello.onboard) set("dialogs", [{ type: "welcome" }])
      })
      await refreshPlan()
    } catch (error) {
      set("fatal", error instanceof Error ? error.message : String(error))
    }
  }

  // -- editing -------------------------------------------------------------

  function addPaths(tokens: string[]) {
    const added = tokens.map((token) => expandHome(token)).filter((token) => !app.state.paths.includes(token))
    if (!added.length) return
    set("state", "paths", (paths) => [...paths, ...added])
    schedulePlan()
  }

  function removePath(path: string) {
    set("state", "paths", (paths) => paths.filter((p) => p !== path))
    schedulePlan()
  }

  function clearPaths() {
    batch(() => {
      set("state", "paths", [])
      set("findings", {})
      set("results", {})
      set("streams", {})
    })
    schedulePlan()
  }

  function applyFlags(tokens: string[]) {
    set("state", "flags", (flags) => mergeFlags(flags, tokens))
    schedulePlan()
  }

  function dropFlag(name: string) {
    set("state", "flags", (flags) => removeFlag(flags, name))
    schedulePlan()
  }

  function presets(): Preset[] {
    return app.hello?.presets ?? []
  }

  function currentPreset(): Preset | undefined {
    return presets().find((p) => p.key === app.state.preset)
  }

  function setPreset(key: string, persist = true) {
    set("state", "preset", key)
    schedulePlan()
    if (persist) void saveSettings({ preset: key })
  }

  function cyclePreset(step: number) {
    const list = presets()
    if (!list.length) return
    const at = list.findIndex((p) => p.key === app.state.preset)
    setPreset(list[(at + step + list.length) % list.length]!.key)
  }

  function setEndpoint(endpoint: Endpoint, persist = true) {
    set("state", "endpoint", endpoint)
    schedulePlan()
    if (persist) void saveSettings({ endpoint })
  }

  async function saveSettings(patch: Partial<Settings>) {
    const current: Settings = { preset: app.state.preset, endpoint: { ...app.state.endpoint }, ...patch }
    try {
      await bridge.request("save_settings", { settings: current })
    } catch (error) {
      toast(`Settings not saved: ${error instanceof Error ? error.message : error}`, "error")
    }
  }

  function select(index: number) {
    const count = app.plan?.files.length ?? 0
    if (!count) return
    set("selected", Math.min(count - 1, Math.max(0, index)))
  }

  // -- dialogs -------------------------------------------------------------

  const openDialog = (dialog: Dialog) => set("dialogs", (stack) => [...stack.filter((d) => d.type !== dialog.type), dialog])
  const closeDialog = () => set("dialogs", (stack) => stack.slice(0, -1))

  function confirm(items: Confirm[]): Promise<boolean> {
    return new Promise((resolve) => openDialog({ type: "confirm", items, resolve }))
  }

  // -- work ----------------------------------------------------------------

  async function inspect() {
    if (app.run) return toast("Already running. Esc stops after the current file.")
    const files = app.plan?.files ?? []
    if (!files.length) return toast("Add a file or folder first.")
    set("run", { kind: "inspect", index: 0, total: files.length, display: files[0]!.display })
    try {
      await bridge.request<{ files: Finding[] }>("inspect", { state: snapshot() }, (message) => {
        if (message.event !== "inspected") return
        const finding = message.data
        batch(() => {
          set("findings", finding.path, finding)
          set("run", (run) => (run ? { ...run, index: run.index + 1, display: finding.display } : run))
        })
      })
      toast(`Inspected ${files.length} file${files.length === 1 ? "" : "s"}.`, "success")
    } catch (error) {
      fail(error)
    } finally {
      set("run", null)
    }
  }

  async function clean(confirmed: string[] = []) {
    if (app.run) return toast("Already running. Esc stops after the current file.")
    const plan = app.plan
    if (!plan?.files.length) return toast("Add a file or folder first.")
    if (!plan.ok) return toast(plan.error ?? "the options are not valid", "error")
    const files = plan.files
    set("run", { kind: "clean", index: 0, total: files.length, display: files[0]!.display })
    set("streams", {})
    try {
      const result = await bridge.request<CleanResult>("clean", { state: snapshot(), confirmed }, (message) => {
        if (message.event === "inspected") return
        const path = files[message.data.index]?.path
        if (!path) return
        if (message.event === "file_start") {
          const { index, display } = message.data
          set("run", (run) => (run ? { ...run, index, display } : run))
        } else if (message.event === "token") {
          const chunk = message.data.text
          set("streams", path, (text) => ((text ?? "") + chunk).slice(-8000))
        } else {
          set("results", path, message.data)
        }
      })
      batch(() => {
        set("history", (entries) => [result.history, ...entries])
        const tail = result.cancelled ? ", then stopped" : ""
        toast(
          `Cleaned ${result.total - result.errors} of ${result.total}${tail}.`,
          result.errors ? "error" : "success",
        )
      })
    } catch (error) {
      if (error instanceof BridgeError && error.code === "needs_confirm") {
        set("run", null)
        const items = error.confirm
        if (await confirm(items)) return clean(items.map((item) => item.kind))
        return toast("Nothing was written.")
      }
      fail(error)
    } finally {
      set("run", null)
    }
  }

  async function cancel() {
    if (!app.run) return
    try {
      await bridge.request("cancel")
      toast("Stopping after the current file.")
    } catch (error) {
      fail(error)
    }
  }

  async function detect() {
    set("scanning", true)
    try {
      const { endpoints } = await bridge.request<{ endpoints: DetectedEndpoint[] }>("detect_endpoints")
      set("endpoints", endpoints)
    } catch (error) {
      fail(error)
      set("endpoints", [])
    } finally {
      set("scanning", false)
    }
  }

  async function loadChecks() {
    try {
      const { checks } = await bridge.request<{ checks: Check[] }>("checks")
      set("checks", checks)
    } catch (error) {
      fail(error)
    }
  }

  async function loadHistory() {
    try {
      const { entries } = await bridge.request<{ entries: HistoryEntry[] }>("history")
      set("history", entries)
    } catch (error) {
      fail(error)
    }
  }

  /** Close the first-run flow and remember it ran, whether finished or skipped. */
  async function finishOnboarding(skipped: boolean) {
    set("dialogs", (stack) => stack.filter((d) => d.type !== "welcome" && !(d.type === "model" && d.onboarding)))
    if (app.hello) set("hello", produce((hello) => void (hello!.onboard = false)))
    await saveSettings({})
    toast(skipped ? "Setup skipped. /setup brings it back." : "Setup saved. /setup changes it.", "success")
  }

  let quitting = false
  function quit() {
    if (quitting) return
    quitting = true
    bridge.close()
    onQuit()
  }

  return {
    app,
    start,
    addPaths,
    removePath,
    clearPaths,
    applyFlags,
    dropFlag,
    presets,
    currentPreset,
    setPreset,
    cyclePreset,
    setEndpoint,
    select,
    openDialog,
    closeDialog,
    inspect,
    clean,
    cancel,
    detect,
    loadChecks,
    loadHistory,
    finishOnboarding,
    toast,
    quit,
  }
}

export type AppController = ReturnType<typeof createApp>

export function endpointLabel(endpoint: Endpoint): string | null {
  if (!endpoint.backend || !endpoint.base_url) return null
  const host = endpoint.base_url.replace(/^https?:\/\//, "")
  return endpoint.model ? `${endpoint.model} on ${host}` : host
}
