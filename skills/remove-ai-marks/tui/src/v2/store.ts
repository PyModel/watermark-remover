import { expandHome, mergeFlags, removeFlag } from "./prompt"
import {
  BridgeError,
  parseFinding,
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
  path?: string
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

function later(ms: number, fn: () => void): ReturnType<typeof setTimeout> {
  const timer = setTimeout(fn, ms)
  timer.unref?.()
  return timer
}

/**
 * Plain state plus one invalidation callback. Views read `app` on each render;
 * only this module talks to the bridge.
 */
export function createApp(bridge: BridgeLike, onQuit: () => void, notify: () => void = () => {}) {
  const app: AppState = {
    hello: null,
    fatal: null,
    state: { paths: [], preset: "hidden", flags: [], endpoint: { ...NO_ENDPOINT }, clean_path: null },
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
  }

  let toastTimer: ReturnType<typeof setTimeout> | undefined
  let planTimer: ReturnType<typeof setTimeout> | undefined
  let planSeq = 0

  const touch = () => notify()
  const snapshot = (): State => JSON.parse(JSON.stringify(app.state))

  function toast(text: string, tone: Toast["tone"] = "info") {
    app.toast = { text, tone }
    clearTimeout(toastTimer)
    toastTimer = later(tone === "error" ? 6000 : 3500, () => {
      app.toast = null
      touch()
    })
    touch()
  }

  function fail(error: unknown) {
    toast(error instanceof Error ? error.message : String(error), "error")
  }

  bridge.onExit((reason) => {
    app.fatal = reason
    touch()
  })

  async function refreshPlan() {
    const seq = ++planSeq
    try {
      const plan = await bridge.request<Plan>("plan", { state: snapshot() })
      if (seq !== planSeq) return
      // A refused plan selects no files. Keep the last list on screen so a
      // typo in a flag shows the error without emptying the view.
      const shown = !plan.ok && app.plan ? { ...plan, files: app.plan.files, files_total: app.plan.files_total } : plan
      app.plan = shown
      if (app.state.clean_path && !shown.files.some((file) => file.path === app.state.clean_path)) {
        app.state.clean_path = null
        toast("The marked file left the list. Mark another with Space.")
      }
      if (app.selected >= shown.files.length) app.selected = Math.max(0, shown.files.length - 1)
      touch()
    } catch (error) {
      if (seq === planSeq) fail(error)
    }
  }

  function schedulePlan() {
    ++planSeq
    clearTimeout(planTimer)
    planTimer = later(60, () => void refreshPlan())
  }

  async function start() {
    try {
      const hello = await bridge.request<Hello>("hello")
      const settings = hello.settings
      const preset = hello.presets.find((p) => p.key === settings.preset)?.key ?? hello.presets[0]!.key
      app.hello = hello
      app.state = {
        paths: hello.initial.paths,
        preset,
        flags: hello.initial.flags,
        endpoint: { ...settings.endpoint },
        clean_path: null,
      }
      if (hello.onboard) app.dialogs = [{ type: "welcome" }]
      touch()
      await refreshPlan()
    } catch (error) {
      app.fatal = error instanceof Error ? error.message : String(error)
      touch()
    }
  }

  function addPaths(tokens: string[]) {
    const added = tokens.map((token) => expandHome(token)).filter((token) => !app.state.paths.includes(token))
    if (!added.length) return
    app.state.paths = [...app.state.paths, ...added]
    touch()
    schedulePlan()
  }

  function removePath(path: string) {
    app.state.paths = app.state.paths.filter((p) => p !== path)
    touch()
    schedulePlan()
  }

  function clearPaths() {
    app.state.paths = []
    app.state.clean_path = null
    app.findings = {}
    app.results = {}
    app.streams = {}
    touch()
    schedulePlan()
  }

  function applyFlags(tokens: string[]) {
    app.state.flags = mergeFlags(app.state.flags, tokens)
    touch()
    schedulePlan()
  }

  function dropFlag(name: string) {
    app.state.flags = removeFlag(app.state.flags, name)
    touch()
    schedulePlan()
  }

  function presets(): Preset[] {
    return app.hello?.presets ?? []
  }

  function currentPreset(): Preset | undefined {
    return presets().find((p) => p.key === app.state.preset)
  }

  function setPreset(key: string, persist = true) {
    app.state.preset = key
    touch()
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
    app.state.endpoint = endpoint
    touch()
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
    app.selected = Math.min(count - 1, Math.max(0, index))
    touch()
  }

  function markFile() {
    if (app.run || app.dialogs.some((dialog) => dialog.type === "confirm")) return
    const file = app.plan?.files[app.selected]
    if (!file) return
    app.state.clean_path = app.state.clean_path === file.path ? null : file.path
    touch()
    schedulePlan()
  }

  const openDialog = (dialog: Dialog) => {
    app.dialogs = [...app.dialogs.filter((d) => d.type !== dialog.type), dialog]
    touch()
  }
  const closeDialog = () => {
    app.dialogs = app.dialogs.slice(0, -1)
    touch()
  }

  function confirm(items: Confirm[]): Promise<boolean> {
    return new Promise((resolve) => openDialog({ type: "confirm", items, resolve }))
  }

  async function inspect() {
    if (app.run) return toast("Already running. Esc stops after the current file.")
    const files = app.plan?.files ?? []
    if (!files.length) return toast("Add a file or folder first.")
    app.run = { kind: "inspect", index: 0, total: files.length, display: files[0]!.display }
    touch()
    try {
      await bridge.request<{ files: Finding[] }>("inspect", { state: snapshot() }, (message) => {
        if (message.event !== "inspected") return
        const finding = parseFinding(message.data)
        if (!finding) return
        app.findings[finding.path] = finding
        if (app.run) app.run = { ...app.run, index: app.run.index + 1, display: finding.display }
        touch()
      })
      toast(`Inspected ${files.length} file${files.length === 1 ? "" : "s"}.`, "success")
    } catch (error) {
      fail(error)
    } finally {
      app.run = null
      touch()
    }
  }

  async function clean(confirmed: string[] = [], state = snapshot()) {
    if (app.run) return toast("Already running. Esc stops after the current file.")
    const plan = app.plan
    if (!plan?.files.length) return toast("Add a file or folder first.")
    if (!plan.ok) return toast(plan.error ?? "the options are not valid", "error")
    const target = plan.files.find((file) => file.path === state.clean_path)
    if (!target) return toast("Mark one file with Space before cleaning.")
    const files = [target]
    app.run = { kind: "clean", index: 0, total: 1, display: target.display, path: target.path }
    app.streams = {}
    touch()
    try {
      const result = await bridge.request<CleanResult>("clean", { state, confirmed }, (message) => {
        if (message.event === "inspected") return
        const path = files[message.data.index]?.path
        if (!path) return
        if (message.event === "file_start") {
          const { index, display } = message.data
          if (app.run) app.run = { ...app.run, index, display }
        } else if (message.event === "token") {
          const chunk = message.data.text
          app.streams[path] = ((app.streams[path] ?? "") + chunk).slice(-8000)
        } else {
          app.results[path] = message.data
        }
        touch()
      })
      app.history = [result.history, ...app.history]
      const tail = result.cancelled ? ", then stopped" : ""
      toast(`Cleaned ${result.total - result.errors} of ${result.total}${tail}.`, result.errors ? "error" : "success")
    } catch (error) {
      if (error instanceof BridgeError && error.code === "needs_confirm") {
        app.run = null
        touch()
        const items = error.confirm
        if (await confirm(items)) return await clean(items.map((item) => item.kind), state)
        return toast("Nothing was written.")
      }
      fail(error)
    } finally {
      app.run = null
      touch()
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
    app.scanning = true
    touch()
    try {
      const { endpoints } = await bridge.request<{ endpoints: DetectedEndpoint[] }>("detect_endpoints")
      app.endpoints = endpoints
    } catch (error) {
      fail(error)
      app.endpoints = []
    } finally {
      app.scanning = false
      touch()
    }
  }

  async function loadChecks() {
    try {
      const { checks } = await bridge.request<{ checks: Check[] }>("checks")
      app.checks = checks
      touch()
    } catch (error) {
      fail(error)
    }
  }

  async function loadHistory() {
    try {
      const { entries } = await bridge.request<{ entries: HistoryEntry[] }>("history")
      app.history = entries
      touch()
    } catch (error) {
      fail(error)
    }
  }

  /** Close the first-run flow and remember it ran, whether finished or skipped. */
  async function finishOnboarding(skipped: boolean) {
    app.dialogs = app.dialogs.filter((d) => d.type !== "welcome" && !(d.type === "model" && d.onboarding))
    if (app.hello) app.hello = { ...app.hello, onboard: false }
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
    markFile,
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
