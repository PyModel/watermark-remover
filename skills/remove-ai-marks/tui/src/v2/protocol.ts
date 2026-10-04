/** Wire types for the bridge. PROTOCOL.md is the source of truth. */

export interface Endpoint {
  backend: string | null
  base_url: string | null
  model: string | null
}

export interface State {
  paths: string[]
  preset: string
  flags: string[]
  endpoint: Endpoint
}

export interface Preset {
  key: string
  label: string
  description: string
  layer: string
  result_class: string
  flags: string[]
  requires_endpoint: boolean
}

export interface Settings {
  preset: string | null
  endpoint: Endpoint
}

export interface Hello {
  version: string
  presets: Preset[]
  settings: Settings
  settings_path: string
  onboard: boolean
  initial: { paths: string[]; flags: string[] }
  backends: string[]
}

export interface FileEntry {
  path: string
  display: string
  kind: string
  /** Null when the file could not be read. */
  size: number | null
}

export interface Confirm {
  kind: string
  message: string
}

export interface Plan {
  ok: boolean
  error: string | null
  command: string
  /** Capped at 2000; `files_total` is the full count. */
  files: FileEntry[]
  files_total: number
  discover_error: string | null
  /** Null when the plan is refused (`ok: false`). */
  layer: string | null
  result_class: string | null
  output: string | null
  confirm: Confirm[]
  estimate_seconds: number
  warnings: string[]
  preflight_error: string | null
}

/** One line of a file with its hidden characters shown as ◆ at `marks`. */
export interface Reveal {
  line: number
  text: string
  marks: [number, number, string][]
}

export interface Finding {
  path: string
  display: string
  kind: string
  suspicious: boolean
  counts: Record<string, number>
  lines: string[]
  reveal: Reveal[]
  error: string | null
}

export interface FileDone {
  index: number
  display: string
  output: string | null
  layer: string
  result_class: string
  exit_code: number
  before: Record<string, number>
  after: Record<string, number>
  lines: string[]
  diff: string | null
  error: string | null
}

export interface HistoryEntry {
  time: string
  command: string
  summary: string
}

export interface CleanResult {
  total: number
  errors: number
  cancelled: boolean
  command: string
  history: HistoryEntry
}

export interface DetectedEndpoint {
  label: string
  backend: string
  base_url: string
  reachable: boolean
  models: string[]
  error: string | null
}

export interface Check {
  name: string
  state: string
  good: boolean
  detail: string
  fix: string
  optional: boolean
}

/** Every event the bridge streams, keyed by name. */
export type BridgeEvent =
  | { event: "inspected"; data: Finding }
  | { event: "file_start"; data: { index: number; total: number; display: string } }
  | { event: "token"; data: { index: number; text: string } }
  | { event: "file_done"; data: FileDone }

export class BridgeError extends Error {
  constructor(
    readonly code: string,
    message: string,
    /** Set for `needs_confirm`: the gates the clean stopped at. */
    readonly confirm: Confirm[] = [],
  ) {
    super(message)
  }
}

/** What the UI needs from a bridge; tests supply a fake. */
export interface BridgeLike {
  request<T>(method: string, params?: object, onEvent?: (event: BridgeEvent) => void): Promise<T>
  onExit(callback: (reason: string) => void): void
  close(): void
}
