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
  /** TUI clean target; null means no marked file. Batch bridge callers may omit it. */
  clean_path?: string | null
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

/**
 * Bridge `reveal.marks` are Unicode code-point indexes into `text`.
 * Python already applied the ellipsis shift (`…` is one code point), so this
 * step must not add another +1. A non-BMP character is one code point and two
 * JavaScript units; rendering slices with JavaScript indexes.
 *
 * BMP text takes the no-op path: the same string and the same marks array.
 * `parseFinding` is the only conversion. Calling it again on the object it
 * returned does not shift marks. Calling it again on the original wire object
 * reconverts from those code-point indexes, so a render may read the wire
 * value every frame. Do not copy a converted excerpt and convert the copy.
 */
const convertedReveals = new WeakSet<Reveal>()
const convertedFindings = new WeakSet<Finding>()

function hasNonBmp(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i)
    if (unit >= 0xd800 && unit <= 0xdbff) return true
  }
  return false
}

function codePointToUnit(text: string, codePointIndex: number): number {
  if (codePointIndex <= 0) return 0
  let units = 0
  let points = 0
  while (points < codePointIndex && units < text.length) {
    const unit = text.charCodeAt(units)
    const step = unit >= 0xd800 && unit <= 0xdbff ? 2 : 1
    if (units + step > text.length) return text.length
    units += step
    points += 1
  }
  return units
}

/** Code-point mark indexes → JavaScript string indexes. BMP returns the same object. */
export function convertReveal(reveal: Reveal): Reveal {
  if (convertedReveals.has(reveal) || !hasNonBmp(reveal.text)) {
    convertedReveals.add(reveal)
    return reveal
  }
  const marks = reveal.marks.map(([start, end, name]) => {
    const jsStart = codePointToUnit(reveal.text, start)
    const jsEnd = codePointToUnit(reveal.text, end)
    return [jsStart, jsEnd < jsStart ? jsStart : jsEnd, name] as [number, number, string]
  })
  const next: Reveal = { ...reveal, marks }
  convertedReveals.add(next)
  return next
}

function usableFinding(value: unknown): value is Finding {
  if (!value || typeof value !== "object") return false
  const finding = value as Partial<Finding>
  if (typeof finding.path !== "string") return false
  if (finding.reveal == null) return true
  if (!Array.isArray(finding.reveal)) return false
  return finding.reveal.every((excerpt) => {
    if (!excerpt || typeof excerpt !== "object") return false
    const reveal = excerpt as Partial<Reveal>
    return typeof reveal.text === "string" && Array.isArray(reveal.marks)
  })
}

/**
 * The one Findings parse step. Wire excerpts are converted here, nowhere else.
 * A BMP finding is returned unchanged. A finding whose reveal text is not a
 * string, or whose reveal is not an array of excerpts, is rejected (`null`)
 * so the caller ignores that frame instead of throwing inside the reader.
 */
export function parseFinding(finding: Finding): Finding | null {
  if (!usableFinding(finding)) return null
  if (!finding.reveal) {
    const next: Finding = { ...finding, reveal: [] }
    convertedFindings.add(next)
    return next
  }
  if (convertedFindings.has(finding) || finding.reveal.length === 0) {
    convertedFindings.add(finding)
    return finding
  }
  let changed = false
  const reveal = finding.reveal.map((excerpt) => {
    const next = convertReveal(excerpt)
    if (next !== excerpt) changed = true
    return next
  })
  if (!changed) {
    convertedFindings.add(finding)
    return finding
  }
  const next: Finding = { ...finding, reveal }
  convertedFindings.add(next)
  return next
}

/** What the UI needs from a bridge; tests supply a fake. */
export interface BridgeLike {
  request<T>(method: string, params?: object, onEvent?: (event: BridgeEvent) => void): Promise<T>
  onExit(callback: (reason: string) => void): void
  close(): void
}
