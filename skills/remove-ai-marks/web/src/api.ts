/**
 * The wm-serve endpoints this page uses (server.py, /openapi.json). Only the
 * fields the page reads are typed; reports pass through otherwise untouched.
 */
export type Kind = "text" | "image" | "container" | "unknown"
export type Confidence = "confirmed" | "probable" | "informational" | "likely_false_positive"
export type DetectionStatus = "DETECTED" | "INCONCLUSIVE" | "NOT_DETECTED" | "NOT_RUN"

export interface TextHit {
  codepoint: string
  label: string
  count: number
  kind: string
  confidence: Confidence
}

export interface Stylometry {
  score: number
  confidence_level: string
  status?: string
  word_count: number
  notes: string[]
}

export interface InspectReport {
  format?: string
  length?: number
  suspicious_total?: number
  hits?: TextHit[]
  stylometry?: Stylometry
  has_c2pa?: boolean
  has_ai_metadata?: boolean
  findings?: string[]
  findings_confidence?: Confidence[]
  layer_a_hits?: TextHit[]
  note?: string
  notes?: string[]
}

export interface Inspection {
  kind: Kind
  suspicious: boolean
  detection_status: DetectionStatus
  report: InspectReport
}

export interface ActionDetail {
  code: string
  effect: string
  text: string
}

export interface CleanReport {
  kind?: string
  rewrite?: Record<string, unknown> & { mode?: string; note?: string; strength?: string; model?: string }
  status?: string
  note?: string
  synthid_removal?: { note?: string; strength?: number }
  pixel_removal?: { available: boolean; error?: string }
  stats?: {
    removed: Record<string, number>
    replaced: Record<string, number>
    preserved_semantic: Record<string, number>
    removed_count: number
    replaced_count: number
    preserved_count: number
  }
  actions?: string[]
  action_details?: ActionDetail[]
  still_has_c2pa?: boolean
  still_has_ai_metadata?: boolean
  post_findings?: string[]
  bytes_in?: number
  bytes_out?: number
}

export interface Cleaned {
  kind: Exclude<Kind, "unknown">
  cleaned: string
  report: CleanReport
}

export type CleanOption =
  | "nfkc"
  | "aggressive_homoglyphs"
  | "keep_non_ai_metadata"
  | "also_layer_a_text"
  | "remove_synthid"
  | "wmct_marker"
  | "detect_before"
  | "detect_after"
export type CleanOptions = Partial<Record<CleanOption, boolean>>

export interface Capabilities {
  version: string
  /** public mode: rate limited, and the endpoints in `disabled` answer 403 */
  public: boolean
  disabled: string[]
  max_input_bytes: number
  tools: Record<"c2patool" | "exiftool" | "qpdf", boolean>
  pixel_backends: Record<"ctrlregen" | "diffusion", boolean>
  scorers: Record<"synthid" | "synthid_http" | "stylometry", boolean>
  text_detectors: Record<string, boolean>
  harnesses: Record<string, boolean>
  extras: Record<string, { available: boolean; hint?: string }>
  layer_b: {
    backend: string | null
    configured: boolean
    base_url_host: string | null
    loopback: boolean
    endpoint_allowed: boolean
  }
}

/** One detector's answer; shapes differ per vendor, so only the shared fields are typed. */
export interface Detection {
  detector?: string
  available?: boolean
  configured?: boolean
  verdict?: string
  is_watermarked?: boolean
  score?: number
  confidence_level?: string
  error?: string
  reason?: string
}

export interface Detected {
  kind: Kind
  detections: Detection[]
}

export const REWRITE_STRENGTHS = ["paraphrase", "backtranslate", "structural", "humanize", "code", "tsapa"] as const
export type Strength = (typeof REWRITE_STRENGTHS)[number]
export type Box = [number, number, number, number]

import { send } from "./engine"
import { BROWSER } from "./target"

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

const KEY = "wm-serve-api-key"

export function apiKey(): string {
  try {
    return sessionStorage.getItem(KEY) ?? ""
  } catch {
    return ""
  }
}

export function setApiKey(value: string) {
  try {
    sessionStorage.setItem(KEY, value)
  } catch {
    // Private mode: the key lives for this page only, which is still correct.
  }
}

type Payload = { ok?: boolean; error?: unknown } | null

/** One request -> (status, parsed JSON). wm-serve over fetch, or the in-page engine. */
async function transport(path: string, body?: unknown): Promise<{ status: number; ok: boolean; payload: Payload }> {
  const json = body === undefined ? undefined : JSON.stringify(body)
  if (BROWSER) {
    const answer = await send({ type: "call", method: json === undefined ? "GET" : "POST", path, body: json })
    if (answer.status === 0) throw new ApiError(UNREACHABLE, 0)
    return { status: answer.status, ok: answer.status < 400, payload: JSON.parse(answer.payload) }
  }
  const headers: Record<string, string> = {}
  const key = apiKey()
  if (key) headers.Authorization = `Bearer ${key}`
  if (json !== undefined) headers["Content-Type"] = "application/json"
  let response: Response
  try {
    response = await fetch(path, { method: json === undefined ? "GET" : "POST", headers, body: json })
  } catch {
    throw new ApiError(UNREACHABLE, 0)
  }
  return { status: response.status, ok: response.ok, payload: await response.json().catch(() => null) }
}

const UNREACHABLE = BROWSER
  ? "The cleaning engine didn't load. Check your connection, then try again."
  : "Can't reach wm-serve. Check that it is still running, then try again."

async function call<T>(path: string, body?: unknown): Promise<T> {
  const { status, ok, payload } = await transport(path, body)
  if (ok && payload?.ok !== false) return payload as unknown as T
  throw new ApiError(errorMessage(status, payload?.error), status)
}

function errorMessage(status: number, serverMessage: unknown): string {
  if (status === 401) return "This server needs an API key."
  if (status === 413) return "This file is over this server's size limit."
  if (status === 429) return "Too many requests from this connection. Wait a minute, then try again."
  if (status === 503) return "The server is busy with other files. Try again in a few seconds."
  if (status === 421) return "This server doesn't answer to this address."
  if (status === 403 && typeof serverMessage === "string" && serverMessage.includes("cross-origin")) {
    return "The server refused a request from another site."
  }
  if (BROWSER && status === 502 && typeof serverMessage === "string" && usingHosted()) {
    const code = /HTTP (\d{3})/.exec(serverMessage)?.[1]
    if (code === "429") return "Today's free hosted rewrites are used up. Try again after midnight UTC, or add your own model server in the Engine view."
    if (code === "413") return `This text is over the hosted model's ${HOSTED_LIMITS.chars.toLocaleString()} characters. Rewrite a shorter part, or use your own model server.`
    if (code === "400" || code === "403") return "The hosted model only takes rewrites from this page. Reload the page and try again."
    if (/timed out/.test(serverMessage)) return "The hosted model didn't answer in time. Try a shorter text."
    return "The hosted model is unavailable right now. Try again in a minute."
  }
  if (BROWSER && status === 502 && typeof serverMessage === "string" && /connection failed|timed out/.test(serverMessage)) {
    return /timed out/.test(serverMessage)
      ? "Your model server didn't answer in time. Try a shorter text or a lighter strength."
      : "Couldn't reach your model server. Check it's running and allows this site; the Engine view lists the steps."
  }
  if (status === 502 && typeof serverMessage === "string") return sentenceCase(serverMessage)
  if (status >= 500) {
    return BROWSER
      ? "The cleaning engine hit an internal error. Reload the page and try again."
      : "wm-serve hit an internal error. Its terminal log has the details."
  }
  if (typeof serverMessage === "string" && serverMessage.includes("binary container as text")) {
    return "This looks like a binary file with a text extension. Rename it with its real extension, such as .docx or .pdf, and drop it again."
  }
  if (typeof serverMessage === "string" && serverMessage.includes("too large for the texture backend")) {
    return "This box is too large to fill with a copied patch. Choose “Blend surrounding colors” and try again."
  }
  if (typeof serverMessage === "string" && serverMessage) return sentenceCase(serverMessage)
  return `wm-serve answered ${status}.`
}

function sentenceCase(message: string): string {
  const text = message.replace(/^error:\s*/i, "")
  return text.charAt(0).toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? "" : ".")
}

export const capabilities = async () => {
  if (BROWSER) {
    const settings = usingHosted() ? { backend: "hosted", base_url: `${location.origin}/api` } : (modelServer() ?? {})
    await send({ type: "configure", settings: JSON.stringify(settings) })
  }
  return call<Capabilities>("/capabilities")
}
export const inspect = (file: string, name: string) => call<Inspection>("/inspect", { file, name })
export const clean = (file: string, name: string, options: CleanOptions) =>
  call<Cleaned>("/clean", { file, name, options })
export const detect = (file: string, name: string) => call<Detected>("/detect", { file, name })
export const rewrite = (file: string, name: string, strength: Strength, lang?: string) =>
  call<Cleaned>("/rewrite", { file, name, strength, ...(lang ? { lang } : {}) })
export const removeVisible = (file: string, name: string, box: Box, backend: "texture" | "simple") =>
  call<Cleaned>("/visible", { file, name, box, backend })

/** The visitor's own model server for Rewrite (static build only); kept in this browser. */
export interface ModelServer {
  /** "hosted": the free model behind this site's /api (pages/worker.ts) */
  backend: "ollama" | "openai-compatible" | "hosted"
  base_url: string
  model: string
  api_key: string
}
const MODEL_SERVER = "wm-model-server"
export const DEFAULT_MODEL_SERVER: ModelServer = { backend: "ollama", base_url: "http://localhost:11434", model: "", api_key: "" }
export const HOSTED_MODEL: ModelServer = { backend: "hosted", base_url: "", model: "hosted", api_key: "" }
export const HOSTED_LIMITS = { perDay: 20, chars: 8000 }

function savedModelServer(): ModelServer | null {
  try {
    const saved = JSON.parse(localStorage.getItem(MODEL_SERVER) ?? "null")
    return saved && typeof saved === "object" ? { ...DEFAULT_MODEL_SERVER, ...saved } : null
  } catch {
    return null
  }
}
// The copy in memory is what the engine gets, so blocked storage only costs persistence.
let current = savedModelServer()

export const modelServer = (): ModelServer | null => current
export const usingHosted = () => current?.backend === "hosted"

export function setModelServer(value: ModelServer | null) {
  current = value
  try {
    if (value) localStorage.setItem(MODEL_SERVER, JSON.stringify(value))
    else localStorage.removeItem(MODEL_SERVER)
  } catch {
    // Private mode: the setting lasts until the tab closes.
  }
}

/** File -> base64 without building a giant binary string by hand. */
export function toBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "")
    reader.onerror = () => reject(new ApiError("This file couldn't be read from disk.", 0))
    reader.readAsDataURL(blob)
  })
}

export function fromBase64(data: string): Uint8Array<ArrayBuffer> {
  const binary = atob(data)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/** notes.md -> notes.cleaned.md, the CLI's naming. Visible-mark output is always PNG. */
export function cleanedName(name: string, extension?: string): string {
  const dot = name.lastIndexOf(".")
  const stem = dot > 0 ? name.slice(0, dot) : name
  const ext = extension ?? (dot > 0 ? name.slice(dot) : "")
  return `${stem}.cleaned${ext}`
}
