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

async function call<T>(path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {}
  const key = apiKey()
  if (key) headers.Authorization = `Bearer ${key}`
  if (body !== undefined) headers["Content-Type"] = "application/json"
  let response: Response
  try {
    response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError("Can't reach wm-serve. Check that it is still running, then try again.", 0)
  }
  const payload = await response.json().catch(() => null)
  if (response.ok && payload?.ok !== false) return payload as T
  throw new ApiError(errorMessage(response.status, payload?.error), response.status)
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
  if (status === 502 && typeof serverMessage === "string") return sentenceCase(serverMessage)
  if (status >= 500) return "wm-serve hit an internal error. Its terminal log has the details."
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

export const capabilities = () => call<Capabilities>("/capabilities")
export const inspect = (file: string, name: string) => call<Inspection>("/inspect", { file, name })
export const clean = (file: string, name: string, options: CleanOptions) =>
  call<Cleaned>("/clean", { file, name, options })
export const detect = (file: string, name: string) => call<Detected>("/detect", { file, name })
export const rewrite = (file: string, name: string, strength: Strength, lang?: string) =>
  call<Cleaned>("/rewrite", { file, name, strength, ...(lang ? { lang } : {}) })
export const removeVisible = (file: string, name: string, box: Box, backend: "texture" | "simple") =>
  call<Cleaned>("/visible", { file, name, box, backend })

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
