/**
 * The free hosted model for Rewrite on watermark.pymodel.com: an OpenAI-compatible
 * /api/v1/chat/completions in front of Cloudflare Workers AI, so Layer B's
 * openai-compatible backend talks to it unchanged. Everything else on the site
 * is static assets; only /api/* reaches this code (assets.run_worker_first; entry: worker.ts).
 *
 * Free means free: Workers AI gives 10,000 neurons a day at no charge, reset at
 * 00:00 UTC, and bills past that on Workers Paid
 * (https://developers.cloudflare.com/workers-ai/platform/pricing/). The account's
 * plan isn't readable here, so this Worker stops itself at DAILY_NEURONS.
 * It never logs or stores the text it rewrites.
 */
import prompts from "./rewrite-prompts.json"

export const MODEL = "@cf/google/gemma-4-26b-a4b-it"
// Pricing page, same model: 9,091 neurons per M input tokens, 27,273 per M output tokens.
const NEURONS_PER_INPUT_TOKEN = 9091 / 1e6
const NEURONS_PER_OUTPUT_TOKEN = 27273 / 1e6
/** 80% of the free 10,000, so estimates that run low can't push the account past it. */
export const DAILY_NEURONS = 8000
export const PER_CLIENT_DAILY = 20
export const MAX_TEXT_CHARS = 8000
const MAX_BODY_BYTES = 64 * 1024
const MAX_OUTPUT_TOKENS = 4096

/**
 * The lead-in of every prompt Layer B sends, generated from rewrite_text.build_prompt by build.ts.
 * __LANG__ / __ORIGINAL_LANG__ mark the language names back-translation fills in.
 */
const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
const ALLOWED = (prompts as { strength: string; lead: string }[]).map(
  (p) => new RegExp(`^${escape(p.lead).replace(/__(?:ORIGINAL_)?LANG__/g, "[^\\n]{1,60}")}$`),
)

interface AiBinding {
  run(model: string, input: Record<string, unknown>): Promise<Record<string, unknown>>
}
interface BudgetStub {
  fetch(url: string, init: { method: string; body: string }): Promise<Response>
}
export interface Env {
  AI: AiBinding
  BUDGET: { idFromName(name: string): unknown; get(id: unknown): BudgetStub }
  ASSETS?: { fetch(request: Request): Promise<Response> }
}

const json = (status: number, body: unknown, extra: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...extra },
  })
const refuse = (status: number, message: string, extra?: Record<string, string>) => json(status, { error: { message } }, extra)

export const utcDay = (now = Date.now()) => new Date(now).toISOString().slice(0, 10)
export const estimateNeurons = (inputTokens: number, outputTokens: number) =>
  inputTokens * NEURONS_PER_INPUT_TOKEN + outputTokens * NEURONS_PER_OUTPUT_TOKEN

/** A prompt is accepted only if its lead-in is one of Layer B's, so this is a rewriter, not a free chatbot. */
export function promptText(content: string): string | null {
  const split = content.indexOf("\n\n---\n")
  if (split < 0) return null
  return ALLOWED.some((pattern) => pattern.test(content.slice(0, split))) ? content.slice(split + 6) : null
}

async function clientKey(request: Request, day: string): Promise<string> {
  // A day-salted hash, so the budget object never holds a raw IP address.
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown"
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${day}:${ip}`))
  return [...new Uint8Array(digest).slice(0, 12)].map((b) => b.toString(16).padStart(2, "0")).join("")
}

export async function handle(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url)
  if (url.pathname !== "/api/v1/chat/completions") return refuse(404, "not found")
  if (request.method !== "POST") return refuse(405, "use POST")
  // Same-origin only: the page's own engine is the one caller.
  // ponytail: Origin stops other sites and casual scripts, not curl (it can send any Origin);
  // the daily budget is what keeps the cost at zero. Add Turnstile if abuse eats the free quota.
  if (request.headers.get("Origin") !== url.origin) return refuse(403, "this endpoint serves the watermark-remover page only")
  const length = Number(request.headers.get("Content-Length") ?? "0")
  if (length > MAX_BODY_BYTES) return refuse(413, "request too large")

  let body: { messages?: unknown; stream?: unknown; temperature?: unknown }
  try {
    const raw = await request.text()
    if (raw.length > MAX_BODY_BYTES) return refuse(413, "request too large")
    body = JSON.parse(raw)
  } catch {
    return refuse(400, "invalid JSON")
  }
  if (body.stream) return refuse(400, "streaming is not supported")
  const messages = body.messages
  const content =
    Array.isArray(messages) && messages.length === 1 && messages[0]?.role === "user" && typeof messages[0]?.content === "string"
      ? (messages[0].content as string)
      : null
  if (content === null) return refuse(400, "send exactly one user message")
  const text = promptText(content)
  if (text === null) return refuse(400, "only watermark-remover rewrite prompts are accepted")
  if (text.length > MAX_TEXT_CHARS) return refuse(413, `text over ${MAX_TEXT_CHARS} characters`)

  // ~3 characters per token is pessimistic for English, so the reservation runs high.
  const inputTokens = Math.ceil(content.length / 3)
  const maxTokens = Math.min(MAX_OUTPUT_TOKENS, Math.ceil(text.length / 2) + 256)
  const reserved = estimateNeurons(inputTokens, maxTokens)
  const day = utcDay()
  const budget = env.BUDGET.get(env.BUDGET.idFromName("global"))
  const call = (op: string, params: Record<string, string | number>) =>
    budget.fetch(`https://budget/${op}?${new URLSearchParams({ day, ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])) })}`, {
      method: "POST",
      body: "",
    })
  const grant = await call("reserve", { client: await clientKey(request, day), neurons: reserved })
  if (!grant.ok) {
    const { reason } = (await grant.json()) as { reason: string }
    return refuse(429, reason === "client" ? "this connection used today's free rewrites" : "today's free rewrites are used up", {
      "Retry-After": String(Math.ceil((Date.parse(`${day}T00:00:00Z`) + 86_400_000 - Date.now()) / 1000)),
    })
  }

  const temperature = typeof body.temperature === "number" ? Math.min(Math.max(body.temperature, 0), 1.5) : 0.9
  let result: Record<string, unknown>
  try {
    // Gemma 4 here reasons by default and can spend the whole budget before answering.
    result = await env.AI.run(MODEL, {
      messages: [{ role: "user", content }],
      max_tokens: maxTokens,
      temperature,
      chat_template_kwargs: { enable_thinking: false },
    })
  } catch {
    // Unknown whether the call was billed, so the reservation stands.
    return refuse(502, "the hosted model is unavailable; try again later")
  }
  const choices = result.choices as { message?: { content?: unknown } }[] | undefined
  const output = typeof result.response === "string" ? result.response : choices?.[0]?.message?.content
  const usage = result.usage as { prompt_tokens?: number; completion_tokens?: number; neurons?: number } | undefined
  // Workers AI reports the neurons it charged; tokens at list price are the fallback.
  const spent =
    typeof usage?.neurons === "number"
      ? usage.neurons
      : usage?.prompt_tokens !== undefined && usage.completion_tokens !== undefined
        ? estimateNeurons(usage.prompt_tokens, usage.completion_tokens)
        : reserved
  await call("settle", { delta: spent - reserved })
  if (typeof output !== "string" || !output.trim()) return refuse(502, "the hosted model returned no text")
  return json(200, {
    object: "chat.completion",
    model: MODEL,
    choices: [{ index: 0, message: { role: "assistant", content: output }, finish_reason: "stop" }],
    usage: usage && { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens },
  })
}

interface Storage {
  get<T>(key: string): Promise<T | undefined>
  put(key: string, value: unknown): Promise<void>
}
interface Day {
  day: string
  neurons: number
  clients: Record<string, number>
}

/**
 * One Durable Object holds today's spend and per-client counts, so every reservation sees
 * every earlier one.
 * ponytail: a single global object serializes every reservation; shard by hash if traffic
 * ever outgrows one object (hundreds of requests per second).
 */
export class Budget {
  constructor(private readonly state: { storage: Storage }) {}

  async fetch(request: Request): Promise<Response> {
    // Inputs come from the URL, not the body: the storage get is the only await before the put,
    // and the runtime's input gate holds other requests while it is in flight. An awaited body
    // read here would let two reservations read the same total and lose one update.
    const url = new URL(request.url)
    const op = url.pathname.slice(1)
    const day = url.searchParams.get("day") ?? ""
    const amount = Number(url.searchParams.get(op === "settle" ? "delta" : "neurons") ?? "0") || 0
    const saved = await this.state.storage.get<Day>("today")
    const today: Day = saved?.day === day ? saved : { day, neurons: 0, clients: {} }
    if (op === "reserve") {
      const client = url.searchParams.get("client") ?? "unknown"
      if ((today.clients[client] ?? 0) >= PER_CLIENT_DAILY) return json(429, { reason: "client" })
      if (today.neurons + amount > DAILY_NEURONS) return json(429, { reason: "budget" })
      today.clients[client] = (today.clients[client] ?? 0) + 1
      today.neurons += amount
    } else if (op === "settle") {
      today.neurons = Math.max(0, today.neurons + amount)
    } else {
      return json(404, { reason: "unknown" })
    }
    await this.state.storage.put("today", today)
    return json(200, { neurons: today.neurons })
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (new URL(request.url).pathname.startsWith("/api/")) return handle(request, env)
    return env.ASSETS ? env.ASSETS.fetch(request) : refuse(404, "not found")
  },
}
