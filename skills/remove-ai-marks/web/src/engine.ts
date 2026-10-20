import type { EngineMessage, EngineRequest } from "./engine.worker"

/** Client side of engine.worker.ts: one worker, requests matched to answers by id. */
type Answer = { status: number; payload: string }
type Request = { type: "call"; method: "GET" | "POST"; path: string; body?: string } | { type: "configure"; settings: string }

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, (answer: Answer) => void>()
const stageListeners = new Set<(stage: string) => void>()

export function onEngineStage(listener: (stage: string) => void): () => void {
  stageListeners.add(listener)
  return () => stageListeners.delete(listener)
}

function engine(): Worker {
  if (worker) return worker
  worker = new Worker("/engine.worker.js", { type: "module" })
  worker.onmessage = (event: MessageEvent<EngineMessage>) => {
    const message = event.data
    if ("stage" in message) {
      for (const listener of stageListeners) listener(message.stage)
      return
    }
    pending.get(message.id)?.(message)
    pending.delete(message.id)
  }
  worker.onerror = () => {
    // The worker script itself failed to load: answer everything waiting as "unreachable".
    for (const resolve of pending.values()) resolve({ status: 0, payload: '{"ok": false}' })
    pending.clear()
    worker = null
  }
  return worker
}

export function send(request: Request): Promise<Answer> {
  const id = nextId++
  return new Promise((resolve) => {
    pending.set(id, resolve)
    engine().postMessage({ id, ...request } as EngineRequest)
  })
}
