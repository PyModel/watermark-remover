import { useCallback, useReducer, useRef } from "react"
import {
  ApiError,
  clean,
  cleanedName,
  detect,
  fromBase64,
  inspect,
  removeVisible,
  rewrite,
  toBase64,
  type Box,
  type CleanOptions,
  type Cleaned,
  type Detection,
  type Inspection,
  type Strength,
} from "./api"

export type Tool = "clean" | "rewrite" | "visible" | "detect"

export interface Result {
  via: Exclude<Tool, "detect">
  cleaned: Cleaned
  cleanedText: string | null
  /** the server's re-inspection of the output: "nothing remains" is its finding, not ours */
  recheck: Inspection
  name: string
  mime: string
}

export interface Item {
  id: number
  file: File
  status: "inspecting" | "ready" | "working" | "done" | "failed"
  /** what is running right now, for the pending label */
  work: Tool | null
  startedAt: number
  data: string
  text: string | null
  inspection: Inspection | null
  result: Result | null
  detections: Detection[] | null
  error: ApiError | null
}

type Action =
  | { type: "add"; items: Item[] }
  | { type: "patch"; id: number; patch: Partial<Item> }
  | { type: "remove"; id: number }
  | { type: "clear" }

function reducer(items: Item[], action: Action): Item[] {
  switch (action.type) {
    case "add":
      return [...items, ...action.items]
    case "patch":
      return items.map((item) => (item.id === action.id ? { ...item, ...action.patch } : item))
    case "remove":
      return items.filter((item) => item.id !== action.id)
    case "clear":
      return []
  }
}

// TextDecoder strips a leading U+FEFF by default; that BOM can be a hidden mark.
export const decodeText = (bytes: BufferSource) => new TextDecoder("utf-8", { ignoreBOM: true }).decode(bytes)

const asApiError = (error: unknown) => (error instanceof ApiError ? error : new ApiError(String(error), 0))

/**
 * The file queue. Every request runs one file at a time against wm-serve; the
 * server is single-host and some backends are slow, so parallel uploads would
 * only queue up there instead.
 */
export function useQueue(onError: (error: ApiError) => void) {
  const [items, dispatch] = useReducer(reducer, [])
  const latest = useRef<Map<number, Item>>(new Map())
  latest.current = new Map(items.map((item) => [item.id, item]))
  const nextId = useRef(1)
  // ponytail: one global chain serializes every request; per-file lanes if a faster server makes it worth it.
  const chain = useRef<Promise<unknown>>(Promise.resolve())

  // Jobs chain on microtasks and can start before React re-renders, so the ref is
  // patched in step with the reducer rather than only on render.
  const patch = useCallback((id: number, value: Partial<Item>) => {
    const current = latest.current.get(id)
    if (current) latest.current.set(id, { ...current, ...value })
    dispatch({ type: "patch", id, patch: value })
  }, [])

  const enqueue = useCallback(<T,>(job: () => Promise<T>): Promise<T> => {
    const run = chain.current.then(job, job)
    chain.current = run.catch(() => undefined)
    return run
  }, [])

  const fail = useCallback(
    (id: number, error: unknown, fallback: Item["status"]) => {
      const apiError = asApiError(error)
      onError(apiError)
      patch(id, { status: fallback, work: null, error: apiError })
    },
    [onError, patch],
  )

  const inspectOne = useCallback(
    (id: number) =>
      enqueue(async () => {
        const item = latest.current.get(id)
        if (!item) return
        patch(id, { status: "inspecting", error: null })
        try {
          const data = item.data || (await toBase64(item.file))
          const inspection = await inspect(data, item.file.name)
          const text = inspection.kind === "text" ? decodeText(await item.file.arrayBuffer()) : null
          patch(id, { status: "ready", data, text, inspection })
        } catch (error) {
          fail(id, error, "failed")
        }
      }),
    [enqueue, fail, patch],
  )

  const add = useCallback(
    (files: File[]) => {
      const added = files.map(
        (file): Item => ({
          id: nextId.current++,
          file,
          status: "inspecting",
          work: null,
          startedAt: 0,
          data: "",
          text: null,
          inspection: null,
          result: null,
          detections: null,
          error: null,
        }),
      )
      dispatch({ type: "add", items: added })
      for (const item of added) {
        latest.current.set(item.id, item)
        void inspectOne(item.id)
      }
      return added[0]?.id ?? null
    },
    [inspectOne],
  )

  /** Run one producing tool, then re-inspect its output on the server. */
  const produce = useCallback(
    (id: number, via: Result["via"], call: (item: Item) => Promise<Cleaned>) =>
      enqueue(async () => {
        const item = latest.current.get(id)
        if (!item?.inspection) return
        patch(id, { status: "working", work: via, startedAt: Date.now(), error: null })
        try {
          const cleaned = await call(item)
          const png = via === "visible"
          const name = cleanedName(item.file.name, png ? ".png" : undefined)
          const recheck = await inspect(cleaned.cleaned, name)
          const cleanedText = cleaned.kind === "text" ? decodeText(fromBase64(cleaned.cleaned)) : null
          const mime = png ? "image/png" : item.file.type || "application/octet-stream"
          patch(id, {
            status: "done",
            work: null,
            result: { via, cleaned, cleanedText, recheck, name, mime },
          })
        } catch (error) {
          fail(id, error, item.result ? "done" : "ready")
        }
      }),
    [enqueue, fail, patch],
  )

  const runClean = useCallback(
    (id: number, options: CleanOptions) => produce(id, "clean", (item) => clean(item.data, item.file.name, options)),
    [produce],
  )
  const runRewrite = useCallback(
    (id: number, strength: Strength, lang?: string) =>
      produce(id, "rewrite", (item) => rewrite(item.data, item.file.name, strength, lang)),
    [produce],
  )
  const runVisible = useCallback(
    (id: number, box: Box, backend: "texture" | "simple") =>
      produce(id, "visible", (item) => removeVisible(item.data, item.file.name, box, backend)),
    [produce],
  )

  const runDetect = useCallback(
    (id: number) =>
      enqueue(async () => {
        const item = latest.current.get(id)
        if (!item?.inspection) return
        const status = item.status
        patch(id, { status: "working", work: "detect", startedAt: Date.now(), error: null })
        try {
          const detected = await detect(item.data, item.file.name)
          patch(id, { status, work: null, detections: detected.detections })
        } catch (error) {
          fail(id, error, status)
        }
      }),
    [enqueue, fail, patch],
  )

  /** Batch: clean every inspected file that has no result yet, with the same options. */
  const cleanAll = useCallback(
    (optionsFor: (item: Item) => CleanOptions) => {
      for (const item of latest.current.values()) {
        if (item.status === "ready" && item.inspection?.kind !== "unknown") void runClean(item.id, optionsFor(item))
      }
    },
    [runClean],
  )

  return {
    items,
    add,
    retry: inspectOne,
    remove: useCallback((id: number) => dispatch({ type: "remove", id }), []),
    clear: useCallback(() => dispatch({ type: "clear" }), []),
    runClean,
    runRewrite,
    runVisible,
    runDetect,
    cleanAll,
  }
}

export type Queue = ReturnType<typeof useQueue>
