/**
 * The cleaning engine for the static build: Pyodide runs the same Python
 * modules wm-serve runs, and pages/bridge.py answers each request exactly as
 * server.Handler would. Everything it loads comes from this site.
 */
export type EngineRequest =
  | { id: number; type: "call"; method: "GET" | "POST"; path: string; body?: string }
  | { id: number; type: "configure"; settings: string }
export type EngineMessage =
  | { type: "progress"; stage: string }
  | { id: number; status: number; payload: string }

// The DOM lib types `self` as a Window; this file only runs as a worker.
declare const self: {
  postMessage(message: EngineMessage): void
  onmessage: ((event: MessageEvent<EngineRequest>) => void) | null
}

const BASE = process.env.WM_PYODIDE ?? "/pyodide/"

type PyFunction = (...args: unknown[]) => { toJs(): unknown; destroy(): void } | undefined
interface Bridge {
  handle: PyFunction
  configure: PyFunction
}
interface Pyodide {
  loadPackage(urls: string[]): Promise<void>
  unpackArchive(buffer: ArrayBuffer, format: string, options: { extractDir: string }): void
  runPython(code: string): unknown
  pyimport(name: string): unknown
}

const progress = (stage: string) => self.postMessage({ type: "progress", stage } satisfies EngineMessage)

async function boot(): Promise<Bridge> {
  progress("Loading Python")
  const url = `${BASE}pyodide.mjs` // a variable, so the bundler leaves this import to the browser
  const { loadPyodide } = await import(/* @vite-ignore */ url)
  const py: Pyodide = await loadPyodide({ indexURL: BASE })
  progress("Loading the cleaners")
  const lock = await (await fetch(`${BASE}pyodide-lock.json`)).json()
  const [zip] = await Promise.all([
    fetch(process.env.WM_CLEANERS ?? "/cleaners.zip").then((r) => r.arrayBuffer()),
    py.loadPackage([`${BASE}${lock.packages.pillow.file_name}`, `${BASE}pypdf-6.19.0-py3-none-any.whl`]),
  ])
  py.unpackArchive(zip, "zip", { extractDir: "/wm" })
  // server.py reports this as its version; read once, at import.
  py.runPython(`import os, sys; sys.path.insert(0, '/wm'); os.environ['WATERMARKS_SERVER_VERSION'] = ${JSON.stringify(process.env.WM_VERSION ?? "dev")}`)
  return py.pyimport("bridge") as Bridge
}

let engine: Promise<Bridge> | null = null

self.onmessage = async (event: MessageEvent<EngineRequest>) => {
  const request = event.data
  const reply = (status: number, payload: string) =>
    self.postMessage({ id: request.id, status, payload } satisfies EngineMessage)
  let bridge: Bridge
  try {
    engine ??= boot()
    bridge = await engine
  } catch (error) {
    // Offline mid-download, or an extension blocked the wasm: status 0 is "can't reach the engine",
    // and the next request starts the boot over.
    engine = null
    reply(0, JSON.stringify({ ok: false, error: String(error) }))
    return
  }
  try {
    if (request.type === "configure") {
      bridge.configure(request.settings)
      reply(200, '{"ok": true}')
      return
    }
    const answer = bridge.handle(request.method, request.path, request.body ?? null)!
    const [status, payload] = answer.toJs() as [number, string]
    answer.destroy()
    reply(status, payload)
  } catch (error) {
    // server.Handler turns its own failures into 500s; reaching here is a bridge bug.
    reply(500, JSON.stringify({ ok: false, error: `engine error: ${String(error)}` }))
  }
}
