import { appendFileSync, mkdirSync } from "node:fs"
import { dirname } from "node:path"
import { BridgeError, type BridgeEvent, type BridgeLike, type Confirm } from "./protocol"

interface Frame {
  id?: number
  event?: string
  data?: unknown
  result?: unknown
  error?: { code?: string; message?: string; data?: { confirm?: Confirm[] } }
}

interface Pending {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
  onEvent?: (event: BridgeEvent) => void
}

/**
 * The Python bridge as a child process speaking JSON Lines.
 *
 * Its stderr never reaches the terminal: the renderer owns every cell, and one
 * stray warning line from the pipeline would scribble over the frame. It goes
 * to WM_TUI_LOG instead, and the last few lines are kept so a crash can be
 * shown on screen rather than as a frozen UI.
 */
export class Bridge implements BridgeLike {
  private readonly child: ReturnType<typeof Bun.spawn>
  private readonly pending = new Map<number, Pending>()
  private nextId = 1
  private exitCallbacks: ((reason: string) => void)[] = []
  private closing = false
  readonly stderrTail: string[] = []

  constructor(
    python: string,
    bridgePath: string,
    private readonly logPath: string | undefined,
  ) {
    if (logPath) {
      try {
        mkdirSync(dirname(logPath), { recursive: true })
      } catch {}
    }
    this.child = Bun.spawn([python, bridgePath], {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
    })
    this.readLines(this.child.stdout as ReadableStream<Uint8Array>, (line) => this.onLine(line))
    this.readLines(this.child.stderr as ReadableStream<Uint8Array>, (line) => this.onStderr(line))
    this.child.exited.then((code) => this.onChildExit(code))
  }

  request<T>(method: string, params: object = {}, onEvent?: (event: BridgeEvent) => void): Promise<T> {
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, onEvent })
      try {
        const stdin = this.child.stdin as import("bun").FileSink
        stdin.write(JSON.stringify({ id, method, params }) + "\n")
        stdin.flush()
      } catch (error) {
        this.pending.delete(id)
        reject(new BridgeError("internal", `bridge is not running: ${error}`))
      }
    })
  }

  onExit(callback: (reason: string) => void): void {
    this.exitCallbacks.push(callback)
  }

  close(): void {
    this.closing = true
    try {
      const stdin = this.child.stdin as import("bun").FileSink
      stdin.write(JSON.stringify({ id: 0, method: "shutdown", params: {} }) + "\n")
      stdin.end()
    } catch {}
    setTimeout(() => this.child.kill(), 500).unref?.()
  }

  private onLine(line: string): void {
    let frame: Frame
    try {
      frame = JSON.parse(line) as Frame
    } catch {
      this.onStderr(`[non-protocol stdout] ${line}`)
      return
    }
    // Unsolicited frames (the startup `ready`) carry no id and need no reply.
    if (frame.id == null) return
    const waiting = this.pending.get(frame.id)
    if (frame.event) {
      waiting?.onEvent?.({ event: frame.event, data: frame.data } as BridgeEvent)
      return
    }
    if (!waiting) return
    this.pending.delete(frame.id)
    if (frame.error) {
      waiting.reject(new BridgeError(frame.error.code ?? "internal", frame.error.message ?? "error", frame.error.data?.confirm))
    } else {
      waiting.resolve(frame.result)
    }
  }

  private onStderr(line: string): void {
    this.stderrTail.push(line)
    if (this.stderrTail.length > 40) this.stderrTail.shift()
    if (this.logPath) {
      try {
        appendFileSync(this.logPath, line + "\n")
      } catch {}
    }
  }

  private onChildExit(code: number | null): void {
    const reason = this.closing ? "closed" : `bridge exited with code ${code}`
    for (const [, waiting] of this.pending) waiting.reject(new BridgeError("internal", reason))
    this.pending.clear()
    if (!this.closing) for (const callback of this.exitCallbacks) callback(reason)
  }

  private async readLines(stream: ReadableStream<Uint8Array>, onLine: (line: string) => void): Promise<void> {
    const decoder = new TextDecoder()
    let buffer = ""
    for await (const chunk of stream) {
      buffer += decoder.decode(chunk, { stream: true })
      let newline = buffer.indexOf("\n")
      while (newline >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, "")
        buffer = buffer.slice(newline + 1)
        if (line) onLine(line)
        newline = buffer.indexOf("\n")
      }
    }
    if (buffer.trim()) onLine(buffer.trim())
  }
}
