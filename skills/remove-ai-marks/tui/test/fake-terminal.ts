import type { Terminal } from "@earendil-works/pi-tui"

/**
 * In-memory terminal for pi-tui. Implements the full Terminal interface,
 * including the methods the README snippet omits (drainInput, kittyProtocolActive,
 * setTitle, setProgress). Input is pushed by tests; nothing reads stdin.
 */
export class FakeTerminal implements Terminal {
  private width: number
  private height: number
  private onInput?: (data: string) => void
  private onResize?: () => void
  private kitty = false
  readonly writes: string[] = []
  readonly titles: string[] = []
  readonly progress: boolean[] = []
  started = false
  cursorHidden = false

  constructor(columns = 80, rows = 24) {
    this.width = columns
    this.height = rows
  }

  start(onInput: (data: string) => void, onResize: () => void): void {
    this.onInput = onInput
    this.onResize = onResize
    this.started = true
  }

  stop(): void {
    this.started = false
  }

  async drainInput(_maxMs?: number, _idleMs?: number): Promise<void> {}

  write(data: string): void {
    this.writes.push(data)
  }

  get columns(): number {
    return this.width
  }

  get rows(): number {
    return this.height
  }

  get kittyProtocolActive(): boolean {
    return this.kitty
  }

  moveBy(_lines: number): void {}

  hideCursor(): void {
    this.cursorHidden = true
  }

  showCursor(): void {
    this.cursorHidden = false
  }

  clearLine(): void {}

  clearFromCursor(): void {}

  clearScreen(): void {}

  setTitle(title: string): void {
    this.titles.push(title)
    // Library Terminal.setTitle writes OSC 0. Capture it once, on writes.
    this.writes.push(`\x1b]0;${title}\x07`)
  }

  setProgress(active: boolean): void {
    this.progress.push(active)
    // Library Terminal.setProgress writes OSC 9;4. Capture it once, on writes.
    this.writes.push(active ? "\x1b]9;4;3\x07" : "\x1b]9;4;0\x07")
  }

  /** Deliver one input sequence to the listener registered by start(). */
  input(data: string): void {
    if (!this.onInput) throw new Error("FakeTerminal.input before start()")
    this.onInput(data)
  }

  resize(columns: number, rows: number): void {
    this.width = columns
    this.height = rows
    this.onResize?.()
  }
}
