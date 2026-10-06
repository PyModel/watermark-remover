/**
 * Split text into plain runs and hidden-character marks.
 *
 * The server reports which code points it found (`hits[].codepoint`), but its
 * sample offsets are Python code-point indices and only samples. The browser
 * already holds the text, so it finds every occurrence itself by walking code
 * points: astral carriers such as tag characters (U+E0000 block) stay one mark
 * instead of two broken surrogate halves.
 */
export type Segment = { kind: "text"; value: string } | { kind: "mark"; cp: string }

export interface Revealed {
  segments: Segment[]
  /** true when `limit` cut the text short; counts in the ledger still cover the whole file */
  truncated: boolean
}

/** "U+200B" -> 0x200b; anything malformed is dropped rather than guessed. */
export function parseCodepoint(label: string): number | null {
  const match = /^U\+([0-9A-F]{4,6})$/i.exec(label.trim())
  return match ? Number.parseInt(match[1]!, 16) : null
}

export function formatCodepoint(cp: number): string {
  return "U+" + cp.toString(16).toUpperCase().padStart(4, "0")
}

export function reveal(text: string, codepoints: Iterable<string>, limit = 100_000): Revealed {
  const wanted = new Set<number>()
  for (const label of codepoints) {
    const cp = parseCodepoint(label)
    if (cp !== null) wanted.add(cp)
  }
  const segments: Segment[] = []
  let run = ""
  let used = 0
  for (const char of text) {
    if (used >= limit) {
      if (run) segments.push({ kind: "text", value: run })
      return { segments, truncated: true }
    }
    used += char.length
    const cp = char.codePointAt(0)!
    if (!wanted.has(cp)) {
      run += char
      continue
    }
    if (run) segments.push({ kind: "text", value: run })
    run = ""
    segments.push({ kind: "mark", cp: formatCodepoint(cp) })
  }
  if (run) segments.push({ kind: "text", value: run })
  return { segments, truncated: false }
}
