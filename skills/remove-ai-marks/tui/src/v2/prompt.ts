/**
 * What the prompt line means.  One input does everything, the way opencode
 * and pi work: a path adds files, a `--flag` adds an option, a `/word` runs a
 * command.  There is no form to learn because the CLI already is the form —
 * every flag `wm` accepts is accepted here, and the bridge validates it with
 * the CLI's own parser.
 */

import { homedir } from "node:os"
import { COMMANDS } from "./commands"

export type Parsed =
  | { kind: "empty" }
  | { kind: "command"; name: string; arg: string }
  | { kind: "flags"; tokens: string[] }
  | { kind: "paths"; tokens: string[] }

/** Split like a POSIX shell would, for quotes and backslash escapes only. */
export function splitArgs(text: string): string[] {
  const out: string[] = []
  let current = ""
  let quote: '"' | "'" | null = null
  let started = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!
    if (quote) {
      if (ch === quote) quote = null
      else if (ch === "\\" && quote === '"' && i + 1 < text.length) current += text[++i]
      else current += ch
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      started = true
    } else if (ch === "\\" && i + 1 < text.length) {
      current += text[++i]
      started = true
    } else if (/\s/.test(ch)) {
      if (started) out.push(current)
      current = ""
      started = false
    } else {
      current += ch
      started = true
    }
  }
  if (started) out.push(current)
  return out
}

export function parsePrompt(text: string): Parsed {
  const trimmed = text.trim()
  if (!trimmed) return { kind: "empty" }
  if (/^\/[a-z-]+(\s|$)/i.test(trimmed)) {
    const [word, ...rest] = trimmed.slice(1).split(/\s+/)
    const name = word!.toLowerCase()
    // `/tmp` is a folder, not a command: only a known name runs.
    if (COMMANDS.some((command) => command.name === name)) return { kind: "command", name, arg: rest.join(" ") }
  }
  const tokens = splitArgs(trimmed)
  if (tokens[0]?.startsWith("-")) return { kind: "flags", tokens }
  return { kind: "paths", tokens }
}

/** `--flag value…` groups, so a valued flag moves with its value. */
export function flagGroups(tokens: string[]): string[][] {
  const groups: string[][] = []
  for (const token of tokens) {
    if (token.startsWith("-") && !/^-\d/.test(token)) groups.push([token])
    else if (groups.length) groups[groups.length - 1]!.push(token)
    else groups.push([token])
  }
  return groups
}

function flagName(group: string[]): string {
  return group[0]!.split("=")[0]!
}

/**
 * Apply typed flags to the current ones.  A bare flag that is already on is
 * turned off — typing `--nfkc` twice is the natural way to undo it — and a
 * valued flag replaces its previous value instead of stacking a second one.
 */
export function mergeFlags(current: string[], tokens: string[]): string[] {
  const groups = flagGroups(current)
  for (const incoming of flagGroups(tokens)) {
    const name = flagName(incoming)
    const at = groups.findIndex((group) => flagName(group) === name)
    const bare = incoming.length === 1 && !incoming[0]!.includes("=")
    if (at >= 0 && bare && groups[at]!.length === 1) groups.splice(at, 1)
    else if (at >= 0) groups[at] = incoming
    else groups.push(incoming)
  }
  return groups.flat()
}

export function removeFlag(current: string[], name: string): string[] {
  return flagGroups(current)
    .filter((group) => flagName(group) !== name)
    .flat()
}

export function expandHome(path: string, home = homedir()): string {
  if (path === "~") return home
  if (path.startsWith("~/")) return home + path.slice(1)
  return path
}

/** A subsequence match, scored so earlier and contiguous hits rank first. */
export function fuzzyScore(query: string, text: string): number | null {
  if (!query) return 0
  const q = query.toLowerCase()
  const t = text.toLowerCase()
  let score = 0
  let last = -1
  for (const ch of q) {
    const at = t.indexOf(ch, last + 1)
    if (at < 0) return null
    score += at === last + 1 ? 2 : 1
    score -= at * 0.01
    last = at
  }
  return score
}

/**
 * Fit a path into `width` columns by eliding the middle, never the file name:
 * `drafts/chapter-01.md` in 16 columns is `…/chapter-01.md`, because the name
 * is what tells two rows apart.
 */
export function shortPath(path: string, width: number): string {
  if (width <= 1) return path.slice(0, Math.max(0, width))
  if (path.length <= width) return path
  const slash = path.lastIndexOf("/")
  const name = slash >= 0 ? path.slice(slash + 1) : path
  if (name.length + 2 > width) return name.slice(0, width - 1) + "…"
  const room = width - name.length - 2
  return (room > 0 ? path.slice(0, room) : "") + "…/" + name
}

/**
 * The cleaned file's name as the user would type it.  A clean writes next to
 * its input by default, so the output reads best relative to the input's own
 * display path; anything else (an --output-dir) stays absolute.
 */
export function outputDisplay(input: { path: string; display: string }, output: string): string {
  const dir = (path: string) => path.slice(0, path.lastIndexOf("/") + 1)
  if (dir(output) !== dir(input.path)) return output
  return dir(input.display) + output.slice(dir(output).length)
}
