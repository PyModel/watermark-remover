import { mkdirSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import type { CapturedFrame } from "@opentui/core"

const hex = (c: any) => {
  const [r, g, b, a] = c.toInts()
  return a === 0 ? null : `#${[r, g, b].map((v: number) => v.toString(16).padStart(2, "0")).join("")}`
}
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/ /g, " ")

/**
 * Write a captured frame as SVG, so layout and colour can be reviewed as an
 * image.  Only when WM_TUI_SNAPSHOTS names a directory; tests never depend
 * on it.
 */
export function saveFrame(name: string, frame: CapturedFrame) {
  const dir = process.env.WM_TUI_SNAPSHOTS
  if (!dir) return
  mkdirSync(dir, { recursive: true })
  const cw = 9, ch = 18
  const out: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${frame.cols * cw}" height="${frame.rows * ch}" font-family="Menlo, monospace" font-size="15">`,
    `<rect width="100%" height="100%" fill="#0f1219"/>`,
  ]
  frame.lines.forEach((line, y) => {
    let x = 0
    for (const span of line.spans) {
      const bg = hex(span.bg)
      if (bg) out.push(`<rect x="${x * cw}" y="${y * ch}" width="${span.width * cw}" height="${ch}" fill="${bg}"/>`)
      const fg = hex(span.fg) ?? "#eeeeee"
      const bold = span.attributes & 1 ? ' font-weight="bold"' : ""
      if (span.text.trim()) out.push(`<text x="${x * cw}" y="${y * ch + 14}" fill="${fg}"${bold} xml:space="preserve">${esc(span.text)}</text>`)
      x += span.width
    }
  })
  out.push("</svg>")
  writeFileSync(join(dir, `${name}.svg`), out.join("\n"))
}
