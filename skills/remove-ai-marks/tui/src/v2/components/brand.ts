import { visibleWidth } from "@earendil-works/pi-tui"
import { paint, theme } from "../theme"

// Three-row, four-column letterforms: terminal-native, not a raster logo.
const letters = {
  W: ["|  |", "|/\\|", "\\/\\/"],
  A: [" /\\ ", "/__\\", "|  |"],
  T: ["___ ", " |  ", " |  "],
  E: ["___ ", "|_  ", "|__ "],
  R: ["__  ", "|_) ", "| \\ "],
  M: ["|\\/|", "|  |", "|  |"],
  K: ["| / ", "|<  ", "| \\ "],
  O: [" __ ", "|  |", "\\__/"],
  V: ["|  |", "\\  /", " \\/ "],
} as const

const words = [
  ["W", "A", "T", "E", "R", "M", "A", "R", "K"],
  ["R", "E", "M", "O", "V", "E", "R"],
] as const

export const WORDMARK = ([0, 1, 2] as const).map((row) =>
  words.map((word) => word.map((letter) => letters[letter][row]).join(" ")).join("   ").trimEnd(),
)

const wordmarkWidth = Math.max(...WORDMARK.map((line) => visibleWidth(line)))

export function renderWordmark(width: number, rows: number) {
  // Preserve task space at 80x24 and below. Never crop ASCII letterforms.
  if (width < wordmarkWidth + 2 || rows < 30) return []
  return WORDMARK.map((line, row) => " " + paint(line, row === 0 ? theme.muted : theme.primary))
}
