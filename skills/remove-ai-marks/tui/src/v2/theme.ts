/**
 * Inspecting a file for watermarks is holding a document under a UV lamp:
 * the page stays dark, and what was invisible lights up.  That is the whole
 * palette.  The lamp (`primary`) marks where you are — focus, selection, the
 * prompt — and `glow` is reserved for the hidden marks themselves, so the one
 * bright pink on screen is always something that was not meant to be seen.
 *
 * Hierarchy comes from surface steps and text weight, never box borders: a
 * border costs two columns and two rows that say nothing, and at 80×24 that
 * is the difference between a file list and a scrollbar.
 */
import { parseColor, styleText, type EditorTheme, type SelectListTheme } from "@earendil-works/pi-tui"

export const theme = {
  /** ink: the darkroom page, blue-black rather than a neutral near-black */
  bg: "#0f1219",
  /** sheet: raised surfaces — the prompt and dialogs */
  panel: "#171b25",
  /** the selected row and input wells */
  element: "#222838",
  /** paper */
  text: "#e4e2da",
  /** graphite */
  muted: "#858b9c",
  faint: "#4b5162",
  /** the lamp */
  primary: "#a996ff",
  /** what the lamp reveals; hidden marks only */
  glow: "#f78fe0",
  /** result classes: sage, wax */
  success: "#8fd6a4",
  warning: "#e8c170",
  error: "#ff6b6b",
} as const

/** The colour a result class is always shown in: the class, never the outcome. */
export function resultClassColor(resultClass: string): string {
  if (resultClass === "Verifiable") return theme.success
  if (resultClass === "Best-effort") return theme.warning
  return theme.muted
}

export function paint(text: string, hex: string, bold = false): string {
  if (!text) return ""
  return styleText(text, { fg: parseColor(hex), bold }, "truecolor")
}

const ink = (hex: string) => (text: string) => paint(text, hex)

const selectList: SelectListTheme = {
  selectedPrefix: (text) => text,
  selectedText: ink(theme.primary),
  description: ink(theme.muted),
  scrollInfo: ink(theme.faint),
  noMatch: ink(theme.muted),
}

/** pi-tui themes are functions, not color tokens. Palette hex stays the source. */
export const editorTheme: EditorTheme = {
  borderColor: ink(theme.primary),
  selectList,
}
