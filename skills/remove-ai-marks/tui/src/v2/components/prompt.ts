import { Editor, truncateToWidth, visibleWidth, type Component, type TUI } from "@earendil-works/pi-tui"
import { commandAutocomplete } from "../autocomplete"
import { type AppController, endpointLabel } from "../store"
import { editorTheme, paint, resultClassColor, theme } from "../theme"

/** Editor has no single-line mode. History is recorded by the app after submit. */
export function createEditor(tui: TUI): Editor {
  const editor = new Editor(tui, editorTheme, { paddingX: 1 })
  editor.setAutocompleteProvider(commandAutocomplete(process.env.WM_TUI_CWD || process.cwd()))
  return editor
}

/** Preset, result class, flags, and the needs-a-model warning under the editor. */
export class PromptStatus implements Component {
  constructor(private ctl: AppController) {}
  invalidate(): void {}
  render(width: number): string[] {
    const preset = this.ctl.currentPreset()
    const result = this.ctl.app.plan?.result_class ?? preset?.result_class ?? ""
    const flags = this.ctl.app.state.flags.join(" ")
    const needs = Boolean(preset?.requires_endpoint && !endpointLabel(this.ctl.app.state.endpoint))
    const right = needs ? "Needs a model. Type /model" : ""
    const room = Math.max(0, width - visibleWidth(right) - (right ? 2 : 0))
    const left =
      paint(preset?.label ?? "…", theme.primary, true) +
      paint(result ? `  ${result}` : "", resultClassColor(result)) +
      paint(flags ? `  ${flags}` : "", theme.muted)
    const line = truncateToWidth(left, room, "") + (right ? "  " + paint(right, theme.warning) : "")
    return [truncateToWidth(line, Math.max(1, width), "")]
  }
}
