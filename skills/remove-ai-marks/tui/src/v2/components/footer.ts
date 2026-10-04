import { truncateToWidth, visibleWidth, type Component, type Terminal } from "@earendil-works/pi-tui"
import { applyCleanChrome } from "../ops"
import type { AppController } from "../store"
import { paint, theme } from "../theme"
import { SPINNER } from "./layout"

/** Footer line, plus setTitle/setProgress while a clean runs. */
export class Footer implements Component {
  private chrome: { title?: string; progress?: boolean } = {}
  constructor(
    private ctl: AppController,
    private terminal: Terminal,
    private frame: () => number,
  ) {}
  invalidate(): void {}
  render(width: number): string[] {
    applyCleanChrome(this.terminal, this.ctl.app.run, this.chrome)
    const app = this.ctl.app
    let left = ""
    let color: string = theme.faint
    if (app.run) {
      const verb = app.run.kind === "clean" ? "Cleaning" : "Inspecting"
      const at = Math.min(app.run.index + 1, app.run.total)
      const spinner = `${SPINNER[this.frame()] ?? SPINNER[0]} ${verb} ${at} of ${app.run.total}: ${app.run.display}`
      left = app.toast ? `${app.toast.text}  ${spinner}` : `${spinner}    esc stops`
      color = theme.primary
    } else if (app.toast) {
      left = app.toast.text
      color = app.toast.tone === "error" ? theme.error : app.toast.tone === "success" ? theme.success : theme.text
    } else if (app.plan && !app.plan.ok) {
      left = app.plan.error ?? "These options are not valid."
      color = theme.error
    } else if (app.plan?.preflight_error) {
      left = app.plan.preflight_error
      color = theme.error
    } else if (app.plan?.warnings.length) {
      left = app.plan.warnings[0]!
      color = theme.warning
    } else if (app.plan?.files.length) {
      left = `$ ${app.plan.command}`
      color = theme.faint
    }
    const hints = width >= 100 ? "ctrl+r clean   ctrl+e inspect   tab preset   ctrl+p commands" : "ctrl+r clean   ctrl+p commands"
    const room = Math.max(0, width - visibleWidth(hints) - 2)
    return [truncateToWidth(paint(truncateToWidth(left, room, ""), color) + "  " + paint(hints, theme.muted), Math.max(1, width), "")]
  }
}
