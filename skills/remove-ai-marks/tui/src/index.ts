import { isViewportTUI, ProcessTerminal, TuiAltScreen } from "@earendil-works/pi-tui"
import { mountApp, type MountedApp } from "./v2/app"
import { Bridge } from "./v2/bridge"

const python = process.env.WM_TUI_PYTHON
const bridgePath = process.env.WM_TUI_BRIDGE
if (!python || !bridgePath) {
  console.error("wm-tui: start this through the `wm-tui` command, which sets WM_TUI_PYTHON and WM_TUI_BRIDGE")
  process.exit(2)
}

const tui = new TuiAltScreen(new ProcessTerminal())
if (!isViewportTUI(tui)) {
  console.error("wm-tui: fullscreen layout is unavailable")
  process.exit(2)
}

const bridge = new Bridge(python, bridgePath, process.env.WM_TUI_LOG)
let quitting = false
let mounted: MountedApp
mounted = mountApp(tui, bridge, () => {
  if (quitting) return
  quitting = true
  mounted.dispose()
  tui.stop()
  process.exit(0)
})
for (const signal of ["SIGTERM", "SIGHUP"] as const) process.on(signal, () => mounted.ctl.quit())
tui.start()
await mounted.ctl.start()
