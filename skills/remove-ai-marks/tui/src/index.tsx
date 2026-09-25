import { createCliRenderer } from "@opentui/core"
import { render } from "@opentui/solid"
import { App } from "./app"
import { Bridge } from "./bridge"
import { createApp } from "./store"
import { theme } from "./theme"

const python = process.env.WM_TUI_PYTHON
const bridgePath = process.env.WM_TUI_BRIDGE
if (!python || !bridgePath) {
  console.error("wm-tui: start this through the `wm-tui` command, which sets WM_TUI_PYTHON and WM_TUI_BRIDGE")
  process.exit(2)
}

const bridge = new Bridge(python, bridgePath, process.env.WM_TUI_LOG)
const renderer = await createCliRenderer({
  exitOnCtrlC: false,
  backgroundColor: theme.bg,
  targetFps: 30,
})
let quitting = false
function quit() {
  if (quitting) return
  quitting = true
  renderer.destroy()
  process.exit(0)
}

const ctl = createApp(bridge, quit)
for (const signal of ["SIGTERM", "SIGHUP"] as const) process.on(signal, () => ctl.quit())

render(() => <App ctl={ctl} />, renderer)
