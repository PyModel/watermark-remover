import type { AppController } from "./store"

/**
 * Every command, once.  The palette lists these, the prompt runs them as
 * `/name`, and help prints them — so a command cannot exist in one place and
 * be missing from another.
 */
export interface Command {
  name: string
  title: string
  /** The key that runs it without the palette, when there is one. */
  key?: string
  /** `arg` is whatever followed `/name` in the prompt; "" from the palette. */
  run: (ctl: AppController, arg: string) => void
}

export const COMMANDS: Command[] = [
  { name: "clean", title: "Clean", key: "ctrl+r", run: (ctl) => void ctl.clean() },
  { name: "inspect", title: "Inspect without writing", key: "ctrl+e", run: (ctl) => void ctl.inspect() },
  { name: "stop", title: "Stop after this file", key: "esc", run: (ctl) => void ctl.cancel() },
  {
    name: "preset",
    title: "Choose what to remove",
    key: "tab",
    run: (ctl, arg) => {
      const wanted = arg.toLowerCase()
      const match = wanted && ctl.presets().find((p) => p.key === wanted || p.label.toLowerCase().startsWith(wanted))
      if (match) ctl.setPreset(match.key)
      else ctl.openDialog({ type: "preset" })
    },
  },
  { name: "model", title: "Choose a model", run: (ctl) => ctl.openDialog({ type: "model" }) },
  { name: "remove", title: "Remove a path (/remove PATH)", run: (ctl, arg) => (arg ? ctl.removePath(arg) : ctl.toast("Type /remove and the path to drop.")) },
  { name: "unset", title: "Remove a flag (/unset --flag)", run: (ctl, arg) => (arg ? ctl.dropFlag(arg) : ctl.toast("Type /unset and the flag to drop.")) },
  { name: "clear", title: "Clear files and results", run: (ctl) => ctl.clearPaths() },
  { name: "history", title: "History", run: (ctl) => ctl.openDialog({ type: "history" }) },
  { name: "doctor", title: "Check this install", run: (ctl) => ctl.openDialog({ type: "doctor" }) },
  { name: "setup", title: "Run setup again", run: (ctl) => ctl.openDialog({ type: "welcome" }) },
  { name: "help", title: "Help", key: "f1", run: (ctl) => ctl.openDialog({ type: "help" }) },
  { name: "quit", title: "Quit", key: "ctrl+c", run: (ctl) => ctl.quit() },
]

export function runCommand(ctl: AppController, name: string, arg = ""): void {
  const command = COMMANDS.find((c) => c.name === name)
  if (command) command.run(ctl, arg)
  else ctl.toast(`No command /${name}. Ctrl+P lists them.`, "error")
}
