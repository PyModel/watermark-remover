import { CombinedAutocompleteProvider, type SlashCommand } from "@earendil-works/pi-tui"
import { COMMANDS } from "./commands"

/** Command registry → slash completions. `fdPath: null` skips the fd walker. */
export function commandAutocomplete(basePath: string): CombinedAutocompleteProvider {
  const commands: SlashCommand[] = COMMANDS.map((command) => ({
    name: command.name,
    description: command.title,
  }))
  return new CombinedAutocompleteProvider(commands, basePath, null)
}
