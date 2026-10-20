import { CombinedAutocompleteProvider, type SlashCommand } from "@earendil-works/pi-tui"
import { COMMANDS } from "./commands"

/** `/remove PATH` and `/unset --flag` titles carry the library argument hint. */
function argumentHint(name: string, title: string): string | undefined {
  const match = title.match(new RegExp(`/${name} (\\S+)`))
  return match?.[1]
}

/** Command registry → slash completions. `fdPath: null` skips the fd walker. */
export function commandAutocomplete(basePath: string): CombinedAutocompleteProvider {
  const commands: SlashCommand[] = COMMANDS.map((command) => ({
    name: command.name,
    description: command.title,
    argumentHint: argumentHint(command.name, command.title),
  }))
  return new CombinedAutocompleteProvider(commands, basePath, null)
}
