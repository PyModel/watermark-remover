import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { expect, test } from "bun:test"
import { commandAutocomplete } from "../src/v2/autocomplete"

test("argument-taking commands pass argumentHint through to slash completions", async () => {
  const provider = commandAutocomplete("/tmp")
  const suggestions = await provider.getSuggestions(["/"], 0, 1, { signal: new AbortController().signal })
  const byName = new Map((suggestions?.items ?? []).map((item) => [item.value, item.description ?? ""]))
  expect(byName.get("remove")).toContain("PATH")
  expect(byName.get("unset")).toContain("--flag")
  expect(byName.get("clean") ?? "").not.toContain("PATH —")
})

test("the terminal UI doc key table documents file selection and prompt history", () => {
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..")
  const readme = readFileSync(resolve(repoRoot, "docs/terminal-ui.md"), "utf8")
  const start = readme.indexOf("| Key | Does |")
  const end = readme.indexOf("**First run**")
  const table = readme.slice(start, end)
  expect(table).toContain("`ctrl+k` / `ctrl+j`")
  expect(table).toContain("previous / next file")
  expect(table).toContain("`up` / `down`")
  expect(table).toContain("`alt+up` / `alt+down`")
  expect(table).toContain("prompt history")
  expect(table).toContain("`space`")
  expect(table).toContain("only the marked file")
})
