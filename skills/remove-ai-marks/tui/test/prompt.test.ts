import { describe, expect, test } from "bun:test"
import {
  expandHome,
  outputDisplay, flagGroups, fuzzyScore, mergeFlags, parsePrompt, removeFlag,
  shortPath,
  splitArgs,
} from "../src/prompt"

describe("prompt", () => {
  test("splits like a shell", () => {
    expect(splitArgs(`a "b c" 'd e' f\\ g`)).toEqual(["a", "b c", "d e", "f g"])
    expect(splitArgs(`--glob "*.md"`)).toEqual(["--glob", "*.md"])
    expect(splitArgs(`""`)).toEqual([""])
  })

  test("classifies commands, flags and paths", () => {
    expect(parsePrompt("  ")).toEqual({ kind: "empty" })
    expect(parsePrompt("/clean")).toEqual({ kind: "command", name: "clean", arg: "" })
    expect(parsePrompt("/Preset rewrite")).toEqual({ kind: "command", name: "preset", arg: "rewrite" })
    expect(parsePrompt("--nfkc --glob *.md")).toEqual({ kind: "flags", tokens: ["--nfkc", "--glob", "*.md"] })
    expect(parsePrompt("drafts notes.md")).toEqual({ kind: "paths", tokens: ["drafts", "notes.md"] })
    // An absolute path is a path, not a command.
    expect(parsePrompt("/Users/me/draft.md").kind).toBe("paths")
  })

  test("groups valued flags with their values", () => {
    expect(flagGroups(["--glob", "*.md", "--nfkc", "--seed", "-1"])).toEqual([["--glob", "*.md"], ["--nfkc"], ["--seed", "-1"]])
  })

  test("a bare flag toggles and a valued flag replaces", () => {
    expect(mergeFlags([], ["--nfkc"])).toEqual(["--nfkc"])
    expect(mergeFlags(["--nfkc"], ["--nfkc"])).toEqual([])
    expect(mergeFlags(["--glob", "*.md"], ["--glob", "*.txt"])).toEqual(["--glob", "*.txt"])
    expect(mergeFlags(["--recursive"], ["--glob", "*.md"])).toEqual(["--recursive", "--glob", "*.md"])
    expect(removeFlag(["--recursive", "--glob", "*.md"], "--glob")).toEqual(["--recursive"])
  })

  test("expands home", () => {
    expect(expandHome("~/x", "/h")).toBe("/h/x")
    expect(expandHome("x~", "/h")).toBe("x~")
  })

  test("fuzzy ranks contiguous prefixes first", () => {
    expect(fuzzyScore("cl", "clean")!).toBeGreaterThan(fuzzyScore("cl", "cancel run")!)
    expect(fuzzyScore("zz", "clean")).toBeNull()
  })

  test("short paths keep the file name", () => {
    expect(shortPath("drafts/chapter-01.md", 40)).toBe("drafts/chapter-01.md")
    expect(shortPath("drafts/chapter-01.md", 15)).toBe("…/chapter-01.md")
    expect(shortPath("drafts/chapter-01.md", 18)).toBe("dra…/chapter-01.md")
    expect(shortPath("a-very-long-file-name.md", 10)).toBe("a-very-lo…")
    expect(shortPath("x", 0)).toBe("")
  })
})

test("outputDisplay keeps a sibling output relative to the input", () => {
  const input = { path: "/home/u/notes/a.md", display: "notes/a.md" }
  expect(outputDisplay(input, "/home/u/notes/a.cleaned.md")).toBe("notes/a.cleaned.md")
  expect(outputDisplay({ path: "/x/a.md", display: "a.md" }, "/x/a.cleaned.md")).toBe("a.cleaned.md")
  expect(outputDisplay(input, "/out/a.cleaned.md")).toBe("/out/a.cleaned.md")
})
