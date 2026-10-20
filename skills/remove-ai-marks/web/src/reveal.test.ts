import { expect, test } from "bun:test"
import { reveal } from "./reveal"

test("marks every occurrence, not just the server's samples", () => {
  const { segments, truncated } = reveal("a​b​c­", ["U+200B", "U+00AD"])
  expect(truncated).toBe(false)
  expect(segments).toEqual([
    { kind: "text", value: "a" },
    { kind: "mark", cp: "U+200B" },
    { kind: "text", value: "b" },
    { kind: "mark", cp: "U+200B" },
    { kind: "text", value: "c" },
    { kind: "mark", cp: "U+00AD" },
  ])
})

test("an astral tag character is one mark, never two surrogate halves", () => {
  const { segments } = reveal("x\u{E0041}y", ["U+E0041"])
  expect(segments).toEqual([
    { kind: "text", value: "x" },
    { kind: "mark", cp: "U+E0041" },
    { kind: "text", value: "y" },
  ])
})

test("unlisted and malformed code points stay plain text", () => {
  expect(reveal("a‍b", ["U+200B", "nope"]).segments).toEqual([{ kind: "text", value: "a‍b" }])
})

test("limit truncates and says so", () => {
  const { segments, truncated } = reveal("abcdef", [], 3)
  expect(truncated).toBe(true)
  expect(segments).toEqual([{ kind: "text", value: "abc" }])
})
