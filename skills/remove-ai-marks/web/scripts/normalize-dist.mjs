// bun build leaves the placeholder line where it rewrote an inline module
// script with whatever indentation the source had. web-check rebuilds dist and
// requires it to match the committed artifact, and `git diff --check` treats
// trailing whitespace on an added line as a defect, so strip trailing
// per-line whitespace from every emitted html file after each build.
import { readdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");

for (const entry of readdirSync(dist)) {
  const path = join(dist, entry);
  if (!statSync(path).isFile() || !entry.endsWith(".html")) continue;
  const text = readFileSync(path, "utf8");
  const clean = text
    .split("\n")
    .map((line) => line.replace(/[ \t]+$/, ""))
    .join("\n");
  if (clean !== text) {
    writeFileSync(path, clean);
    console.log("[normalize-dist] stripped trailing whitespace:", entry);
  }
}
