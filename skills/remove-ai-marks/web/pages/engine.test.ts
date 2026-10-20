/**
 * The static site's engine (Pyodide + bridge.py) must answer exactly what
 * native Python answers through the same bridge. Needs `bun run build:pages`.
 */
import { describe, expect, test } from "bun:test"
import { existsSync, readFileSync, readdirSync } from "node:fs"
import { join } from "node:path"
import { loadPyodide } from "pyodide"

const WEB = join(import.meta.dir, "..")
const REPO = join(WEB, "..", "..", "..")
const RUNTIME = join(WEB, "dist-pages", "pyodide", `v${JSON.parse(readFileSync(join(WEB, "node_modules/pyodide/package.json"), "utf8")).version}`)
// Reference Python: needs Pillow and pypdf, like the wheels the browser loads.
const NATIVE = process.env.WM_PYTHON ?? join(REPO, ".venv", "bin", "python")
const fixture = (name: string) => readFileSync(join(REPO, "tests", "fixtures", name)).toString("base64")

// Fixtures the repo has no file for, made by native Python so both sides read identical bytes.
const made = JSON.parse(
  Bun.spawnSync([
    NATIVE,
    "-c",
    `
import base64, io, json, struct, zlib
from PIL import Image
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
png = b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"tEXt", b"Comment\\x00c2pa test contentcredentials") + chunk(b"IDAT", zlib.compress(b"\\x00\\x00\\x00")) + chunk(b"IEND", b"")
photo = Image.new("RGB", (96, 64), (200, 180, 160))
for x in range(60, 90):
    for y in range(40, 56): photo.putpixel((x, y), (255, 255, 255))
jpeg = io.BytesIO(); photo.save(jpeg, "JPEG", quality=92, comment=b"Made with Google AI")
pdf = (b"%PDF-1.4\\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\\n"
       b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]>>endobj\\n4 0 obj<</Producer(DALL-E 3)/Creator(OpenAI)>>endobj\\n"
       b"trailer<</Root 1 0 R/Info 4 0 R>>\\n%%EOF\\n")
print(json.dumps({k: base64.b64encode(v).decode() for k, v in {"png": png, "jpeg": jpeg.getvalue(), "pdf": pdf}.items()}))
`,
  ]).stdout.toString(),
)

const CASES: [string, "GET" | "POST", string, Record<string, unknown> | null][] = [
  ["text inspect", "POST", "/inspect", { file: fixture("sample_watermarked.txt"), name: "a.txt" }],
  ["text clean", "POST", "/clean", { file: fixture("sample_watermarked.txt"), name: "a.txt", options: { nfkc: true } }],
  ["html clean", "POST", "/clean", { file: fixture("sample_ai.html"), name: "a.html" }],
  ["avif clean", "POST", "/clean", { file: fixture("sample_c2pa.avif"), name: "a.avif" }],
  ["png clean + marker", "POST", "/clean", { file: made.png, name: "a.png", options: { wmct_marker: true } }],
  ["jpeg inspect", "POST", "/inspect", { file: made.jpeg, name: "a.jpg" }],
  ["jpeg visible", "POST", "/visible", { file: made.jpeg, name: "a.jpg", box: [58, 38, 34, 20], backend: "simple" }],
  ["pdf clean", "POST", "/clean", { file: made.pdf, name: "a.pdf" }],
  ["detect, nothing configured", "POST", "/detect", { file: fixture("stylometry_ai_sample.txt"), name: "a.txt" }],
  ["rewrite, not set up", "POST", "/rewrite", { file: fixture("sample_watermarked.txt"), name: "a.txt", strength: "paraphrase" }],
  ["bad option", "POST", "/clean", { file: made.png, name: "a.png", options: { nope: true } }],
  ["unknown path", "GET", "/nope", null],
]

// Each run gets its own temp dir; the name is the only thing allowed to differ.
const normalize = (payload: string) => JSON.parse(payload.replace(/\/tmp\/wm-[a-z]+-[a-z0-9_]+/g, "/tmp/wm-*"))

function native(): [number, string][] {
  const requests = CASES.map(([, method, path, body]) => [method, path, body && JSON.stringify(body)])
  const run = Bun.spawnSync(
    [NATIVE, "-c", "import json, sys, bridge; print(json.dumps([bridge.handle(*r) for r in json.load(sys.stdin)]))"],
    {
      stdin: new TextEncoder().encode(JSON.stringify(requests)),
      // Same environment the browser has: no PATH tools, no Layer B settings, no detector keys.
      env: { PATH: "/nonexistent", HOME: "/nonexistent", PYTHONPATH: `${join(WEB, "..", "scripts")}:${import.meta.dir}` },
    },
  )
  if (run.exitCode !== 0) throw new Error(run.stderr.toString())
  return JSON.parse(run.stdout.toString())
}

describe("browser engine", async () => {
  const zip = readdirSync(join(WEB, "dist-pages")).find((name) => /^cleaners-[0-9a-f]{16}\.zip$/.test(name))
  if (!zip || !existsSync(join(RUNTIME, "pyodide.asm.wasm"))) throw new Error("run `bun run build:pages` first")
  const py = await loadPyodide({ indexURL: `${RUNTIME}/` })
  await py.loadPackage(["pillow-12.2.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl", "pypdf-6.19.0-py3-none-any.whl"].map((f) => join(RUNTIME, f)))
  py.unpackArchive(new Uint8Array(readFileSync(join(WEB, "dist-pages", zip))).buffer, "zip", { extractDir: "/wm" })
  py.runPython("import sys, os; sys.path.insert(0, '/wm'); os.environ.clear()")
  const bridge = py.pyimport("bridge")
  const expected = native()

  CASES.forEach(([label, method, path, body], index) => {
    test(label, () => {
      const answer = bridge.handle(method, path, body && JSON.stringify(body)).toJs() as [number, string]
      expect(answer[0]).toBe(expected[index][0])
      const [got, want] = [normalize(answer[1]), normalize(expected[index][1])]
      if (label === "jpeg visible") {
        // Pyodide's Pillow decodes JPEG with libjpeg 9, native wheels with libjpeg-turbo: pixels
        // differ by rounding. Same report, same PNG header (size, depth, colour type) is the bar.
        const ihdr = (b64: string) => Buffer.from(b64, "base64").subarray(0, 33).toString("hex")
        expect(ihdr(got.cleaned)).toBe(ihdr(want.cleaned))
        delete got.cleaned
        delete want.cleaned
      }
      expect(got).toEqual(want)
    })
  })

  test("capabilities: nothing external is claimed", () => {
    const [status, payload] = bridge.handle("GET", "/capabilities", null).toJs() as [number, string]
    const caps = JSON.parse(payload)
    expect(status).toBe(200)
    expect(Object.values(caps.tools)).toEqual([false, false, false])
    expect(caps.layer_b.configured).toBe(false)
    expect(caps.max_input_bytes).toBe(40 * 1024 * 1024)
  })

  test("a redirect to another origin counts as another origin", () => {
    const same = (a: string, b: string) => py.runPython(`import bridge; bridge._origin(${JSON.stringify(a)}) == bridge._origin(${JSON.stringify(b)})`)
    expect(same("http://localhost:11434/api/chat", "http://LOCALHOST:11434/v1")).toBe(true)
    expect(same("http://localhost/api", "http://localhost:80/api")).toBe(true)
    expect(same("http://localhost:11434/api", "http://localhost:11435/api")).toBe(false)
    expect(same("http://localhost:11434/api", "http://127.0.0.1:11434/api")).toBe(false)
    expect(same("http://localhost/api", "https://localhost/api")).toBe(false)
  })

  test("rewrite settings reach Layer B", () => {
    bridge.configure(JSON.stringify({ backend: "ollama", model: "qwen3:4b", base_url: "http://127.0.0.1:11434" }))
    const caps = JSON.parse((bridge.handle("GET", "/capabilities", null).toJs() as [number, string])[1])
    expect(caps.layer_b).toMatchObject({ backend: "ollama", configured: true, loopback: true })
    bridge.configure("{}")
  })
})
