/**
 * Builds the Cloudflare Pages site: the same page with its requests answered by
 * the Python cleaners running in a Web Worker (Pyodide), served from this site
 * only. Output: web/dist-pages (gitignored). Run: bun run build:pages
 */
import { createHash } from "node:crypto"
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"

const WEB = join(import.meta.dir, "..")
const SCRIPTS = join(WEB, "..", "scripts")
const OUT = join(WEB, "dist-pages")
const CACHE = join(WEB, ".cache")
const CORE = join(WEB, "node_modules", "pyodide")
const VERSION: string = JSON.parse(readFileSync(join(CORE, "package.json"), "utf8")).version
const PYODIDE = `/pyodide/v${VERSION}/`
const RUNTIME = join(OUT, PYODIDE)
const APP_VERSION = readFileSync(join(WEB, "..", "..", "..", "pyproject.toml"), "utf8").match(/^version = "([^"]+)"/m)![1]

// Pillow is the only compiled package the cleaners need (JPEG decoding for visible-mark removal).
// Its hash comes from pyodide-lock.json; pypdf (pure Python, PDF metadata) is pinned from PyPI.
const lock = JSON.parse(readFileSync(join(CORE, "pyodide-lock.json"), "utf8"))
const WHEELS = [
  {
    file: lock.packages.pillow.file_name as string,
    url: `https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/${lock.packages.pillow.file_name}`,
    sha256: lock.packages.pillow.sha256 as string,
  },
  {
    file: "pypdf-6.19.0-py3-none-any.whl",
    url: "https://files.pythonhosted.org/packages/3c/2c/c43c03eaf630435f023f1dc61ec4a4a78951ad5530a62c71cc89bde307b7/pypdf-6.19.0-py3-none-any.whl",
    sha256: "7e5d6e730e7dae87d560a2cee218b852f6498c8be61966f3cd02ead971e48d14",
  },
]

const sha256 = (data: Uint8Array) => createHash("sha256").update(data).digest("hex")

async function wheel({ file, url, sha256: expected }: (typeof WHEELS)[number]): Promise<Uint8Array> {
  const cached = join(CACHE, file)
  let data = existsSync(cached) ? new Uint8Array(readFileSync(cached)) : null
  if (!data || sha256(data) !== expected) {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`${file}: download failed with HTTP ${response.status}`)
    data = new Uint8Array(await response.arrayBuffer())
    if (sha256(data) !== expected) throw new Error(`${file}: sha256 mismatch, refusing to ship it`)
    mkdirSync(CACHE, { recursive: true })
    writeFileSync(cached, data)
  }
  return data
}

/**
 * Every cleaner module plus server.py and the bridge, as one zip the worker unpacks. The name
 * carries its hash: it changes with every cleaner release, unlike the Pyodide runtime's path.
 */
function cleaners(): string {
  const files = readdirSync(SCRIPTS).filter((name) => name.endsWith(".py"))
  const staging = join(OUT, "cleaners.zip")
  const zip = Bun.spawnSync(
    ["python3", "-m", "zipfile", "-c", staging, ...files.map((f) => join(SCRIPTS, f)), join(import.meta.dir, "bridge.py")],
    { stderr: "inherit" },
  )
  if (zip.exitCode !== 0) throw new Error("zipping the cleaners failed")
  const name = `cleaners-${sha256(new Uint8Array(readFileSync(staging))).slice(0, 16)}.zip`
  renameSync(staging, join(OUT, name))
  return `/${name}`
}

// CSP: wm-serve's page policy plus what Pyodide needs (wasm compile, a same-origin worker) and the
// visitor's own model server on this machine for Rewrite. Nothing else can be fetched, which is what
// makes "your files never leave this device" checkable in the browser's network panel.
const CSP = [
  "default-src 'none'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "worker-src 'self'",
  "style-src 'self'",
  "img-src 'self' blob: data:",
  "connect-src 'self' http://localhost:* http://127.0.0.1:*",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join("; ")

const HEADERS = `/*
  Content-Security-Policy: ${CSP}
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=()
${PYODIDE}*
  Cache-Control: public, max-age=31536000, immutable
/chunk-*
  Cache-Control: public, max-age=31536000, immutable
/cleaners-*
  Cache-Control: public, max-age=31536000, immutable
`

rmSync(OUT, { recursive: true, force: true })
mkdirSync(RUNTIME, { recursive: true })
const CLEANERS = cleaners()
const build = await Bun.build({
  entrypoints: [join(WEB, "index.html"), join(WEB, "src", "engine.worker.ts")],
  outdir: OUT,
  minify: true,
  naming: { entry: "[name].[ext]", chunk: "chunk-[hash].[ext]", asset: "chunk-[hash].[ext]" },
  define: { "process.env.WM_TARGET": '"browser"', "process.env.WM_PYODIDE": JSON.stringify(PYODIDE),
    "process.env.WM_VERSION": JSON.stringify(APP_VERSION),
    "process.env.WM_CLEANERS": JSON.stringify(CLEANERS),
  },
})
if (!build.success) {
  for (const log of build.logs) console.error(log)
  process.exit(1)
}
for (const file of ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) {
  cpSync(join(CORE, file), join(RUNTIME, file))
}
for (const spec of WHEELS) writeFileSync(join(RUNTIME, spec.file), await wheel(spec))
writeFileSync(join(OUT, "_headers"), HEADERS)
console.log(`dist-pages ready: watermark-remover ${APP_VERSION}, Pyodide ${VERSION}, ${WHEELS.length} wheels, cleaners at ${CLEANERS}`)
