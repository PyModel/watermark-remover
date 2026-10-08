# HTTP service and browser app

`wm-serve` exposes the whole pipeline over HTTP, with a web UI at `/`.

The whole pipeline is also exposed over HTTP (stdlib-only, no web framework). The agent skill and any web app can call it instead of running the CLIs locally.

Open `http://127.0.0.1:8765/` for the web UI. Drop one file or many: each is inspected, with every hidden character lit in place in the text (or the metadata found in an image or document). Clean one file or the whole queue, then download `NAME.cleaned.EXT`. Images also get visible-mark removal (drag a box over a logo) and optional SynthID band suppression. Text gets a Layer B rewrite and the watermark detectors, when the server has them configured. After every step the page has the server re-inspect the new bytes, so "nothing remains" is the inspector's finding. The Server view lists every capability with what to set to turn it on. The page is prebuilt React served by `wm-serve` itself (no Bun or Node needed to use it), loads nothing from other origins, and asks for the bearer key when the server has one.

### In the browser

[watermark.pymodel.com](https://watermark.pymodel.com) is the same page with the server replaced by the same Python modules running in the tab (Pyodide, Python compiled to WebAssembly). `pages/bridge.py` drives `server.Handler` without a socket, so every answer matches `wm-serve`; `make pages-check` compares them on text, HTML, AVIF, PNG, JPEG and PDF fixtures. Files are never uploaded: the page's Content-Security-Policy allows only its own origin and `localhost`, which you can confirm in the browser's network panel. The first visit downloads about 8 MB compressed (Python, Pillow, pypdf, the cleaners), cached afterwards.

What runs there: hidden characters, metadata for every format, visible-mark removal, SynthID band suppression, the style score, and Rewrite through your own model server on `localhost` (Ollama, LM Studio or any OpenAI-compatible server; the Engine view lists the steps, including `OLLAMA_ORIGINS`). Vendor detectors, the SynthID scorer and `exiftool`/`qpdf`/`c2patool` need keys or binaries, so they stay in `wm-serve` and the CLI. The browser limit is 40 MB per file.

```bash
make pages-check     # build web/dist-pages and compare the browser engine with native Python
make pages-deploy    # pages-check, then wrangler deploy to Workers static assets (needs Cloudflare login)
```

`/rewrite` uses only the model in the server's environment (`WATERMARKS_REWRITE_BACKEND`, `WATERMARKS_REWRITE_MODEL`, `WATERMARKS_REWRITE_BASE_URL`); a request that names its own endpoint, model or key is refused. `/visible` runs the built-in texture and blend fills only; the external inpainting command stays CLI-only.

```bash
wm-serve                     # http://127.0.0.1:8765 (web UI at /), optional bearer key
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/capabilities
curl -s http://127.0.0.1:8765/openapi.json
# POST /inspect, /detect, /clean with {"file": "<base64>", "name": "x.png", "options": {...}}
# POST /rewrite with {"file": ..., "name": "x.txt", "strength": "paraphrase"}
# POST /visible with {"file": ..., "name": "x.png", "box": [x, y, w, h], "backend": "texture"}
```

Every deployment refuses requests for an unknown `Host` (a loopback bind answers only to `127.0.0.1`, `localhost` and `::1`, which stops DNS rebinding), refuses browser POSTs from another origin, drops idle sockets after 60 s, and runs at most 4 jobs at once (503 with `Retry-After` past that). To run it as a public service:

| Variable | Effect |
| --- | --- |
| `WATERMARKS_SERVER_PUBLIC=1` | 30 POSTs per minute per client (429 past that), client address from the last `X-Forwarded-For` hop, and `/rewrite` and `/detect` turned off so visitors can't spend your model or vendor keys |
| `WATERMARKS_SERVER_ALLOWED_HOSTS` | Comma-separated host names to answer to, such as `wm.example.com,healthcheck.railway.app` |
| `WATERMARKS_SERVER_RATE_LIMIT` | POSTs per minute per client; `0` turns it off |
| `WATERMARKS_SERVER_MAX_JOBS` | Concurrent jobs (default 4) |
| `WATERMARKS_MAX_INPUT_BYTES` | Largest accepted file (default 256 MB) |
| `PORT` | Listening port when `WATERMARKS_SERVER_PORT` is unset, as Railway and Cloud Run set it |

`railway.json` deploys the core `Dockerfile` with `/health` as the health check.

Docker (core image with exiftool + qpdf + c2patool baked in) and compose profiles for the heavy harnesses:

```bash
make docker-core-build
docker compose up --build -d                  # core only
docker compose --profile harness --profile heavy up --build -d
./compose-check.sh
```

See [`skills/remove-ai-marks/references/service-mode.md`](https://github.com/PyModel/watermark-remover/blob/main/skills/remove-ai-marks/references/service-mode.md) for the thin-client curl flow and [`docs/windows-autostart.md`](https://github.com/PyModel/watermark-remover/blob/main/docs/windows-autostart.md) for a Windows login task.
