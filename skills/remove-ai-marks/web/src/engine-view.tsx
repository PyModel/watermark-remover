import { useState } from "react"
import { DEFAULT_MODEL_SERVER, HOSTED_LIMITS, HOSTED_MODEL, modelServer, setModelServer, usingHosted, type ModelServer } from "./api"
import type { Connection } from "./App"
import { Icon } from "./icons"
import { CapabilityTable } from "./server"
import { SITE_ORIGIN } from "./target"
import { Card, Chip, StatRow, formatBytes } from "./ui"

/** Ports the common local model servers listen on by default. */
type Local = Exclude<ModelServer["backend"], "hosted">
const DEFAULT_URL: Record<Local, string> = {
  ollama: "http://localhost:11434",
  "openai-compatible": "http://localhost:1234",
}

/** Only this computer: the page's CSP allows nothing else, so anything else would just fail later. */
export function localAddressError(value: string): string | null {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return "Enter a full address, such as http://localhost:11434."
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return "The address must start with http:// or https://."
  if (!["localhost", "127.0.0.1"].includes(url.hostname)) {
    return "Only a model server on this computer can be used here: localhost or 127.0.0.1."
  }
  return null
}

/** The static build's settings view: the engine in this tab, and the visitor's own model server. */
export function EngineView({ connection, stage, onRetry }: { connection: Connection; stage: string; onRetry: () => void }) {
  const caps = connection.caps
  const status =
    connection.state === "online" ? <Chip tone="ok">Ready</Chip> : connection.state === "offline" ? <Chip tone="bad">Didn't load</Chip> : <Chip tone="info">{stage}</Chip>
  return (
    <>
      <section className="hero">
        <div>
          <p className="eyebrow">Cleaning runs in this tab. Files are never uploaded.</p>
          <h1>Engine</h1>
        </div>
        {connection.state === "offline" && (
          <div className="hero-controls">
            <button type="button" className="button" onClick={onRetry}>
              <Icon name="refresh" size={16} />
              Load again
            </button>
          </div>
        )}
      </section>

      <div className="grid-3">
        <Card title="Cleaning engine" icon="cpu">
          <div className="rows">
            <StatRow icon="pulse" label="Status">{status}</StatRow>
            <StatRow icon="cube" label="Cleaners">{caps ? `watermark-remover ${caps.version}` : "—"}</StatRow>
            <StatRow icon="layers" label="Runs on">Python in WebAssembly</StatRow>
            <StatRow icon="upload" label="File size limit">{caps ? formatBytes(caps.max_input_bytes) : "—"}</StatRow>
          </div>
          <p className="card-note">
            The same Python modules as the wm command line. The first visit downloads about 8 MB; later visits load it from
            the browser cache.
          </p>
        </Card>
        <Card title="Where data goes" icon="shield">
          <div className="rows">
            <StatRow icon="doc" label="Your files">Stay in this tab</StatRow>
            <StatRow icon="link" label="This page connects to">This site only</StatRow>
            <StatRow icon="layers" label="Rewrite sends text to">
              {usingHosted() ? "The hosted model (Cloudflare Workers AI)" : modelServer() ? "Your model server" : "Nowhere until you choose a model"}
            </StatRow>
          </div>
          <p className="card-note">
            The page's security policy blocks every other address, so you can confirm this in your browser's network panel.
            The hosted model is this site's own /api; it keeps no copy of your text.
          </p>
        </Card>
        <ModelServerCard connection={connection} onSaved={onRetry} />
      </div>

      <SetupSteps />

      {caps && <CapabilityTable caps={caps} />}
    </>
  )
}

function ModelServerCard({ connection, onSaved }: { connection: Connection; onSaved: () => void }) {
  const saved = modelServer()
  const [mode, setMode] = useState<"hosted" | "local">(saved?.backend === "hosted" ? "hosted" : "local")
  const [draft, setDraft] = useState<ModelServer>(saved && saved.backend !== "hosted" ? saved : DEFAULT_MODEL_SERVER)
  const [error, setError] = useState<string | null>(null)
  const configured = connection.caps?.layer_b.configured ?? false
  const set = (patch: Partial<ModelServer>) => setDraft((previous) => ({ ...previous, ...patch }))
  const save = (value: ModelServer | null) => {
    setModelServer(value)
    onSaved()
  }
  return (
    <Card
      title="Model for Rewrite"
      icon="layers"
      badge={<Chip tone={configured ? "ok" : "off"}>{configured ? (usingHosted() ? "Hosted model" : "Your server") : "Not set up"}</Chip>}
    >
      <div className="segmented" role="radiogroup" aria-label="Which model rewrites">
        <label>
          <input type="radio" name="model-mode" checked={mode === "hosted"} onChange={() => setMode("hosted")} />
          Free hosted model
        </label>
        <label>
          <input type="radio" name="model-mode" checked={mode === "local"} onChange={() => setMode("local")} />
          Your own server
        </label>
      </div>
      {mode === "hosted" ? (
        <HostedChoice onUse={() => save(HOSTED_MODEL)} onStop={usingHosted() ? () => save(null) : undefined} />
      ) : (
        <form
          className="model-form"
          onSubmit={(event) => {
            event.preventDefault()
            const problem = localAddressError(draft.base_url.trim()) ?? (draft.model.trim() ? null : "Enter the model name, such as qwen3:4b.")
            setError(problem)
            if (problem) return
            save({ ...draft, base_url: draft.base_url.trim(), model: draft.model.trim(), api_key: draft.api_key.trim() })
          }}
        >
          <label className="field">
            <span>Server</span>
            <select
              value={draft.backend}
              onChange={(event) => {
                const backend = event.currentTarget.value as Local
                const previous = draft.backend as Local
                set({ backend, base_url: draft.base_url === DEFAULT_URL[previous] ? DEFAULT_URL[backend] : draft.base_url })
              }}
            >
              <option value="ollama">Ollama</option>
              <option value="openai-compatible">OpenAI-compatible (LM Studio, llama.cpp, vLLM)</option>
            </select>
          </label>
          <label className="field">
            <span>Address</span>
            <input value={draft.base_url} onChange={(event) => set({ base_url: event.currentTarget.value })} inputMode="url" spellCheck={false} />
          </label>
          <label className="field">
            <span>Model</span>
            <input value={draft.model} onChange={(event) => set({ model: event.currentTarget.value })} placeholder="qwen3:4b" spellCheck={false} />
          </label>
          {draft.backend === "openai-compatible" && (
            <label className="field">
              <span>API key, if your server asks for one</span>
              <input type="password" autoComplete="off" value={draft.api_key} onChange={(event) => set({ api_key: event.currentTarget.value })} />
            </label>
          )}
          {error && (
            <p className="error-text" role="alert">
              {error}
            </p>
          )}
          <div className="form-actions">
            <button type="submit" className="button primary">
              Save model server
            </button>
            {saved && saved.backend !== "hosted" && (
              <button type="button" className="button" onClick={() => (setDraft(DEFAULT_MODEL_SERVER), save(null))}>
                Remove
              </button>
            )}
          </div>
          <p className="card-note">Saved in this browser only. Text goes straight from this tab to that address.</p>
        </form>
      )}
    </Card>
  )
}

/** The free hosted model, as a choice the visitor makes knowing the text leaves the device. */
export function HostedChoice({ onUse, onStop }: { onUse: () => void; onStop?: () => void }) {
  return (
    <div className="hosted-choice">
      <p className="card-text">
        Rewrites run on a small model (Gemma 4) on Cloudflare Workers AI, at no cost. The text you rewrite is sent there; your
        files and everything else stay in this tab.
      </p>
      <p className="card-note">
        Up to {HOSTED_LIMITS.perDay} rewrites per connection per day, {HOSTED_LIMITS.chars.toLocaleString()} characters each, while the
        day's free allowance lasts. TSAPA search isn't offered here.
      </p>
      <div className="form-actions">
        {onStop ? (
          <button type="button" className="button" onClick={onStop}>
            Stop using the hosted model
          </button>
        ) : (
          <button type="button" className="button primary" onClick={onUse}>
            Use the free hosted model
          </button>
        )}
      </div>
    </div>
  )
}

/** Browsers block a page from calling a local server unless the server allows that page. */
function SetupSteps() {
  return (
    <Card title="Let your model server answer this page" icon="queue" className="span-all">
      <div className="grid-2 setup">
        <div>
          <h3 className="sub">Ollama</h3>
          <ol className="steps">
            <li>
              Allow this site, then quit and reopen Ollama. On macOS:
              <code className="command">launchctl setenv OLLAMA_ORIGINS "{SITE_ORIGIN}"</code>
              On Linux or Windows, set <code>OLLAMA_ORIGINS={SITE_ORIGIN}</code> for the Ollama service.
            </li>
            <li>
              Download a small instruct model, for example:
              <code className="command">ollama pull qwen3:4b</code>
            </li>
            <li>Add it above with the address http://localhost:11434 and the model name you pulled.</li>
          </ol>
        </div>
        <div>
          <h3 className="sub">LM Studio and other OpenAI-compatible servers</h3>
          <ol className="steps">
            <li>Start the local server and turn on CORS in its server settings (LM Studio: Developer, then Settings).</li>
            <li>Load a model, then add the server above with its address and the model's name.</li>
          </ol>
          <h3 className="sub">If the browser asks</h3>
          <p className="card-text">
            Chrome may ask whether this site can reach devices on your local network. Choose Allow, or Rewrite can't reach
            your server.
          </p>
        </div>
      </div>
    </Card>
  )
}
