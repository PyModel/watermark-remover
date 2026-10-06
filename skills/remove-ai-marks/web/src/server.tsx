import { useState } from "react"
import { apiKey, type Capabilities } from "./api"
import type { Connection } from "./App"
import { Icon } from "./icons"
import { Card, Check, Chip, StatRow, type Tone } from "./ui"

interface Row {
  name: string
  enables: string
  status: [Tone, string]
  hint?: string
}

/** Every entry of /capabilities as one row: what it is for, and what to do if it is off. */
export function capabilityRows(caps: Capabilities): Row[] {
  const ready: [Tone, string] = ["ok", "Ready"]
  const setup: [Tone, string] = ["off", "Needs setup"]
  const layerB = caps.layer_b
  const rows: Row[] = [
    { name: "Hidden characters", enables: "Find and remove zero-width and other invisible Unicode in text", status: ready },
    { name: "Metadata", enables: "Strip C2PA, EXIF, XMP and AI tags from images, PDF, Office, EPUB, HTML and SVG", status: ready },
    { name: "Style score", enables: "Stylometry on text over 30 words; a signal, never proof", status: caps.scorers.stylometry ? ready : setup },
    { name: "Visible marks", enables: "Paint out a logo or label inside a box you draw (PNG and JPEG)", status: ready },
    { name: "SynthID band suppression", enables: "Best-effort DCT pass on PNG output; can't be verified without a scorer", status: ["warn", "Best-effort"] },
    {
      name: "Rewrite (Layer B)",
      enables: "Reword text with a model to weaken token-level watermarks",
      status: layerB.configured ? (layerB.endpoint_allowed ? ready : ["bad", "Blocked"]) : setup,
      hint: layerB.configured
        ? layerB.endpoint_allowed
          ? `${layerB.backend} at ${layerB.base_url_host}${layerB.loopback ? " (this machine)" : ""}`
          : `${layerB.base_url_host} is not this machine. Set WATERMARKS_REWRITE_ALLOW_REMOTE=1 to send text there.`
        : "Set WATERMARKS_REWRITE_BACKEND, WATERMARKS_REWRITE_MODEL and WATERMARKS_REWRITE_BASE_URL, then restart wm-serve.",
    },
    {
      name: "SynthID scorer",
      enables: "Score images for a SynthID watermark before and after cleaning",
      status: caps.scorers.synthid || caps.scorers.synthid_http ? ready : setup,
      hint: "Set WATERMARKS_SYNTHID_SCORER_URL or REVERSE_SYNTHID_DIR.",
    },
    ...Object.entries(caps.text_detectors).map(([name, on]): Row => ({
      name: `Detector: ${name}`,
      enables: "Check text for a vendor watermark. Sends the text to that detector.",
      status: on ? ready : setup,
      hint: on ? undefined : "Configure this detector's environment variables, then restart wm-serve.",
    })),
    ...(["c2patool", "exiftool", "qpdf"] as const).map((tool): Row => ({
      name: tool,
      enables: { c2patool: "Read and verify C2PA manifests in depth", exiftool: "Extra metadata coverage for camera formats", qpdf: "Rebuild PDFs so removed objects are really gone" }[tool],
      status: caps.tools[tool] ? ready : ["off", "Not installed"],
      hint: caps.tools[tool] ? undefined : `Optional. Built-in cleaning still runs without ${tool}.`,
    })),
    ...Object.entries(caps.extras).map(([name, extra]): Row => ({
      name: `Extra: ${name}`,
      enables: `Optional Python extra watermark-remover[${name}]`,
      status: extra.available ? ready : ["off", "Not installed"],
      hint: extra.available ? undefined : extra.hint,
    })),
    {
      name: "Pixel regeneration",
      enables: "CtrlRegen and diffusion backends for pixel watermarks",
      status: ["off", "CLI only"],
      hint: "Not exposed over HTTP yet. Use the wm command line.",
    },
  ]
  return rows
}

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/health", "Liveness and version"],
  ["GET", "/capabilities", "What this server can run"],
  ["GET", "/openapi.json", "OpenAPI 3 description of every endpoint"],
  ["POST", "/inspect", "Findings for one file; read-only"],
  ["POST", "/clean", "Cleaned bytes plus a report of every change"],
  ["POST", "/detect", "Run configured watermark detectors"],
  ["POST", "/rewrite", "Layer B rewrite with the server's model"],
  ["POST", "/visible", "Inpaint a box in a PNG or JPEG"],
]

export function ServerView({ connection, onRetry, onKey }: { connection: Connection; onRetry: () => void; onKey: (key: string) => void }) {
  const caps = connection.caps
  return (
    <>
      <section className="hero">
        <div>
          <p className="eyebrow">What this wm-serve instance can run, and how to turn the rest on</p>
          <h1>Server</h1>
        </div>
        <div className="hero-controls">
          <button type="button" className="button" onClick={onRetry}>
            <Icon name="refresh" size={16} />
            Check again
          </button>
        </div>
      </section>

      {!caps ? (
        <Card title="Not connected" icon="alert" className="span-all">
          <p className="card-text">
            {connection.state === "connecting"
              ? "Asking wm-serve what it can do…"
              : "wm-serve didn't answer. Start it with wm-serve (or make serve), then check again."}
          </p>
        </Card>
      ) : (
        <>
          <div className="grid-3">
            <Card title="Connection" icon="link">
              <div className="rows">
                <StatRow icon="pulse" label="Status">
                  <Chip tone="ok">Connected</Chip>
                </StatRow>
                <StatRow icon="cube" label="Version">{caps.version}</StatRow>
                <StatRow icon="link" label="Address">{location.host}</StatRow>
                <StatRow icon="key" label="API key">{apiKey() ? "Sent with each request" : "Not required"}</StatRow>
              </div>
            </Card>
            <Card title="Rewrite model" icon="layers" badge={<Chip tone={caps.layer_b.configured ? "ok" : "off"}>{caps.layer_b.configured ? "Configured" : "Not set up"}</Chip>}>
              <div className="rows">
                <StatRow label="Backend">{caps.layer_b.backend ?? "None"}</StatRow>
                <StatRow label="Endpoint host">{caps.layer_b.base_url_host ?? "None"}</StatRow>
                <StatRow label="On this machine">{caps.layer_b.base_url_host ? (caps.layer_b.loopback ? "Yes" : "No") : "—"}</StatRow>
                <StatRow label="Allowed">{caps.layer_b.endpoint_allowed ? "Yes" : "No"}</StatRow>
              </div>
              <p className="card-note">Chosen on the server only. This page can't change the model or where text is sent.</p>
            </Card>
            <Card title="Detectors" icon="shield">
              <div className="rows">
                {Object.entries(caps.text_detectors).map(([name, on]) => (
                  <div className="health-row" key={name}>
                    <span className="health-label">
                      <Check ok={on} label={on ? "Configured" : "Not configured"} />
                      {name}
                    </span>
                    <span>{on ? "Ready" : "Off"}</span>
                  </div>
                ))}
                <div className="health-row">
                  <span className="health-label">
                    <Check ok={caps.scorers.synthid || caps.scorers.synthid_http} label="SynthID scorer" />
                    SynthID image scorer
                  </span>
                  <span>{caps.scorers.synthid || caps.scorers.synthid_http ? "Ready" : "Off"}</span>
                </div>
              </div>
            </Card>
          </div>

          <Card title="Capabilities" icon="database" className="table-card">
            <div className="table" role="table" aria-label="Capabilities">
              <div className="table-row table-head" role="row">
                <span role="columnheader">Capability</span>
                <span role="columnheader">What it does</span>
                <span role="columnheader">Status</span>
              </div>
              {capabilityRows(caps).map((row) => (
                <div className="table-row" role="row" key={row.name}>
                  <span role="cell" className="table-name">
                    {row.name}
                  </span>
                  <span role="cell" className="table-desc">
                    {row.enables}
                    {row.hint && <small>{row.hint}</small>}
                  </span>
                  <span role="cell">
                    <Chip tone={row.status[0]}>{row.status[1]}</Chip>
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}

      <div className="grid-2">
        <Card title="Endpoints" icon="queue">
          <ul className="endpoints">
            {ENDPOINTS.map(([method, path, what]) => (
              <li key={path}>
                <code className={`method method-${method.toLowerCase()}`}>{method}</code>
                <code>{path}</code>
                <span>{what}</span>
              </li>
            ))}
          </ul>
        </Card>
        <Card title="API key" icon="key">
          <p className="card-text">
            If wm-serve was started with --api-key, enter that key here. It is kept for this tab only and sent as a bearer
            token.
          </p>
          <KeyForm rejected={false} onSave={onKey} compact />
        </Card>
      </div>
    </>
  )
}

export function KeyForm({ onSave, rejected, compact }: { onSave: (key: string) => void; rejected: boolean; compact?: boolean }) {
  const [key, setKey] = useState("")
  return (
    <form
      className={`keyform ${compact ? "" : "keyform-gate"}`}
      onSubmit={(event) => {
        event.preventDefault()
        if (key.trim()) onSave(key.trim())
      }}
    >
      <label htmlFor={compact ? "api-key-settings" : "api-key"}>API key</label>
      {rejected && (
        <p className="error-text" role="alert">
          The server didn't accept that key. Check the value passed to --api-key.
        </p>
      )}
      <div className="keyform-row">
        <input
          id={compact ? "api-key-settings" : "api-key"}
          type="password"
          autoComplete="off"
          value={key}
          onChange={(event) => setKey(event.currentTarget.value)}
        />
        <button type="submit" className="button primary">
          {compact ? "Save key" : "Unlock"}
        </button>
      </div>
    </form>
  )
}
