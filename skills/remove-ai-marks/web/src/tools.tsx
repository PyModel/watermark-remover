import { useEffect, useState } from "react"
import { fromBase64, REWRITE_STRENGTHS, type Box, type Capabilities, type CleanOption, type CleanOptions, type Detection, type Kind, type Strength } from "./api"
import { Icon } from "./icons"
import type { Item, Queue, Tool } from "./queue"
import { Card, Chip, Segmented, plural, sentence, type Tone } from "./ui"

const TOOL_LABEL: Record<Tool, string> = { clean: "Clean", rewrite: "Rewrite", visible: "Visible mark", detect: "Detect" }

export function toolsFor(kind: Kind): Tool[] {
  if (kind === "text") return ["clean", "rewrite", "detect"]
  if (kind === "image") return ["clean", "visible", "detect"]
  if (kind === "container") return ["clean"]
  return []
}

interface ToolsProps {
  item: Item
  caps: Capabilities | null
  tool: Tool
  setTool: (tool: Tool) => void
  options: CleanOptions
  setOptions: (options: CleanOptions) => void
  box: Box | null
  setBox: (box: Box | null) => void
  queue: Queue
}

export function ToolsCard(props: ToolsProps) {
  const { item, tool, setTool } = props
  const kind = item.inspection!.kind
  const tools = toolsFor(kind)
  const busy = item.status === "working" || item.status === "inspecting"
  return (
    <Card title="Actions" icon="pen" className="tools">
      {tools.length > 1 && (
        <Segmented label="Action" value={tool} onChange={setTool} options={tools.map((id) => ({ id, label: TOOL_LABEL[id] }))} />
      )}
      {item.error && item.inspection && (
        <div className="error" role="alert">
          <Icon name="alert" size={16} />
          <p>{item.error.message}</p>
        </div>
      )}
      <fieldset className="tool-panel" disabled={busy}>
        <legend className="visually-hidden">{TOOL_LABEL[tool]}</legend>
        {tool === "clean" && <CleanPanel {...props} />}
        {tool === "rewrite" && <RewritePanel {...props} />}
        {tool === "visible" && <VisiblePanel {...props} />}
        {tool === "detect" && <DetectPanel {...props} />}
      </fieldset>
      <Pending item={item} />
    </Card>
  )
}

/** Live elapsed time, because a rewrite or a TSAPA run can take minutes. */
function Pending({ item }: { item: Item }) {
  const [now, setNow] = useState(Date.now())
  const working = item.status === "working"
  useEffect(() => {
    if (!working) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [working])
  if (!working) return null
  const seconds = Math.max(0, Math.round((now - item.startedAt) / 1000))
  const what = { clean: "Cleaning, then checking the result", rewrite: "Rewriting with the server's model", visible: "Painting out the mark", detect: "Running detectors" }[item.work ?? "clean"]
  return (
    <p className="pending" role="status">
      <span className="spinner" aria-hidden="true" />
      {what}… {seconds}s
    </p>
  )
}

function Option({ id, label, hint, options, setOptions, fallback = false, disabled }: {
  id: CleanOption
  label: string
  hint: string
  options: CleanOptions
  setOptions: (options: CleanOptions) => void
  fallback?: boolean
  disabled?: boolean
}) {
  return (
    <label className="option" data-disabled={disabled || undefined}>
      <input
        type="checkbox"
        checked={!disabled && (options[id] ?? fallback)}
        disabled={disabled}
        onChange={(event) => setOptions({ ...options, [id]: event.currentTarget.checked })}
      />
      <span>
        {label}
        <small>{hint}</small>
      </span>
    </label>
  )
}

function CleanPanel({ item, caps, options, setOptions, queue }: ToolsProps) {
  const kind = item.inspection!.kind
  const png = item.inspection!.report.format === "png"
  const textDetectors = caps ? Object.values(caps.text_detectors).some(Boolean) : false
  const scorer = caps ? caps.scorers.synthid || caps.scorers.synthid_http : false
  const shared = { options, setOptions }
  return (
    <>
      <div className="options">
        {kind === "text" && (
          <>
            <Option id="nfkc" label="Normalize compatibility characters" hint="NFKC folds full-width letters and ligatures into plain ones." {...shared} />
            <Option id="aggressive_homoglyphs" label="Replace look-alike letters" hint="Swaps Cyrillic or Greek look-alikes for Latin. Can change real non-Latin text." {...shared} />
            {textDetectors && (
              <Option id="detect_after" label="Run detectors on the result" hint="Sends the cleaned text to the configured detectors." {...shared} />
            )}
          </>
        )}
        {kind === "image" && (
          <>
            <Option id="keep_non_ai_metadata" label="Keep camera and editing metadata" hint="Removes only AI and provenance entries." {...shared} />
            <Option
              id="remove_synthid"
              label="Suppress SynthID frequency band"
              hint={png ? "Best-effort pass on the pixels. It can't be confirmed without a SynthID scorer." : "PNG files only."}
              disabled={!png}
              {...shared}
            />
            <Option
              id="wmct_marker"
              label="Add a “cleaned by wm” marker"
              hint={png ? "A plain PNG text entry that says this file was cleaned. Off by default." : "PNG files only."}
              disabled={!png}
              {...shared}
            />
            {scorer && (
              <Option id="detect_after" label="Score SynthID after cleaning" hint="Uses the server's SynthID scorer." {...shared} />
            )}
          </>
        )}
        {kind === "container" && (
          <Option id="also_layer_a_text" label="Also remove hidden characters in the text" hint="Cleans the text inside the document as well as its metadata." fallback {...shared} />
        )}
      </div>
      <button type="button" className="button primary wide" onClick={() => void queue.runClean(item.id, options)}>
        {item.result?.via === "clean" ? "Clean again" : "Clean file"}
      </button>
      <p className="card-note">Writes a new file for you to download. The original is never changed.</p>
    </>
  )
}

const STRENGTH_INFO: Record<Strength, [string, string]> = {
  paraphrase: ["Paraphrase", "Rewords each sentence and keeps the structure."],
  backtranslate: ["Back-translate", "Translates to another language and back."],
  structural: ["Restructure", "Outlines every claim, then writes the document again from the outline."],
  humanize: ["Humanize", "Rewrites so it reads less like model output."],
  code: ["Code", "For source code: rewrites comments and names, keeps behavior."],
  tsapa: ["TSAPA search", "Evolves many candidate rewrites. Slow: many model calls, often minutes."],
}

function RewritePanel({ item, caps, queue }: ToolsProps) {
  const [strength, setStrength] = useState<Strength>("paraphrase")
  const [lang, setLang] = useState("French")
  const layerB = caps?.layer_b
  const ready = !!layerB?.configured && layerB.endpoint_allowed
  return (
    <>
      {!ready && (
        <div className="notice">
          <Icon name="alert" size={16} />
          <p>
            {layerB?.configured
              ? `The rewrite model at ${layerB.base_url_host} isn't on this machine, so wm-serve won't send text to it. Set WATERMARKS_REWRITE_ALLOW_REMOTE=1 to allow it.`
              : "No rewrite model is set up. Set WATERMARKS_REWRITE_BACKEND, WATERMARKS_REWRITE_MODEL and WATERMARKS_REWRITE_BASE_URL, then restart wm-serve."}
          </p>
        </div>
      )}
      <div className="choices" role="radiogroup" aria-label="Rewrite strength">
        {REWRITE_STRENGTHS.map((id) => (
          <label className="choice" key={id}>
            <input type="radio" name={`strength-${item.id}`} checked={strength === id} onChange={() => setStrength(id)} disabled={!ready} />
            <span>
              {STRENGTH_INFO[id][0]}
              <small>{STRENGTH_INFO[id][1]}</small>
            </span>
          </label>
        ))}
      </div>
      {strength === "backtranslate" && (
        <label className="field">
          <span>Language to go through</span>
          <input value={lang} onChange={(event) => setLang(event.currentTarget.value)} maxLength={40} disabled={!ready} />
        </label>
      )}
      <button
        type="button"
        className="button primary wide"
        disabled={!ready || (strength === "backtranslate" && !lang.trim())}
        onClick={() => void queue.runRewrite(item.id, strength, strength === "backtranslate" ? lang.trim() : undefined)}
      >
        Rewrite text
      </button>
      <p className="card-note">
        Best-effort: a rewrite weakens statistical watermarks but can't certify they are gone. Hidden characters are removed from
        the output as well.
      </p>
    </>
  )
}

function VisiblePanel({ item, box, setBox, queue }: ToolsProps) {
  const [backend, setBackend] = useState<"texture" | "simple">("texture")
  const format = item.inspection!.report.format
  const supported = format === "png" || format === "jpeg"
  const value = box ?? [0, 0, 0, 0]
  const tooLarge = item.error?.message.includes("too large to fill with a copied patch")
  const set = (index: number, raw: string) => {
    const next = [...value] as Box
    next[index] = Math.max(0, Math.floor(Number(raw) || 0))
    setBox(next)
  }
  if (!supported)
    return (
      <div className="notice">
        <Icon name="alert" size={16} />
        <p>Visible-mark removal reads PNG and JPEG. Convert this {format?.toUpperCase() ?? "image"} first.</p>
      </div>
    )
  return (
    <>
      <p className="card-text">Drag a box over the logo or label in the image, or type it in pixels.</p>
      <div className="box-fields">
        {(["X", "Y", "Width", "Height"] as const).map((label, index) => (
          <label className="field" key={label}>
            <span>{label}</span>
            <input type="number" min={0} inputMode="numeric" value={value[index]} onChange={(event) => set(index, event.currentTarget.value)} />
          </label>
        ))}
      </div>
      <div className="choices" role="radiogroup" aria-label="Fill method">
        <label className="choice">
          <input type="radio" name={`backend-${item.id}`} checked={backend === "texture"} onChange={() => setBackend("texture")} />
          <span>
            Copy matching texture
            <small>Fills the box with a patch from elsewhere in the image. Best for small marks.</small>
          </span>
        </label>
        <label className="choice">
          <input type="radio" name={`backend-${item.id}`} checked={backend === "simple"} onChange={() => setBackend("simple")} />
          <span>
            Blend surrounding colors
            <small>Smooth fill from the box edges. Works for any size.</small>
          </span>
        </label>
      </div>
      {tooLarge && backend === "texture" && (
        <button type="button" className="button" onClick={() => setBackend("simple")}>
          Use “Blend surrounding colors” instead
        </button>
      )}
      <button
        type="button"
        className="button primary wide"
        disabled={!(value[2] > 0 && value[3] > 0)}
        onClick={() => void queue.runVisible(item.id, value, backend)}
      >
        Remove mark
      </button>
      <p className="card-note">Best-effort: check the result by eye. Output is a new PNG without the original metadata.</p>
    </>
  )
}

function DetectPanel({ item, caps, queue }: ToolsProps) {
  const [consent, setConsent] = useState(false)
  const kind = item.inspection!.kind
  const configured = caps ? Object.entries(caps.text_detectors).filter(([, on]) => on).map(([name]) => name) : []
  const scorer = caps ? caps.scorers.synthid || caps.scorers.synthid_http : false
  const needsConsent = kind === "text" && configured.length > 0
  const blocked = kind === "image" && !scorer
  return (
    <>
      {kind === "text" ? (
        <p className="card-text">
          {configured.length
            ? `Runs ${configured.join(", ")} and the style score.`
            : "No vendor detectors are set up on this server, so only the style score runs."}
        </p>
      ) : (
        <p className="card-text">Scores the image with the server's SynthID scorer.</p>
      )}
      {blocked && (
        <div className="notice">
          <Icon name="alert" size={16} />
          <p>No SynthID scorer is set up. Set WATERMARKS_SYNTHID_SCORER_URL or REVERSE_SYNTHID_DIR, then restart wm-serve.</p>
        </div>
      )}
      {needsConsent && (
        <label className="option">
          <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.currentTarget.checked)} />
          <span>
            Send this text to the detectors
            <small>Some detectors are vendor APIs. The text leaves this server when they run.</small>
          </span>
        </label>
      )}
      <button
        type="button"
        className="button primary wide"
        disabled={blocked || (needsConsent && !consent)}
        onClick={() => void queue.runDetect(item.id)}
      >
        Run detectors
      </button>
      {item.detections && <Detections detections={item.detections} />}
    </>
  )
}

function verdictOf(detection: Detection): [Tone, string] {
  if (detection.available === false) return ["off", "Not run"]
  if (detection.is_watermarked === true || detection.verdict === "DETECTED") return ["mark", "Detected"]
  if (detection.verdict === "NOT_DETECTED" || detection.is_watermarked === false) return ["ok", "Not detected"]
  if (typeof detection.score === "number" && detection.verdict === undefined) return ["info", `Score ${detection.score.toFixed(2)}`]
  return ["warn", "Inconclusive"]
}

function Detections({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) return <p className="card-note">No detector answered for this file type.</p>
  return (
    <ul className="detections" aria-label="Detector results">
      {detections.map((detection, index) => {
        const [tone, label] = verdictOf(detection)
        const detail = detection.error ?? detection.reason ?? (detection.confidence_level ? sentence(detection.confidence_level.toLowerCase().replaceAll("_", " ")) : null)
        return (
          <li key={index}>
            <span className="detector-name">{detection.detector ?? "Detector"}</span>
            <Chip tone={tone}>{label}</Chip>
            {detail && <small>{detail}</small>}
          </li>
        )
      })}
      <li className="card-note">Inconclusive or not run is not the same as clean.</li>
    </ul>
  )
}

/* ---------- result ---------- */

export function ResultCard({ item }: { item: Item }) {
  const result = item.result!
  const { cleaned, recheck, via } = result
  const [download, setDownload] = useState<string>()
  useEffect(() => {
    const url = URL.createObjectURL(new Blob([fromBase64(cleaned.cleaned)], { type: result.mime }))
    setDownload(url)
    return () => URL.revokeObjectURL(url)
  }, [cleaned, result.mime])

  const remainingHits = recheck.report.hits ?? recheck.report.layer_a_hits ?? []
  const hiddenLeft = recheck.kind === "text" ? (recheck.report.suspicious_total ?? 0) : remainingHits.reduce((sum, hit) => sum + hit.count, 0)
  const metaLeft = recheck.report.findings?.length ?? 0
  const left = hiddenLeft + metaLeft
  const bestEffort = via !== "clean" || !!cleaned.report.synthid_removal

  return (
    <Card title="Result" icon="check" badge={<Chip tone={bestEffort ? "warn" : "ok"}>{bestEffort ? "Best-effort" : "Verifiable"}</Chip>}>
      <p className="verdict">{headline(item)}</p>
      <Changes item={item} />
      <div className={`recheck ${left === 0 ? "is-ok" : "is-left"}`}>
        <h3 className="sub">Checked again</h3>
        <p>
          {left === 0
            ? `wm-serve re-inspected the new file: no hidden characters${recheck.kind === "text" ? "" : " or provenance metadata"} remain.`
            : `wm-serve re-inspected the new file: ${left} ${plural(left, "finding")} remain.`}
        </p>
        {metaLeft > 0 && (
          <ul className="evidence">
            {recheck.report.findings!.map((finding, index) => (
              <li key={index}>
                <code>{finding}</code>
              </li>
            ))}
          </ul>
        )}
      </div>
      {bestEffort && <p className="card-note">{bestEffortNote(item)}</p>}
      <a className="button primary wide" href={download} download={result.name}>
        <Icon name="download" size={16} />
        <span className="truncate">Download {result.name}</span>
      </a>
    </Card>
  )
}

function headline(item: Item): string {
  const { cleaned, via } = item.result!
  if (via === "rewrite") return `Rewritten (${cleaned.report.rewrite?.strength ?? "rewrite"})`
  if (via === "visible") return "Mark painted out"
  const stats = cleaned.report.stats
  if (stats) {
    const parts = []
    if (stats.removed_count) parts.push(`Removed ${stats.removed_count}`)
    if (stats.replaced_count) parts.push(`replaced ${stats.replaced_count}`)
    return parts.length ? sentence(parts.join(", ")) : "Nothing needed removing"
  }
  const changes = cleaned.report.action_details?.length ?? 0
  return changes ? `Made ${changes} ${plural(changes, "change")}` : "Nothing needed removing"
}

function bestEffortNote(item: Item): string {
  const report = item.result!.cleaned.report
  return (
    report.rewrite?.note ??
    report.synthid_removal?.note?.replace(/^./, (c) => c.toUpperCase()) ??
    report.note ??
    "This step can't be verified exactly. Check the result yourself."
  )
}

function Changes({ item }: { item: Item }) {
  const report = item.result!.cleaned.report
  const stats = report.stats
  if (stats) {
    const groups: [string, Record<string, number>][] = [
      ["Removed", stats.removed],
      ["Replaced with a plain space", stats.replaced],
      ["Kept because they carry meaning", stats.preserved_semantic],
    ]
    return (
      <>
        {groups
          .filter(([, entries]) => Object.keys(entries).length > 0)
          .map(([title, entries]) => (
            <div key={title}>
              <h3 className="sub">{title}</h3>
              <ul className="changes">
                {Object.entries(entries).map(([label, count]) => (
                  <li key={label}>
                    <mark className="hit-cp muted">{label.split(" ", 1)[0]!.slice(2)}</mark>
                    <span>{sentence(label.replace(/^U\+[0-9A-F]+\s*/i, "").replace(/\s*\([A-Za-z]{2}\)$/, "").toLowerCase())}</span>
                    <span className="hit-count">×{count}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
      </>
    )
  }
  const rewrite = report.rewrite
  if (rewrite)
    return (
      <div className="rows">
        <div className="stat-row">
          <span className="stat-label">Model</span>
          <span className="stat-value">{String(rewrite.model ?? "—")}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Characters</span>
          <span className="stat-value">
            {String(rewrite.input_chars ?? "?")} → {String(rewrite.output_chars ?? "?")}
          </span>
        </div>
      </div>
    )
  const actions = report.action_details ?? []
  if (actions.length === 0) return <p className="card-note">Nothing needed changing.</p>
  return (
    <ul className="changes plain">
      {actions.map((action, index) => (
        <li key={index}>{sentence(action.text)}</li>
      ))}
    </ul>
  )
}
