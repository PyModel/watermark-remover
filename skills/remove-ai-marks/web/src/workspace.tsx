import { useEffect, useMemo, useRef, useState } from "react"
import { fromBase64, type Box, type Capabilities, type CleanOptions, type Inspection, type TextHit } from "./api"
import { Icon, type IconName } from "./icons"
import type { Item, Queue, Tool } from "./queue"
import { reveal } from "./reveal"
import { capabilityRows } from "./server"
import { ResultCard, ToolsCard, toolsFor } from "./tools"
import { Card, Chip, MiniStat, Progress, Segmented, formatBytes, plural, smooth, unicodeName, type Tone } from "./ui"

type Filter = "all" | "text" | "image" | "container"

interface Props {
  queue: Queue
  caps: Capabilities | null
  offline: boolean
  onReconnect: () => void
  selected: number | null
  setSelected: (id: number | null) => void
  addFiles: (files: File[]) => void
}

export function Workspace({ queue, caps, offline, onReconnect, selected, setSelected, addFiles }: Props) {
  const [filter, setFilter] = useState<Filter>("all")
  const [search, setSearch] = useState("")
  const [options, setOptions] = useState<CleanOptions>({ also_layer_a_text: true })
  const [tool, setTool] = useState<Tool>("clean")
  const [boxes, setBoxes] = useState<Record<number, Box | null>>({})
  const [active, setActive] = useState<string | null>(null)

  const items = queue.items
  const item = items.find((candidate) => candidate.id === selected) ?? null
  const kind = item?.inspection?.kind

  // A new file starts on its first available tool, never on one left over from the last file.
  useEffect(() => {
    setActive(null)
    if (kind) setTool(toolsFor(kind, caps?.disabled)[0] ?? "clean")
  }, [selected, kind, caps])

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase()
    return items.filter(
      (candidate) =>
        (filter === "all" || candidate.inspection?.kind === filter) && (!q || candidate.file.name.toLowerCase().includes(q)),
    )
  }, [items, filter, search])

  const select = (id: number | null) => setSelected(id)

  return (
    <>
      <section className="hero">
        <div>
          <p className="eyebrow">Find and remove AI provenance marks, then check the result again</p>
          <h1>Workspace</h1>
        </div>
        {items.length > 0 && (
          <div className="hero-controls">
            <Segmented<Filter>
              label="Show files"
              value={filter}
              onChange={setFilter}
              options={[
                { id: "all", label: "All" },
                { id: "text", label: "Text" },
                { id: "image", label: "Images" },
                { id: "container", label: "Documents" },
              ]}
            />
            <label className="search">
              <Icon name="search" />
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.currentTarget.value)}
                placeholder="Search files"
                aria-label="Search files"
              />
            </label>
          </div>
        )}
      </section>

      {offline && (
        <div className="banner" role="alert">
          <Icon name="alert" />
          <span>Can't reach wm-serve. Check that it is still running.</span>
          <button type="button" className="button" onClick={onReconnect}>
            Try again
          </button>
        </div>
      )}

      {items.length > 0 && <Metrics items={items} caps={caps} />}

      {items.length === 0 ? (
        <>
          <Intake onFiles={addFiles} limit={caps?.max_input_bytes ?? null} />
          {caps && <CapabilityCards caps={caps} />}
        </>
      ) : (
        <>
          <FilesCard
            items={visible}
            total={items.length}
            selected={selected}
            onSelect={select}
            queue={queue}
            options={options}
            onFiles={addFiles}
            onCleared={() => {
              queue.clear()
              select(null)
            }}
          />
          {item ? (
            <div className="detail">
              <DocumentCard
                key={item.id}
                item={item}
                active={active}
                boxing={tool === "visible"}
                box={boxes[item.id] ?? null}
                setBox={(box) => setBoxes((all) => ({ ...all, [item.id]: box }))}
              />
              <div className="side">
                <FindingsCard item={item} active={active} setActive={setActive} onRetry={() => void queue.retry(item.id)} />
                {item.inspection && item.inspection.kind !== "unknown" && (
                  <ToolsCard
                    key={item.id}
                    item={item}
                    caps={caps}
                    tool={tool}
                    setTool={setTool}
                    options={options}
                    setOptions={setOptions}
                    box={boxes[item.id] ?? null}
                    setBox={(box) => setBoxes((all) => ({ ...all, [item.id]: box }))}
                    queue={queue}
                  />
                )}
                {item.result && <ResultCard item={item} />}
              </div>
            </div>
          ) : (
            <p className="empty-note">Select a file above to see what was found in it.</p>
          )}
        </>
      )}
    </>
  )
}

/* ---------- metrics ---------- */

function hiddenCount(inspection: Inspection | null): number {
  if (!inspection) return 0
  if (inspection.kind === "text") return inspection.report.suspicious_total ?? 0
  return (inspection.report.layer_a_hits ?? []).reduce((sum, hit) => sum + hit.count, 0)
}

function Metrics({ items, caps }: { items: Item[]; caps: Capabilities | null }) {
  const done = items.filter((item) => item.result).length
  const busy = items.filter((item) => item.status === "inspecting" || item.status === "working").length
  const failed = items.filter((item) => item.status === "failed").length
  const hiddenFound = items.reduce((sum, item) => sum + hiddenCount(item.inspection), 0)
  const hiddenLeft = items.reduce((sum, item) => sum + (item.result ? hiddenCount(item.result.recheck) : hiddenCount(item.inspection)), 0)
  const metaFound = items.reduce((sum, item) => sum + (item.inspection?.report.findings?.length ?? 0), 0)
  const metaLeft = items.reduce(
    (sum, item) => sum + ((item.result ? item.result.recheck : item.inspection)?.report.findings?.length ?? 0),
    0,
  )
  const rewritten = items.filter((item) => item.result?.via === "rewrite").length
  const detectors = caps && !caps.disabled.includes("detect") ? Object.values(caps.text_detectors).filter(Boolean).length : 0
  const rewriteOn = !!caps?.layer_b.configured && !caps.disabled.includes("rewrite")
  const pct = (part: number, whole: number) => (whole ? (part / whole) * 100 : 0)

  return (
    <section className="metrics" aria-label="Session totals">
      <article className="metric">
        <div className="metric-head">
          <h2>Files cleaned</h2>
          <span className="round-icon">
            <Icon name="queue" />
          </span>
        </div>
        <p className="metric-value">
          <strong>{done}</strong>
          <span>of {items.length}</span>
        </p>
        <Progress value={pct(done, items.length)} label="Files cleaned" />
        <div className="metric-foot">
          <MiniStat label="In progress" value={busy} />
          <MiniStat label="Failed" value={failed} align="right" />
        </div>
      </article>

      <article className="metric">
        <div className="metric-head">
          <h2>Hidden characters</h2>
          <span className="round-icon">
            <Icon name="eye" />
          </span>
        </div>
        <p className="metric-value">
          <strong>{hiddenFound.toLocaleString()}</strong>
          <span>found</span>
        </p>
        <Progress value={pct(hiddenFound - hiddenLeft, hiddenFound)} label="Hidden characters removed" tone="mark" />
        <div className="metric-foot">
          <MiniStat label="Removed" value={(hiddenFound - hiddenLeft).toLocaleString()} />
          <MiniStat label="Still there" value={hiddenLeft.toLocaleString()} align="right" />
        </div>
      </article>

      <article className="metric">
        <div className="metric-head">
          <h2>Metadata entries</h2>
          <span className="round-icon">
            <Icon name="database" />
          </span>
        </div>
        <p className="metric-value">
          <strong>{metaFound.toLocaleString()}</strong>
          <span>found</span>
        </p>
        <Progress value={pct(metaFound - metaLeft, metaFound)} label="Metadata entries removed" tone="green" />
        <div className="metric-foot">
          <MiniStat label="Removed" value={Math.max(0, metaFound - metaLeft)} />
          <MiniStat label="Still there" value={metaLeft} align="right" />
        </div>
      </article>

      <article className="metric">
        <div className="metric-head">
          <h2>Wording</h2>
          <span className="round-icon">
            <Icon name="layers" />
          </span>
        </div>
        <p className="metric-value">
          <strong className="metric-word">{caps ? (rewriteOn ? "Rewrite ready" : "Rewrite off") : "—"}</strong>
        </p>
        <p className="metric-note">
          {caps?.disabled.includes("rewrite")
            ? "This public server doesn't run rewrites. Run wm-serve yourself to use one."
            : rewriteOn
              ? "Token-level watermarks need a rewrite; hidden characters and metadata don't."
              : "Token-level watermarks need a rewrite model set up on the server."}
        </p>
        <div className="metric-foot">
          <MiniStat label="Rewritten" value={rewritten} />
          <MiniStat label="Detectors on" value={caps ? detectors : "—"} align="right" />
        </div>
      </article>
    </section>
  )
}

/* ---------- intake ---------- */

// A real reveal() of a real string: the example shows exactly what a dropped file will look like.
const SAMPLE = reveal("Looks like plain text.​ Copied from a chat­ window.", ["U+200B", "U+00AD"])

function FilePicker({ onFiles, children, className }: { onFiles: (files: File[]) => void; children: React.ReactNode; className: string }) {
  return (
    <label className={className}>
      <input
        type="file"
        multiple
        className="visually-hidden"
        onChange={(event) => {
          onFiles([...(event.currentTarget.files ?? [])])
          event.currentTarget.value = ""
        }}
      />
      {children}
    </label>
  )
}

function Intake({ onFiles, limit }: { onFiles: (files: File[]) => void; limit: number | null }) {
  return (
    <section className="intake" aria-labelledby="intake-title">
      <div className="intake-icon">
        <Icon name="upload" size={24} />
      </div>
      <h2 id="intake-title">Drop files to inspect them</h2>
      <p>
        Text, Markdown and code, PNG, JPEG, WebP and other images, PDF, Word, EPUB, HTML and SVG.
        {limit !== null && ` Up to ${formatBytes(limit)} each.`}
      </p>
      <p className="sample" aria-label="Example: a sentence with two hidden characters, each shown as a lit code point">
        {SAMPLE.segments.map((segment, index) =>
          segment.kind === "text" ? segment.value : <mark key={index}>{segment.cp.slice(2)}</mark>,
        )}
      </p>
      <FilePicker onFiles={onFiles} className="button primary large">
        <Icon name="upload" size={18} />
        Choose files
      </FilePicker>
      <p className="intake-help">Inspecting is read-only. Nothing changes until you clean, and every result is a new file.</p>
    </section>
  )
}

const CARD_INFO: Record<string, { icon: IconName; title: string; works: string }> = {
  "Hidden characters": { icon: "eye", title: "Hidden characters", works: "Text, Markdown, code, and the text inside documents" },
  Metadata: { icon: "database", title: "Metadata", works: "Images, PDF, Office, EPUB, HTML, SVG" },
  "Visible marks": { icon: "crop", title: "Visible marks", works: "PNG and JPEG, saved as PNG" },
  "SynthID band suppression": { icon: "image", title: "Pixel watermarks", works: "PNG output only" },
  "Rewrite (Layer B)": { icon: "pen", title: "Wording", works: "Text files" },
}

function CapabilityCards({ caps }: { caps: Capabilities }) {
  const rows = capabilityRows(caps).filter((row) => row.name in CARD_INFO)
  return (
    <section className="capabilities" aria-label="What this server can remove">
      {rows.map((row) => (
        <article className="capability" key={row.name}>
          <div className="capability-head">
            <span className="tile-icon">
              <Icon name={CARD_INFO[row.name]!.icon} />
            </span>
            <Chip tone={row.status[0]}>{row.status[1]}</Chip>
          </div>
          <h3>{CARD_INFO[row.name]!.title}</h3>
          <p>{row.enables}.</p>
          {row.status[1] === "Needs setup" && <p className="capability-hint">Set up on the server. The Server view lists what to set.</p>}
          {row.status[1] === "Off on this server" && <p className="capability-hint">{row.hint}</p>}
          <p className="capability-works">{CARD_INFO[row.name]!.works}</p>
        </article>
      ))}
    </section>
  )
}

/* ---------- files table ---------- */

const KIND_LABEL = { text: "Text", image: "Image", container: "Document", unknown: "Unknown" } as const

function statusOf(item: Item): [Tone, string] {
  if (item.status === "inspecting") return ["info", "Inspecting"]
  if (item.status === "working") return ["info", { clean: "Cleaning", rewrite: "Rewriting", visible: "Inpainting", detect: "Detecting" }[item.work ?? "clean"]]
  if (item.status === "failed") return ["bad", "Failed"]
  if (item.error) return ["bad", "Error"]
  if (item.result) return ["ok", "Cleaned"]
  if (item.inspection?.kind === "unknown") return ["off", "Not recognized"]
  return item.inspection?.suspicious ? ["mark", "Marks found"] : ["off", "Nothing found"]
}

function findingsCount(inspection: Inspection | null): number {
  if (!inspection) return 0
  return hiddenCount(inspection) + (inspection.report.findings?.length ?? 0)
}

function FilesCard({
  items,
  total,
  selected,
  onSelect,
  queue,
  options,
  onFiles,
  onCleared,
}: {
  items: Item[]
  total: number
  selected: number | null
  onSelect: (id: number) => void
  queue: Queue
  options: CleanOptions
  onFiles: (files: File[]) => void
  onCleared: () => void
}) {
  const cleanable = queue.items.filter((item) => item.status === "ready" && item.inspection?.kind !== "unknown").length
  const downloadable = queue.items.filter((item) => item.result)
  const max = Math.max(1, ...items.map((item) => findingsCount(item.inspection)))
  return (
    <Card
      title={
        <>
          Files <span className="count">{total}</span>
        </>
      }
      icon="queue"
      className="table-card"
      actions={
        <div className="card-actions">
          <FilePicker onFiles={onFiles} className="button">
            <Icon name="upload" size={16} />
            Add files
          </FilePicker>
          {downloadable.length > 1 && (
            <button type="button" className="button" onClick={() => downloadAll(downloadable)}>
              <Icon name="download" size={16} />
              Download all
            </button>
          )}
          <button type="button" className="button" onClick={onCleared}>
            Clear
          </button>
          <button
            type="button"
            className="button primary"
            disabled={cleanable === 0}
            onClick={() => queue.cleanAll(() => options)}
          >
            Clean {cleanable > 1 ? `all ${cleanable}` : cleanable === 1 ? "1 file" : "all"}
          </button>
        </div>
      }
    >
      <div className="table files" role="table" aria-label="Files">
        <div className="table-row table-head" role="row">
          <span role="columnheader">Name</span>
          <span role="columnheader">Type</span>
          <span role="columnheader">Findings</span>
          <span role="columnheader">Status</span>
          <span role="columnheader" className="bar-col">
            <span className="visually-hidden">Share of findings</span>
          </span>
        </div>
        {items.length === 0 && <p className="table-empty">No files match this filter.</p>}
        {items.map((item) => {
          const [tone, label] = statusOf(item)
          const count = findingsCount(item.inspection)
          return (
            <div className="table-row" role="row" key={item.id} aria-current={item.id === selected ? "true" : undefined}>
              <span role="cell" className="table-name">
                <button type="button" className="row-select" onClick={() => onSelect(item.id)}>
                  <Icon name={item.inspection?.kind === "image" ? "image" : "doc"} size={15} />
                  <span className="truncate">{item.file.name}</span>
                </button>
              </span>
              <span role="cell">{item.inspection ? KIND_LABEL[item.inspection.kind] : "—"}</span>
              <span role="cell" className="num">
                {item.inspection ? count.toLocaleString() : "—"}
              </span>
              <span role="cell">
                <Chip tone={tone}>{label}</Chip>
              </span>
              <span role="cell" className="bar-col">
                <span className="bar-track" aria-hidden="true">
                  <span className={`bar-fill ${item.result ? "bar-done" : count ? "bar-mark" : ""}`} style={{ width: `${(count / max) * 100}%` }} />
                </span>
              </span>
            </div>
          )
        })}
      </div>
    </Card>
  )
}

function downloadAll(items: Item[]) {
  // ponytail: one anchor click per file; the browser may ask once to allow several downloads. Zip when that annoys.
  for (const item of items) {
    const url = URL.createObjectURL(new Blob([fromBase64(item.result!.cleaned.cleaned)], { type: item.result!.mime }))
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = item.result!.name
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
}

/* ---------- document ---------- */

function DocumentCard({
  item,
  active,
  boxing,
  box,
  setBox,
}: {
  item: Item
  active: string | null
  boxing: boolean
  box: Box | null
  setBox: (box: Box | null) => void
}) {
  const [view, setView] = useState<"original" | "cleaned">("original")
  useEffect(() => {
    if (item.result) setView("cleaned")
  }, [item.result])

  const { inspection, result } = item
  const showCleaned = view === "cleaned" && result !== null
  const detail = [inspection?.report.format?.toUpperCase(), formatBytes(item.file.size)].filter(Boolean).join(", ")

  return (
    <Card
      title={<span className="truncate">{item.file.name}</span>}
      icon={inspection?.kind === "image" ? "image" : "doc"}
      className="document"
      badge={<span className="card-meta">{detail}</span>}
      actions={
        result && (
          <Segmented
            label="Show version"
            value={view}
            onChange={setView}
            options={[
              { id: "original", label: "Original" },
              { id: "cleaned", label: "Cleaned" },
            ]}
          />
        )
      }
    >
      {!inspection ? (
        <p className="placeholder" aria-busy={item.status === "inspecting"}>
          {item.status === "inspecting" ? "Reading the file…" : "This file couldn't be inspected."}
        </p>
      ) : inspection.kind === "text" && item.text !== null ? (
        <TextView
          text={showCleaned ? (result!.cleanedText ?? "") : item.text}
          hits={(showCleaned ? result!.recheck : inspection).report.hits ?? []}
          active={active}
          label={showCleaned ? "Cleaned text" : "Original text"}
        />
      ) : inspection.kind === "image" ? (
        <ImageView
          item={item}
          cleaned={showCleaned}
          boxing={boxing && !showCleaned}
          box={box}
          setBox={setBox}
        />
      ) : inspection.kind === "unknown" ? (
        <p className="placeholder">wm-serve doesn't recognize this format. Rename the file with its real extension, such as .txt, .png or .pdf, and add it again.</p>
      ) : (
        <Evidence inspection={inspection} recheck={showCleaned ? result!.recheck : null} />
      )}
    </Card>
  )
}

function TextView({ text, hits, active, label }: { text: string; hits: TextHit[]; active: string | null; label: string }) {
  const labels = useMemo(() => new Map(hits.map((hit) => [hit.codepoint, hit.label])), [hits])
  const revealed = useMemo(() => reveal(text, labels.keys()), [text, labels])
  const ref = useRef<HTMLPreElement>(null)

  // Built once per text; hovering a finding only flips classes, so a long file never re-renders.
  const nodes = useMemo(
    () =>
      revealed.segments.map((segment, index) =>
        segment.kind === "text" ? (
          segment.value
        ) : (
          <mark key={index} data-cp={segment.cp} title={labels.get(segment.cp) ?? segment.cp}>
            {segment.cp.slice(2)}
          </mark>
        ),
      ),
    [revealed, labels],
  )

  useEffect(() => {
    const root = ref.current
    if (!root) return
    root.classList.toggle("focused", active !== null)
    for (const mark of root.querySelectorAll("mark.on")) mark.classList.remove("on")
    if (active !== null) for (const mark of root.querySelectorAll(`mark[data-cp="${active}"]`)) mark.classList.add("on")
  }, [active, nodes])

  if (text.length === 0) return <p className="placeholder">This file is empty.</p>
  return (
    <>
      <pre className="doc" ref={ref} tabIndex={0} aria-label={label}>
        {nodes}
      </pre>
      <p className="card-note">
        {[...text].length.toLocaleString()} characters.
        {revealed.truncated && " Showing the first 100,000; counts cover the whole file."}
      </p>
    </>
  )
}

function ImageView({ item, cleaned, boxing, box, setBox }: { item: Item; cleaned: boolean; boxing: boolean; box: Box | null; setBox: (box: Box | null) => void }) {
  const [url, setUrl] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [natural, setNatural] = useState<[number, number] | null>(null)
  const img = useRef<HTMLImageElement>(null)
  const drag = useRef<[number, number] | null>(null)

  useEffect(() => {
    setFailed(false)
    const blob = cleaned && item.result ? new Blob([fromBase64(item.result.cleaned.cleaned)], { type: item.result.mime }) : item.file
    const next = URL.createObjectURL(blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [cleaned, item.file, item.result])

  /** Pointer position in the image's own pixels, whatever size it is drawn at. */
  const toPixels = (event: React.PointerEvent): [number, number] => {
    const rect = img.current!.getBoundingClientRect()
    const [w, h] = natural ?? [rect.width, rect.height]
    return [
      Math.round(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * w),
      Math.round(Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) * h),
    ]
  }

  const onDown = (event: React.PointerEvent) => {
    if (!boxing || !natural) return
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = toPixels(event)
    setBox(null)
  }
  const onMove = (event: React.PointerEvent) => {
    if (!drag.current) return
    const [x0, y0] = drag.current
    const [x1, y1] = toPixels(event)
    setBox([Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)])
  }
  const onUp = () => {
    drag.current = null
  }

  if (failed)
    return (
      <p className="placeholder">
        No preview: this browser can't decode the image. {boxing && "Type the box in pixels in the Visible mark panel instead."}
      </p>
    )
  return (
    <div className="image-stage" data-boxing={boxing || undefined}>
      <div className="image-frame" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        {url && (
          <img
            ref={img}
            src={url}
            alt={`${cleaned ? "Cleaned" : "Original"} ${item.file.name}`}
            draggable={false}
            onLoad={(event) => setNatural([event.currentTarget.naturalWidth, event.currentTarget.naturalHeight])}
            onError={() => setFailed(true)}
          />
        )}
        {boxing && box && natural && box[2] > 0 && box[3] > 0 && (
          <span
            className="box"
            aria-hidden="true"
            style={{
              left: `${(box[0] / natural[0]) * 100}%`,
              top: `${(box[1] / natural[1]) * 100}%`,
              width: `${(box[2] / natural[0]) * 100}%`,
              height: `${(box[3] / natural[1]) * 100}%`,
            }}
          />
        )}
      </div>
      <p className="card-note">
        {natural && `${natural[0]} × ${natural[1]} pixels. `}
        {boxing ? "Drag over the mark to box it." : cleaned ? "Showing the cleaned file." : ""}
      </p>
      {!cleaned && item.inspection && <Evidence inspection={item.inspection} recheck={null} compact />}
    </div>
  )
}

function Evidence({ inspection, recheck, compact }: { inspection: Inspection; recheck: Inspection | null; compact?: boolean }) {
  const findings = inspection.report.findings ?? []
  const remaining = new Set(recheck?.report.findings ?? [])
  if (findings.length === 0) return compact ? null : <p className="placeholder">No provenance metadata in this file.</p>
  return (
    <ul className="evidence" aria-label="Metadata found">
      {findings.map((finding, index) => {
        const gone = recheck !== null && !remaining.has(finding)
        return (
          <li key={index} className={gone ? "gone" : undefined}>
            <code>{finding}</code>
            {gone && <Chip tone="ok">Removed</Chip>}
          </li>
        )
      })}
    </ul>
  )
}

/* ---------- findings ---------- */

function FindingsCard({
  item,
  active,
  setActive,
  onRetry,
}: {
  item: Item
  active: string | null
  setActive: (cp: string | null) => void
  onRetry: () => void
}) {
  const { inspection } = item
  if (!inspection)
    return (
      <Card title="Findings" icon="search">
        {item.error ? (
          <>
            <div className="error" role="alert">
              <Icon name="alert" size={16} />
              <p>{item.error.message}</p>
            </div>
            <button type="button" className="button" onClick={onRetry}>
              <Icon name="refresh" size={16} />
              Inspect again
            </button>
          </>
        ) : (
          <p className="placeholder" aria-busy="true">
            Inspecting…
          </p>
        )}
      </Card>
    )
  const { kind, report } = inspection
  if (kind === "unknown")
    return (
      <Card title="Format not recognized" icon="alert">
        <p className="card-text">Rename the file with its real extension and add it again.</p>
      </Card>
    )

  const hits = kind === "text" ? (report.hits ?? []) : (report.layer_a_hits ?? [])
  const total = hiddenCount(inspection)
  const headline =
    kind === "text"
      ? total === 0
        ? "No hidden characters"
        : `${total.toLocaleString()} hidden ${plural(total, "character")}`
      : report.has_c2pa
        ? "C2PA manifest found"
        : report.has_ai_metadata
          ? "AI metadata found"
          : (report.findings ?? []).length
            ? `${report.findings!.length} metadata ${report.findings!.length === 1 ? "entry" : "entries"}`
            : "No provenance metadata"
  return (
    <Card title="Findings" icon="search" badge={<Chip tone={inspection.suspicious ? "mark" : "ok"}>{inspection.suspicious ? "Marks found" : "Clean so far"}</Chip>}>
      <p className="verdict">{headline}</p>
      <p className="class-line">
        {kind === "text" ? "Layer A" : "Layer M"}, <strong>verifiable</strong>: every finding is counted exactly.
      </p>
      {hits.length > 0 && (
        <>
          {kind !== "text" && <h3 className="sub">Hidden characters in the text</h3>}
          <HitList hits={hits} active={active} setActive={setActive} />
        </>
      )}
      <div className="rows">
        {kind === "text" && (
          <>
            <div className="stat-row">
              <span className="stat-label">Watermark detectors</span>
              <span className="stat-value">{detectionLabel(inspection.detection_status)}</span>
            </div>
            {report.stylometry && (
              <div className="stat-row">
                <span className="stat-label">Style score</span>
                <span className="stat-value">
                  {report.stylometry.status === "insufficient_length"
                    ? `Needs 30 words (has ${report.stylometry.word_count})`
                    : `${report.stylometry.score.toFixed(2)}, ${report.stylometry.confidence_level.toLowerCase().replaceAll("_", " ")}`}
                </span>
              </div>
            )}
          </>
        )}
        {(report.findings_confidence ?? []).length > 0 && (
          <div className="stat-row">
            <span className="stat-label">Confidence</span>
            <span className="stat-value">{confidenceSummary(report.findings_confidence!)}</span>
          </div>
        )}
      </div>
      {kind === "text" && (
        <p className="card-note">
          Watermarks in the word choice itself can't be seen here. A rewrite with a model is the only way to weaken them.
        </p>
      )}
    </Card>
  )
}

function HitList({ hits, active, setActive }: { hits: TextHit[]; active: string | null; setActive: (cp: string | null) => void }) {
  return (
    <ul className="hits">
      {hits.map((hit) => (
        <li key={hit.codepoint}>
          <button
            type="button"
            className="hit"
            aria-pressed={active === hit.codepoint}
            onMouseEnter={() => setActive(hit.codepoint)}
            onMouseLeave={() => setActive(null)}
            onFocus={() => setActive(hit.codepoint)}
            onBlur={() => setActive(null)}
            onClick={() =>
              document.querySelector(`mark[data-cp="${hit.codepoint}"]`)?.scrollIntoView({ block: "center", behavior: smooth() })
            }
          >
            <mark className="hit-cp">{hit.codepoint.slice(2)}</mark>
            <span className="hit-name">{unicodeName(hit.label)}</span>
            <span className="hit-count">×{hit.count.toLocaleString()}</span>
            <span className="hit-confidence">{hit.confidence.replaceAll("_", " ")}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

export function detectionLabel(status: Inspection["detection_status"]): string {
  return {
    DETECTED: "Watermark detected",
    INCONCLUSIVE: "Inconclusive: can't rule one out",
    NOT_DETECTED: "None detected",
    NOT_RUN: "Not run",
  }[status]
}

function confidenceSummary(levels: string[]): string {
  const counts = new Map<string, number>()
  for (const level of levels) counts.set(level, (counts.get(level) ?? 0) + 1)
  return [...counts].map(([level, count]) => `${count} ${level.replaceAll("_", " ")}`).join(", ")
}
