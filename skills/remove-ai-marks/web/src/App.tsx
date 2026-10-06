import { useCallback, useEffect, useState } from "react"
import { ApiError, apiKey, capabilities, setApiKey, type Capabilities } from "./api"
import { BrandMark, Icon, type IconName } from "./icons"
import { useQueue } from "./queue"
import { KeyForm, ServerView } from "./server"
import { Workspace } from "./workspace"

type View = "workspace" | "server"
export type Connection = { state: "connecting" | "online" | "offline" | "locked"; caps: Capabilities | null; rejected: boolean }

const VIEWS: { id: View; icon: IconName; label: string }[] = [
  { id: "workspace", icon: "grid", label: "Workspace" },
  { id: "server", icon: "cpu", label: "Server and capabilities" },
]

export function App() {
  const [view, setView] = useState<View>("workspace")
  const [connection, setConnection] = useState<Connection>({ state: "connecting", caps: null, rejected: false })
  const [dragging, setDragging] = useState(false)

  const connect = useCallback(() => {
    capabilities()
      .then((caps) => setConnection({ state: "online", caps, rejected: false }))
      .catch((error: ApiError) =>
        setConnection({
          state: error.status === 401 ? "locked" : "offline",
          caps: null,
          rejected: error.status === 401 && apiKey() !== "",
        }),
      )
  }, [])
  useEffect(connect, [connect])

  const onError = useCallback((error: ApiError) => {
    if (error.status === 401) setConnection({ state: "locked", caps: null, rejected: apiKey() !== "" })
    if (error.status === 0) setConnection((previous) => ({ ...previous, state: "offline" }))
  }, [])
  const queue = useQueue(onError)
  const [selected, setSelected] = useState<number | null>(null)

  const addFiles = useCallback(
    (files: File[]) => {
      if (files.length === 0) return
      const first = queue.add(files)
      setSelected((current) => current ?? first)
      setView("workspace")
    },
    [queue.add],
  )

  // The whole window is the drop target; the drop zone is only the invitation.
  useEffect(() => {
    let depth = 0
    const hasFiles = (event: DragEvent) => event.dataTransfer?.types.includes("Files") ?? false
    const over = (event: DragEvent) => hasFiles(event) && event.preventDefault()
    const enter = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth++
      setDragging(true)
    }
    const leave = () => {
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDragging(false)
    }
    const drop = (event: DragEvent) => {
      event.preventDefault()
      depth = 0
      setDragging(false)
      addFiles([...(event.dataTransfer?.files ?? [])])
    }
    window.addEventListener("dragover", over)
    window.addEventListener("dragenter", enter)
    window.addEventListener("dragleave", leave)
    window.addEventListener("drop", drop)
    return () => {
      window.removeEventListener("dragover", over)
      window.removeEventListener("dragenter", enter)
      window.removeEventListener("dragleave", leave)
      window.removeEventListener("drop", drop)
    }
  }, [addFiles])

  const unlock = (key: string) => {
    setApiKey(key)
    connect()
  }

  return (
    <div className="canvas" data-dragging={dragging || undefined}>
      <div className="shell">
        <header className="topbar">
          <div className="brand-cluster">
            <BrandMark size={48} />
            <span className="brand">watermark-remover</span>
            <StatusPill connection={connection} />
          </div>
          <div className="runtime-meta">
            {connection.caps && (
              <span className="pill">
                <Icon name="layers" />
                {connection.caps.public
                  ? "Public server"
                  : `Rewrite ${connection.caps.layer_b.configured ? "ready" : "not set up"}`}
              </span>
            )}
            {connection.caps && (
              <span className="pill">
                <Icon name="cube" />
                wm-serve {connection.caps.version}
              </span>
            )}
            <span className="pill pill-address">
              <Icon name="link" />
              {location.host}
            </span>
          </div>
        </header>

        <div className="body">
          <nav className="sidebar" aria-label="Views">
            {VIEWS.map(({ id, icon, label }) => (
              <button
                key={id}
                type="button"
                className="nav-button"
                aria-label={label}
                title={label}
                aria-current={view === id ? "page" : undefined}
                onClick={() => setView(id)}
              >
                <Icon name={icon} />
                {id === "server" && connection.state !== "online" && connection.state !== "connecting" && (
                  <span className="nav-dot" aria-hidden="true" />
                )}
              </button>
            ))}
          </nav>

          <main className="content">
            {connection.state === "locked" ? (
              <section className="hero">
                <div>
                  <p className="eyebrow">This server asks for a key before it cleans anything</p>
                  <h1>Unlock</h1>
                  <KeyForm rejected={connection.rejected} onSave={unlock} />
                </div>
              </section>
            ) : view === "server" ? (
              <ServerView connection={connection} onRetry={connect} onKey={unlock} />
            ) : (
              <Workspace
                queue={queue}
                caps={connection.caps}
                offline={connection.state === "offline"}
                onReconnect={connect}
                selected={selected}
                setSelected={setSelected}
                addFiles={addFiles}
              />
            )}
          </main>
        </div>
      </div>
      <div className="drop-veil" aria-hidden="true">
        <p>Drop to inspect</p>
      </div>
    </div>
  )
}

function StatusPill({ connection }: { connection: Connection }) {
  const label = {
    connecting: "Connecting",
    online: "Connected",
    offline: "Not connected",
    locked: "Locked",
  }[connection.state]
  return (
    <span className="pill" role="status">
      <span className={`dot dot-${connection.state}`} aria-hidden="true" />
      {label}
    </span>
  )
}
