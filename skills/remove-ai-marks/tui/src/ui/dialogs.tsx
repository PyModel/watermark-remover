import { useKeyboard, useRenderer } from "@opentui/solid"
import { createSignal, For, onMount, Show } from "solid-js"
import type { Confirm, DetectedEndpoint } from "../protocol"
import { COMMANDS, runCommand } from "../commands"
import type { AppController } from "../store"
import { endpointLabel, NO_ENDPOINT } from "../store"
import { resultClassColor, theme } from "../theme"
import { DialogFrame, Keys, type ListItem, ListDialog } from "./dialog"

type Props = { ctl: AppController; active: boolean }

// -- command palette --------------------------------------------------------

export function Palette(props: Props) {
  const items = (): ListItem[] => COMMANDS.map((c) => ({ id: c.name, title: c.title, right: c.key ?? `/${c.name}` }))
  return (
    <ListDialog
      title="Commands"
      items={items()}
      active={props.active}
      filter
      placeholder="search"
      onSelect={(item) => {
        props.ctl.closeDialog()
        runCommand(props.ctl, item.id)
      }}
      onClose={() => props.ctl.closeDialog()}
    />
  )
}

// -- preset -------------------------------------------------------------------

export function PresetDialog(props: Props & { onboarding?: boolean }) {
  const items = (): ListItem[] =>
    props.ctl.presets().map((preset) => ({
      id: preset.key,
      title: preset.label,
      right: preset.result_class,
      rightColor: resultClassColor(preset.result_class),
      detail: preset.flags.length
        ? `${preset.description}\nadds ${preset.flags.join(" ")}`
        : preset.description,
    }))
  return (
    <ListDialog
      title="What to remove"
      steps={props.onboarding ? [3, 3] : undefined}
      items={items()}
      initial={props.ctl.app.state.preset}
      active={props.active}
      width={70}
      footer="Labels are promises: Verifiable is counted before and after; Best-effort has no detector guarantee."
      onSelect={(item) => {
        props.ctl.setPreset(item.id)
        props.ctl.closeDialog()
        if (props.onboarding) void props.ctl.finishOnboarding(false)
      }}
      onClose={() => {
        props.ctl.closeDialog()
        if (props.onboarding) void props.ctl.finishOnboarding(true)
      }}
    />
  )
}

// -- model --------------------------------------------------------------------

export function ModelDialog(props: Props & { onboarding?: boolean }) {
  const ctl = props.ctl
  const [picking, setPicking] = createSignal<DetectedEndpoint | null>(null)
  const [manual, setManual] = createSignal(false)

  onMount(() => {
    if (!ctl.app.endpoints && !ctl.app.scanning) void ctl.detect()
  })

  const next = () => {
    ctl.closeDialog()
    if (props.onboarding) ctl.openDialog({ type: "preset", onboarding: true })
  }

  const choose = (endpoint: DetectedEndpoint, model: string | null) => {
    ctl.setEndpoint({ backend: endpoint.backend, base_url: endpoint.base_url, model })
    ctl.toast(`Using ${model ?? endpoint.base_url}.`, "success")
    next()
  }

  const serverItems = (): ListItem[] => {
    const found = ctl.app.endpoints ?? []
    const current = endpointLabel(ctl.app.state.endpoint)
    return [
      ...found.map((endpoint) => ({
        id: `${endpoint.backend}|${endpoint.base_url}`,
        title: `${endpoint.label}  ${endpoint.base_url.replace(/^https?:\/\//, "")}`,
        right: endpoint.reachable
          ? `${endpoint.models.length} model${endpoint.models.length === 1 ? "" : "s"}`
          : "Not running",
        rightColor: endpoint.reachable ? theme.success : theme.faint,
        titleColor: endpoint.reachable ? theme.text : theme.muted,
        disabled: !endpoint.reachable,
      })),
      { id: "rescan", title: ctl.app.scanning ? "Scanning…" : "Scan again", right: "This machine only", disabled: ctl.app.scanning },
      { id: "manual", title: "Enter a server", right: "Any URL" },
      {
        id: "none",
        title: props.onboarding ? "Skip, clean hidden marks only" : "No model",
        right: current ? `Now ${current}` : "Rewrite off",
      },
    ]
  }

  const select = (item: ListItem) => {
    if (item.id === "rescan") return void ctl.detect()
    if (item.id === "manual") return setManual(true)
    if (item.id === "none") {
      ctl.setEndpoint({ ...NO_ENDPOINT })
      return next()
    }
    const endpoint = (ctl.app.endpoints ?? []).find((e) => `${e.backend}|${e.base_url}` === item.id)
    if (!endpoint) return
    if (endpoint.models.length > 1) return setPicking(endpoint)
    choose(endpoint, endpoint.models[0] ?? null)
  }

  const close = () => {
    ctl.closeDialog()
    if (props.onboarding) void ctl.finishOnboarding(true)
  }

  return (
    <Show
      when={!manual()}
      fallback={<ManualEndpoint ctl={ctl} active={props.active} onDone={next} onBack={() => setManual(false)} />}
    >
      <Show
        when={picking()}
        fallback={
          <ListDialog
            title="A model for rewriting"
            steps={props.onboarding ? [2, 3] : undefined}
            items={serverItems()}
            active={props.active}
            width={72}
            empty="Scanning local servers…"
            footer={
              "Only the LLM rewrite needs a model. The scan asks this machine's own servers for their model list. " +
              "It never sends a document and never contacts another machine."
            }
            onSelect={select}
            onClose={close}
          />
        }
      >
        {(endpoint: () => DetectedEndpoint) => (
          <ListDialog
            title={`Models on ${endpoint().label}`}
            items={endpoint().models.map((model: string) => ({ id: model, title: model }))}
            initial={ctl.app.state.endpoint.model ?? undefined}
            active={props.active}
            filter
            width={72}
            onSelect={(item) => choose(endpoint(), item.id)}
            onClose={() => setPicking(null)}
          />
        )}
      </Show>
    </Show>
  )
}

/**
 * A server typed by hand.  The backend is chosen, not guessed from the port:
 * a port is a convention, and a wrong guess fails later as an opaque
 * connection error on the first file.
 */
function ManualEndpoint(props: Props & { onDone: () => void; onBack: () => void }) {
  const ctl = props.ctl
  const backends = () => ctl.app.hello?.backends ?? []
  const current = ctl.app.state.endpoint
  const [field, setField] = createSignal<0 | 1 | 2>(0)
  const [backend, setBackend] = createSignal(current.backend ?? backends()[0] ?? "")
  const [baseUrl, setBaseUrl] = createSignal(current.base_url ?? "")
  const [model, setModel] = createSignal(current.model ?? "")
  const [error, setError] = createSignal<string | null>(null)

  const submit = () => {
    const url = baseUrl().trim().replace(/\/+$/, "")
    if (!/^https?:\/\/[^/\s]+/.test(url)) return setError("base URL needs http:// or https:// and a host")
    ctl.setEndpoint({ backend: backend(), base_url: url, model: model().trim() || null })
    props.onDone()
  }

  const cycleBackend = (step: number) => {
    const list = backends()
    if (!list.length) return
    const at = Math.max(0, list.indexOf(backend()))
    setBackend(list[(at + step + list.length) % list.length]!)
  }

  useKeyboard((key) => {
    if (!props.active) return
    if (key.name === "escape") {
      key.preventDefault()
      props.onBack()
    } else if (key.name === "tab" || key.name === "down") {
      key.preventDefault()
      setField((f) => ((f + 1) % 3) as 0 | 1 | 2)
    } else if (key.name === "up") {
      key.preventDefault()
      setField((f) => ((f + 2) % 3) as 0 | 1 | 2)
    } else if (field() === 0 && (key.name === "left" || key.name === "right")) {
      key.preventDefault()
      cycleBackend(key.name === "left" ? -1 : 1)
    } else if (field() === 0 && key.name === "return") {
      key.preventDefault()
      setField(1)
    }
  })

  const label = (text: string, index: number) => <text fg={field() === index ? theme.primary : theme.muted}>{text}</text>

  const fieldBox = (index: 1 | 2, value: () => string, onInput: (v: string) => void, placeholder: string) => (
    <box height={1} backgroundColor={theme.element} paddingX={1} marginBottom={1}>
      <input
        focused={props.active && field() === index}
        value={value()}
        placeholder={placeholder}
        onInput={(v: string) => {
          setError(null)
          onInput(v)
        }}
        onSubmit={() => (index === 1 ? setField(2) : submit())}
        backgroundColor={theme.element}
        focusedBackgroundColor={theme.element}
        textColor={theme.text}
        focusedTextColor={theme.text}
        placeholderColor={theme.faint}
        cursorColor={theme.primary}
      />
    </box>
  )

  return (
    <DialogFrame title="Enter a server" width={72}>
      {label("Backend", 0)}
      <box height={1} marginBottom={1} flexDirection="row" gap={2}>
        <For each={backends()}>
          {(name) => (
            <text fg={name === backend() ? (field() === 0 ? theme.primary : theme.text) : theme.faint}>
              {name === backend() ? `● ${name}` : `○ ${name}`}
            </text>
          )}
        </For>
      </box>
      {label("Base URL", 1)}
      {fieldBox(1, baseUrl, setBaseUrl, "http://127.0.0.1:11434")}
      {label("Model", 2)}
      {fieldBox(2, model, setModel, "e.g. qwen3:14b")}
      <text fg={theme.muted} wrapMode="word">
        A host other than this machine is saved as typed; each clean asks before any text leaves.
      </text>
      <Show when={error()}>
        <text fg={theme.error}>{error()}</text>
      </Show>
      <box marginTop={1}>
        <Keys items={[["tab", "next"], ["← →", "backend"], ["enter", "save"], ["esc", "back"]]} />
      </box>
    </DialogFrame>
  )
}

// -- welcome (first run) ------------------------------------------------------

export function WelcomeDialog(props: Props) {
  const ctl = props.ctl
  useKeyboard((key) => {
    if (!props.active) return
    if (key.name === "return") {
      key.preventDefault()
      ctl.closeDialog()
      ctl.openDialog({ type: "model", onboarding: true })
    } else if (key.name === "escape") {
      key.preventDefault()
      void ctl.finishOnboarding(true)
    }
  })
  const row = (label: string, color: string, text: string) => (
    <box flexDirection="row" height={1}>
      <text width={16} fg={color}>
        {label}
      </text>
      <text fg={theme.muted} truncate wrapMode="none">
        {text}
      </text>
    </box>
  )
  return (
    <DialogFrame title="Welcome to wm" steps={[1, 3]} width={70}>
      <text fg={theme.text} wrapMode="word">
        wm finds the watermarks and provenance marks hidden in files you own, removes them, and tells you how sure it is.
      </text>
      <box marginTop={1} flexDirection="column">
        {row("Verifiable", theme.success, "Counted before and after the clean")}
        {row("Best-effort", theme.warning, "No detector can confirm the result")}
      </box>
      <box marginTop={1}>
        <text fg={theme.muted} wrapMode="word">
          Removing hidden characters and metadata is Verifiable and needs nothing installed. Next, pick an optional local model for rewriting, then the default clean.
        </text>
      </box>
      <box marginTop={1}>
        <Keys items={[["enter", "continue"], ["esc", "skip setup"]]} />
      </box>
    </DialogFrame>
  )
}

// -- confirm ------------------------------------------------------------------

export function ConfirmDialog(props: Props & { items: Confirm[]; resolve: (ok: boolean) => void }) {
  const answer = (ok: boolean) => {
    props.ctl.closeDialog()
    props.resolve(ok)
  }
  useKeyboard((key) => {
    if (!props.active) return
    if (key.name === "return" || key.name === "y") {
      key.preventDefault()
      answer(true)
    } else if (key.name === "escape" || key.name === "n") {
      key.preventDefault()
      answer(false)
    }
  })
  return (
    <DialogFrame title="Before anything is written" width={72}>
      <box flexDirection="column">
        <For each={props.items}>
          {(item) => (
            <box flexDirection="row" marginBottom={1}>
              <text fg={theme.warning} width={3}>
                !
              </text>
              <text fg={theme.text} wrapMode="word" flexGrow={1}>
                {item.message}
              </text>
            </box>
          )}
        </For>
      </box>
      <Keys items={[["enter", "go ahead"], ["esc", "cancel"]]} />
    </DialogFrame>
  )
}

// -- help ---------------------------------------------------------------------

export function HelpDialog(props: Props) {
  useKeyboard((key) => {
    if (!props.active) return
    if (key.name === "escape" || key.name === "return" || key.name === "f1") {
      key.preventDefault()
      props.ctl.closeDialog()
    }
  })
  const row = (left: string, right: string, color: string = theme.text) => (
    <box flexDirection="row" height={1}>
      <text width={12} fg={color}>
        {left}
      </text>
      <text fg={theme.muted} truncate wrapMode="none">
        {right}
      </text>
    </box>
  )
  return (
    <DialogFrame title="Help" width={72}>
      {row("enter", "add the path, --flag or /command typed in the prompt")}
      {row("↑ ↓", "move through files")}
      {row("ctrl+p", "search every command")}
      <box marginTop={1} flexDirection="column">
        <For each={COMMANDS}>{(command) => row(command.key ?? `/${command.name}`, command.title)}</For>
      </box>
      <box marginTop={1}>
        <text fg={theme.muted} wrapMode="word">
          Any flag `wm` accepts works in the prompt: `--nfkc`, `--glob *.md`, `-o out/`. A bare flag typed again turns
          it off. The command in the footer is the exact `wm` run, so a clean here can be repeated anywhere.
        </text>
      </box>
      <box marginTop={1} flexDirection="column">
        <text fg={theme.faint}>{`settings  ${props.ctl.app.hello?.settings_path ?? ""}`}</text>
        <text fg={theme.faint}>API key   WATERMARKS_REWRITE_API_KEY, read at run time, never shown or saved</text>
      </box>
    </DialogFrame>
  )
}

// -- doctor -------------------------------------------------------------------

export function DoctorDialog(props: Props) {
  onMount(() => void props.ctl.loadChecks())
  useKeyboard((key) => {
    if (!props.active) return
    if (key.name === "escape" || key.name === "return") {
      key.preventDefault()
      props.ctl.closeDialog()
    }
  })
  const checks = () => props.ctl.app.checks
  return (
    <DialogFrame title="This install" width={80}>
      <Show when={checks()} fallback={<text fg={theme.muted}>checking…</text>}>
        <For each={checks()!}>
          {(check) => (
            <box flexDirection="column" marginBottom={check.fix ? 1 : 0}>
              <box flexDirection="row">
                <text width={3} flexShrink={0} fg={check.good ? theme.success : theme.warning}>
                  {check.good ? "✓" : "·"}
                </text>
                <text width={20} flexShrink={0} fg={theme.text}>
                  {check.name}
                </text>
                <text fg={theme.muted} flexGrow={1} flexShrink={1}>
                  {`${check.state}.  ${check.detail}`}
                </text>
              </box>
              <Show when={check.fix}>
                <box paddingLeft={23}>
                  <text fg={theme.primary} selectable>
                    {check.fix}
                  </text>
                </box>
              </Show>
            </box>
          )}
        </For>
      </Show>
    </DialogFrame>
  )
}

// -- history ------------------------------------------------------------------

export function HistoryDialog(props: Props) {
  const renderer = useRenderer()
  onMount(() => void props.ctl.loadHistory())
  const items = (): ListItem[] =>
    props.ctl.app.history.map((entry, i) => ({
      id: String(i),
      title: entry.command,
      right: entry.time,
      detail: entry.summary,
    }))
  return (
    <ListDialog
      title="History"
      items={items()}
      active={props.active}
      width={96}
      empty="Nothing cleaned yet in this session."
      footer="Enter copies the command. Some terminals ignore the copy."
      onSelect={(item) => {
        const entry = props.ctl.app.history[Number(item.id)]
        if (!entry) return
        const ok = renderer.copyToClipboardOSC52(entry.command)
        props.ctl.toast(ok ? "Command copied." : "This terminal did not accept the copy.", ok ? "success" : "error")
      }}
      onClose={() => props.ctl.closeDialog()}
    />
  )
}
