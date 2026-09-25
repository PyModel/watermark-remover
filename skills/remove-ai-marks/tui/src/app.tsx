import { type InputRenderable, TextAttributes } from "@opentui/core"
import { useKeyboard, useTerminalDimensions } from "@opentui/solid"
import { createMemo, createSignal, For, onCleanup, onMount, Show } from "solid-js"
import { runCommand } from "./commands"
import { outputDisplay, parsePrompt, shortPath } from "./prompt"
import type { FileDone, FileEntry, Finding, Reveal } from "./protocol"
import { type AppController, type Dialog, endpointLabel } from "./store"
import { resultClassColor, theme } from "./theme"
import { Keys } from "./ui/dialog"
import {
  ConfirmDialog,
  DoctorDialog,
  HelpDialog,
  HistoryDialog,
  ModelDialog,
  Palette,
  PresetDialog,
  WelcomeDialog,
} from "./ui/dialogs"

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

/**
 * One screen, no tabs.  Files on the left, the selected file on the right, a
 * prompt at the bottom that takes a path, a `--flag` or a `/command`.  The
 * file list is also the progress view and the result table.
 */
export function App(props: { ctl: AppController }) {
  const ctl = props.ctl
  const app = ctl.app
  const dims = useTerminalDimensions()
  const [frame, setFrame] = createSignal(0)
  let input: InputRenderable | undefined

  const timer = setInterval(() => {
    if (app.run) setFrame((f) => (f + 1) % SPINNER.length)
  }, 80)
  onCleanup(() => clearInterval(timer))
  onMount(() => void ctl.start())

  const dialogOpen = () => app.dialogs.length > 0
  const files = () => app.plan?.files ?? []
  const selectedFile = () => files()[app.selected]

  const submit = (value: string) => {
    const parsed = parsePrompt(value)
    if (input) input.value = ""
    if (parsed.kind === "empty") return
    if (parsed.kind === "paths") return ctl.addPaths(parsed.tokens)
    if (parsed.kind === "flags") return ctl.applyFlags(parsed.tokens)
    runCommand(ctl, parsed.name, parsed.arg)
  }

  useKeyboard((key) => {
    if (key.ctrl && (key.name === "c" || key.name === "d")) {
      key.preventDefault()
      return ctl.quit()
    }
    if (dialogOpen()) return
    const name = key.name
    if (key.ctrl && name === "p") {
      key.preventDefault()
      ctl.openDialog({ type: "palette" })
    } else if (key.ctrl && name === "r") {
      key.preventDefault()
      void ctl.clean()
    } else if (key.ctrl && name === "e") {
      key.preventDefault()
      void ctl.inspect()
    } else if (name === "f1") {
      ctl.openDialog({ type: "help" })
    } else if (name === "tab") {
      key.preventDefault()
      ctl.cyclePreset(key.shift ? -1 : 1)
    } else if (name === "up" || (key.ctrl && name === "k")) {
      key.preventDefault()
      ctl.select(app.selected - 1)
    } else if (name === "down" || (key.ctrl && name === "j")) {
      key.preventDefault()
      ctl.select(app.selected + 1)
    } else if (name === "pageup") {
      ctl.select(app.selected - 10)
    } else if (name === "pagedown") {
      ctl.select(app.selected + 10)
    } else if (name === "escape") {
      if (app.run) void ctl.cancel()
      else if (input) input.value = ""
    }
  })

  const listWidth = () => {
    const w = dims().width - 4
    return w < 70 ? w : Math.min(56, Math.max(30, Math.floor(w * 0.42)))
  }
  const showDetail = () => dims().width - 4 >= 70

  return (
    <box width={dims().width} height={dims().height} flexDirection="column" backgroundColor={theme.bg}>
      <Header ctl={ctl} />
      <Show when={!app.fatal} fallback={<Fatal ctl={ctl} />}>
        <Show
          when={files().length || app.plan?.discover_error}
          fallback={
            <box flexGrow={1}>
              <Empty ctl={ctl} />
            </box>
          }
        >
          <box flexGrow={1} flexDirection="row" paddingX={2} paddingTop={1} gap={3}>
            <box width={listWidth()} flexDirection="column">
              <FileList ctl={ctl} frame={frame()} width={listWidth()} />
            </box>
            <Show when={showDetail()}>
              <box flexGrow={1} flexDirection="column">
                <Show when={selectedFile()}>{(file: () => FileEntry) => <Detail ctl={ctl} file={file()} width={dims().width - 7 - listWidth()} />}</Show>
              </box>
            </Show>
          </box>
        </Show>
        <Prompt ctl={ctl} focused={!dialogOpen()} onSubmit={submit} ref={(r) => (input = r)} />
        <Footer ctl={ctl} frame={frame()} />
      </Show>
      <For each={app.dialogs}>
        {(dialog, i) => <DialogView ctl={ctl} dialog={dialog} active={i() === app.dialogs.length - 1} />}
      </For>
    </box>
  )
}

function DialogView(props: { ctl: AppController; dialog: Dialog; active: boolean }) {
  const d = props.dialog
  switch (d.type) {
    case "palette":
      return <Palette ctl={props.ctl} active={props.active} />
    case "preset":
      return <PresetDialog ctl={props.ctl} active={props.active} onboarding={d.onboarding} />
    case "model":
      return <ModelDialog ctl={props.ctl} active={props.active} onboarding={d.onboarding} />
    case "welcome":
      return <WelcomeDialog ctl={props.ctl} active={props.active} />
    case "help":
      return <HelpDialog ctl={props.ctl} active={props.active} />
    case "doctor":
      return <DoctorDialog ctl={props.ctl} active={props.active} />
    case "history":
      return <HistoryDialog ctl={props.ctl} active={props.active} />
    case "confirm":
      return <ConfirmDialog ctl={props.ctl} active={props.active} items={d.items} resolve={d.resolve} />
  }
}

// -- header -------------------------------------------------------------------

function Header(props: { ctl: AppController }) {
  const app = props.ctl.app
  const endpoint = () => endpointLabel(app.state.endpoint)
  return (
    <box height={1} flexDirection="row" justifyContent="space-between" paddingX={2} marginTop={1}>
      <text>
        <span style={{ fg: theme.primary }}>
          <b>wm</b>
        </span>
        <span style={{ fg: theme.muted }}>{"  watermark remover"}</span>
      </text>
      <text>
        <Show when={endpoint()} fallback={<span style={{ fg: theme.faint }}>No model</span>}>
          <span style={{ fg: theme.primary }}>● </span>
          <span style={{ fg: theme.muted }}>{endpoint()!}</span>
        </Show>
      </text>
    </box>
  )
}

// -- empty state --------------------------------------------------------------

function Empty(props: { ctl: AppController }) {
  const hint = (key: string, text: string) => (
    <box flexDirection="row" height={1}>
      <text width={10} fg={theme.text}>
        {key}
      </text>
      <text fg={theme.muted}>{text}</text>
    </box>
  )
  return (
    <box flexGrow={1} flexDirection="column" alignItems="center" justifyContent="center">
      <box flexDirection="column" width={52}>
      <text fg={theme.text}>Find what is hidden in a file, then remove it.</text>
      <box marginTop={1}>
        <text fg={theme.faint}>Type a file or folder below and press enter.</text>
      </box>
      <box marginTop={1} flexDirection="column">
        {hint("ctrl+e", "See what is hidden, without writing")}
        {hint("ctrl+r", "Clean, leaving the originals untouched")}
        {hint("tab", "Choose what to remove")}
        {hint("ctrl+p", "Every command")}
      </box>
      </box>
    </box>
  )
}

// -- file list ----------------------------------------------------------------

function status(ctl: AppController, file: FileEntry, frame: number): { glyph: string; color: string; text: string } {
  const app = ctl.app
  const done = app.results[file.path]
  const run = app.run
  const index = app.plan?.files.findIndex((f) => f.path === file.path) ?? -1
  if (run && run.kind === "clean" && index === run.index && !done) {
    return { glyph: SPINNER[frame]!, color: theme.primary, text: "cleaning" }
  }
  if (done) {
    if (done.error) return { glyph: "✗", color: theme.error, text: "error" }
    const left = Object.values(done.after).reduce((a, b) => a + b, 0)
    return left
      ? { glyph: "◆", color: theme.glow, text: `${left} left` }
      : { glyph: "✓", color: theme.success, text: "cleaned" }
  }
  const finding = app.findings[file.path]
  if (finding) {
    if (finding.error) return { glyph: "✗", color: theme.error, text: "error" }
    const found = Object.values(finding.counts).reduce((a, b) => a + b, 0)
    return finding.suspicious || found
      ? { glyph: "◆", color: theme.glow, text: found ? `${found} mark${found === 1 ? "" : "s"}` : "suspicious" }
      : { glyph: "○", color: theme.success, text: "no marks" }
  }
  return { glyph: "·", color: theme.faint, text: "" }
}

function fileCount(total: number, shown: number) {
  const noun = `${total} file${total === 1 ? "" : "s"}`
  return shown < total ? `${noun}, first ${shown} shown` : noun
}

function FileList(props: { ctl: AppController; frame: number; width: number }) {
  const app = props.ctl.app
  const dims = useTerminalDimensions()
  const files = () => app.plan?.files ?? []
  const rows = () => Math.max(3, dims().height - 13)
  const offset = createMemo(() => {
    const i = app.selected
    return Math.max(0, Math.min(i - Math.floor(rows() / 2), files().length - rows()))
  })
  return (
    <box flexDirection="column">
      <box flexDirection="row" height={1} marginBottom={1} justifyContent="space-between">
        <text fg={theme.muted}>{fileCount(app.plan?.files_total ?? 0, files().length)}</text>
        <Show when={files().length > rows()}>
          <text fg={theme.faint}>{`${app.selected + 1}/${files().length}`}</text>
        </Show>
      </box>
      <Show when={app.plan?.discover_error}>
        <text fg={theme.error} wrapMode="word">
          {app.plan!.discover_error!}
        </text>
      </Show>
      <For each={files().slice(offset(), offset() + rows())}>
        {(file) => {
          const selected = () => files()[app.selected]?.path === file.path
          const state = () => status(props.ctl, file, props.frame)
          return (
            <box
              flexDirection="row"
              height={1}
              backgroundColor={selected() ? theme.element : theme.bg}
              onMouseDown={() => props.ctl.select(files().findIndex((f) => f.path === file.path))}
            >
              <text width={1} fg={theme.primary}>
                {selected() ? "▌" : " "}
              </text>
              <text width={2} fg={state().color}>
                {state().glyph}
              </text>
              <text flexGrow={1} truncate fg={selected() ? theme.text : theme.muted}>
                {shortPath(file.display, props.width - 14)}
              </text>
              <text width={11} textAlign="right" fg={state().color} paddingRight={1}>
                {state().text}
              </text>
            </box>
          )
        }}
      </For>
    </box>
  )
}

// -- detail -------------------------------------------------------------------

function Detail(props: { ctl: AppController; file: FileEntry; width: number }) {
  const app = props.ctl.app
  const dims = useTerminalDimensions()
  const finding = () => app.findings[props.file.path]
  const result = () => app.results[props.file.path]
  const stream = () => app.streams[props.file.path]
  const room = () => Math.max(3, dims().height - 14)

  // The two file headers and the hunk headers repeat what the title says.
  const diffLines = () =>
    (result()?.diff ?? "")
      .split("\n")
      .slice(2)
      .filter((line) => line && !line.startsWith("@@"))
  const diffColor = (line: string) =>
    line.startsWith("+") ? theme.success : line.startsWith("-") ? theme.error : theme.faint

  return (
    <box flexDirection="column">
      <box flexDirection="row" height={1} marginBottom={1}>
        <text fg={theme.text} attributes={TextAttributes.BOLD} truncate flexGrow={1}>
          {props.file.display}
        </text>
      </box>

      <Show when={stream() && !result()}>
        <text fg={theme.primary}>Rewriting…</text>
        <text fg={theme.muted} wrapMode="word">
          {stream()!.split("\n").slice(-room()).join("\n")}
        </text>
      </Show>

      <Show when={result()}>
        {(done: () => FileDone) => (
          <box flexDirection="column">
            <Show
              when={!done().error}
              fallback={
                <text fg={theme.error} wrapMode="word">
                  {done().error!}
                </text>
              }
            >
              <text>
                <span style={{ fg: resultClassColor(done().result_class) }}>{done().result_class}</span>
                <span style={{ fg: theme.faint }}>{`  layer ${done().layer}`}</span>
              </text>
              <Show when={done().output}>
                <text fg={theme.muted} truncate>{shortPath(`Saved to ${outputDisplay(props.file, done().output!)}`, props.width)}</text>
              </Show>
              <box marginTop={1} flexDirection="column">
                <For each={done().lines.slice(0, 8)}>{(line) => <text fg={theme.text} truncate>{line}</text>}</For>
              </box>
              <Show when={diffLines().length}>
                <box marginTop={1} flexDirection="column">
                  <For each={diffLines().slice(0, Math.max(0, room() - done().lines.length - 4))}>
                    {(line) => <DiffLine line={line} color={diffColor(line)} />}
                  </For>
                </box>
              </Show>
            </Show>
          </box>
        )}
      </Show>

      <Show when={!result() && !stream()}>
        <Show
          when={finding()}
          fallback={<text fg={theme.faint}>Not inspected yet. Ctrl+E shows what is hidden.</text>}
        >
          {(found: () => Finding) => <FindingView finding={found()} room={room()} />}
        </Show>
      </Show>
    </box>
  )
}

/**
 * What inspection found, with the hidden characters lit up in the file's own
 * lines.  This is the screen the tool exists for, so it is the only place the
 * glow colour is used at any size.
 */
function FindingView(props: { finding: Finding; room: number }) {
  const hidden = () => props.finding.counts.hidden ?? 0
  const reveal = () => props.finding.reveal ?? []
  const excerpts = () => reveal().slice(0, Math.max(1, Math.floor((props.room - 4) / 2)))
  return (
    <box flexDirection="column">
      <Show when={props.finding.error}>
        <text fg={theme.error} wrapMode="word">
          {props.finding.error!}
        </text>
      </Show>
      <Show when={!props.finding.error}>
        <Show
          when={hidden() || props.finding.lines.length}
          fallback={<text fg={theme.success}>Nothing hidden found.</text>}
        >
          <Show when={hidden()}>
            <text fg={theme.glow}>{`${hidden()} hidden character${hidden() === 1 ? "" : "s"}`}</text>
          </Show>
          <Show when={excerpts().length}>
            <box flexDirection="column" marginTop={1}>
              <For each={excerpts()}>{(excerpt) => <RevealLine reveal={excerpt} />}</For>
            </box>
          </Show>
          <box flexDirection="column" marginTop={1}>
            <For each={props.finding.lines.slice(0, Math.max(0, props.room - 3 - excerpts().length * 2))}>
              {(line) => (
                <text fg={theme.muted} truncate>
                  {line}
                </text>
              )}
            </For>
          </box>
        </Show>
      </Show>
    </box>
  )
}

/** How the bridge writes an invisible character inside a diff. */
const ESCAPED = /(<U\+[0-9A-F]{4,6}>)/

/**
 * A diff line with its invisible characters drawn as a glowing ◆, the same
 * mark the reveal view uses, instead of the bridge's `<U+200B>` escape.
 */
function DiffLine(props: { line: string; color: string }) {
  const parts = () => props.line.split(ESCAPED).filter(Boolean)
  return (
    <text truncate wrapMode="none">
      <For each={parts()}>
        {(part) =>
          ESCAPED.test(part) ? (
            <span style={{ fg: theme.glow }}>
              <b>◆</b>
            </span>
          ) : (
            <span style={{ fg: props.color }}>{part}</span>
          )
        }
      </For>
    </text>
  )
}

const GUTTER = 6

/** One excerpt: the line, each ◆ glowing, and its name set under it. */
function RevealLine(props: { reveal: Reveal }) {
  const segments = () => {
    const out: { text: string; mark: boolean }[] = []
    let at = 0
    for (const [start, end] of props.reveal.marks) {
      if (start > at) out.push({ text: props.reveal.text.slice(at, start), mark: false })
      out.push({ text: props.reveal.text.slice(start, end), mark: true })
      at = end
    }
    if (at < props.reveal.text.length) out.push({ text: props.reveal.text.slice(at), mark: false })
    return out
  }
  // Names go under their marks while they fit; a name that would collide
  // with the one before it moves to the end of the row instead.
  const labels = () => {
    let row = ""
    const late: string[] = []
    for (const [start, , name] of props.reveal.marks) {
      if (start >= row.length + (row ? 1 : 0)) row = row.padEnd(start) + name
      else late.push(name)
    }
    return late.length ? `${row}  ${late.join(" ")}` : row
  }
  return (
    <box flexDirection="column">
      <box flexDirection="row" height={1}>
        <text width={GUTTER} fg={theme.faint}>
          {String(props.reveal.line).padStart(GUTTER - 2)}
        </text>
        <text truncate flexGrow={1}>
          <For each={segments()}>
            {(segment) =>
              segment.mark ? (
                <span style={{ fg: theme.glow }}>
                  <b>{segment.text}</b>
                </span>
              ) : (
                <span style={{ fg: theme.text }}>{segment.text}</span>
              )
            }
          </For>
        </text>
      </box>
      <box flexDirection="row" height={1}>
        <text width={GUTTER}> </text>
        <text truncate flexGrow={1} fg={theme.glow}>
          {labels()}
        </text>
      </box>
    </box>
  )
}

// -- prompt -------------------------------------------------------------------

function Prompt(props: {
  ctl: AppController
  focused: boolean
  onSubmit: (value: string) => void
  ref: (input: InputRenderable) => void
}) {
  const app = props.ctl.app
  let local: InputRenderable | undefined
  const preset = () => props.ctl.currentPreset()
  const needsModel = () => preset()?.requires_endpoint && !endpointLabel(app.state.endpoint)
  return (
    <box
      marginX={2}
      marginTop={1}
      paddingX={2}
      paddingY={1}
      flexDirection="column"
      backgroundColor={theme.panel}
      border={["left"]}
      borderStyle="heavy"
      borderColor={app.run ? theme.faint : theme.primary}
    >
      <box height={1}>
        <input
          ref={(r: InputRenderable) => {
            local = r
            props.ref(r)
          }}
          focused={props.focused}
          placeholder="Add a file or folder, a --flag, or a /command"
          onSubmit={() => props.onSubmit(local?.value ?? "")}
          backgroundColor={theme.panel}
          focusedBackgroundColor={theme.panel}
          textColor={theme.text}
          focusedTextColor={theme.text}
          placeholderColor={theme.faint}
          cursorColor={theme.primary}
        />
      </box>
      <box height={1} marginTop={1} flexDirection="row" justifyContent="space-between">
        <text truncate flexGrow={1}>
          <span style={{ fg: theme.primary }}>
            <b>{preset()?.label ?? "…"}</b>
          </span>
          <span style={{ fg: resultClassColor(app.plan?.result_class ?? preset()?.result_class ?? "") }}>
            {`  ${app.plan?.result_class ?? preset()?.result_class ?? ""}`}
          </span>
          <Show when={app.state.flags.length}>
            <span style={{ fg: theme.muted }}>{`  ${app.state.flags.join(" ")}`}</span>
          </Show>
        </text>
        <Show when={needsModel()}>
          <text fg={theme.warning}>Needs a model. Type /model</text>
        </Show>
      </box>
    </box>
  )
}

// -- footer -------------------------------------------------------------------

function Footer(props: { ctl: AppController; frame: number }) {
  const app = props.ctl.app
  const dims = useTerminalDimensions()
  const left = () => {
    if (app.run) {
      const verb = app.run.kind === "clean" ? "Cleaning" : "Inspecting"
      return {
        text: `${SPINNER[props.frame]} ${verb} ${Math.min(app.run.index + 1, app.run.total)} of ${app.run.total}: ${app.run.display}    esc stops`,
        color: theme.primary,
      }
    }
    if (app.toast) {
      const color = app.toast.tone === "error" ? theme.error : app.toast.tone === "success" ? theme.success : theme.text
      return { text: app.toast.text, color }
    }
    if (app.plan && !app.plan.ok) return { text: app.plan.error ?? "These options are not valid.", color: theme.error }
    if (app.plan?.preflight_error) return { text: app.plan.preflight_error, color: theme.error }
    if (app.plan?.warnings.length) return { text: app.plan.warnings[0]!, color: theme.warning }
    if (app.plan?.files.length) return { text: `$ ${app.plan.command}`, color: theme.faint }
    return { text: "", color: theme.faint }
  }
  const wide = () => dims().width >= 100
  return (
    <box height={1} flexDirection="row" justifyContent="space-between" paddingX={2} marginTop={1} marginBottom={0}>
      <text fg={left().color} truncate flexGrow={1}>
        {left().text}
      </text>
      <box paddingLeft={2}>
        <Show
          when={wide()}
          fallback={<Keys items={[["ctrl+r", "clean"], ["ctrl+p", "commands"]]} />}
        >
          <Keys items={[["ctrl+r", "clean"], ["ctrl+e", "inspect"], ["tab", "preset"], ["ctrl+p", "commands"]]} />
        </Show>
      </box>
    </box>
  )
}

// -- fatal --------------------------------------------------------------------

function Fatal(props: { ctl: AppController }) {
  return (
    <box flexGrow={1} flexDirection="column" paddingX={2} paddingTop={1}>
      <text fg={theme.error} attributes={TextAttributes.BOLD}>
        The Python side stopped.
      </text>
      <text fg={theme.muted} wrapMode="word">
        {props.ctl.app.fatal ?? ""}
      </text>
      <box marginTop={1}>
        <text fg={theme.faint} wrapMode="word">
          {`Details are in ${process.env.WM_TUI_LOG ?? "the log"}. ctrl+c quits.`}
        </text>
      </box>
    </box>
  )
}
