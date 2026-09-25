import { RGBA, TextAttributes } from "@opentui/core"
import { useKeyboard, useTerminalDimensions } from "@opentui/solid"
import { createEffect, createMemo, createSignal, For, type JSX, on, Show } from "solid-js"
import { fuzzyScore } from "../prompt"
import { theme } from "../theme"

const SCRIM = RGBA.fromInts(0, 0, 0, 160)

/**
 * A floating panel over a dimmed screen.  No border: the panel is one step
 * lighter than the page, which separates it without spending a column on
 * each side.
 */
export function DialogFrame(props: {
  title: string
  width?: number
  children: JSX.Element
  /** [current, total] when the dialog is one step of a sequence. */
  steps?: [number, number]
}) {
  const dims = useTerminalDimensions()
  const width = () => Math.max(30, Math.min(props.width ?? 64, dims().width - 4))
  return (
    <box
      position="absolute"
      left={0}
      top={0}
      width={dims().width}
      height={dims().height}
      zIndex={20}
      backgroundColor={SCRIM}
      alignItems="center"
      paddingTop={Math.max(1, Math.floor(dims().height / 6))}
    >
      <box
        width={width()}
        maxHeight={dims().height - 2}
        backgroundColor={theme.panel}
        flexDirection="column"
        paddingX={2}
        paddingY={1}
      >
        <box flexDirection="row" justifyContent="space-between" height={1} marginBottom={1}>
          <text fg={theme.text} attributes={TextAttributes.BOLD}>
            {props.title}
          </text>
          <text>
            <Show when={props.steps}>
              {(steps: () => [number, number]) => (
                <span style={{ fg: theme.primary }}>
                  {"●".repeat(steps()[0]) + "○".repeat(steps()[1] - steps()[0]) + "   "}
                </span>
              )}
            </Show>
            <span style={{ fg: theme.faint }}>esc</span>
          </text>
        </box>
        {props.children}
      </box>
    </box>
  )
}

export interface ListItem {
  id: string
  title: string
  /** Shown under the highlighted row only, so the list stays one line per item. */
  detail?: string
  right?: string
  rightColor?: string
  titleColor?: string
  disabled?: boolean
}

/**
 * The one list every dialog uses: optional type-to-filter, arrows or
 * ctrl+n/p to move, enter to choose, esc to close.  The highlighted row is a
 * solid accent bar, the way opencode marks the current choice.
 */
export function ListDialog(props: {
  title: string
  steps?: [number, number]
  items: ListItem[]
  active: boolean
  filter?: boolean
  placeholder?: string
  initial?: string
  width?: number
  footer?: string
  empty?: string
  onSelect: (item: ListItem) => void
  onClose: () => void
}) {
  const dims = useTerminalDimensions()
  const [query, setQuery] = createSignal("")
  const [index, setIndex] = createSignal(0)

  const visible = createMemo(() => {
    const q = query().trim()
    if (!q) return props.items
    return props.items
      .map((item) => ({ item, score: fuzzyScore(q, item.title) }))
      .filter((entry) => entry.score !== null)
      .sort((a, b) => b.score! - a.score!)
      .map((entry) => entry.item)
  })

  createEffect(
    on(
      () => props.items,
      () => {
        const at = props.initial ? props.items.findIndex((item) => item.id === props.initial) : -1
        setIndex(Math.max(0, at))
      },
    ),
  )
  createEffect(on(query, () => setIndex(0), { defer: true }))

  const rows = () => Math.max(3, Math.min(12, dims().height - 12))
  const offset = () => {
    const i = index()
    return Math.max(0, Math.min(i - Math.floor(rows() / 2), visible().length - rows()))
  }
  const current = () => visible()[index()]

  const move = (step: number) => {
    const count = visible().length
    if (!count) return
    setIndex((i) => (i + step + count) % count)
  }

  useKeyboard((key) => {
    if (!props.active) return
    if (key.name === "escape") {
      key.preventDefault()
      props.onClose()
    } else if (key.name === "up" || (key.ctrl && key.name === "p")) {
      key.preventDefault()
      move(-1)
    } else if (key.name === "down" || (key.ctrl && key.name === "n")) {
      key.preventDefault()
      move(1)
    } else if (key.name === "pageup") {
      move(-rows())
    } else if (key.name === "pagedown") {
      move(rows())
    } else if (key.name === "return") {
      key.preventDefault()
      const item = current()
      if (item && !item.disabled) props.onSelect(item)
    }
  })

  return (
    <DialogFrame title={props.title} steps={props.steps} width={props.width}>
      <Show when={props.filter}>
        <box height={1} marginBottom={1}>
          <input
            focused={props.active}
            placeholder={props.placeholder ?? "Type to filter"}
            onInput={setQuery}
            backgroundColor={theme.panel}
            focusedBackgroundColor={theme.panel}
            textColor={theme.text}
            focusedTextColor={theme.text}
            placeholderColor={theme.faint}
            cursorColor={theme.primary}
          />
        </box>
      </Show>
      <Show when={visible().length} fallback={<text fg={theme.muted}>{props.empty ?? "Nothing matches."}</text>}>
        <box flexDirection="column">
          <For each={visible().slice(offset(), offset() + rows())}>
            {(item) => {
              const selected = () => current()?.id === item.id
              return (
                <box
                  flexDirection="row"
                  height={1}
                  paddingX={1}
                  backgroundColor={selected() ? theme.primary : theme.panel}
                  onMouseDown={() => {
                    const at = visible().findIndex((entry) => entry.id === item.id)
                    if (at >= 0) setIndex(at)
                    if (!item.disabled) props.onSelect(item)
                  }}
                >
                  <text
                    flexGrow={1}
                    truncate
                    wrapMode="none"
                    fg={selected() ? theme.bg : item.disabled ? theme.faint : (item.titleColor ?? theme.text)}
                    attributes={selected() ? TextAttributes.BOLD : TextAttributes.NONE}
                  >
                    {item.title}
                  </text>
                  <Show when={item.right}>
                    <text marginLeft={2} flexShrink={0} fg={selected() ? theme.bg : (item.rightColor ?? theme.muted)}>
                      {item.right}
                    </text>
                  </Show>
                </box>
              )
            }}
          </For>
        </box>
      </Show>
      <Show when={current()?.detail}>
        <box marginTop={1}>
          <text fg={theme.muted} wrapMode="word">
            {current()!.detail}
          </text>
        </box>
      </Show>
      <Show when={props.footer}>
        <box marginTop={1}>
          <text fg={theme.faint} wrapMode="word">
            {props.footer}
          </text>
        </box>
      </Show>
    </DialogFrame>
  )
}

/** Key hints in the house style: the key bright, the action dim. */
export function Keys(props: { items: [string, string][] }) {
  return (
    <text>
      <For each={props.items}>
        {([key, label], i) => (
          <>
            <span style={{ fg: theme.text }}>{key}</span>
            <span style={{ fg: theme.muted }}>{` ${label}${i() < props.items.length - 1 ? "   " : ""}`}</span>
          </>
        )}
      </For>
    </text>
  )
}
