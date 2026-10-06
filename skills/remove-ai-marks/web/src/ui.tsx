import type { ReactNode } from "react"
import { Icon, type IconName } from "./icons"

/** ds4's card: a title row, an optional icon or badge on the right, then rows. */
export function Card({
  title,
  icon,
  badge,
  actions,
  className = "",
  children,
  labelledBy,
}: {
  title: ReactNode
  icon?: IconName
  badge?: ReactNode
  actions?: ReactNode
  className?: string
  children: ReactNode
  labelledBy?: string
}) {
  return (
    <section className={`card ${className}`} aria-labelledby={labelledBy}>
      <div className="card-head">
        <h2 id={labelledBy} className="card-title">
          {icon && (
            <span className="title-icon">
              <Icon name={icon} />
            </span>
          )}
          {title}
        </h2>
        <div className="card-head-right">
          {badge}
          {actions}
        </div>
      </div>
      {children}
    </section>
  )
}

export function StatRow({ icon, label, children }: { icon?: IconName; label: ReactNode; children: ReactNode }) {
  return (
    <div className="stat-row">
      <span className="stat-label">
        {icon && <Icon name={icon} size={15} />}
        {label}
      </span>
      <span className="stat-value">{children}</span>
    </div>
  )
}

export type Tone = "ok" | "off" | "warn" | "bad" | "mark" | "info"

/** Status chip: the text carries the meaning, the color only repeats it. */
export function Chip({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`chip chip-${tone}`}>{children}</span>
}

export function Check({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`check-dot ${ok ? "is-ok" : "is-off"}`} role="img" aria-label={label}>
      <Icon name={ok ? "check" : "x"} size={12} />
    </span>
  )
}

export function Progress({ value, label, tone = "blue" }: { value: number; label: string; tone?: "blue" | "green" | "mark" }) {
  const clamped = Math.max(0, Math.min(100, value))
  return (
    <div className={`progress progress-${tone}`} role="progressbar" aria-label={label} aria-valuenow={Math.round(clamped)} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-fill" style={{ width: `${clamped}%` }} />
    </div>
  )
}

export function MiniStat({ label, value, align }: { label: string; value: ReactNode; align?: "right" }) {
  return (
    <div className={`mini-stat ${align === "right" ? "align-right" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { id: T; label: string; disabled?: boolean }[]
  onChange: (id: T) => void
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          aria-pressed={value === option.id}
          disabled={option.disabled}
          onClick={() => onChange(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

/* ---------- copy helpers ---------- */

export function sentence(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1)
}

export function plural(count: number, word: string): string {
  return count === 1 ? word : `${word}s`
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** "U+200B ZERO WIDTH SPACE (Cf)" -> "Zero width space" */
export function unicodeName(label: string): string {
  const name = label.replace(/^U\+[0-9A-F]+\s*/i, "").replace(/\s*\([A-Za-z]{2}\)$/, "")
  return sentence(name.toLowerCase())
}

export function smooth(): ScrollBehavior {
  return matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"
}
