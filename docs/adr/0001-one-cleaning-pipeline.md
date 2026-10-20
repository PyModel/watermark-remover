# One canonical cleaning pipeline; CLI, TUI, and HTTP are adapters only

The HTTP service carried a second, parallel cleaning path: it validated options
against its own allow-list and called `clean_text` / `clean_image` /
`clean_container` directly, so its guarantees silently diverged from the CLIs
even though the docs claimed parity. We decided there is exactly one semantic
cleaning pipeline: every surface normalizes its input into a `CleanRequest`,
crosses `plan_work`, and executes through the single execution seam
(`clean_pipeline.execute`); no surface imports a concrete cleaner, and only the
execution layer does. The trade-off is adapter freedom: an HTTP-only flag or a
TUI shortcut cannot exist as such — a genuinely semantic option must become a
`CleanRequest` field or it does not ship. This is hard to reverse (public CLI,
TUI, and HTTP contracts) and surprising without context (a future contributor
will wonder why the server may not just call `clean_image`), so it is recorded
here and enforced by an architecture import-graph test rather than by
convention.

## Considered options

- Keep per-surface validation and let HTTP call cleaners directly — rejected:
  that is the divergence this ADR exists to kill.
- A backend registry with entry-point discovery for pixel backends — rejected:
  the external adapters are opt-in checkouts, not installed distributions;
  a small table in the execution layer covers the real capability-result need.
