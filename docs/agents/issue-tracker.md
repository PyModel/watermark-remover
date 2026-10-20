# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in the to-auto control plane, **not** in this repo: `<state>/tracker/<feature-slug>/`, where `<state>` is the path `goal.mjs control` prints (`~/.local/state/to-auto/<repo-id>` by default). The control plane is its own private git repo, committed after every write, and is never merged into this repo.

## Conventions

- One feature per directory: `<state>/tracker/<feature-slug>/`
- The spec is `<state>/tracker/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `<state>/tracker/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- The only status line the frontier parser accepts, exactly:
  `**Status:** ready-for-agent | in-flight | done | stuck | blocked | needs-human`
- Blocking: `**Blocked by:** 01, 03` or `**Blocked by:** None` (a parenthetical or a title after a number is ignored)
- Claims: `**Claims:** exclusive: a/b.ts, c/ ; shared-regenerate: pnpm-lock.yaml ; guarded: .github/**`; `**Claims amended:** +<paths> (<kind>): <why>` lines are written only by `goal.mjs claim`
- Optional capability line: `**Capability:** advanced/high` (tier `lightweight | standard | advanced`, intensity `low | medium | high`; absent means `standard/medium`)
- Triage state strings come from `triage-labels.md`
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `<state>/tracker/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced full path. Never resolve a bare `#3`: always a full path or full reference.
