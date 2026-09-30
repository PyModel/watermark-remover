# Agent skills

This repo gitignores `CLAUDE.md` and `AGENTS.md` (`.gitignore:19-20`): each checkout keeps local copies. The canonical `## Agent skills` block therefore lives in this tracked file; local copies may mirror it.

## Agent skills

### Issue tracker

Local markdown in the to-auto control plane at `<state>/tracker/<feature-slug>/` (the path `goal.mjs control` prints), never inside this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical role strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) used as-is. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
