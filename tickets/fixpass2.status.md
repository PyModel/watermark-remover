# Stage 8 fix pass 2 — status

- Branch: `goal/execute-phases-3-4-of-tasks-porting-plan`
- Head: `d1384cde1b8a081afb94206248a147aa987ba616`
- Scope: findings F1–F11 from the stage-8 review; one commit per finding (F3+F7 and F8+F9+F10 merged — adjacent edits to the same report dict / action loop).

## Findings

| Finding | Result | Commit |
| --- | --- | --- |
| F1 mark+auto-tile crash | fixed: mark plans never tile (report says `skipped-for-mark`); dest-less mark call raises labeled `ValueError`, never an assert | 6977775 |
| F2 migan/lama extras | fixed: extras declared (onnxruntime + huggingface-hub), KNOWN_EXTRAS + extras_status/capabilities surface them, uv.lock regenerated; `pip install --dry-run -e .[migan]` resolves (exit 0) | e0d7171 |
| F3 external+mark honesty | fixed: report `backend` names the backend that actually filled; labeled warning when external cannot apply | c8150f3 |
| F4 weight revisions | fixed: LaMa `c3c0c9e4…`, MI-GAN `406830d0…` pinned in `_get_session`; recorded in PROVENANCE.md (e86e668 reformat) | d5fc250 |
| F5 provenance dead config | fixed (option a): `metadata_evidence()` applies each row's declared signals/tokens to file metadata; merges with TC260 keys in the mark pass; still-declarative fields relabeled | 29b8bac |
| F6 docs | fixed: README Layer V row + "Automatic mark detection" section (factual) | 896df79 |
| F7 dilate honesty | fixed: mark report states `dilation: engine-controlled (0 user-applied)` | c8150f3 |
| F8 no-mark aliasing | fixed: auto pass returns a copy when nothing fires | 4f32fbe |
| F9 per-mark containment | fixed: `MarkError` rows, per-mark localize/fill try/except, aggregate `error` status | 4f32fbe |
| F10 duplicate order stacks | fixed: registry preference derived from `inpaint_backends.FILL_RESOLVE_ORDER` via `_FILL_STACK_TIERS` | 4f32fbe |
| F11 pill wrapper dupes | fixed: adapters derive from the shared wrapper helpers; behavior pinned unchanged | ba99fbd |

## Validation

- Full suite: `2012 passed, 3 skipped` (no previously-passing test fails).
- `node goal.mjs run -- make check`: exit 0 at final head `88a2f7663d14…`
  log `/Users/panda/Projects/active/watermark-remover/.git/goal-logs/88a2f7663d14-1791225193023-49102.log`
  sha256 `8c12cb662622ba6c354aa51daab37b7551f099702ae97b9517f76f32448b8da3`
- Worktree clean; no push, no PR, no tracker edits.

## Blockers

None.
