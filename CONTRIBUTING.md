# Contributing to watermark-remover

Thanks for helping keep the skill accurate and the cleaners reliable. The
project is a small Python skill (`skills/remove-ai-marks/`) plus tests —
focused PRs land fastest.

## Who can do what

| Action | Who |
| --- | --- |
| Open issues | Anyone |
| Suggest a release | Anyone (use the **Release suggestion** issue template) |
| Open pull requests | Anyone (fork the repo) |
| Approve and merge pull requests | Maintainer only (`@elkaix`) |

`main` is protected. A change needs a pull request, a passing **CI** check
(`test`), and an approving review from the code owner before merge. Only the
maintainer can give that approval. Direct pushes to `main` are blocked for
non-admins.

To suggest a release without a code change: open a **Release suggestion**
issue.

## Prerequisites

- **Python 3.10+** (stdlib only for the skill scripts; optional rewrite backends
  use HTTP to local Ollama / OpenAI-compatible endpoints)
- Install test extras: `python3 -m pip install -r requirements-test.txt`
- From the repo root: `make check` must pass before you open a PR
- Optional for manual checks: [`c2patool`](https://github.com/contentauth/c2pa-rs/tree/main/cli), [`exiftool`](https://exiftool.org/), and external visible-mark adapters

## Layout

| Path | Role |
| --- | --- |
| `skills/remove-ai-marks/SKILL.md` | Agent skill entry (workflow, ethics) |
| `skills/remove-ai-marks/scripts/` | Layer A/B hooks + image/container cleaners |
| `skills/remove-ai-marks/references/` | Vendors, mark classes, matrix, ethics |
| `DESIGN.md` | Module seams, invariants, guarantee classes, roadmap |
| `tests/` | Pytest suite and fixtures |
| `.github/workflows/ci.yml` | CI job `test` |

## Layers (what to change where)

1. **Layer A (Unicode / format controls)** — deterministic scripts under
   `scripts/` (`text_unicode.py`, `clean_text.py`, `inspect_text.py`). Prefer
   tests with fixtures in `tests/fixtures/`.
2. **Layer B (statistical rewrite)** — `rewrite_text.py` is the backend seam;
   `tsapa.py` owns evolutionary optimization. No bundled model.
3. **Layer V (visible images)** — `morphomod.py` owns masks, dilation, PNG
   restore, and external detector/inpainter adapters. Never guess a region.
4. **Layer M (C2PA / EXIF / XMP / props)** — `image_meta.py` delegates HEIF/
   AVIF to `heif_meta.py`; `container_meta.py` handles SVG/PDF/DOCX/ODT/HTML/MD.
5. **Soft-binding risk** — `inspect_soft_binding.py` detects and warns; removal
   remains out of scope.
6. **Batch orchestration** — share discovery through `batch_inputs.py`; do not
   duplicate traversal/output-path logic in CLI modules.

## Checklist for a change

- [ ] Behaviour matches `SKILL.md` / `references/removal-matrix.md` when relevant
- [ ] Unit tests updated or added under `tests/`
- [ ] `make check` passes (compile + tests + CLI smoke)
- [ ] Docs updated (README and/or skill references) if user-facing behaviour
      changes
- [ ] No drive-by refactors unrelated to the fix or feature

## PR expectations

- Stay focused and match existing style (stdlib-first scripts, clear CLI flags)
- Do not commit secrets, private user files, or large binary fixtures unless
  needed and redacted
- Respect `references/ethics.md`: this tool is for content the user owns

Questions? Open an issue describing the input type (text / image / document)
and which layer failed or is missing.

## Community

- [Code of Conduct](CODE_OF_CONDUCT.md) — expected behaviour in the project
- [Security policy](SECURITY.md) — how to report vulnerabilities privately
- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md) and
  [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates
