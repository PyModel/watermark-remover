# Development



## Optional demo

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo.py
```

The demo calls the same modules as the CLI, it is not a second implementation. Layer B prompt generation is limited to plain-text uploads, and binary document extraction is deliberately not guessed.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt
make check           # Python lint/format/tests/smoke + independent residual verifier
make tui-check       # frozen Bun install, TypeScript check + mounted UI tests
make web-check       # web UI typecheck + tests, rebuild, fail if committed dist/ is stale
make pages-check     # browser build (Pyodide) + engine parity with native Python
make packaging       # build/install wheel, verify all commands and read-only TUI boot
# make ci            # all of the above; requires Bun 1.4+
```

CI runs Python/workflow checks, a Linux/Windows Python matrix, Linux/macOS TUI tests and a dependency audit in parallel. Independent cleaning and installed-wheel gates depend on those fast checks. **CI gate** fails if any required job fails or is skipped. Weekly/manual `deps-latest.yml` tests the newest dependencies within declared ranges, without provider credentials or document/model calls.

Maintainer-pushed `v*` tags run the same full gates before Trusted Publishing. The wheel installed by the packaging test and its source distribution are promoted unchanged to PyPI, then attached to the GitHub release. No release-please or automatic version/tag creation. `WM_PACKAGING_DIST` exports verified artifacts into a **new** directory; ordinary tests keep their artifacts in temporary storage.
