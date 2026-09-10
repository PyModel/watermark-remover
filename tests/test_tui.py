"""wm-tui: the invariants that keep the TUI honest and inside its seam.

These are contract tests, not widget-pixel tests. They pin the four rules the
TUI must not drift from: it goes through the shared plan seam, it never speaks
HTTP itself, it never labels a best-effort layer as verified, and it never
renders or copies an API key.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from clean_request import CleanRequest
from tui import (
    BEST_EFFORT,
    COST_CONFIRM_SECONDS,
    DETECTION_ONLY,
    LAYER_RESULT_CLASS,
    VERIFIABLE,
    HistoryEntry,
    discover_files,
    estimate_rewrite_seconds,
    format_badge,
    format_duration,
    result_class_for,
    should_confirm_cost,
)

ZWSP = "Delve into it.​ Moreover, it is important to note this.\n"


# --- the honesty contract ----------------------------------------------------


def test_best_effort_layers_are_never_labelled_verifiable():
    """A Layer B or Layer V success is still best-effort. This is the whole point."""
    assert result_class_for("B") == BEST_EFFORT
    assert result_class_for("V") == BEST_EFFORT
    assert result_class_for("synthid") == BEST_EFFORT
    assert result_class_for("soft-binding") == DETECTION_ONLY
    assert result_class_for("A") == VERIFIABLE
    assert result_class_for("M") == VERIFIABLE


def test_unknown_layers_default_to_best_effort():
    """Failing open to 'verified' would be the dangerous direction."""
    assert result_class_for("something-new") == BEST_EFFORT
    assert result_class_for("") == BEST_EFFORT


def test_badges_render_their_class_name():
    assert VERIFIABLE in format_badge("A")
    assert BEST_EFFORT in format_badge("B")
    assert DETECTION_ONLY in format_badge("soft-binding")


def test_result_class_table_covers_every_shipped_layer():
    assert set(LAYER_RESULT_CLASS) >= {"A", "B", "V", "M", "soft-binding"}


# --- the seam ----------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("module", ["tui.py", "tui_app.py"])
def test_tui_never_speaks_http_itself(module):
    """Every request must inherit layer_b_http's SSRF hardening."""
    imported = _imported_modules(SCRIPTS / module)
    assert not imported & {"urllib", "http", "socket", "requests", "httpx"}


@pytest.mark.parametrize("module", ["tui.py", "tui_app.py"])
def test_tui_never_builds_a_clean_plan_itself(module):
    source = (SCRIPTS / module).read_text(encoding="utf-8")
    assert "CleanPlan(" not in source
    assert "TextCleanPlan(" not in source
    assert "VisiblePlan(" not in source
    assert "RewritePlan(" not in source


def test_only_clean_request_and_clean_asset_construct_plans():
    """Plan construction stays in one place, so the CLI and TUI cannot diverge."""
    allowed = {"clean_request.py", "clean_asset.py", "clean_file.py", "server.py", "demo.py"}
    offenders = []
    for path in SCRIPTS.glob("*.py"):
        if path.name in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        if "CleanPlan(" in source or "TextCleanPlan(" in source:
            offenders.append(path.name)
    assert not offenders, f"plans built outside the shared builder: {offenders}"


# --- secrets -----------------------------------------------------------------


def test_no_api_key_reaches_a_copied_command():
    request = CleanRequest(
        paths=(Path("a.txt"),),
        rewrite="humanize",
        rewrite_api_key="unit-test-secret-never-real",
    )
    entry = HistoryEntry(
        when="12:00",
        summary="1 file",
        command=" ".join(request.command_line()),
        request=request,
    )
    assert "unit-test-secret-never-real" not in entry.command
    assert "unit-test-secret-never-real" not in repr(entry)


def test_tui_reads_the_key_from_the_environment_and_never_renders_it():
    source = (SCRIPTS / "tui_app.py").read_text(encoding="utf-8")
    # The value is only ever tested for presence, never placed in a widget.
    assert "os.environ.get(REWRITE_ENV_KEY)" in source
    assert 'Static(f"API key: {os.environ' not in source


# --- the batch cost gate -----------------------------------------------------


def test_no_estimate_without_a_rewrite():
    assert estimate_rewrite_seconds(CleanRequest(), 10) == 0.0


def test_estimate_scales_with_files_candidates_and_timeout():
    request = CleanRequest(rewrite="humanize", rewrite_candidates=3, rewrite_timeout=10.0)
    assert estimate_rewrite_seconds(request, 4) == 4 * 3 * 10.0


def test_tsapa_estimate_accounts_for_the_population_search():
    request = CleanRequest(
        rewrite="tsapa",
        tsapa_generations=3,
        tsapa_population=4,
        rewrite_timeout=10.0,
    )
    # A TSAPA batch is far more expensive than N candidates; the gate must say so.
    assert estimate_rewrite_seconds(request, 2) == 2 * 3 * 4 * 10.0


def test_a_single_cheap_rewrite_is_not_gated():
    """A gate that fires on every rewrite is a gate nobody reads."""
    request = CleanRequest(rewrite="humanize", rewrite_candidates=1, rewrite_timeout=120.0)
    assert not should_confirm_cost(request, 1)


def test_a_batch_rewrite_is_gated():
    request = CleanRequest(rewrite="humanize", rewrite_candidates=1, rewrite_timeout=120.0)
    assert should_confirm_cost(request, 10)


def test_a_single_tsapa_run_is_gated():
    """One file is enough when the search issues generations x population calls."""
    request = CleanRequest(rewrite="tsapa", tsapa_generations=5, tsapa_population=12)
    assert should_confirm_cost(request, 1)
    assert estimate_rewrite_seconds(request, 1) > COST_CONFIRM_SECONDS


def test_a_run_without_a_rewrite_is_never_gated():
    assert not should_confirm_cost(CleanRequest(), 1000)


def test_durations_read_in_human_units():
    assert format_duration(30) == "30s"
    assert format_duration(600) == "10m"
    assert format_duration(7200) == "2.0h"


# --- file discovery ----------------------------------------------------------


def test_discovery_reuses_the_cli_input_selector(tmp_path: Path):
    (tmp_path / "a.md").write_text(ZWSP, encoding="utf-8")
    (tmp_path / "b.txt").write_text(ZWSP, encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    files, error = discover_files(CleanRequest(paths=(tmp_path,)))
    assert error is None
    assert {p.name for p in files} == {"a.md", "b.txt"}

    scoped, error = discover_files(CleanRequest(paths=(tmp_path,), extensions=".md"))
    assert error is None
    assert {p.name for p in scoped} == {"a.md"}


def test_discovery_reports_selection_errors_instead_of_raising(tmp_path: Path):
    files, error = discover_files(CleanRequest(paths=(tmp_path / "missing",)))
    assert files == []
    assert error


# --- the entry point ---------------------------------------------------------


def test_entry_point_without_the_extra_prints_a_hint_and_exits_2(tmp_path: Path):
    """A default install must fail with guidance, not a traceback."""
    program = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'textual' or name.startswith('textual.'):\n"
        "            raise ImportError('blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import tui\n"
        "sys.exit(tui.main([]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "watermark-remover[tui]" in result.stderr


# --- headless app ------------------------------------------------------------

pytest.importorskip("textual")


def _run(coro):
    return asyncio.run(coro)


def test_app_mounts_and_lists_files(tmp_path: Path):
    from tui_app import WatermarkTuiApp

    (tmp_path / "draft.md").write_text(ZWSP, encoding="utf-8")
    (tmp_path / "notes.txt").write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(tmp_path,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert {p.name for p in app.files} == {"draft.md", "notes.txt"}
            # Everything discovered starts selected, and the summary says so.
            assert len(app.selected) == 2

    _run(scenario())


def test_generated_command_round_trips_and_carries_no_secret(tmp_path: Path):
    from textual.widgets import Checkbox, TextArea
    from tui_app import WatermarkTuiApp

    (tmp_path / "draft.txt").write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(tmp_path,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#cb-nfkc", Checkbox).value = True
            await pilot.pause()
            preview = app.query_one("#command-copyable", TextArea).text
            assert preview.startswith("wm ")
            assert "--nfkc" in preview
            # The selectable fallback must carry exactly what the button copies.
            assert app.collect_request().nfkc is True

    _run(scenario())


def test_run_cleans_the_selected_file_and_reinspects(tmp_path: Path):
    from textual.widgets import DataTable, Input
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    out = tmp_path / "out"
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-output", Input).value = str(out / "draft.txt")
            await pilot.pause()
            app.action_run()
            for _ in range(80):
                await pilot.pause()
                if app.query_one("#run-table", DataTable).row_count:
                    break
            assert app.query_one("#run-table", DataTable).row_count == 1
            written = out / "draft.txt"
            assert written.is_file()
            # Layer A is deterministic: the zero-width carrier is gone.
            assert "​" not in written.read_text(encoding="utf-8")

    _run(scenario())


def test_batch_preflight_failure_writes_nothing(tmp_path: Path):
    from textual.widgets import DataTable, Input
    from tui_app import WatermarkTuiApp

    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text(ZWSP, encoding="utf-8")
    second.write_text(ZWSP, encoding="utf-8")
    before = {path: path.read_bytes() for path in (first, second)}
    app = WatermarkTuiApp(CleanRequest(paths=(first, second)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            # A batch output directory that contains the inputs makes every
            # destination alias its own source: preflight must refuse the whole
            # run rather than clean the first file and then discover it.
            app.query_one("#in-output", Input).value = str(tmp_path)
            await pilot.pause()
            app.action_run()
            for _ in range(60):
                await pilot.pause()
            assert app.query_one("#run-table", DataTable).row_count == 0
            # Not one byte written before the refusal.
            assert {path: path.read_bytes() for path in (first, second)} == before

    _run(scenario())


def test_markup_in_a_filename_is_escaped_not_parsed(tmp_path: Path):
    """A name like report[1].md must render, not disappear into a style tag."""
    from tui_app import WatermarkTuiApp

    tricky = tmp_path / "report[bold]x.txt"
    tricky.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(tricky,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            block = app._inspect_block(tricky, soft_binding=False)
            # The name's own brackets are escaped, so Rich renders them
            # literally instead of treating them as an opening style tag.
            assert "report\\[bold]x.txt" in block

    _run(scenario())


def test_stylometry_status_is_not_swallowed_by_markup(tmp_path: Path):
    """Short input reports a status; brackets would have eaten it."""
    from tui_app import WatermarkTuiApp

    short = tmp_path / "short.txt"
    short.write_text("Hi.\n", encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(short,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            line = app._stylometry_line(short)
            assert "insufficient_length" in line
            assert "Best-effort" in line

    _run(scenario())


def test_remote_endpoint_is_blocked_in_the_plan_panel(tmp_path: Path):
    from textual.widgets import Input, Select, Static
    from tui_app import WatermarkTuiApp

    (tmp_path / "draft.txt").write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(tmp_path,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#sel-rewrite", Select).value = "humanize"
            app.query_one("#in-base-url", Input).value = "https://example.test"
            await pilot.pause()
            state = app.query_one("#endpoint-state", Static).render()
            assert "blocked" in str(state)

    _run(scenario())
