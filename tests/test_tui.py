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

#: Fields whose request value is tri-state but whose Plan control is not.
#: The CLI has no way to say "no" to a store_true flag, so "not stated" has to
#: mean "ask the environment"; the TUI has a visible checkbox, so unchecked is
#: a stated no. Reuse therefore turns a None into an explicit False, and the
#: endpoint policy the pane displays stays the one the run will apply.
TRI_STATE_BINDINGS = {"rewrite_allow_remote"}


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
            # The box is refreshed by Checkbox.Changed, so wait for the event
            # rather than assume one pump cycle delivered it -- a single pause
            # is enough on a fast machine and not on a loaded CI runner.
            preview = ""
            for _ in range(20):
                await pilot.pause()
                preview = app.query_one("#command-copyable", TextArea).text
                if "--nfkc" in preview:
                    break
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


def _tiny_png() -> bytes:
    import struct
    import zlib

    def chunk(ctype: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(payload, zlib.crc32(ctype)) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


def test_dry_run_describes_the_visible_clean_and_writes_nothing(tmp_path: Path):
    """The CLI's --dry-run has a TUI surface, and it is the CLI's own preview."""
    from textual.widgets import Checkbox, DataTable, Input, RichLog, TabbedContent, TextArea
    from tui_app import WatermarkTuiApp

    source = tmp_path / "shot.png"
    source.write_bytes(_tiny_png())
    destination = tmp_path / "shot.cleaned.png"
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-output", Input).value = str(destination)
            app.query_one("#in-box", Input).value = "0,0,1,1"
            app.query_one("#cb-dry-run", Checkbox).value = True
            await pilot.pause()
            assert "--dry-run" in app.query_one("#command-copyable", TextArea).text

            # RichLog defers writes until it has been laid out, so the Run pane
            # has to be the visible tab for the log assertions below to mean
            # anything.
            app.query_one(TabbedContent).active = "tab-run"
            await pilot.pause()
            app.action_run()
            for _ in range(200):
                await pilot.pause()
                if app.query_one("#run-table", DataTable).row_count:
                    break
            assert app.query_one("#run-table", DataTable).row_count == 1
            # The whole point: a preview leaves the filesystem alone.
            assert not destination.exists()
            assert not any(tmp_path.glob("*.mask.pgm"))
            log = app.query_one("#run-log", RichLog)
            rendered = "".join(strip.text for strip in log.lines)
            assert "inpaint" in rendered

    _run(scenario())


def test_dry_run_is_refused_on_a_text_asset(tmp_path: Path):
    """--dry-run is image-only in the CLI; the TUI inherits the refusal."""
    from textual.widgets import Checkbox, Input, RichLog, TabbedContent
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-output", Input).value = str(tmp_path / "out.txt")
            app.query_one("#cb-dry-run", Checkbox).value = True
            app.query_one(TabbedContent).active = "tab-run"
            await pilot.pause()
            app.action_run()
            for _ in range(60):
                await pilot.pause()
            log = app.query_one("#run-log", RichLog)
            rendered = "".join(strip.text for strip in log.lines)
            assert "only valid for image assets" in rendered
            assert not (tmp_path / "out.txt").exists()

    _run(scenario())


def test_a_bracketed_filename_survives_the_run_log(tmp_path: Path):
    """Operator-controlled strings reach Rich, which treats [x] as a style tag.

    ``notes[bold]x.md`` rendering as ``notesx.md`` is not cosmetic: the log is
    the record of which file was touched.
    """
    from textual.widgets import Checkbox, Input, RichLog, TabbedContent
    from tui_app import WatermarkTuiApp

    source = tmp_path / "notes[bold]x.md"
    source.write_text("---\nai_generated: true\n---\nhi" + ZWSP + "\n", encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-output", Input).value = str(tmp_path / "out.md")
            # A text-body flag on a container asset triggers the skipped-
            # transform note, which is the line that carries the file name.
            app.query_one("#cb-nfkc", Checkbox).value = True
            app.query_one(TabbedContent).active = "tab-run"
            await pilot.pause()
            app.action_run()
            for _ in range(200):
                await pilot.pause()
                if app.history:
                    break
            for _ in range(10):
                await pilot.pause()
            rendered = "".join(strip.text for strip in app.query_one("#run-log", RichLog).lines)
            assert "notes[bold]x.md" in rendered
            assert "notesx.md" not in rendered

    _run(scenario())


def test_the_copyable_command_is_shell_safe(tmp_path: Path):
    import shlex

    from textual.widgets import TextArea
    from tui_app import WatermarkTuiApp

    source = tmp_path / "report[1] draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            command = app.query_one("#command-copyable", TextArea).text
            assert shlex.split(command) == app.collect_request().command_line()

    _run(scenario())


def test_the_backends_pane_is_current_when_you_look_at_it(tmp_path: Path):
    """Its Layer B rows describe the current plan, so a mount-time snapshot lies."""
    from textual.widgets import DataTable, Input, TabbedContent
    from tui_app import WatermarkTuiApp

    (tmp_path / "draft.txt").write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(tmp_path,)))

    def endpoint_row(table: DataTable) -> tuple:
        for row in table.rows:
            cells = table.get_row(row)
            if cells[0] == "layer B endpoint":
                return cells
        raise AssertionError("no layer B endpoint row")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#backend-table", DataTable)
            assert endpoint_row(table)[1] == "blocked"

            app.query_one("#in-base-url", Input).value = "http://127.0.0.1:11434"
            app.query_one(TabbedContent).active = "tab-backends"
            for _ in range(6):
                await pilot.pause()
            assert endpoint_row(table)[1] == "allowed"

    _run(scenario())


def test_a_probe_result_survives_leaving_and_returning_to_the_pane(tmp_path: Path):
    """Rebuilding the table must not throw away the probe the operator just ran."""
    from textual.widgets import DataTable, TabbedContent
    from tui_app import WatermarkTuiApp

    (tmp_path / "draft.txt").write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(tmp_path,)))

    class _Probe:
        backend = "ollama"
        reachable = True
        summary = "reachable, 2 model(s)"
        models = ()

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app._apply_probe(_Probe())
            table = app.query_one("#backend-table", DataTable)
            app.query_one(TabbedContent).active = "tab-plan"
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab-backends"
            for _ in range(6):
                await pilot.pause()
            labels = [table.get_row(row)[0] for row in table.rows]
            assert "probe: ollama" in labels

    _run(scenario())


def test_every_clean_request_field_is_bound_or_documented_as_absent():
    """The CLI's flag surface must be reachable from the TUI, or say why not.

    Adding a ``CleanRequest`` field without a Plan control silently makes the
    TUI a weaker front end than the CLI over the same seam. This is the test
    that turned up ``--as``/``--force-text``, ``--audit``, ``--dilate``,
    ``--degrade`` and the rest having no interactive surface at all.
    """
    import dataclasses

    from clean_request import CleanRequest
    from tui_app import PLAN_BINDINGS, UNBOUND_REQUEST_FIELDS

    fields = {f.name for f in dataclasses.fields(CleanRequest)}
    bound = {binding.field for binding in PLAN_BINDINGS}
    assert bound <= fields, f"bindings name fields that do not exist: {bound - fields}"
    assert not (fields - bound - set(UNBOUND_REQUEST_FIELDS)), (
        "unbound CleanRequest fields: "
        + ", ".join(sorted(fields - bound - set(UNBOUND_REQUEST_FIELDS)))
    )
    assert not (set(UNBOUND_REQUEST_FIELDS) & bound), "a field cannot be both bound and excluded"
    assert all(UNBOUND_REQUEST_FIELDS.values()), "every exclusion needs a stated reason"


def test_plan_bindings_have_no_duplicate_widgets_or_fields():
    from tui_app import PLAN_BINDINGS

    selectors = [binding.selector for binding in PLAN_BINDINGS]
    fields = [binding.field for binding in PLAN_BINDINGS]
    assert len(set(selectors)) == len(selectors)
    assert len(set(fields)) == len(fields)


def test_reuse_restores_every_bound_option(tmp_path: Path):
    """ "Reuse" must repopulate the plan it was given, not a subset of it."""
    from tui_app import PLAN_BINDINGS, WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    saved = CleanRequest(
        paths=(source,),
        output=tmp_path / "out.txt",
        force_type="text",
        force_text=True,
        audit="record.json",
        nfkc=True,
        aggressive_homoglyphs=True,
        rewrite="humanize",
        rewrite_backend="ollama",
        rewrite_base_url="http://127.0.0.1:11434",
        rewrite_model="qwen3",
        rewrite_candidates=3,
        rewrite_temperature=0.4,
        rewrite_timeout=90.0,
        rewrite_lang="fr",
        rewrite_original_lang="en",
        tsapa_generations=7,
        tsapa_population=14,
        char_perturb=True,
        char_mode="confusable",
        char_strength=0.25,
        seed=11,
        visible_mask=tmp_path / "m.pgm",
        visible_backend="external",
        dilate=4,
        detect_command="detect {input} {mask}",
        inpaint_command="fill {input} {mask} {output}",
        visible_prompt="erase the logo",
        timeout=60.0,
        quality="high",
        degrade="freq-dct",
        morpho="grid",
        remove_synthid=True,
        synthid_strength=0.8,
        degrade_strength=0.3,
        degrade_seed=5,
        keep_artifacts=True,
        wmct_marker=True,
    )
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.apply_request(saved)
            await pilot.pause()
            restored = app.collect_request()
            for binding in PLAN_BINDINGS:
                if binding.field in TRI_STATE_BINDINGS:
                    continue
                assert getattr(restored, binding.field) == getattr(saved, binding.field), (
                    binding.field
                )

    _run(scenario())


# --- numeric plan fields ------------------------------------------------------


def test_a_zero_is_a_value_not_an_empty_field(tmp_path: Path):
    """0, 0.0 and a temperature of 0.0 are all things an operator can mean.

    ``PlanBinding.read`` fell back to the binding default on any falsy parse,
    so a deliberately-zero seed, temperature or strength silently ran as
    something else — and the copyable command showed the substitute.
    """
    from textual.widgets import Input
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-temperature", Input).value = "0"
            app.query_one("#in-seed", Input).value = "0"
            app.query_one("#in-perturb-strength", Input).value = "0"
            app.query_one("#in-degrade-strength", Input).value = "0"
            await pilot.pause()
            request = app.collect_request()
            assert request.rewrite_temperature == 0.0
            assert request.seed == 0
            assert request.char_strength == 0.0
            assert request.degrade_strength == 0.0

    _run(scenario())


def test_an_unparseable_number_is_refused_not_silently_defaulted(tmp_path: Path):
    """A field typed wrong must stop the plan, not become the default.

    A timeout of "soon" used to collapse to 1800.0: the run proceeded with a
    value nobody chose, and the copyable command advertised it as chosen.
    """
    from textual.widgets import Input, Static
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-timeout", Input).value = "soon"
            await pilot.pause()
            with pytest.raises(ValueError, match="timeout: not a number"):
                app.collect_request()
            preview = str(app.query_one("#command-preview", Static).render())
            assert "not a number" in preview
            assert "1800" not in preview

    _run(scenario())


def test_a_bad_number_names_the_field_it_came_from(tmp_path: Path):
    """``lstrip("#in-")`` strips a character set, not a prefix.

    "#in-inpaint-command" came out as "paint-command", naming a field that
    does not exist.
    """
    from textual.widgets import Input
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-dilate", Input).value = "wide"
            with pytest.raises(ValueError, match=r"^dilate: not a number"):
                app._number("#in-dilate", int)
            app.query_one("#in-inpaint-command", Input).value = "fill {input}"
            with pytest.raises(ValueError, match=r"^inpaint-command: not a number"):
                app._number("#in-inpaint-command", int)

    _run(scenario())


def test_a_bad_number_does_not_break_the_backends_pane(tmp_path: Path):
    """Half-typed input must not blank a pane that reports the endpoint policy."""
    from textual.widgets import Input, TabbedContent
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#in-timeout", Input).value = "soon"
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab-backends"
            await pilot.pause()
            rows = app.query_one("#backend-table").row_count
            assert rows > 0

    _run(scenario())


def test_an_invalid_form_never_leaves_a_runnable_command_behind(tmp_path: Path):
    """A stale copyable command is the same defect as an unserialised flag.

    The box is what the operator pastes into a shell. While the form is
    invalid it must say so, not keep offering the last command that parsed.
    """
    from textual.widgets import Input, TextArea
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#command-copyable", TextArea).text.startswith("wm ")
            app.query_one("#in-timeout", Input).value = "soon"
            copyable = ""
            for _ in range(20):
                await pilot.pause()
                copyable = app.query_one("#command-copyable", TextArea).text
                if not copyable.startswith("wm "):
                    break
            assert copyable.startswith("# invalid options")
            assert "timeout: not a number" in copyable

    _run(scenario())


def test_the_allow_remote_checkbox_states_a_choice_either_way(tmp_path: Path):
    """Unchecked must mean "no", not "let the environment decide".

    The pane renders the endpoint policy from this field. If unchecked meant
    "unstated", a set WATERMARKS_REWRITE_ALLOW_REMOTE would permit an egress
    the pane was still calling blocked.
    """
    from textual.widgets import Checkbox
    from tui_app import WatermarkTuiApp

    source = tmp_path / "draft.txt"
    source.write_text(ZWSP, encoding="utf-8")
    app = WatermarkTuiApp(CleanRequest(paths=(source,)))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.collect_request().rewrite_allow_remote is False
            app.query_one("#cb-allow-remote", Checkbox).value = True
            await pilot.pause()
            assert app.collect_request().rewrite_allow_remote is True

    _run(scenario())
