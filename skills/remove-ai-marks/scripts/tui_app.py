#!/usr/bin/env python3
"""Textual widgets for wm-tui.

Split from ``tui.py`` so the entry point can print an install hint on a default
install where textual is absent, and so the pure logic in ``tui`` stays testable
without a terminal.

Every rule in ``tui``'s module docstring applies here.  In particular: no
``urllib``, no ``CleanPlan`` construction, no verified badge on a best-effort
layer, and no API key ever reaching a widget or a copied string.
"""

from __future__ import annotations

import difflib
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets._select import NoSelection

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_kind import SUPPORTED_EXTENSIONS
from batch_inputs import select_inputs
from clean_asset import DEGRADE_CLI_CHOICES, MORPHO_CLI_CHOICES
from clean_file import dry_run_payload, run_clean_item
from clean_request import (
    FORCED_KINDS,
    QUALITY_PROFILES,
    REWRITE_CLI_CHOICES,
    CleanPlanPreflightError,
    CleanRequest,
    build_rewrite_plan,
    describe_dropped_text_transforms,
    dropped_text_transforms,
    plan_work,
    resolve_kind,
)
from common import atomic_write_text
from inspect_file import inspect_asset
from layer_b_discovery import classify_endpoint, probe_backend
from morphomod import VISIBLE_CLEAN_BACKENDS
from optional_deps import KNOWN_EXTRAS, check_optional
from perturb_text import MODES as PERTURB_MODES
from rewrite_text import (
    LIVE_REWRITE_BACKENDS,
    REASONING_EFFORTS,
    RewriteConfigurationError,
    generate_candidates,
)
from score_stylometry import score_text_stylometry
from tui import (
    DEFAULT_REWRITE_TIMEOUT,
    IDLE_STREAM_TITLE,
    PRESETS,
    STREAM_VIEW_CHARS,
    HistoryEntry,
    Preset,
    TuiSettings,
    apply_preset,
    discover_files,
    estimate_rewrite_seconds,
    format_badge,
    format_duration,
    preset_for,
    result_class_for,
    save_settings,
    should_confirm_cost,
)

REWRITE_ENV_KEY = "WATERMARKS_REWRITE_API_KEY"

#: Column label and share of the leftover width, per table.  ``DataTable``
#: sizes columns to their content, which leaves a table of short rows stopping
#: a third of the way across its pane with a dark strip after it; these weights
#: are what spread the columns over the whole width instead.
CAPABILITY_COLUMNS: tuple[tuple[str, int], ...] = (
    ("capability", 2),
    ("state", 1),
    ("detail", 5),
)
RUN_COLUMNS: tuple[tuple[str, int], ...] = (
    ("file", 4),
    ("kind", 1),
    ("result", 1),
    ("class", 2),
    ("residual", 1),
    ("note", 4),
)
HISTORY_COLUMNS: tuple[tuple[str, int], ...] = (
    ("time", 1),
    ("summary", 8),
)


def _column_widths(total: int, spec: tuple[tuple[str, int], ...]) -> list[int]:
    """Split ``total`` columns across ``spec``, exactly and without overflow.

    Every column gets at least its own label, so a narrow terminal degrades to
    a readable header rather than a row of ellipses.  Anything left over is
    shared by weight and the remainder lands on the last column, which is what
    makes the widths add up to ``total`` rather than to one or two less.
    """
    floors = [max(len(label), 4) for label, _ in spec]
    if total <= sum(floors):
        return floors
    weights = [weight for _, weight in spec]
    extra = total - sum(floors)
    share = sum(weights) or 1
    widths = [
        floor + extra * weight // share for floor, weight in zip(floors, weights, strict=True)
    ]
    widths[-1] += total - sum(widths)
    return widths


@dataclass
class _TableModel:
    """The rows a ``DataTable`` is showing, kept so it can be re-laid out.

    Column widths can only be given when a column is added, so filling the
    pane's width after a resize means re-adding the columns — and therefore
    re-adding the rows.  Holding them here is what makes that possible without
    reading them back out of the widget.
    """

    spec: tuple[tuple[str, int], ...]
    rows: list[tuple[str, ...]] = field(default_factory=list)
    #: Width the columns were last laid out for. Negative forces a redraw.
    width: int = 0


class ConfirmModal(ModalScreen[bool]):
    """A yes/no gate that defaults to No.

    Used for every irreversible or outward-facing action: remote egress,
    in-place overwrite, semantic stripping, and the batch cost ceiling.
    """

    BINDINGS: ClassVar[list] = [("escape", "dismiss_false", "Cancel")]

    def __init__(self, title: str, body: str, confirm_label: str = "Proceed") -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(self._title, id="modal-title")
            yield Static(self._body, id="modal-text")
            with Horizontal(id="modal-buttons"):
                yield Button("No", variant="primary", id="modal-no")
                yield Button(self._confirm_label, variant="warning", id="modal-yes")

    def on_mount(self) -> None:
        # Default focus lands on No: confirmation must be a deliberate act.
        self.query_one("#modal-no", Button).focus()

    @on(Button.Pressed, "#modal-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#modal-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class CandidateModal(ModalScreen[int | None]):
    """Pick among generated rewrites instead of accepting the auto-selected one."""

    BINDINGS: ClassVar[list] = [("escape", "dismiss_none", "Cancel")]

    def __init__(self, candidates) -> None:
        super().__init__()
        self._candidates = candidates

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label(
                f"{len(self._candidates)} candidates — {format_badge('B')}",
                id="modal-title",
            )
            table = DataTable(id="candidate-table", cursor_type="row")
            yield table
            yield TextArea("", read_only=True, id="candidate-preview")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cand-cancel")
                yield Button("Use selected", variant="primary", id="cand-ok")

    def on_mount(self) -> None:
        table = self.query_one("#candidate-table", DataTable)
        table.add_columns("#", "divergence", "score", "auto-pick", "chars")
        for candidate in self._candidates:
            table.add_row(
                str(candidate.index + 1),
                f"{candidate.lexical_divergence:.3f}",
                f"{candidate.selection_score:.3f}",
                "yes" if candidate.selected else "",
                str(len(candidate.text)),
            )
        table.focus()
        self._show(0)

    def _show(self, row: int) -> None:
        if 0 <= row < len(self._candidates):
            self.query_one("#candidate-preview", TextArea).text = self._candidates[row].text

    @on(DataTable.RowHighlighted, "#candidate-table")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        self._show(event.cursor_row)

    @on(Button.Pressed, "#cand-ok")
    def _ok(self) -> None:
        self.dismiss(self.query_one("#candidate-table", DataTable).cursor_row)

    @on(Button.Pressed, "#cand-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class WatermarkTuiApp(App):
    """inspect -> clean -> re-inspect, over the same seam the CLI uses."""

    TITLE = "wm-tui"
    SUB_TITLE = "watermark-remover"

    CSS = """
    Screen { layout: vertical; }
    /* Header and Footer are docked, so the flow area is two rows shorter than
       the screen.  Left at ``height: auto`` the tab container claimed all of
       it and pushed the status bar out, which cost the screen a permanent
       vertical scrollbar -- two columns stolen from every pane, on a screen
       that had nothing to scroll. */
    #tabs { height: 1fr; }

    /* -- the shared row grid --------------------------------------------- */
    .row { height: auto; }
    /* ``.muted``, not a bare ``Static``: ``Checkbox`` subclasses ``Static``,
       so the type selector also caught every checkbox and stretched it to
       fill the row -- "recursive" rendered 46 columns wide next to a
       16-column button. These are the inline status labels only. */
    .row > .muted { width: 1fr; height: 3; content-align: left middle; padding: 0 1; }
    /* Checkboxes are ``width: auto`` by default, which left-packs them and
       leaves the rest of the row empty. Give them the same bounded share as
       the input fields so a row of controls reads as one grid. */
    .row > Checkbox { width: 1fr; }
    /* Share the row rather than claiming a fixed 32 columns each: four
       fields at a fixed width overflow an 80- or 120-column terminal and
       the last one is simply unreachable.  ``1fr`` cannot overflow, so no
       ``max-width`` is needed to stay safe -- and capping it left a ragged
       gap at the end of every row on a wide terminal while the uncapped
       neighbours stretched past it. */
    .field { width: 1fr; }
    .wide { width: 1fr; }

    /* -- type ------------------------------------------------------------- */
    .muted { color: $text-muted; }
    /* Names the controls in the row beneath it.  A placeholder is gone the
       moment a field is filled, and "10" in a row of three inputs says
       nothing about which field it is. */
    .caption { color: $text-muted; height: 1; padding: 0 1; }
    /* Breathing room between groups; a form with no rhythm reads as one
       undifferentiated wall of controls. */
    .section { text-style: bold; margin-top: 1; }
    .section:first-of-type { margin-top: 0; }

    /* -- panes ------------------------------------------------------------ */
    TabPane { padding: 1 1 0 1; }
    /* Every scrolling surface gets the same titled frame, so a pane reads as
       a labelled thing rather than text floating on the terminal. */
    DataTable { border: round $panel; }
    DataTable:focus { border: round $accent; }
    #files-list { height: 1fr; border: round $panel; }
    #files-list:focus { border: round $accent; }
    #inspect-report { height: 1fr; border: round $panel; padding: 1; }
    #command-preview { height: auto; min-height: 3; border: round $accent; padding: 0 1; }
    #command-copyable { height: 5; border: round $panel; }
    #run-table { height: 1fr; min-height: 6; }
    #run-log { height: 1fr; min-height: 6; border: round $panel; }
    #history-table { height: 1fr; }
    /* Side by side: the stream is what the model is writing now, the diff is
       what changed — reading one without the other is half the answer, and
       stacking them starved the result table above. */
    #run-panes { height: 10; }
    #diff-view { width: 1fr; border: round $panel; }
    #stream-view { width: 1fr; border: round $accent; }
    #status-bar { height: 1; padding: 0 1; background: $panel; color: $text-muted; }

    /* -- the Start pane --------------------------------------------------- */
    /* One bordered, titled block per step.  The onboarding is four things to
       do in order; a flat column of controls does not say that. */
    .step { height: auto; border: round $primary 50%; padding: 0 1 1 1; margin-bottom: 1; }
    .step:focus-within { border: round $accent; }
    /* One line, not three: a summary padded to the height of a button row
       left a hole in the middle of the first thing the operator reads. */
    #start-files { width: 1fr; height: auto; padding: 0 1; }
    #start-ready { width: 1fr; height: 3; content-align: left middle; padding: 0 1; }
    #preset-detail { height: auto; min-height: 2; padding: 0 1; }
    #backend-table { height: auto; max-height: 12; }
    #install-command { height: 3; border: round $panel; }
    #btn-start-run { min-width: 22; }

    /* -- modals ----------------------------------------------------------- */
    #modal-body {
        width: 84; height: auto; max-height: 90%;
        border: thick $warning; background: $surface; padding: 1 2;
    }
    #modal-title { text-style: bold; }
    #modal-text { padding: 1 0; }
    #modal-buttons { height: auto; align-horizontal: right; }
    #candidate-table { height: 10; }
    #candidate-preview { height: 12; }
    """

    BINDINGS: ClassVar[list] = [
        ("q", "quit", "Quit"),
        ("r", "rescan", "Rescan"),
        ("i", "inspect", "Inspect"),
        ("ctrl+r", "run", "Run"),
    ]

    def __init__(self, request: CleanRequest, *, preset: Preset | None = None) -> None:
        super().__init__()
        self.request = request
        # The saved setup's preset, or the safest one.  Landing on a preset is
        # what makes "add files, press Clean" work without a tour of the Plan
        # tab; the pane names the preset and the command preview shows exactly
        # what it turned on, so nothing about it is implicit.
        self.preset: Preset = preset or PRESETS[0]
        self.files: list[Path] = []
        self.selected: list[Path] = []
        self.history: list[HistoryEntry] = []
        self._cancel_requested = False
        # Namespaced deliberately: textual's App owns a private `_running`, and
        # reusing that name silently reads the framework's lifecycle state.
        self._clean_running = False
        # Kept so rebuilding the capability table does not discard a probe the
        # operator just ran.
        self._last_probe = None
        # Install command per capability row, by row index; "" for the rows
        # that are not an extra.
        self._install_commands: list[str] = []
        # Which extras are installed, cached.  The endpoint row is rebuilt on
        # every keystroke in the base-URL field — the control and the row it
        # describes are on the same pane now, so a stale row would contradict
        # the field right next to it — and re-importing five optional packages
        # per keystroke to learn nothing new is not worth that.
        self._extra_rows: list[tuple[tuple[str, ...], str]] | None = None
        # Asset kind per discovered file, from the last rescan.  Classifying
        # reads the file, so it happens once per rescan rather than on every
        # keystroke that redraws the readiness line.
        self._kinds: dict[Path, str] = {}
        self._tables: dict[str, _TableModel] = {
            "#run-table": _TableModel(RUN_COLUMNS),
            "#backend-table": _TableModel(CAPABILITY_COLUMNS),
            "#history-table": _TableModel(HISTORY_COLUMNS),
        }

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        # Start first, and it is the setup pane: the old landing tab was a file
        # list over a plan nobody had configured yet, and the capability table
        # that told you what was missing was five tabs away with no way to act
        # on any of it.
        with TabbedContent(initial="tab-start", id="tabs"):
            with TabPane("Start", id="tab-start"):
                yield from self._compose_start()
            with TabPane("Files", id="tab-files"):
                yield from self._compose_files()
            with TabPane("Inspect", id="tab-inspect"):
                yield from self._compose_inspect()
            with TabPane("Plan", id="tab-plan"):
                yield from self._compose_plan()
            with TabPane("Run", id="tab-run"):
                yield from self._compose_run()
            with TabPane("History", id="tab-history"):
                yield from self._compose_history()
        yield Static("", id="status-bar")
        yield Footer()

    def _compose_start(self) -> ComposeResult:
        """Add files, choose what to remove, point at a backend, clean."""
        with VerticalScroll(id="start"):
            with Vertical(classes="step") as step:
                step.border_title = "1 · Add files"
                yield Static(
                    "add as many as you like · filters on the Files tab", classes="caption"
                )
                with Horizontal(classes="row"):
                    yield Input(
                        placeholder="path to a file or a folder",
                        id="in-add-path",
                        classes="wide",
                    )
                    yield Button("Add", variant="primary", id="btn-add-path")
                    yield Button("Clear", id="btn-clear-paths")
                yield Static("", id="start-files", classes="muted")

            with Vertical(classes="step") as step:
                step.border_title = "2 · Choose what to remove"
                with Horizontal(classes="row"):
                    yield Select(
                        [(preset.label, preset.key) for preset in PRESETS],
                        value=self.preset.key,
                        allow_blank=False,
                        id="sel-preset",
                        classes="field",
                    )
                    yield Static(
                        "every option a preset sets stays visible on the Plan tab",
                        classes="muted",
                    )
                yield Static("", id="preset-detail")

            with Vertical(classes="step") as step:
                step.border_title = "3 · Clean"
                with Horizontal(classes="row"):
                    yield Button("Clean now", variant="primary", id="btn-start-run")
                    yield Button("Advanced options", id="btn-advanced")
                    yield Static("", id="start-ready", classes="muted")

            with Vertical(classes="step") as step:
                step.border_title = "Layer B endpoint — only the rewrite preset needs one"
                yield Static("backend · base URL · model", classes="caption")
                with Horizontal(classes="row"):
                    yield Select(
                        [(name, name) for name in LIVE_REWRITE_BACKENDS],
                        prompt="backend",
                        allow_blank=True,
                        id="sel-backend",
                        classes="field",
                    )
                    yield Input(
                        placeholder="base URL (http://127.0.0.1:11434)",
                        id="in-base-url",
                        classes="field",
                    )
                    # Free text, not a discovery-only picker: an endpoint that
                    # does not list models (or is not reachable yet) must still
                    # be usable.
                    yield Input(placeholder="model", id="in-model", classes="field")
                    yield Button("Probe", id="btn-probe")
                yield Static("discovered models · reasoning effort", classes="caption")
                with Horizontal(classes="row"):
                    yield Select(
                        [], prompt="discovered", allow_blank=True, id="sel-model", classes="field"
                    )
                    yield Select(
                        [(name, name) for name in REASONING_EFFORTS],
                        prompt="reasoning effort",
                        allow_blank=True,
                        id="sel-effort",
                        classes="field",
                    )
                    yield Checkbox("disable thinking", False, id="cb-disable-thinking")
                with Horizontal(classes="row"):
                    yield Checkbox("allow remote endpoint", False, id="cb-allow-remote")
                    yield Button("Save setup", id="btn-save-settings")
                    yield Static("", id="endpoint-state", classes="muted")
                yield Static("", id="settings-state", classes="muted")
                yield Static("", id="api-key-state", classes="muted")

            with Vertical(classes="step") as step:
                step.border_title = "Installed capabilities"
                yield DataTable(id="backend-table", cursor_type="row", zebra_stripes=True)
                yield Static("select a row for the command that installs it", classes="caption")
                yield TextArea("", read_only=True, id="install-command")
                with Horizontal(classes="row"):
                    yield Button("Refresh", id="btn-refresh-backends")
                    yield Static(
                        "the core clean needs nothing extra; the rows above are opt-in backends",
                        classes="muted",
                    )

    def _compose_files(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Input(value=self.request.glob, placeholder="glob", id="in-glob", classes="field")
            yield Input(
                value=self.request.extensions or "",
                placeholder="extensions (.md,.txt)",
                id="in-extensions",
                classes="field",
            )
            yield Checkbox("recursive", self.request.recursive, id="cb-recursive")
            yield Button("Rescan", id="btn-rescan")
        listing = SelectionList[str](id="files-list")
        listing.border_title = "space toggles · every file is selected after a rescan"
        yield listing
        yield Static("", id="files-summary", classes="muted")

    def _compose_inspect(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Button("Inspect selected", variant="primary", id="btn-inspect")
            yield Checkbox("soft binding", False, id="cb-inspect-soft")
        yield VerticalScroll(Static("", id="inspect-report"))

    def _compose_plan(self) -> ComposeResult:
        with VerticalScroll():
            yield Label("Routing", classes="section")
            yield Static("asset kind · force text · audit record", classes="caption")
            with Horizontal(classes="row"):
                yield Select(
                    [(name, name) for name in FORCED_KINDS],
                    value="auto",
                    allow_blank=False,
                    id="sel-force-type",
                    classes="field",
                )
                # The skipped-transform note tells the operator to force text on
                # a container; it would be a poor UI that said so and then made
                # them leave for the CLI.
                yield Checkbox("force text", False, id="cb-force-text")
                yield Input(placeholder="audit JSON path", id="in-audit", classes="field")

            yield Label("Layer A — hidden Unicode  " + format_badge("A"), classes="section")
            with Horizontal(classes="row"):
                yield Checkbox("NFKC", self.request.nfkc, id="cb-nfkc")
                yield Checkbox("aggressive homoglyphs", False, id="cb-homoglyphs")
                yield Checkbox("strip semantic format", False, id="cb-semantic")

            yield Label("Layer M — metadata  " + format_badge("M"), classes="section")
            with Horizontal(classes="row"):
                yield Checkbox("keep non-AI metadata", False, id="cb-keep-meta")
                yield Checkbox("detect soft binding", False, id="cb-soft")

            yield Label("Layer B — LLM rewrite  " + format_badge("B"), classes="section")
            # The endpoint, the model and the key state are set-once settings
            # and live on the Start tab; what stays here is what changes per
            # run.  Splitting them that way is what lets Start be short enough
            # to read.
            yield Static(
                "strength · candidates · temperature · per-call timeout (s) — "
                "backend and model are on the Start tab",
                classes="caption",
            )
            with Horizontal(classes="row"):
                yield Select(
                    [(name, name) for name in REWRITE_CLI_CHOICES],
                    prompt="strength (off)",
                    allow_blank=True,
                    id="sel-rewrite",
                    classes="field",
                )
                yield Input(
                    placeholder="candidates", value="1", id="in-candidates", classes="field"
                )
                yield Input(placeholder="temperature", id="in-temperature", classes="field")
                yield Input(placeholder="timeout s", id="in-rewrite-timeout", classes="field")
            yield Static(
                "pivot language · original language · tsapa generations · tsapa population",
                classes="caption",
            )
            with Horizontal(classes="row"):
                yield Input(placeholder="pivot lang", id="in-lang", classes="field")
                yield Input(placeholder="original lang", id="in-original-lang", classes="field")
                yield Input(placeholder="tsapa generations", id="in-generations", classes="field")
                yield Input(placeholder="tsapa population", id="in-population", classes="field")
            with Horizontal(classes="row"):
                yield Button("Pick from candidates", id="btn-candidates")
                yield Static(
                    "generates N candidates for one text file and lets you choose",
                    classes="muted",
                )

            yield Label("Character perturbation", classes="section")
            with Horizontal(classes="row"):
                yield Checkbox("char perturb", False, id="cb-perturb")
                yield Select(
                    [(name, name) for name in PERTURB_MODES],
                    prompt="mode",
                    allow_blank=True,
                    id="sel-perturb-mode",
                    classes="field",
                )
                yield Input(placeholder="strength 0-1", id="in-perturb-strength", classes="field")
                yield Input(placeholder="seed", id="in-seed", classes="field")

            yield Label("Layer V — visible marks  " + format_badge("V"), classes="section")
            yield Static("mask · box · inpaint backend · dilation radius", classes="caption")
            with Horizontal(classes="row"):
                yield Input(placeholder="mask path", id="in-mask", classes="field")
                yield Input(placeholder="box x,y,w,h", id="in-box", classes="field")
                yield Select(
                    [(name, name) for name in VISIBLE_CLEAN_BACKENDS],
                    prompt="visible backend",
                    allow_blank=True,
                    id="sel-visible-backend",
                    classes="field",
                )
                yield Input(placeholder="dilate radius", id="in-dilate", classes="field")
            yield Static(
                "detector command · inpainter command · prompt · external timeout (s)",
                classes="caption",
            )
            with Horizontal(classes="row"):
                yield Input(
                    placeholder="detect cmd {input} {mask}",
                    id="in-detect-command",
                    classes="field",
                )
                yield Input(
                    placeholder="inpaint cmd {input} {mask} {output}",
                    id="in-inpaint-command",
                    classes="field",
                )
                yield Input(placeholder="visible prompt", id="in-visible-prompt", classes="field")
                yield Input(placeholder="external timeout", id="in-timeout", classes="field")
            with Horizontal(classes="row"):
                yield Select(
                    [(name, name) for name in QUALITY_PROFILES],
                    prompt="quality",
                    allow_blank=True,
                    id="sel-quality",
                    classes="field",
                )
                yield Checkbox("remove SynthID", False, id="cb-synthid")
                yield Checkbox("dry run", False, id="cb-dry-run")
                yield Static("", id="visible-state", classes="muted")

            yield Label("Image degradation  " + format_badge("V"), classes="section")
            yield Static(
                "frequency · morphological · strength · seed · SynthID strength",
                classes="caption",
            )
            with Horizontal(classes="row"):
                yield Select(
                    [(name, name) for name in DEGRADE_CLI_CHOICES],
                    prompt="degrade",
                    allow_blank=True,
                    id="sel-degrade",
                    classes="field",
                )
                yield Select(
                    [(name, name) for name in MORPHO_CLI_CHOICES],
                    prompt="morpho",
                    allow_blank=True,
                    id="sel-morpho",
                    classes="field",
                )
                yield Input(
                    placeholder="degrade strength", id="in-degrade-strength", classes="field"
                )
                yield Input(placeholder="degrade seed", id="in-degrade-seed", classes="field")
                yield Input(
                    placeholder="synthid strength", id="in-synthid-strength", classes="field"
                )

            yield Label("Output", classes="section")
            with Horizontal(classes="row"):
                yield Input(placeholder="output path or directory", id="in-output", classes="wide")
                yield Checkbox("in place", False, id="cb-in-place")
                yield Checkbox("keep artifacts", False, id="cb-artifacts")
                yield Checkbox("wmCt marker", False, id="cb-wmct")

            yield Label("Equivalent command", classes="section")
            yield Static("", id="command-preview")
            with Horizontal(classes="row"):
                yield Button("Copy command", id="btn-copy-command")
                yield Static("", id="copy-state", classes="muted")
            yield TextArea("", read_only=True, id="command-copyable")

    def _compose_run(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Button("Run", variant="primary", id="btn-run")
            yield Button("Stop after current file", id="btn-cancel", disabled=True)
            yield Static("", id="run-state", classes="muted")
        results = DataTable(id="run-table", zebra_stripes=True)
        results.border_title = "results — one row per file, badged by layer"
        yield results
        with Horizontal(id="run-panes"):
            stream = TextArea("", read_only=True, id="stream-view")
            stream.border_title = IDLE_STREAM_TITLE
            yield stream
            diff = TextArea("", read_only=True, id="diff-view")
            diff.border_title = "before / after"
            yield diff
        log = RichLog(id="run-log", markup=True, wrap=True)
        log.border_title = "log — the after-state of every file, re-inspected"
        yield log

    def _compose_history(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Button("Copy", id="btn-history-copy")
            yield Button("Reuse", id="btn-history-reuse")
            yield Static("in-memory, this session only", classes="muted")
        table = DataTable(id="history-table", cursor_type="row", zebra_stripes=True)
        table.border_title = "every command this session generated"
        yield table
        yield TextArea("", read_only=True, id="history-copyable")

    # -- lifecycle ---------------------------------------------------------

    def on_mount(self) -> None:
        for selector in self._tables:
            self._relayout_table(selector)
        self._refresh_api_key_state()
        # The preset is applied, not merely displayed: the Plan widgets and the
        # command preview must say what pressing "Clean now" would actually do.
        self.apply_request(apply_preset(self.request, self.preset))
        self.action_rescan()
        self.refresh_backends()
        self._sync_preview()

    def on_resize(self) -> None:
        """Re-lay the tables when the terminal changes size."""
        self._relayout_tables()

    def _relayout_tables(self) -> None:
        """Re-lay every table that now knows how wide it is.

        A table in a hidden ``TabPane`` has no width to divide up, so the one
        on the tab you open second would keep the fallback columns it was given
        at mount — the very dark strip this is here to remove.
        """
        if not self.is_mounted:
            return
        for selector in self._tables:
            if self.query(selector):
                self._relayout_table(selector)

    # -- tables ------------------------------------------------------------

    def _relayout_table(self, selector: str) -> bool:
        """Rebuild a table so its columns span the pane. True when it redrew.

        ``DataTable`` sizes each column to its content and column widths can
        only be given when the column is added, so a table of short rows stops
        halfway across the pane and the header band ends in a dark strip.
        Filling the width means re-adding the columns, which means re-adding
        the rows — which is why the rows are kept in ``_TableModel``.
        """
        model = self._tables[selector]
        table = self.query_one(selector, DataTable)
        width = table.size.width
        if width == model.width and table.columns:
            return False
        model.width = width
        usable = width - 2 * table.cell_padding * len(model.spec)
        # Before the first layout the table has no width to divide up; add the
        # columns anyway so a row written now is not dropped for want of one.
        widths = _column_widths(usable, model.spec) if usable > 0 else [None] * len(model.spec)
        table.clear(columns=True)
        for (label, _), column_width in zip(model.spec, widths, strict=True):
            table.add_column(label, width=column_width)
        for row in model.rows:
            table.add_row(*row)
        return True

    def _add_table_row(self, selector: str, *cells: str) -> None:
        self._tables[selector].rows.append(cells)
        # A relayout re-adds every row itself; appending twice would double it.
        if not self._relayout_table(selector):
            self.query_one(selector, DataTable).add_row(*cells)

    def _set_table_rows(self, selector: str, rows: list[tuple[str, ...]]) -> None:
        model = self._tables[selector]
        model.rows = rows
        model.width = -1  # no real width is negative, so this forces the redraw
        self._relayout_table(selector)

    def _status(self, message: str) -> None:
        self.query_one("#status-bar", Static).update(message)

    def _log(self, message: str) -> None:
        self.query_one("#run-log", RichLog).write(message)

    # -- files -------------------------------------------------------------

    def action_rescan(self) -> None:
        self.request = replace(
            self.request,
            glob=self.query_one("#in-glob", Input).value or "*",
            extensions=self.query_one("#in-extensions", Input).value or None,
            recursive=self.query_one("#cb-recursive", Checkbox).value,
        )
        files, error = discover_files(self.request)
        listing = self.query_one("#files-list", SelectionList)
        listing.clear_options()
        if error:
            self._status(f"selection error: {error}")
            self.files = []
            self.selected = []
            self._update_files_summary()
            self._sync_preview()
            return
        self.files = files
        self._kinds = {path: self._classify(path) for path in files}
        listing.add_options([(str(path), str(path), True) for path in files])
        # Everything the rescan found is selected: the list it replaced no
        # longer matches, so carrying a previous deselection forward would
        # silently apply it to different files.  ``_update_files_summary``
        # says so rather than leaving it to be discovered.
        self.selected = list(files)
        self._update_files_summary()
        self._sync_preview()

    def _classify(self, path: Path) -> str:
        """Best-effort asset kind. A file we cannot classify is not a warning."""
        try:
            return resolve_kind(path, self.request)
        except (ValueError, OSError):
            return "unknown"

    def _update_files_summary(self) -> None:
        filters = [f"glob {self.request.glob}"]
        if self.request.recursive:
            filters.append("recursive")
        if self.request.extensions:
            filters.append(f"ext {self.request.extensions}")
        self.query_one("#files-summary", Static).update(
            f"{len(self.selected)} of {len(self.files)} selected · " + " · ".join(filters)
        )
        self._sync_start()

    # -- start pane --------------------------------------------------------

    def _sync_start(self) -> None:
        """Keep the onboarding pane's three answers current."""
        roots = len(self.request.paths)
        self.query_one("#start-files", Static).update(
            f"{len(self.files)} file(s) under {roots} path(s) · {len(self.selected)} selected"
        )
        self.query_one("#preset-detail", Static).update(
            f"{self.preset.badge()}  {escape(self.preset.description)}"
        )
        self.query_one("#start-ready", Static).update(self._readiness())
        # The Run pane's own state line, which nothing wrote to before: the
        # operator arrives there from another tab and has to be told what
        # pressing Run would act on.
        self.query_one("#run-state", Static).update(
            f"{len(self.selected)} file(s) · {escape(self.preset.label)} {self.preset.badge()}"
        )
        # The endpoint controls and the row that reports the endpoint policy
        # are on the same pane; a row that lags the field above it by a tab
        # switch is a pane arguing with itself.
        if self.is_mounted:
            self.refresh_backends(recheck=False)

    def _readiness(self) -> str:
        """What still stands between this form and a run. Never a bare "ready"."""
        blockers = []
        if not self.selected:
            blockers.append("step 1: no files selected")
        if self.preset.requires_endpoint and not self._value("#in-base-url"):
            blockers.append("this preset needs a Layer B endpoint — set one below")
        extra = self.preset.requires_extra
        if extra and not check_optional(extra).available:
            blockers.append(f"needs watermark-remover[{extra}]")
        blockers.extend(self._dropped_transform_warnings())
        if blockers:
            return "[yellow]" + escape(" · ".join(blockers)) + "[/]"
        return f"[green]ready[/] — {len(self.selected)} file(s), {self.preset.badge()}"

    def _dropped_transform_warnings(self) -> list[str]:
        """Name text transforms this selection would silently drop.

        ``.md`` and ``.html`` route to the container pipeline, so a rewrite
        chosen in step 2 is skipped for them — the run says so afterwards, in
        one row of a results table. A preset that promises a rewrite has to say
        it will not happen *before* the run, not report it after.
        """
        try:
            request = self.collect_request()
        except ValueError:
            return []
        counted: dict[str, int] = {}
        for path in self.selected:
            for name in dropped_text_transforms(request, self._kinds.get(path, "unknown")):
                counted[name] = counted.get(name, 0) + 1
        return [
            f"{count} file(s) would skip {name} — tick “force text” on Plan"
            for name, count in sorted(counted.items())
        ]

    @on(Button.Pressed, "#btn-add-path")
    @on(Input.Submitted, "#in-add-path")
    def _add_path(self) -> None:
        raw = self._value("#in-add-path")
        if raw is None:
            self._status("type a file or folder path first")
            return
        path = Path(raw).expanduser()
        if not path.exists():
            self._status(f"no such path: {path}")
            return
        if path in self.request.paths:
            self._status(f"already added: {path}")
            return
        self.request = replace(self.request, paths=(*self.request.paths, path))
        self.query_one("#in-add-path", Input).value = ""
        self.action_rescan()
        self._status(f"added {path}")

    @on(Button.Pressed, "#btn-clear-paths")
    def _clear_paths(self) -> None:
        self.request = replace(self.request, paths=())
        self.action_rescan()
        self._status("cleared — add a file or folder to start again")

    @on(Select.Changed, "#sel-preset")
    def _preset_changed(self, event: Select.Changed) -> None:
        """Apply a preset through the same widgets everything else reads."""
        chosen = preset_for(None if isinstance(event.value, NoSelection) else str(event.value))
        if chosen is None:
            return
        self.preset = chosen
        # Round-trip through the form so the preset lands in the Plan widgets:
        # a preset that only changed a private field would not show up in the
        # command preview, and the run would not match what the pane claims.
        try:
            current = self.collect_request()
        except ValueError:
            current = self.request
        self.apply_request(apply_preset(current, chosen))
        self._status(f"{chosen.label} — {result_class_for(chosen.layer)}")

    @on(Button.Pressed, "#btn-advanced")
    def _show_advanced(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-plan"

    @on(Button.Pressed, "#btn-start-run")
    def _start_run(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-run"
        self.action_run()

    @on(Button.Pressed, "#btn-save-settings")
    def _save_setup(self) -> None:
        """Remember the endpoint. Never the key — it has no field to land in."""
        try:
            request = self.collect_request()
        except ValueError as error:
            self._status(f"invalid options: {error}")
            return
        settings = TuiSettings(
            preset=self.preset.key,
            rewrite_backend=request.rewrite_backend,
            rewrite_base_url=request.rewrite_base_url,
            rewrite_model=request.rewrite_model,
            rewrite_reasoning_effort=request.rewrite_reasoning_effort,
            rewrite_allow_remote=request.rewrite_allow_remote,
        )
        try:
            where = save_settings(settings)
        except OSError as error:
            self.query_one("#settings-state", Static).update(
                f"[red]could not save: {escape(str(error))}[/]"
            )
            return
        self.query_one("#settings-state", Static).update(
            f"saved to {escape(str(where))} — the API key is never written"
        )

    @on(DataTable.RowHighlighted, "#backend-table")
    def _capability_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the command that installs the highlighted capability.

        ``refresh_backends`` strips the shared "Install watermark-remover[...]"
        preamble so the import that actually failed stays visible in a narrow
        detail column — but that preamble was the actionable half.  It belongs
        here, in something selectable.
        """
        row = event.cursor_row
        command = self._install_commands[row] if 0 <= row < len(self._install_commands) else ""
        self.query_one("#install-command", TextArea).text = (
            command or "# nothing to install for this row"
        )

    @on(SelectionList.SelectedChanged, "#files-list")
    def _files_changed(self) -> None:
        listing = self.query_one("#files-list", SelectionList)
        self.selected = [Path(value) for value in listing.selected]
        self._update_files_summary()
        self._sync_preview()

    @on(Button.Pressed, "#btn-rescan")
    def _rescan_pressed(self) -> None:
        self.action_rescan()

    # -- inspect -----------------------------------------------------------

    def action_inspect(self) -> None:
        if not self.selected:
            self._status("nothing selected")
            return
        self._status("inspecting…")
        self.inspect_worker(list(self.selected))

    @on(Button.Pressed, "#btn-inspect")
    def _inspect_pressed(self) -> None:
        self.action_inspect()

    @work(thread=True, exclusive=True, group="inspect")
    def inspect_worker(self, paths: list[Path]) -> None:
        soft = self.query_one("#cb-inspect-soft", Checkbox).value
        blocks = [self._inspect_block(path, soft_binding=soft) for path in paths]
        self.call_from_thread(self.query_one("#inspect-report", Static).update, "\n\n".join(blocks))
        self.call_from_thread(self._status, f"inspected {len(paths)} file(s)")

    def _inspect_block(self, path: Path, *, soft_binding: bool, label: str = "") -> str:
        # Filenames, findings and error text are not ours: a name like
        # "report[1].md" would otherwise be parsed as Rich markup and vanish.
        name = escape(path.name)
        try:
            report = inspect_asset(path, soft_binding=soft_binding)
        except Exception as error:  # inspection must never take the UI down
            return f"[bold]{name}[/] — [red]inspect failed: {escape(str(error))}[/]"
        kind = escape(str(report.get("kind", "unknown")))
        lines = [f"[bold]{name}[/]{label}  kind: {kind}"]

        if kind == "text":
            total = report.get("suspicious_total", 0)
            lines.append(f"  A hidden Unicode: {total} carrier(s)  {format_badge('A')}")
            for carrier, count in sorted((report.get("counts") or {}).items()):
                if count:
                    lines.append(f"      {escape(str(carrier))} x{count}")
            lines.append(self._stylometry_line(path))
        else:
            lines.append(
                f"  M metadata: C2PA={report.get('has_c2pa')} "
                f"AI={report.get('has_ai_metadata')}  {format_badge('M')}"
            )
            for finding in (report.get("findings") or [])[:12]:
                lines.append(f"      {escape(str(finding))}")
            if report.get("note"):
                lines.append(f"      {escape(str(report['note']))}")

        soft = (report.get("soft_binding") or {}).get("soft_binding")
        if soft:
            state = "found" if soft.get("found") else "none"
            lines.append(f"  soft binding: {state}  {format_badge('soft-binding')}")
        return "\n".join(lines)

    def _stylometry_line(self, path: Path) -> str:
        """Layer B's detector view. Always labelled best-effort."""
        try:
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
            report = score_text_stylometry(text, str(path))
        except Exception as error:
            return f"  B token watermark: unavailable ({escape(str(error))})  {format_badge('B')}"
        markers = len(report.matched_markers or ())
        detail = f"  B token watermark: stylometry {report.score:.2f} ({report.confidence_level})"
        if report.status != "ok":
            # Parentheses, not brackets: brackets are Rich markup and would be
            # parsed away, leaving a stray gap where the status should be.
            detail += f" ({escape(report.status)})"
        return f"{detail}, {markers} AI phrase(s)  {format_badge('B')}"

    # -- plan --------------------------------------------------------------

    def _refresh_api_key_state(self) -> None:
        # The key itself is never rendered — only whether one is present.
        present = "set" if os.environ.get(REWRITE_ENV_KEY) else "not set"
        self.query_one("#api-key-state", Static).update(
            f"API key: {present} (read from {REWRITE_ENV_KEY}; never displayed or copied)"
        )

    def _value(self, widget_id: str) -> str | None:
        raw = self.query_one(widget_id, Input).value.strip()
        return raw or None

    def _number(self, widget_id: str, cast) -> object | None:
        """Parse a numeric field: empty means "unset", garbage means invalid.

        Raising rather than returning ``None`` on garbage is the point.  A
        field the operator typed something into but got wrong must not quietly
        become the default — that runs a clean with a timeout or a strength
        nobody asked for, and the copyable command would show the substituted
        value as though it had been chosen.
        """
        raw = self._value(widget_id)
        if raw is None:
            return None
        try:
            return cast(raw)
        except ValueError as error:
            field = widget_id.removeprefix("#in-")
            raise ValueError(f"{field}: not a number ({raw})") from error

    def _selected_value(self, widget_id: str) -> str | None:
        """Read a Select, treating "nothing chosen" as None.

        Textual's empty sentinel is ``Select.NULL`` (a ``NoSelection``);
        ``Select.BLANK`` is the bool ``False`` and is *not* it.  Comparing
        against the wrong one lets the sentinel leak into the request and
        stringify as "Select.NULL".
        """
        value = self.query_one(widget_id, Select).value
        return None if isinstance(value, NoSelection) else str(value)

    def collect_request(self) -> CleanRequest:
        """Read every Plan widget into a CleanRequest. The only request builder."""
        values: dict[str, object] = {}
        for binding in PLAN_BINDINGS:
            values[binding.field] = binding.read(self)
        return replace(
            self.request,
            paths=tuple(self.selected),
            visible_box=self._box(),
            # ``--tsapa`` is an alias for ``--rewrite tsapa``; the picker above
            # already carries it, so setting both would double the flag.
            tsapa=False,
            rewrite_disable_thinking=(
                True if self.query_one("#cb-disable-thinking", Checkbox).value else None
            ),
            **values,
        )

    def _box(self) -> tuple[int, int, int, int] | None:
        raw = self._value("#in-box")
        if not raw:
            return None
        try:
            parts = tuple(int(part) for part in raw.split(","))
        except ValueError:
            parts = ()
        if len(parts) != 4:
            self._status("box must be x,y,w,h")
            return None
        return parts

    @on(Input.Changed)
    @on(Select.Changed)
    @on(Checkbox.Changed)
    def _any_change(self) -> None:
        self._sync_preview()

    def _sync_preview(self) -> None:
        try:
            request = self.collect_request()
        except ValueError as error:
            message = f"invalid options: {error}"
            self.query_one("#command-preview", Static).update(f"[red]{escape(message)}[/]")
            # Clear the copyable box rather than leave the last valid command
            # in it: a box that still offers a runnable command while the form
            # is invalid hands the operator something the form no longer says.
            self.query_one("#command-copyable", TextArea).text = f"# {message}"
            return
        command = request.command_string()
        self.query_one("#command-preview", Static).update(escape(command))
        # The selectable copy of the command is the fallback for terminals that
        # ignore OSC 52 — it must always carry the same string as the button.
        self.query_one("#command-copyable", TextArea).text = command
        self._sync_endpoint_state(request)
        self._sync_visible_state(request)
        self._sync_start()

    def _sync_endpoint_state(self, request: CleanRequest) -> None:
        if request.rewrite_strength is None:
            self.query_one("#endpoint-state", Static).update("Layer B off")
            return
        policy = classify_endpoint(
            request.rewrite_base_url, allow_remote=request.rewrite_allow_remote
        )
        if policy.loopback:
            state = f"[green]loopback {escape(policy.host)}[/]"
        elif policy.allowed:
            state = f"[yellow]REMOTE {escape(policy.host)} — text leaves this machine[/]"
        else:
            state = f"[red]blocked: {escape(str(policy.reason))}[/]"
        self.query_one("#endpoint-state", Static).update(state)

    def _sync_visible_state(self, request: CleanRequest) -> None:
        note = ""
        if len(self.selected) > 1 and (request.visible_mask or request.visible_box):
            note = "[yellow]mask/box are single-file; use --detect-command for batch[/]"
        elif not check_optional("visible").available and request.visible_requested():
            note = check_optional("visible").hint
        self.query_one("#visible-state", Static).update(note)

    @on(Select.Changed, "#sel-model")
    def _discovered_model_chosen(self, event: Select.Changed) -> None:
        """A discovered model fills the free-text field, which stays the source."""
        if not isinstance(event.value, NoSelection):
            self.query_one("#in-model", Input).value = str(event.value)

    @on(Button.Pressed, "#btn-copy-command")
    def _copy_command(self) -> None:
        command = self.query_one("#command-copyable", TextArea).text
        self.copy_to_clipboard(command)
        # OSC 52 is fire-and-forget: the terminal never acknowledges it, and
        # macOS Terminal.app ignores it outright. Never claim a verified copy.
        self.query_one("#copy-state", Static).update(
            "copied via OSC 52 — if your terminal ignored it, the box below is selectable"
        )

    # -- candidate picker ---------------------------------------------------

    @on(Button.Pressed, "#btn-candidates")
    def _candidates_pressed(self) -> None:
        try:
            request = self.collect_request()
        except ValueError as error:
            self._status(f"invalid options: {error}")
            return
        if request.rewrite_strength is None:
            self._status("choose a Layer B strength first")
            return
        if len(self.selected) != 1:
            self._status("candidate picking works on exactly one file")
            return
        path = self.selected[0]
        try:
            if resolve_kind(path, request) != "text":
                self._status(f"{path.name} is not a text asset")
                return
        except ValueError as error:
            self._status(str(error))
            return
        try:
            plan = build_rewrite_plan(request)
        except RewriteConfigurationError as error:
            self._status(str(error))
            return
        self.run_worker(self._pick_candidate(path, plan), exclusive=False)

    async def _pick_candidate(self, path: Path, plan) -> None:
        policy = classify_endpoint(plan.base_url, allow_remote=plan.allow_remote)
        if not policy.loopback and policy.allowed:
            approved = await self.push_screen_wait(
                ConfirmModal(
                    "Send document text to a remote endpoint?",
                    f"{policy.warning}\n\nEndpoint host: {policy.host}",
                    "Send anyway",
                )
            )
            if not approved:
                self._status("cancelled: remote endpoint not approved")
                return
        self._status(f"generating {plan.candidates} candidate(s)…")
        self._begin_stream(path.name)
        candidates = await self._generate_candidates(path, plan)
        if not candidates:
            return
        chosen = await self.push_screen_wait(CandidateModal(candidates))
        if chosen is None:
            self._status("candidate discarded; nothing written")
            return
        destination = path.with_name(f"{path.stem}.rewritten{path.suffix}")
        try:
            atomic_write_text(destination, candidates[chosen].text)
        except OSError as error:
            self._status(f"write failed: {error}")
            return
        self._log(
            f"wrote {destination} from candidate {chosen + 1} "
            f"{format_badge('B')} — no detector guarantee"
        )
        self._status(f"wrote {destination.name}")

    @work(thread=True, exclusive=True, group="candidates")
    async def _generate_candidates(self, path: Path, plan):
        """Generate off the UI thread; a model call must never block the terminal."""
        try:
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
            return generate_candidates(text, plan, on_token=self._append_stream)
        except Exception as error:
            self.call_from_thread(
                self._status, f"candidate generation failed: {escape(str(error))}"
            )
            return []

    def _append_stream(self, fragment: str) -> None:
        """Token sink: called from the reader thread, so hop to the UI thread."""
        self.call_from_thread(self._write_stream, fragment)

    def _write_stream(self, fragment: str) -> None:
        view = self.query_one("#stream-view", TextArea)
        # Bounded on purpose: a long document would otherwise grow the widget's
        # document without limit while the run is still going.
        view.text = (view.text + fragment)[-STREAM_VIEW_CHARS:]
        view.scroll_end(animate=False)

    def _begin_stream(self, filename: str | None) -> None:
        """Reset the stream box, and label why it is empty when nothing streams.

        An always-visible box that only ever fills on one code path reads as
        broken; naming the reason is cheaper than hiding it.
        """
        view = self.query_one("#stream-view", TextArea)
        view.text = ""
        view.border_title = IDLE_STREAM_TITLE if filename is None else f"generating {filename}"

    # -- backends ----------------------------------------------------------

    @on(Button.Pressed, "#btn-refresh-backends")
    def _refresh_pressed(self) -> None:
        self.refresh_backends()

    @on(TabbedContent.TabActivated, "#tabs")
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Rebuild the capability table when it comes into view.

        Its Layer B rows are derived from the *current* plan, so a table left
        as it was at mount reports "no base URL set" over a plan that has one —
        stale state presented as fact.
        """
        if event.pane.id == "tab-start":
            self.refresh_backends()
        # A pane that was hidden until now has only just been given a width.
        self.call_after_refresh(self._relayout_tables)

    def refresh_backends(self, *, recheck: bool = True) -> None:
        """Rebuild the capability table. ``recheck`` re-imports the extras."""
        if recheck or self._extra_rows is None:
            self._extra_rows = [
                (
                    (
                        f"extra: {extra}",
                        "available" if availability.available else "missing",
                        # The hint's shared "install watermark-remover[...]"
                        # preamble repeats on every row and pushes the part that
                        # differs — the import that actually failed — off the
                        # visible width.  The preamble is not lost: it becomes
                        # the row's install command.
                        availability.hint.replace("Reason: ", "").split(". ")[-1],
                    ),
                    ("" if availability.available else f'pip install "watermark-remover[{extra}]"'),
                )
                for extra, availability in ((name, check_optional(name)) for name in KNOWN_EXTRAS)
            ]
        rows: list[tuple[str, ...]] = [row for row, _ in self._extra_rows]
        commands: list[str] = [command for _, command in self._extra_rows]
        # A half-typed number must not blank the capability pane: fall back to
        # the last valid request so the endpoint row still says something true.
        try:
            request = self.collect_request() if self.is_mounted else self.request
        except ValueError:
            request = self.request
        policy = classify_endpoint(
            request.rewrite_base_url, allow_remote=request.rewrite_allow_remote
        )
        rows.append(
            (
                "layer B endpoint",
                "allowed" if policy.allowed else "blocked",
                policy.reason or f"host {policy.host or '-'} (loopback={policy.loopback})",
            )
        )
        commands.append("")
        rows.append(
            (
                "layer B api key",
                "set" if os.environ.get(REWRITE_ENV_KEY) else "not set",
                f"{REWRITE_ENV_KEY} — never displayed",
            )
        )
        # An export line, not the key: this is the shape of the thing to set,
        # and the value stays somewhere this process never reads it back out.
        commands.append(f"export {REWRITE_ENV_KEY}=...   # set it in your shell, not here")
        if self._last_probe is not None:
            rows.append(
                (
                    f"probe: {escape(self._last_probe.backend)}",
                    "reachable" if self._last_probe.reachable else "unreachable",
                    escape(self._last_probe.summary),
                )
            )
            commands.append("")
        self._install_commands = commands
        self._set_table_rows("#backend-table", rows)

    @on(Button.Pressed, "#btn-probe")
    def _probe_pressed(self) -> None:
        try:
            request = self.collect_request()
        except ValueError as error:
            self._status(f"invalid options: {error}")
            return
        backend = request.rewrite_backend
        if not backend:
            self._status("choose a Layer B backend first")
            return
        self._status(f"probing {backend}…")
        self.probe_worker(backend, request.rewrite_base_url, request.rewrite_allow_remote)

    @work(thread=True, exclusive=True, group="probe")
    def probe_worker(self, backend: str, base_url: str | None, allow_remote: bool) -> None:
        probe = probe_backend(
            backend,
            base_url,
            api_key=os.environ.get(REWRITE_ENV_KEY),
            allow_remote=allow_remote,
        )
        self.call_from_thread(self._apply_probe, probe)

    def _apply_probe(self, probe) -> None:
        self._status(f"{probe.backend}: {probe.summary}")
        if probe.models:
            model_select = self.query_one("#sel-model", Select)
            model_select.set_options([(name, name) for name in probe.models])
        self._last_probe = probe
        self.refresh_backends()

    # -- run ---------------------------------------------------------------

    def action_run(self) -> None:
        if self._clean_running:
            self._status("already running")
            return
        try:
            request = self.collect_request()
        except ValueError as error:
            self._status(f"invalid options: {error}")
            return
        if not request.paths:
            self._status("nothing selected")
            return
        self.run_worker(self._confirm_then_run(request), exclusive=False)

    @on(Button.Pressed, "#btn-run")
    def _run_pressed(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#btn-cancel")
    def _cancel_pressed(self) -> None:
        self._cancel_requested = True
        self._status("stopping after the current file…")

    async def _confirm_then_run(self, request: CleanRequest) -> None:
        """Gate every irreversible or outward-facing aspect before writing."""
        policy = classify_endpoint(
            request.rewrite_base_url, allow_remote=request.rewrite_allow_remote
        )
        if request.rewrite_strength is not None and not policy.loopback and policy.allowed:
            # Render the rewrite path's own warning verbatim: the operator is
            # agreeing to that statement, not to a paraphrase of it.
            approved = await self.push_screen_wait(
                ConfirmModal(
                    "Send document text to a remote endpoint?",
                    f"{policy.warning}\n\nEndpoint host: {policy.host}\n"
                    "The full text of every selected file will be sent there.",
                    "Send anyway",
                )
            )
            if not approved:
                self._status("cancelled: remote endpoint not approved")
                return

        if request.in_place:
            approved = await self.push_screen_wait(
                ConfirmModal(
                    "Overwrite the source files?",
                    f"{len(request.paths)} file(s) will be rewritten in place. "
                    "A .bak backup is created for each.",
                    "Overwrite",
                )
            )
            if not approved:
                self._status("cancelled: in-place not approved")
                return

        if request.strip_semantic_format:
            approved = await self.push_screen_wait(
                ConfirmModal(
                    "Strip semantic formatting?",
                    "Contextual ZWJ, variation selectors, and balanced bidi controls "
                    "are preserved by default because removing them can change how "
                    "text renders or what it means.",
                    "Strip anyway",
                )
            )
            if not approved:
                self._status("cancelled: semantic stripping not approved")
                return

        estimate = estimate_rewrite_seconds(request, len(request.paths))
        if should_confirm_cost(request, len(request.paths)):
            approved = await self.push_screen_wait(
                ConfirmModal(
                    "This run calls a model repeatedly",
                    f"{len(request.paths)} file(s) run sequentially.\n"
                    f"Worst case: {format_duration(estimate)} "
                    f"(files x calls x {request.rewrite_timeout or DEFAULT_REWRITE_TIMEOUT:.0f}s "
                    "timeout).\nCancelling stops after the current file; an in-flight "
                    "model call cannot be interrupted.",
                    "Run",
                )
            )
            if not approved:
                self._status("cancelled: cost not approved")
                return

        self.clean_worker(request)

    @work(thread=True, exclusive=True, group="clean")
    def clean_worker(self, request: CleanRequest) -> None:
        self._cancel_requested = False
        self.call_from_thread(self._set_running, True)
        self.call_from_thread(self._set_table_rows, "#run-table", [])

        batch = len(request.paths) > 1
        # Preflight every destination before the first write: a failing plan
        # aborts the whole run rather than cleaning the first N files.
        try:
            # The selection is already resolved to explicit files, but it still
            # goes through select_inputs so the TUI inherits the CLI's symlink
            # and regular-file refusals rather than reimplementing them.
            selection = select_inputs(
                request.paths,
                recursive=request.recursive,
                pattern=request.glob,
                extensions=request.allowed_extensions(SUPPORTED_EXTENSIONS),
            )
            work_items = plan_work(selection.items, request, batch)
        except CleanPlanPreflightError as error:
            self._fail_preflight(
                f"preflight failed on {escape(str(error.path))}: {escape(str(error))}"
            )
            return
        except Exception as error:
            # Anything raised before the first write means nothing was written;
            # surface it rather than letting the worker die silently.
            self._fail_preflight(f"preflight failed: {escape(str(error))}")
            return

        if request.dry_run:
            # Mirrors clean_file.main: a dry run describes and returns before
            # any directory is created or any byte is written.
            self.call_from_thread(self._render_dry_run, request, work_items)
            return

        if batch and request.output and not request.in_place:
            request.output.mkdir(parents=True, exist_ok=True)

        # json=True keeps run_clean_item silent so this UI renders the payload
        # itself instead of the CLI printing over the terminal.
        silent = replace(request, json=True, quiet=True)
        done = 0
        for item, output, plan in work_items:
            if self._cancel_requested:
                self.call_from_thread(self._log, "[yellow]stopped after the current file[/]")
                break
            before = self._read_text(item.path)
            self.call_from_thread(self._status, f"cleaning {item.path.name}…")
            streaming = plan.text.rewrite_plan is not None
            self.call_from_thread(self._begin_stream, item.path.name if streaming else None)
            payload = run_clean_item(
                item.path,
                output,
                silent,
                plan,
                on_token=self._append_stream if streaming else None,
            )
            done += 1
            self.call_from_thread(self._render_result, request, item.path, payload, before)

        self.call_from_thread(self._set_running, False)
        self.call_from_thread(self._status, f"done: {done} of {len(work_items)} file(s)")
        self.call_from_thread(self._record_history, request)

    def _render_dry_run(self, request: CleanRequest, work_items) -> None:
        """Show what a visible-mark clean would do. Nothing is written."""
        self._begin_stream(None)
        for item, output, plan in work_items:
            payload = dry_run_payload(item.path, output, plan, request.in_place)
            self._add_table_row(
                "#run-table",
                escape(item.path.name),
                "image",
                "dry-run",
                format_badge("V"),
                "-",
                escape(str(payload["output"])),
            )
            self._log(f"[bold]dry-run {escape(str(payload['input']))}[/]")
            for action in payload["actions"]:
                self._log(f"  - {escape(str(action))}")
        self._set_running(False)
        self._status(f"dry run: {len(work_items)} file(s) described, nothing written")
        self._record_history(request)

    def _fail_preflight(self, message: str) -> None:
        self.call_from_thread(self._log, f"[red]{message}[/]")
        self.call_from_thread(self._status, f"{message} — nothing was written")
        self.call_from_thread(self._set_running, False)

    def _set_running(self, running: bool) -> None:
        self._clean_running = running
        self.query_one("#btn-run", Button).disabled = running
        self.query_one("#btn-cancel", Button).disabled = not running

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="surrogateescape")
        except (OSError, UnicodeDecodeError):
            return None

    def _render_result(
        self,
        request: CleanRequest,
        path: Path,
        payload: dict,
        before: str | None,
    ) -> None:
        kind = payload.get("kind", "unknown")
        failed = payload.get("exit_code", 0) != 0
        # The badge follows the layer that did the work, never the outcome.
        layer = (
            "B"
            if request.rewrite_strength and kind == "text"
            else (
                "V"
                if request.visible_requested() and kind == "image"
                else ("A" if kind == "text" else "M")
            )
        )
        note = payload.get("error") or ""
        skipped = payload.get("skipped_text_transforms")
        if skipped:
            note = f"skipped {', '.join(skipped)}"
        self._add_table_row(
            "#run-table",
            escape(path.name),
            escape(str(kind)),
            "error" if failed else "written",
            "-" if failed else result_class_for(layer),
            "yes" if payload.get("residual") else "",
            escape(note),
        )
        if failed:
            self._log(f"[red]{escape(path.name)}: {escape(str(payload.get('error')))}[/]")
            return
        if skipped:
            # Carries the file name, so it is operator-controlled text: escape
            # it or a name like report[1].md is swallowed as a style tag.
            self._log(f"[yellow]{escape(describe_dropped_text_transforms(request, kind, path))}[/]")
        output = payload.get("output")
        if output and before is not None:
            after = self._read_text(Path(output))
            if after is not None and after != before:
                diff = "\n".join(
                    difflib.unified_diff(
                        before.splitlines(),
                        after.splitlines(),
                        fromfile=f"{path.name} (before)",
                        tofile=f"{path.name} (after)",
                        lineterm="",
                        n=2,
                    )
                )
                self.query_one("#diff-view", TextArea).text = diff or "(no line-level change)"
        # Re-inspect: the point of the loop is the measured after-state.
        self._log(self._inspect_block(Path(output), soft_binding=False, label=" (after)"))

    def _record_history(self, request: CleanRequest) -> None:
        summary_parts = [f"{len(request.paths)} file(s)"]
        layers = []
        if request.nfkc or request.aggressive_homoglyphs or request.strip_semantic_format:
            layers.append("A+")
        if request.rewrite_strength:
            layers.append(f"B({request.rewrite_strength})")
        if request.visible_requested():
            layers.append("V")
        if not request.keep_non_ai_metadata:
            layers.append("M")
        if layers:
            summary_parts.append("·".join(layers))
        entry = HistoryEntry(
            when=time.strftime("%H:%M"),
            summary=" · ".join(summary_parts),
            command=request.command_string(),
            request=request,
        )
        self.history.insert(0, entry)
        self._set_table_rows("#history-table", [(item.when, item.summary) for item in self.history])
        self.query_one("#history-copyable", TextArea).text = entry.command

    # -- history -----------------------------------------------------------

    @on(DataTable.RowHighlighted, "#history-table")
    def _history_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.history):
            self.query_one("#history-copyable", TextArea).text = self.history[
                event.cursor_row
            ].command

    @on(Button.Pressed, "#btn-history-copy")
    def _history_copy(self) -> None:
        command = self.query_one("#history-copyable", TextArea).text
        if not command:
            return
        self.copy_to_clipboard(command)
        self._status("copied via OSC 52 — the box below is selectable if it was ignored")

    @on(Button.Pressed, "#btn-history-reuse")
    def _history_reuse(self) -> None:
        row = self.query_one("#history-table", DataTable).cursor_row
        if not (0 <= row < len(self.history)):
            return
        self.apply_request(self.history[row].request)
        self._status("plan repopulated from history")

    def apply_request(self, request: CleanRequest) -> None:
        """Push a saved request back into the Plan widgets.

        Driven by the same table ``collect_request`` reads, so "Reuse" cannot
        quietly drop an option that only one of the two knows about.
        """
        for binding in PLAN_BINDINGS:
            binding.write(self, getattr(request, binding.field))
        self.query_one("#cb-disable-thinking", Checkbox).value = bool(
            request.rewrite_disable_thinking
        )
        self.query_one("#in-box", Input).value = (
            ",".join(str(part) for part in request.visible_box) if request.visible_box else ""
        )
        self._sync_preview()


@dataclass(frozen=True)
class PlanBinding:
    """One Plan widget bound to one ``CleanRequest`` field.

    Declaring the binding once is what keeps reading the form and repopulating
    it symmetric.  Two hand-written lists drift, and the drift is invisible:
    an option that only ``collect_request`` knows about is silently dropped by
    "Reuse", and an option only ``apply_request`` knows about is never read.
    A test asserts every field is bound here or listed as deliberately absent.
    """

    selector: str
    field: str
    kind: str
    default: object = None

    def read(self, app: WatermarkTuiApp) -> object:
        if self.kind == "bool":
            return app.query_one(self.selector, Checkbox).value
        if self.kind == "select":
            return app._selected_value(self.selector) or self.default
        if self.kind == "text":
            return app._value(self.selector) or self.default
        if self.kind == "path":
            raw = app._value(self.selector)
            return Path(raw) if raw else None
        if self.kind in ("int", "float"):
            # ``is None``, not truthiness: 0, 0.0 and a temperature of 0.0 are
            # all values an operator can legitimately mean, and falling back to
            # the default for them silently runs something else.
            value = app._number(self.selector, int if self.kind == "int" else float)
            return self.default if value is None else value
        raise AssertionError(f"unknown binding kind: {self.kind}")

    def write(self, app: WatermarkTuiApp, value: object) -> None:
        if self.kind == "bool":
            app.query_one(self.selector, Checkbox).value = bool(value)
            return
        if self.kind == "select":
            app.query_one(self.selector, Select).value = value if value is not None else Select.NULL
            return
        app.query_one(self.selector, Input).value = "" if value is None else str(value)


#: Every ``CleanRequest`` field the Plan pane owns.  Fields absent from this
#: table are listed in ``UNBOUND_REQUEST_FIELDS`` with the reason.
PLAN_BINDINGS: tuple[PlanBinding, ...] = (
    # Routing
    PlanBinding("#sel-force-type", "force_type", "select", "auto"),
    PlanBinding("#cb-force-text", "force_text", "bool"),
    PlanBinding("#in-audit", "audit", "text"),
    # Layer A
    PlanBinding("#cb-nfkc", "nfkc", "bool"),
    PlanBinding("#cb-homoglyphs", "aggressive_homoglyphs", "bool"),
    PlanBinding("#cb-semantic", "strip_semantic_format", "bool"),
    # Layer M
    PlanBinding("#cb-keep-meta", "keep_non_ai_metadata", "bool"),
    PlanBinding("#cb-soft", "soft_binding", "bool"),
    # Layer B
    PlanBinding("#sel-rewrite", "rewrite", "select"),
    PlanBinding("#sel-backend", "rewrite_backend", "select"),
    PlanBinding("#in-base-url", "rewrite_base_url", "text"),
    PlanBinding("#in-model", "rewrite_model", "text"),
    PlanBinding("#in-candidates", "rewrite_candidates", "int"),
    PlanBinding("#in-temperature", "rewrite_temperature", "float"),
    PlanBinding("#in-rewrite-timeout", "rewrite_timeout", "float"),
    PlanBinding("#sel-effort", "rewrite_reasoning_effort", "select"),
    PlanBinding("#cb-allow-remote", "rewrite_allow_remote", "bool"),
    PlanBinding("#in-lang", "rewrite_lang", "text"),
    PlanBinding("#in-original-lang", "rewrite_original_lang", "text"),
    PlanBinding("#in-generations", "tsapa_generations", "int", 5),
    PlanBinding("#in-population", "tsapa_population", "int", 12),
    # Character perturbation
    PlanBinding("#cb-perturb", "char_perturb", "bool"),
    PlanBinding("#sel-perturb-mode", "char_mode", "select", "zero-width"),
    PlanBinding("#in-perturb-strength", "char_strength", "float", 0.1),
    PlanBinding("#in-seed", "seed", "int"),
    # Layer V
    PlanBinding("#in-mask", "visible_mask", "path"),
    PlanBinding("#sel-visible-backend", "visible_backend", "select", "texture"),
    PlanBinding("#in-dilate", "dilate", "int"),
    PlanBinding("#in-detect-command", "detect_command", "text"),
    PlanBinding("#in-inpaint-command", "inpaint_command", "text"),
    PlanBinding(
        "#in-visible-prompt",
        "visible_prompt",
        "text",
        "Remove watermark, fill with background",
    ),
    PlanBinding("#in-timeout", "timeout", "float", 1800.0),
    PlanBinding("#sel-quality", "quality", "select", "balanced"),
    PlanBinding("#cb-synthid", "remove_synthid", "bool"),
    PlanBinding("#in-synthid-strength", "synthid_strength", "float", 0.6),
    PlanBinding("#cb-dry-run", "dry_run", "bool"),
    # Image degradation
    PlanBinding("#sel-degrade", "degrade", "select"),
    PlanBinding("#sel-morpho", "morpho", "select"),
    PlanBinding("#in-degrade-strength", "degrade_strength", "float", 0.6),
    PlanBinding("#in-degrade-seed", "degrade_seed", "int"),
    # Output
    PlanBinding("#in-output", "output", "path"),
    PlanBinding("#cb-in-place", "in_place", "bool"),
    PlanBinding("#cb-artifacts", "keep_artifacts", "bool"),
    PlanBinding("#cb-wmct", "wmct_marker", "bool"),
)

#: Fields the Plan pane deliberately does not own, and why.
UNBOUND_REQUEST_FIELDS: dict[str, str] = {
    "paths": "the Files pane's selection",
    "recursive": "the Files pane's filters",
    "glob": "the Files pane's filters",
    "extensions": "the Files pane's filters",
    "visible_box": "parsed from x,y,w,h rather than read straight through",
    "rewrite_disable_thinking": "tri-state: unchecked means unset, not False",
    "tsapa": "an alias for --rewrite tsapa, which the strength picker carries",
    "rewrite_api_key": "read from the environment; never rendered or persisted",
    "json": "CLI presentation; the TUI renders payloads itself",
    "quiet": "CLI presentation; the TUI renders payloads itself",
}
