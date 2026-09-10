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
from dataclasses import replace
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
from clean_file import run_clean_item
from clean_request import (
    QUALITY_PROFILES,
    REWRITE_CLI_CHOICES,
    CleanPlanPreflightError,
    CleanRequest,
    build_rewrite_plan,
    describe_dropped_text_transforms,
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
    HistoryEntry,
    discover_files,
    estimate_rewrite_seconds,
    format_badge,
    format_duration,
    result_class_for,
    should_confirm_cost,
)

REWRITE_ENV_KEY = "WATERMARKS_REWRITE_API_KEY"


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
    #files-list { height: 1fr; border: round $primary; }
    #inspect-report { height: 1fr; border: round $primary; padding: 1; }
    #command-preview { height: auto; min-height: 3; border: round $accent; padding: 0 1; }
    #command-copyable { height: 5; border: round $panel; }
    #run-table { height: 1fr; }
    #run-log { height: 12; border: round $panel; }
    #backend-table { height: 1fr; }
    #history-table { height: 1fr; }
    #diff-view { height: 14; border: round $panel; }
    .row { height: auto; }
    .row > Static { width: 1fr; height: 3; content-align: left middle; padding: 0 1; }
    .field { width: 32; }
    .wide { width: 1fr; }
    #modal-body {
        width: 84; height: auto; max-height: 90%;
        border: thick $warning; background: $surface; padding: 1 2;
    }
    #modal-title { text-style: bold; }
    #modal-text { padding: 1 0; }
    #modal-buttons { height: auto; align-horizontal: right; }
    #candidate-table { height: 10; }
    #candidate-preview { height: 12; }
    .muted { color: $text-muted; }
    """

    BINDINGS: ClassVar[list] = [
        ("q", "quit", "Quit"),
        ("r", "rescan", "Rescan"),
        ("i", "inspect", "Inspect"),
        ("ctrl+r", "run", "Run"),
    ]

    def __init__(self, request: CleanRequest) -> None:
        super().__init__()
        self.request = request
        self.files: list[Path] = []
        self.selected: list[Path] = []
        self.history: list[HistoryEntry] = []
        self._cancel_requested = False
        # Namespaced deliberately: textual's App owns a private `_running`, and
        # reusing that name silently reads the framework's lifecycle state.
        self._clean_running = False

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-files"):
            with TabPane("Files", id="tab-files"):
                yield from self._compose_files()
            with TabPane("Inspect", id="tab-inspect"):
                yield from self._compose_inspect()
            with TabPane("Plan", id="tab-plan"):
                yield from self._compose_plan()
            with TabPane("Run", id="tab-run"):
                yield from self._compose_run()
            with TabPane("Backends", id="tab-backends"):
                yield from self._compose_backends()
            with TabPane("History", id="tab-history"):
                yield from self._compose_history()
        yield Static("", id="status-bar")
        yield Footer()

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
        yield SelectionList[str](id="files-list")
        yield Static("", id="files-summary", classes="muted")

    def _compose_inspect(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Button("Inspect selected", variant="primary", id="btn-inspect")
            yield Checkbox("soft binding", False, id="cb-inspect-soft")
        yield VerticalScroll(Static("", id="inspect-report"))

    def _compose_plan(self) -> ComposeResult:
        with VerticalScroll():
            yield Label("Layer A — hidden Unicode  " + format_badge("A"))
            with Horizontal(classes="row"):
                yield Checkbox("NFKC", self.request.nfkc, id="cb-nfkc")
                yield Checkbox("aggressive homoglyphs", False, id="cb-homoglyphs")
                yield Checkbox("strip semantic format", False, id="cb-semantic")

            yield Label("Layer M — metadata  " + format_badge("M"))
            with Horizontal(classes="row"):
                yield Checkbox("keep non-AI metadata", False, id="cb-keep-meta")
                yield Checkbox("detect soft binding", False, id="cb-soft")

            yield Label("Layer B — LLM rewrite  " + format_badge("B"))
            with Horizontal(classes="row"):
                yield Select(
                    [(name, name) for name in REWRITE_CLI_CHOICES],
                    prompt="strength (off)",
                    allow_blank=True,
                    id="sel-rewrite",
                    classes="field",
                )
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
            with Horizontal(classes="row"):
                # Free text, not a discovery-only picker: an endpoint that does
                # not list models (or is not reachable yet) must still be usable.
                yield Input(placeholder="model", id="in-model", classes="field")
                yield Select(
                    [], prompt="discovered", allow_blank=True, id="sel-model", classes="field"
                )
                yield Button("Discover models", id="btn-discover")
                yield Input(
                    placeholder="candidates", value="1", id="in-candidates", classes="field"
                )
            with Horizontal(classes="row"):
                yield Input(placeholder="temperature", id="in-temperature", classes="field")
                yield Input(placeholder="timeout s", id="in-rewrite-timeout", classes="field")
                yield Select(
                    [(name, name) for name in REASONING_EFFORTS],
                    prompt="reasoning effort",
                    allow_blank=True,
                    id="sel-effort",
                    classes="field",
                )
            with Horizontal(classes="row"):
                yield Checkbox("disable thinking", False, id="cb-disable-thinking")
                yield Checkbox("allow remote endpoint", False, id="cb-allow-remote")
                yield Static("", id="endpoint-state", classes="muted")
            with Horizontal(classes="row"):
                yield Input(placeholder="pivot lang", id="in-lang", classes="field")
                yield Input(placeholder="tsapa generations", id="in-generations", classes="field")
                yield Input(placeholder="tsapa population", id="in-population", classes="field")
            with Horizontal(classes="row"):
                yield Button("Pick from candidates", id="btn-candidates")
                yield Static(
                    "generates N candidates for one text file and lets you choose",
                    classes="muted",
                )
            yield Static("", id="api-key-state", classes="muted")

            yield Label("Character perturbation")
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

            yield Label("Layer V — visible marks  " + format_badge("V"))
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
            with Horizontal(classes="row"):
                yield Select(
                    [(name, name) for name in QUALITY_PROFILES],
                    prompt="quality",
                    allow_blank=True,
                    id="sel-quality",
                    classes="field",
                )
                yield Checkbox("remove SynthID", False, id="cb-synthid")
                yield Static("", id="visible-state", classes="muted")

            yield Label("Output")
            with Horizontal(classes="row"):
                yield Input(placeholder="output path or directory", id="in-output", classes="wide")
                yield Checkbox("in place", False, id="cb-in-place")
                yield Checkbox("keep artifacts", False, id="cb-artifacts")

            yield Label("Equivalent command")
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
        yield DataTable(id="run-table")
        yield TextArea("", read_only=True, id="diff-view")
        yield RichLog(id="run-log", markup=True, wrap=True)

    def _compose_backends(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Button("Refresh", id="btn-refresh-backends")
            yield Button("Probe Layer B endpoint", id="btn-probe")
        yield DataTable(id="backend-table")

    def _compose_history(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Button("Copy", id="btn-history-copy")
            yield Button("Reuse", id="btn-history-reuse")
            yield Static("in-memory, this session only", classes="muted")
        yield DataTable(id="history-table", cursor_type="row")
        yield TextArea("", read_only=True, id="history-copyable")

    # -- lifecycle ---------------------------------------------------------

    def on_mount(self) -> None:
        self.query_one("#run-table", DataTable).add_columns(
            "file", "kind", "result", "class", "residual", "note"
        )
        self.query_one("#backend-table", DataTable).add_columns("capability", "state", "detail")
        self.query_one("#history-table", DataTable).add_columns("time", "summary")
        self._refresh_api_key_state()
        self.action_rescan()
        self.refresh_backends()
        self._sync_preview()

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
            self._sync_preview()
            return
        self.files = files
        listing.add_options([(str(path), str(path), True) for path in files])
        self.selected = list(files)
        self._update_files_summary()
        self._sync_preview()

    def _update_files_summary(self) -> None:
        filters = [f"glob {self.request.glob}"]
        if self.request.recursive:
            filters.append("recursive")
        if self.request.extensions:
            filters.append(f"ext {self.request.extensions}")
        self.query_one("#files-summary", Static).update(
            f"{len(self.selected)} of {len(self.files)} selected · " + " · ".join(filters)
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
        raw = self._value(widget_id)
        if raw is None:
            return None
        try:
            return cast(raw)
        except ValueError:
            self._status(f"{widget_id.lstrip('#in-')}: not a number ({raw})")
            return None

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
        box_raw = self._value("#in-box")
        box = None
        if box_raw:
            try:
                parts = tuple(int(part) for part in box_raw.split(","))
                box = parts if len(parts) == 4 else None
            except ValueError:
                box = None
            if box is None:
                self._status("box must be x,y,w,h")

        mask_raw = self._value("#in-mask")
        output_raw = self._value("#in-output")
        perturb_mode = self._selected_value("#sel-perturb-mode")
        visible_backend = self._selected_value("#sel-visible-backend") or "texture"

        return replace(
            self.request,
            paths=tuple(self.selected),
            output=Path(output_raw) if output_raw else None,
            in_place=self.query_one("#cb-in-place", Checkbox).value,
            nfkc=self.query_one("#cb-nfkc", Checkbox).value,
            aggressive_homoglyphs=self.query_one("#cb-homoglyphs", Checkbox).value,
            strip_semantic_format=self.query_one("#cb-semantic", Checkbox).value,
            keep_non_ai_metadata=self.query_one("#cb-keep-meta", Checkbox).value,
            soft_binding=self.query_one("#cb-soft", Checkbox).value,
            rewrite=self._selected_value("#sel-rewrite"),
            rewrite_backend=self._selected_value("#sel-backend"),
            rewrite_model=self._value("#in-model"),
            rewrite_base_url=self._value("#in-base-url"),
            rewrite_lang=self._value("#in-lang"),
            rewrite_timeout=self._number("#in-rewrite-timeout", float),
            rewrite_temperature=self._number("#in-temperature", float),
            rewrite_candidates=self._number("#in-candidates", int),
            rewrite_reasoning_effort=self._selected_value("#sel-effort"),
            rewrite_disable_thinking=(
                True if self.query_one("#cb-disable-thinking", Checkbox).value else None
            ),
            rewrite_allow_remote=self.query_one("#cb-allow-remote", Checkbox).value,
            tsapa=False,
            tsapa_generations=self._number("#in-generations", int) or 5,
            tsapa_population=self._number("#in-population", int) or 12,
            char_perturb=self.query_one("#cb-perturb", Checkbox).value,
            char_mode=perturb_mode or "zero-width",
            char_strength=self._number("#in-perturb-strength", float) or 0.1,
            visible_mask=Path(mask_raw) if mask_raw else None,
            visible_box=box,
            visible_backend=visible_backend,
            quality=self._selected_value("#sel-quality") or "balanced",
            remove_synthid=self.query_one("#cb-synthid", Checkbox).value,
            keep_artifacts=self.query_one("#cb-artifacts", Checkbox).value,
        )

    @on(Input.Changed)
    @on(Select.Changed)
    @on(Checkbox.Changed)
    def _any_change(self) -> None:
        self._sync_preview()

    def _sync_preview(self) -> None:
        try:
            request = self.collect_request()
        except ValueError as error:
            self.query_one("#command-preview", Static).update(f"[red]{escape(str(error))}[/]")
            return
        command = " ".join(request.command_line())
        self.query_one("#command-preview", Static).update(escape(command))
        # The selectable copy of the command is the fallback for terminals that
        # ignore OSC 52 — it must always carry the same string as the button.
        self.query_one("#command-copyable", TextArea).text = command
        self._sync_endpoint_state(request)
        self._sync_visible_state(request)

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
        request = self.collect_request()
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
            return generate_candidates(text, plan)
        except Exception as error:
            self.call_from_thread(self._status, f"candidate generation failed: {error}")
            return []

    # -- backends ----------------------------------------------------------

    @on(Button.Pressed, "#btn-refresh-backends")
    def _refresh_pressed(self) -> None:
        self.refresh_backends()

    def refresh_backends(self) -> None:
        table = self.query_one("#backend-table", DataTable)
        table.clear()
        for extra in KNOWN_EXTRAS:
            availability = check_optional(extra)
            table.add_row(
                f"extra: {extra}",
                "available" if availability.available else "missing",
                availability.hint,
            )
        request = self.collect_request() if self.is_mounted else self.request
        policy = classify_endpoint(
            request.rewrite_base_url, allow_remote=request.rewrite_allow_remote
        )
        table.add_row(
            "layer B endpoint",
            "allowed" if policy.allowed else "blocked",
            policy.reason or f"host {policy.host or '-'} (loopback={policy.loopback})",
        )
        table.add_row(
            "layer B api key",
            "set" if os.environ.get(REWRITE_ENV_KEY) else "not set",
            f"{REWRITE_ENV_KEY} — never displayed",
        )

    @on(Button.Pressed, "#btn-probe")
    @on(Button.Pressed, "#btn-discover")
    def _probe_pressed(self) -> None:
        request = self.collect_request()
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
        table = self.query_one("#backend-table", DataTable)
        table.add_row(
            f"probe: {probe.backend}",
            "reachable" if probe.reachable else "unreachable",
            probe.summary,
        )

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
        table_clear = self.query_one("#run-table", DataTable).clear
        self.call_from_thread(table_clear)

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
            payload = run_clean_item(item.path, output, silent, plan)
            done += 1
            self.call_from_thread(self._render_result, request, item.path, payload, before)

        self.call_from_thread(self._set_running, False)
        self.call_from_thread(self._status, f"done: {done} of {len(work_items)} file(s)")
        self.call_from_thread(self._record_history, request)

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
        self.query_one("#run-table", DataTable).add_row(
            path.name,
            kind,
            "error" if failed else "written",
            "-" if failed else result_class_for(layer),
            "yes" if payload.get("residual") else "",
            note,
        )
        if failed:
            self._log(f"[red]{escape(path.name)}: {escape(str(payload.get('error')))}[/]")
            return
        if skipped:
            self._log(f"[yellow]{describe_dropped_text_transforms(request, kind, path)}[/]")
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
            command=" ".join(request.command_line()),
            request=request,
        )
        self.history.insert(0, entry)
        table = self.query_one("#history-table", DataTable)
        table.clear()
        for item in self.history:
            table.add_row(item.when, item.summary)
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
        """Push a saved request back into the Plan widgets."""
        self.query_one("#cb-nfkc", Checkbox).value = request.nfkc
        self.query_one("#cb-homoglyphs", Checkbox).value = request.aggressive_homoglyphs
        self.query_one("#cb-semantic", Checkbox).value = request.strip_semantic_format
        self.query_one("#cb-keep-meta", Checkbox).value = request.keep_non_ai_metadata
        self.query_one("#cb-soft", Checkbox).value = request.soft_binding
        self.query_one("#cb-perturb", Checkbox).value = request.char_perturb
        self.query_one("#cb-synthid", Checkbox).value = request.remove_synthid
        self.query_one("#cb-in-place", Checkbox).value = request.in_place
        self.query_one("#cb-artifacts", Checkbox).value = request.keep_artifacts
        self.query_one("#cb-allow-remote", Checkbox).value = request.rewrite_allow_remote
        self.query_one("#in-base-url", Input).value = request.rewrite_base_url or ""
        self.query_one("#in-output", Input).value = str(request.output or "")
        self.query_one("#in-candidates", Input).value = str(request.rewrite_candidates or "")
        self.query_one("#in-model", Input).value = request.rewrite_model or ""
        self.query_one("#sel-rewrite", Select).value = request.rewrite or Select.NULL
        self.query_one("#sel-backend", Select).value = request.rewrite_backend or Select.NULL
        self._sync_preview()
