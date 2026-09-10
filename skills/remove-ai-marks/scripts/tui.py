#!/usr/bin/env python3
"""wm-tui — interactive terminal UI over the watermark-remover pipeline.

The loop this exists for is inspect -> clean -> re-inspect, with Layer A's exact
removal counts sitting next to Layer B's detector scores so "cleanly" is a
measured before/after rather than a claim.

Design rules this module is held to (see ``tasks/tui-plan.md``):

* It never builds a ``CleanPlan`` itself.  Every run fills a ``CleanRequest``
  and goes through ``clean_request.plan_work`` and ``clean_file.run_clean_item``
  — the same seam, and the same refusals, as the CLI.
* It never speaks HTTP.  Rewrites go through ``rewrite_text``; model discovery
  goes through ``layer_b_discovery``, which goes through ``layer_b_http``.  There
  is no ``urllib`` import in this file, and a test enforces that.
* It never renders a best-effort result as verified.  Layer A and Layer M are
  Verifiable, Layer B and Layer V are Best-effort, soft binding is
  Detection-only, and the badge follows the layer, not the outcome.
* It never displays or persists an API key.

This module is deliberately one file: ``[tool.setuptools] packages`` is an
explicit list, so a subpackage would silently not ship in the wheel.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_kind import SUPPORTED_EXTENSIONS
from batch_inputs import select_inputs
from clean_request import CleanRequest
from optional_deps import check_optional

TUI_EXTRA = "tui"

#: Result classes from CONTEXT.md.  The badge follows the *layer*, never the
#: outcome: a Layer B rewrite that "worked" is still best-effort.
VERIFIABLE = "Verifiable"
BEST_EFFORT = "Best-effort"
DETECTION_ONLY = "Detection-only"

LAYER_RESULT_CLASS: dict[str, str] = {
    "A": VERIFIABLE,
    "M": VERIFIABLE,
    "B": BEST_EFFORT,
    "V": BEST_EFFORT,
    "soft-binding": DETECTION_ONLY,
    "synthid": BEST_EFFORT,
}

RESULT_CLASS_STYLE = {
    VERIFIABLE: "bold green",
    BEST_EFFORT: "bold yellow",
    DETECTION_ONLY: "bold cyan",
}

#: Default Layer B timeout used for the batch cost estimate when none is set.
DEFAULT_REWRITE_TIMEOUT = 120.0

#: Border title for the Layer B stream pane when no rewrite is running.  An
#: always-visible box that only fills on one code path reads as broken, so it
#: says why it is empty rather than hiding.
IDLE_STREAM_TITLE = "Layer B stream — no live rewrite in this plan"

#: Characters kept in the live Layer B stream view.  A rewrite of a long
#: document would otherwise grow the widget without bound while it runs.
STREAM_VIEW_CHARS = 8000

#: Worst-case seconds above which a run is worth stopping to confirm.  A single
#: file with one candidate sits under this; a batch, or any TSAPA search, does
#: not.  A gate that fires on every rewrite is a gate nobody reads.
COST_CONFIRM_SECONDS = 300.0


def result_class_for(layer: str) -> str:
    """The honesty label for a layer. Unknown layers are never called verified."""
    return LAYER_RESULT_CLASS.get(layer, BEST_EFFORT)


def format_badge(layer: str) -> str:
    """Rich markup badge naming the layer's result class."""
    label = result_class_for(layer)
    return f"[{RESULT_CLASS_STYLE[label]}]{label}[/]"


def estimate_rewrite_seconds(request: CleanRequest, file_count: int) -> float:
    """Worst-case wall clock for a batch that runs a live rewrite.

    Sequential execution (matching ``clean_file.main``) is what makes this
    honest: files x candidates x per-call timeout is a real ceiling, not an
    optimistic one.
    """
    if request.rewrite_strength is None or file_count <= 0:
        return 0.0
    timeout = request.rewrite_timeout or DEFAULT_REWRITE_TIMEOUT
    candidates = max(1, request.rewrite_candidates or 1)
    if request.rewrite_strength == "tsapa":
        # TSAPA issues roughly population calls per generation, per file.
        calls = max(1, request.tsapa_generations) * max(2, request.tsapa_population)
    else:
        calls = candidates
    return float(file_count) * calls * timeout


def should_confirm_cost(request: CleanRequest, file_count: int) -> bool:
    """Whether this run is expensive enough to stop and confirm."""
    return estimate_rewrite_seconds(request, file_count) > COST_CONFIRM_SECONDS


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


@dataclass
class HistoryEntry:
    """One command this session generated, kept in memory only.

    Persisting history would turn it into a preset store and put the
    secret-serialization question on the table; it stays in memory in v1.
    """

    when: str
    summary: str
    command: str
    request: CleanRequest = field(repr=False)


def discover_files(request: CleanRequest) -> tuple[list[Path], str | None]:
    """Resolve the request's selection through the CLI's own input selector."""
    if not request.paths:
        return [], None
    try:
        selection = select_inputs(
            request.paths,
            recursive=request.recursive,
            pattern=request.glob,
            extensions=request.allowed_extensions(SUPPORTED_EXTENSIONS),
        )
    except ValueError as error:
        return [], str(error)
    return [item.path for item in selection.items], None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wm-tui",
        description="Interactive terminal UI for watermark-remover.",
    )
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        help="File(s) or director(ies) to open. Defaults to the current directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into directories when listing files",
    )
    parser.add_argument("--glob", default="*", help="Directory glob for the file list")
    parser.add_argument("--extensions", default=None, help="Comma-separated extension allow-list")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    availability = check_optional(TUI_EXTRA)
    if not availability.available:
        print(availability.hint, file=sys.stderr)
        return 2

    paths = tuple(args.path) if args.path else (Path.cwd(),)
    request = CleanRequest(
        paths=paths,
        recursive=args.recursive,
        glob=args.glob,
        extensions=args.extensions,
    )
    # Imported here, not at module scope: the guard above must be able to print
    # an install hint on a default install where textual is absent.
    from tui_app import WatermarkTuiApp

    WatermarkTuiApp(request).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
