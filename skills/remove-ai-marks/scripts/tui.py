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
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import get_args, get_type_hints

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_kind import SUPPORTED_EXTENSIONS
from batch_inputs import select_inputs
from clean_request import CleanRequest
from common import atomic_write_text
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
    # Character perturbation adds noise to defeat a detector rather than
    # removing a carrier that can be counted afterwards. There is nothing to
    # verify, so it cannot be badged with the layer that strips zero-width.
    "perturb": BEST_EFFORT,
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


def layer_for_result(request: CleanRequest, kind: str) -> str:
    """The layer that did the work on one asset. Follows the work, not the outcome.

    ``CleanRequest.visible_requested`` covers mask, box, dilation and the
    external inpainter, but ``--degrade``, ``--morpho`` and
    ``--remove-synthid`` are pixel-domain operations too: routing them to the
    metadata layer badged a frequency-domain perturbation *Verifiable*, which
    is exactly the claim this project does not make.
    """
    if kind == "text":
        if request.rewrite_strength:
            return "B"
        return "perturb" if request.char_perturb else "A"
    if kind == "image":
        if request.visible_requested() or request.degrade or request.morpho:
            return "V"
        if request.remove_synthid:
            return "synthid"
    return "M"


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


# --- presets -----------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """One named starting point for a clean.

    A preset is a claim, not just a shortcut.  Choosing "Deep clean" is
    choosing a best-effort result, so the result class is part of the label
    the operator reads *before* running — not something they only learn from
    the results table afterwards.
    """

    key: str
    label: str
    description: str
    #: The weakest layer this preset turns on.  A preset is only as verifiable
    #: as its least verifiable step, so this is the honest badge for the whole
    #: thing: adding a Layer B rewrite to a Layer A clean makes it best-effort.
    layer: str
    overrides: dict[str, object]
    #: True when the preset cannot run without a reachable Layer B endpoint.
    requires_endpoint: bool = False
    #: The extra this preset needs installed, if any.
    requires_extra: str | None = None

    def badge(self) -> str:
        return format_badge(self.layer)

    def headline(self) -> str:
        """Label and result class together, for the point of choice."""
        return f"{self.label} — {result_class_for(self.layer)}"


#: Every field a preset is allowed to set.  Each preset assigns all of them, so
#: switching presets replaces the previous choice instead of layering on top of
#: it — a half-applied preset runs something nobody selected.
PRESET_FIELDS: tuple[str, ...] = (
    "nfkc",
    "aggressive_homoglyphs",
    "keep_non_ai_metadata",
    "rewrite",
    "char_perturb",
    "remove_synthid",
    "degrade",
    "morpho",
)

#: Fields a preset must never set.  Each one either overwrites the operator's
#: input, changes what the text means, or turns the run into a description
#: instead of a clean.  They are deliberate, per-run decisions with their own
#: confirmation gates; a one-click convenience control must not reach for them.
PRESET_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "in_place",
    "strip_semantic_format",
    "dry_run",
)

PRESETS: tuple[Preset, ...] = (
    Preset(
        key="hidden",
        label="Hidden marks",
        description=(
            "Zero-width carriers, bidi controls and AI metadata. "
            "Counted before and after — nothing is rephrased. "
            "Identical to a bare `wm FILE`."
        ),
        layer="A",
        overrides={
            "nfkc": False,
            "aggressive_homoglyphs": False,
            "keep_non_ai_metadata": False,
            "rewrite": None,
            "char_perturb": False,
            "remove_synthid": False,
            "degrade": None,
            "morpho": None,
        },
    ),
    Preset(
        key="hidden-aggressive",
        label="Hidden marks, aggressive",
        description=(
            "Adds NFKC normalisation and homoglyph folding: Cyrillic and Greek "
            "look-alikes become ASCII. Can change genuinely mixed-script text."
        ),
        layer="A",
        overrides={
            "nfkc": True,
            "aggressive_homoglyphs": True,
            "keep_non_ai_metadata": False,
            "rewrite": None,
            "char_perturb": False,
            "remove_synthid": False,
            "degrade": None,
            "morpho": None,
        },
    ),
    Preset(
        key="rewrite",
        label="Deep clean (LLM rewrite)",
        description=(
            "Hidden marks, then a local model rephrases the text to break "
            "token-level watermarks. No detector guarantee. Needs an endpoint."
        ),
        layer="B",
        overrides={
            "nfkc": False,
            "aggressive_homoglyphs": False,
            "keep_non_ai_metadata": False,
            "rewrite": "paraphrase",
            "char_perturb": False,
            "remove_synthid": False,
            "degrade": None,
            "morpho": None,
        },
        requires_endpoint=True,
    ),
    Preset(
        key="image",
        label="Images: metadata + degrade",
        description=(
            "Strips C2PA and AI metadata, then perturbs the frequency domain "
            "where invisible image marks live. Best-effort; the pixels change."
        ),
        layer="V",
        overrides={
            "nfkc": False,
            "aggressive_homoglyphs": False,
            "keep_non_ai_metadata": False,
            "rewrite": None,
            "char_perturb": False,
            "remove_synthid": False,
            "degrade": "freq-dct",
            "morpho": None,
        },
    ),
)


def preset_for(key: str | None) -> Preset | None:
    """The preset with this key, or None. An unknown key is never guessed at."""
    for preset in PRESETS:
        if preset.key == key:
            return preset
    return None


def apply_preset(request: CleanRequest, preset: Preset) -> CleanRequest:
    """Return ``request`` with the preset's fields — and only those — applied."""
    return replace(request, **preset.overrides)


# --- persisted setup ---------------------------------------------------------

#: Environment override for the settings file, so a test never touches the
#: real one and an operator can keep per-project setups side by side.
SETTINGS_ENV = "WATERMARKS_TUI_SETTINGS"


@dataclass(frozen=True)
class TuiSettings:
    """The setup wm-tui remembers between runs.

    Deliberately not routed through ``configuration``: that module is the
    shared CLI/server config seam with its own precedence rules, and this is
    one UI's memory of which endpoint you last pointed it at.  The generated
    command still carries every value explicitly, so a command copied out of
    the TUI runs the same way on a machine that has no settings file.

    There is no API key field, and there never will be one.  The key is read
    from the environment at run time and is never rendered, copied or written
    to disk — ``rewrite_api_key`` exists on ``CleanRequest`` and is absent
    here on purpose.
    """

    preset: str | None = None
    rewrite_backend: str | None = None
    rewrite_base_url: str | None = None
    rewrite_model: str | None = None
    rewrite_reasoning_effort: str | None = None
    rewrite_allow_remote: bool | None = None

    def seed(self, request: CleanRequest) -> CleanRequest:
        """Apply the remembered endpoint to a fresh request."""
        remembered = {
            field_name: value
            for field_name, value in asdict(self).items()
            if field_name != "preset" and value is not None
        }
        return replace(request, **remembered)


def settings_path() -> Path:
    """Where the setup file lives, honouring the usual per-platform roots."""
    override = os.environ.get(SETTINGS_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    return root / "watermark-remover" / "tui.json"


def load_settings(path: Path | None = None) -> TuiSettings:
    """Read the setup file. Anything unreadable means "no saved setup".

    Fail-soft on purpose: a corrupt or hand-edited settings file must not stop
    the operator from starting the TUI, and every value in it is a convenience
    with a visible control behind it.
    """
    target = path or settings_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return TuiSettings()
    if not isinstance(raw, dict):
        return TuiSettings()
    hints = get_type_hints(TuiSettings)
    allowed = {
        name: tuple(kind for kind in get_args(hint) if kind is not type(None))
        for name, hint in hints.items()
    }
    # A value of the wrong type is as unusable as an absent key: drop it, or
    # it reaches CleanRequest and fails late in classify_endpoint instead of
    # failing soft here.
    return TuiSettings(
        **{
            key: value
            for key, value in raw.items()
            if key in allowed and isinstance(value, allowed[key])
        }
    )


def save_settings(settings: TuiSettings, path: Path | None = None) -> Path:
    """Write the setup file atomically and return where it went."""
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n")
    return target


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
    settings = load_settings()
    # The saved setup only seeds the endpoint fields. Anything the operator
    # typed on the command line stays exactly as typed.
    request = settings.seed(
        CleanRequest(
            paths=paths,
            recursive=args.recursive,
            glob=args.glob,
            extensions=args.extensions,
        )
    )
    # Imported here, not at module scope: the guard above must be able to print
    # an install hint on a default install where textual is absent.
    from tui_app import WatermarkTuiApp

    WatermarkTuiApp(request, preset=preset_for(settings.preset)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
