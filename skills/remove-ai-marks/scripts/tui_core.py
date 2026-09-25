#!/usr/bin/env python3
"""Pure decision logic behind ``wm-tui``, shared by the bridge and its tests.

``wm-tui`` is a TypeScript frontend over a Python bridge (see
``tui/PROTOCOL.md``).  The frontend only presents; every decision about a clean
is made here, in Python, next to the pipeline it describes.  Keeping the logic
in a module with no I/O loop of its own is what lets it be unit-tested without
spawning anything.

Design rules this module is held to:

* A request is only ever built through ``clean_file._build_parser`` and
  ``CleanRequest.from_args``.  Presets are flag lists for that reason: a preset
  is exactly what a person could type after ``wm``, so the command preview is
  the command that runs.  Plans come from ``clean_request.plan_work``; this
  module never assembles one itself.
* It never speaks HTTP.  Discovery goes through ``layer_b_discovery``, which
  goes through ``layer_b_http``, so every probe inherits that module's
  hardening.  A test enforces that no HTTP library is imported here.
* It never labels a best-effort result as verified.  The badge follows the
  layer that did the work, never the outcome.
* It never holds an API key in anything it returns.  The key is read from the
  environment only when a run starts.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import sys
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_inputs import InputItem, InputSelection
from clean_asset import CleanPlan
from clean_file import _build_parser as _build_clean_parser
from clean_request import (
    CleanPlanPreflightError,
    CleanRequest,
    dropped_text_transforms,
    plan_work,
)
from common import atomic_write_text, looks_binary
from layer_b_discovery import MODEL_ROUTES, EndpointPolicy, classify_endpoint, probe_backend
from optional_deps import KNOWN_EXTRAS, check_optional
from pipeline_actions import ActionCode, Effect
from rewrite_text import DEFAULT_BASE_URL, LIVE_REWRITE_BACKENDS, RewritePlan, resolve_base_url

#: The environment variable the key is read from at run time, and only then.
API_KEY_ENV = "WATERMARKS_REWRITE_API_KEY"

# --- result classes ----------------------------------------------------------

#: Result classes from CONTEXT.md.  The badge follows the *layer*, never the
#: outcome: a Layer B rewrite that "worked" is still best-effort.
VERIFIABLE = "Verifiable"
BEST_EFFORT = "Best-effort"

LAYER_RESULT_CLASS: dict[str, str] = {
    "A": VERIFIABLE,
    "M": VERIFIABLE,
    "B": BEST_EFFORT,
    "V": BEST_EFFORT,
    "synthid": BEST_EFFORT,
    # Character perturbation adds noise to defeat a detector rather than
    # removing a carrier that can be counted afterwards. There is nothing to
    # verify, so it cannot be badged with the layer that strips zero-width.
    "perturb": BEST_EFFORT,
}


def result_class_for(layer: str) -> str:
    """The honesty label for a layer. Unknown layers are never called verified."""
    return LAYER_RESULT_CLASS.get(layer, BEST_EFFORT)


def _changes_pixels(request: CleanRequest) -> bool:
    """Whether the request edits image pixels rather than image metadata.

    ``CleanRequest.visible_requested`` covers mask, box, dilation and the
    external inpainter; ``--degrade`` and ``--morpho`` are pixel-domain
    operations too, and badging them as metadata work would call a
    frequency-domain perturbation *Verifiable*.
    """
    return request.visible_requested() or bool(request.degrade) or bool(request.morpho)


def layer_for_result(request: CleanRequest, kind: str) -> str:
    """The layer that did the work on one asset. Follows the work, not the outcome."""
    if kind == "text":
        if request.rewrite_strength:
            return "B"
        return "perturb" if request.char_perturb else "A"
    if kind == "image":
        if _changes_pixels(request):
            return "V"
        if request.remove_synthid:
            return "synthid"
    return "M"


#: ``layer_for_result``'s layers as a plan reports them (the wire carries A,
#: B or V there), weakest last.
_PLAN_LAYER = {"A": "A", "M": "A", "V": "V", "perturb": "V", "synthid": "V", "B": "B"}
_PLAN_LAYER_ORDER = ("A", "V", "B")


def plan_layer(request: CleanRequest, kinds: Collection[str] = ()) -> str:
    """The weakest layer a whole plan turns on.

    A plan is only as verifiable as its least verifiable step, so this is the
    honest badge to show *before* a run: adding a rewrite to a Layer A clean
    makes the whole thing best-effort.  Given the resolved kinds of the
    selected files, only the work those files will get counts, so a rewrite
    over Markdown files alone (which it skips) does not make the plan
    best-effort.  Without kinds, every step the request asks for counts.
    """
    layers = {_PLAN_LAYER[layer_for_result(request, kind)] for kind in kinds or ("text", "image")}
    return max(layers, key=_PLAN_LAYER_ORDER.index)


# --- the batch cost gate -----------------------------------------------------

#: ``RewritePlan``'s own defaults, so the estimate uses the values a run would.
_REWRITE_DEFAULTS: dict[str, Any] = {item.name: item.default for item in fields(RewritePlan)}

#: Worst-case seconds above which a run is worth stopping to confirm.  A single
#: file with one candidate sits under this; a batch, or any TSAPA search, does
#: not.  A gate that fires on every rewrite is a gate nobody reads.
COST_CONFIRM_SECONDS = 300.0


def _rewrite_timeout(request: CleanRequest) -> float:
    return request.rewrite_timeout or _REWRITE_DEFAULTS["timeout"]


def estimate_rewrite_seconds(request: CleanRequest, file_count: int) -> float:
    """Worst-case wall clock for a batch that runs a live rewrite.

    Sequential execution (matching ``clean_file.main``) is what makes this
    honest: files x candidates x per-call timeout is a real ceiling, not an
    optimistic one.
    """
    if request.rewrite_strength is None or file_count <= 0:
        return 0.0
    if request.rewrite_strength == "tsapa":
        # TSAPA issues roughly population calls per generation, per file.
        calls = max(1, request.tsapa_generations) * max(2, request.tsapa_population)
    else:
        calls = max(1, request.rewrite_candidates or _REWRITE_DEFAULTS["candidates"])
    return float(file_count) * calls * _rewrite_timeout(request)


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# --- presets -----------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """One named starting point for a clean, expressed as ``wm`` flags.

    A preset is a claim, not just a shortcut.  Choosing "Deep clean" is
    choosing a best-effort result, so the result class travels with the preset
    and the operator reads it *before* running.

    The flags are CLI flags rather than request fields so a preset can only do
    what a person could type, and so the command preview shows it verbatim.
    A preset never carries a flag that has its own confirmation gate.
    """

    key: str
    label: str
    description: str
    #: The weakest layer this preset turns on.
    layer: str
    flags: tuple[str, ...]
    #: True when the preset cannot run without a reachable Layer B endpoint.
    requires_endpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "layer": self.layer,
            "result_class": result_class_for(self.layer),
            "flags": list(self.flags),
            "requires_endpoint": self.requires_endpoint,
        }


PRESETS: tuple[Preset, ...] = (
    Preset(
        key="hidden",
        label="Hidden marks",
        description=(
            "Zero-width carriers, bidi controls and AI metadata. "
            "Counted before and after; nothing is rephrased. "
            "Identical to a bare `wm FILE`."
        ),
        layer="A",
        flags=(),
    ),
    Preset(
        key="hidden-aggressive",
        label="Hidden marks, aggressive",
        description=(
            "Adds NFKC normalisation and homoglyph folding: Cyrillic and Greek "
            "look-alikes become ASCII. Can change genuinely mixed-script text."
        ),
        layer="A",
        flags=("--nfkc", "--aggressive-homoglyphs"),
    ),
    Preset(
        key="rewrite",
        label="Deep clean (LLM rewrite)",
        description=(
            "Hidden marks, then a local model rephrases the text to break "
            "token-level watermarks. No detector guarantee. Needs an endpoint."
        ),
        layer="B",
        flags=("--rewrite", "paraphrase"),
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
        flags=("--degrade", "freq-dct"),
    ),
)


def preset_for(key: str | None) -> Preset | None:
    """The preset with this key, or None. An unknown key is never guessed at."""
    return next((preset for preset in PRESETS if preset.key == key), None)


# --- composing a request -----------------------------------------------------


class BadRequest(ValueError):
    """The message itself is malformed: wrong types, unknown keys."""


class InvalidOptions(ValueError):
    """argparse or ``CleanRequest`` refused the options. The message is theirs."""


#: Wire endpoint key -> ``CleanRequest`` field and ``tui.json`` key (they are
#: the same names).  An allow-list, so nothing else a client sends (an
#: ``api_key``, say) can be poured into a request or a settings file.
ENDPOINT_FIELDS: dict[str, str] = {
    "backend": "rewrite_backend",
    "base_url": "rewrite_base_url",
    "model": "rewrite_model",
}


@dataclass(frozen=True)
class Composition:
    """A composed request plus the ``wm`` command that is equivalent to it.

    ``request`` has ``json`` and ``quiet`` forced on so ``run_clean_item`` stays
    silent; ``command`` is rendered *before* that, so the preview is what an
    operator would type, not what the bridge needs.  Neither carries a key.
    """

    request: CleanRequest
    command: str


def _string_list(value: object, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BadRequest(f"state.{name} must be a list of strings")
    return list(value)


def _read_state(state: object) -> tuple[list[str], list[str], str | None]:
    """The state's paths, its flags with the preset's in front, and the preset key.

    Only shapes are checked here.  An unknown preset contributes no flags and
    is refused by ``compose_request``, so a refused plan still previews
    everything else that was typed.
    """
    if not isinstance(state, Mapping):
        raise BadRequest("state must be an object")
    paths = _string_list(state.get("paths"), "paths")
    flags = _string_list(state.get("flags"), "flags")
    key = state.get("preset")
    if key is not None and not isinstance(key, str):
        raise BadRequest("state.preset must be a string or null")
    preset = preset_for(key)
    return paths, [*(preset.flags if preset else ()), *flags], key


def _raising_parser() -> argparse.ArgumentParser:
    """``clean_file``'s own parser, made to raise instead of printing and exiting.

    argparse reports a bad flag by printing usage and calling ``sys.exit``, and
    ``--help`` (or any abbreviation of it) prints the whole help first.  In a
    long-lived bridge the exit would kill the process and the text would land
    in the log; the message is what the operator needs, so it is carried out
    as ``InvalidOptions`` instead.
    """
    parser = _build_clean_parser()
    parser.prog = "wm"

    def refuse(message: str) -> NoReturn:
        raise InvalidOptions(message)

    def leave(status: int = 0, message: str | None = None) -> NoReturn:
        refuse(message or "--help is not available here")

    parser.error = refuse  # type: ignore[method-assign]
    parser.exit = leave  # type: ignore[method-assign]
    parser.print_help = lambda file=None: None  # type: ignore[method-assign]
    return parser


def preview_command(state: Mapping[str, object]) -> str:
    """The ``wm ...`` line for a state that did not compose.

    A refused flag should still be visible in the preview, next to the error
    that names it, rather than vanishing.
    """
    paths, flags, _ = _read_state(state)
    return shlex.join(["wm", *paths, *flags])


def compose_request(state: Mapping[str, object], *, allow_remote: bool = False) -> Composition:
    """Compose the ``CleanRequest`` for one frontend ``state`` (PROTOCOL "State").

    1. Parse ``[*preset.flags, *flags, "--", *paths]`` with ``clean_file``'s
       parser, so a later flag wins over an earlier one and any CLI flag is
       valid.  Paths go after ``--`` so a file named ``-x.md`` is a path, not
       an option; argparse reads the same request either way.
    2. Build the request with ``CleanRequest.from_args``.
    3. Apply ``endpoint`` only when the request rewrites, which keeps the
       preview free of endpoint flags that would do nothing.  *allow_remote*
       is applied the same way: it is how ``clean`` passes a per-run
       ``confirmed: ["remote"]`` in, and it is never persisted.
    4. Force ``json`` and ``quiet``.

    The API key is never part of a composition, which is shown and
    remembered: the rewrite reads ``WATERMARKS_REWRITE_API_KEY`` itself, at
    run time, exactly as it does under ``wm``.

    Raises ``BadRequest`` for a malformed state and ``InvalidOptions`` for
    anything argparse or ``CleanRequest`` refuses.
    """
    paths, flags, preset_key = _read_state(state)
    if preset_key is not None and preset_for(preset_key) is None:
        raise InvalidOptions(f"unknown preset: {preset_key}")
    endpoint = state.get("endpoint")
    if endpoint is not None and not isinstance(endpoint, Mapping):
        raise BadRequest("state.endpoint must be an object or null")
    # Validated even when it will not be applied, so a malformed endpoint is
    # reported while the operator is still editing, not first on the run that
    # turns a rewrite on.  An unset field leaves WATERMARKS_REWRITE_* in charge.
    overrides = {
        target: value
        for target, value in _endpoint_values(endpoint or {}, "state.endpoint").items()
        if value is not None
    }

    args = _raising_parser().parse_args([*flags, "--", *paths])
    try:
        request = CleanRequest.from_args(args)
    except (ValueError, TypeError) as error:
        raise InvalidOptions(str(error)) from error

    # Egress permission comes from the per-run confirmation and nowhere else:
    # not a typed --rewrite-allow-remote, and not WATERMARKS_REWRITE_ALLOW_REMOTE
    # (an explicit False outranks the environment in live_from_environment).
    if request.rewrite_strength is not None:
        request = replace(request, **overrides, rewrite_allow_remote=allow_remote)
    else:
        request = replace(request, rewrite_allow_remote=None)

    command = request.command_string()
    return Composition(request=replace(request, json=True, quiet=True), command=command)


def _endpoint_values(endpoint: Mapping[str, object], where: str) -> dict[str, str | None]:
    """Validate an endpoint object and map it onto request/settings fields.

    ``null`` and ``""`` both mean "not stated".  Remote egress is deliberately
    not a field: it is a per-run confirmation.
    """
    unknown = sorted(str(key) for key in endpoint if key not in ENDPOINT_FIELDS)
    if unknown:
        raise BadRequest(f"unknown {where} field(s): {', '.join(unknown)}")
    values: dict[str, str | None] = {}
    for key, target in ENDPOINT_FIELDS.items():
        value = endpoint.get(key)
        if value is not None and not isinstance(value, str):
            raise BadRequest(f"{where}.{key} must be a string or null")
        values[target] = value or None
    backend = values["rewrite_backend"]
    if backend is not None and backend not in LIVE_REWRITE_BACKENDS:
        raise InvalidOptions(
            f"unknown rewrite backend: {backend} (choose from {', '.join(LIVE_REWRITE_BACKENDS)})"
        )
    return values


# --- selection ---------------------------------------------------------------


def preflight_work(
    request: CleanRequest, selection: InputSelection
) -> list[tuple[InputItem, Path | None, CleanPlan]]:
    """Every destination and plan, validated before the first write.

    ``clean_request.plan_work`` with its per-file error folded into one
    ``ValueError`` that names the offending file.
    """
    try:
        return plan_work(selection.items, request, selection.batch)
    except CleanPlanPreflightError as error:
        raise ValueError(f"{error.path}: {error}") from error


def display_path(path: Path) -> str:
    """A path as the operator would type it: relative to cwd when under it."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return str(path)


def describe_output(request: CleanRequest, file_count: int) -> str:
    """Where a run will write, in one line, before anything is written."""
    if request.dry_run:
        return "Dry run: describes the plan and writes nothing."
    if request.in_place:
        return "Overwrites each file in place and keeps a .bak copy. Asks first."
    if request.output is not None:
        if file_count > 1:
            return f"Writes into {request.output}/, keeping each file's relative path."
        return f"Writes to {request.output}."
    return "Writes NAME.cleaned.EXT next to each file. Originals are never touched."


# --- confirmation gates ------------------------------------------------------


def _endpoint_policy(request: CleanRequest) -> EndpointPolicy | None:
    """How the endpoint a rewrite would really contact classifies; None without one.

    Mirrors ``RewritePlan.live_from_environment``'s precedence (explicit value,
    then ``WATERMARKS_REWRITE_BASE_URL``, then default) so the verdict is about
    the endpoint the run will contact, not the one the form happens to show.
    Classified with remote allowed, so an off-machine host reads as a question
    to ask rather than a refusal.
    """
    if request.rewrite_strength is None:
        return None
    return classify_endpoint(resolve_base_url(request.rewrite_base_url), allow_remote=True)


def confirm_gates(request: CleanRequest, file_count: int, text_count: int) -> list[dict[str, str]]:
    """Every gate a clean stops at before the first write.

    One function feeds both ``plan`` (the preview) and ``clean`` (the gate), so
    the two can never disagree about what needs a yes.  *text_count* is how
    many of the *file_count* files read as text: only those are rewritten, so
    only those can leave the machine or cost model time.

    The remote gate fires for any off-machine endpoint, whatever
    ``--rewrite-allow-remote`` or ``WATERMARKS_REWRITE_ALLOW_REMOTE`` say: in
    the TUI, sending a document away is agreed to per run, never standing.
    """
    gates: list[dict[str, str]] = []
    policy = _endpoint_policy(request) if text_count else None
    if policy is not None and policy.allowed and not policy.loopback:
        # The rewrite path's own warning: the operator agrees to that
        # statement, not to a paraphrase of it.
        gates.append(
            {
                "kind": "remote",
                "message": f"{_sentence(str(policy.warning).removeprefix('warning: '))}. "
                f"The full text of {_plural(text_count, 'file')} will be sent to {policy.host}.",
            }
        )
    if request.in_place:
        gates.append(
            {
                "kind": "in_place",
                "message": f"{_plural(file_count, 'file')} will be overwritten in place. "
                "A .bak backup is kept for each.",
            }
        )
    if request.strip_semantic_format:
        gates.append(
            {
                "kind": "semantic",
                "message": "Contextual ZWJ, variation selectors and balanced bidi controls "
                "are preserved by default because removing them can change how text "
                "renders or what it means.",
            }
        )
    estimate = estimate_rewrite_seconds(request, text_count)
    if estimate > COST_CONFIRM_SECONDS:
        gates.append(
            {
                "kind": "cost",
                "message": f"Rewriting {_plural(text_count, 'file')} one after another can "
                f"take up to {format_duration(estimate)} (files x calls x "
                f"{_rewrite_timeout(request):.0f}s timeout). "
                "Cancelling stops after the current file; an in-flight model call "
                "cannot be interrupted.",
            }
        )
    return gates


#: How a non-text kind is named to someone who picked the files.
_NOT_TEXT_LABELS = {"container": "Markdown and other documents", "image": "images"}


def _joined(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def plan_warnings(request: CleanRequest, kinds: Sequence[str] | None) -> list[str]:
    """What the preview should say that no confirmation can change.

    * A malformed or non-http(s) base URL is not a gate (there is nothing to
      agree to) but every rewritten file would fail on it.  An off-machine
      host is not listed here: that is the ``remote`` gate.
    * Text-body transforms are no-ops on files that do not read as text
      (``clean_request.dropped_text_transforms``); without a word here, a
      Markdown file would look rewritten.

    *kinds* are the selected files' resolved kinds, or None when unknown.
    """
    warnings: list[str] = []
    rewrites = kinds is None or "text" in kinds
    policy = _endpoint_policy(request) if rewrites else None
    if policy is not None and not policy.allowed:
        warnings.append(f"Layer B endpoint refused: {policy.reason}.")
    skipping = [kind for kind in kinds or () if dropped_text_transforms(request, kind)]
    if skipping:
        names = dropped_text_transforms(request, skipping[0])
        which = " and ".join(label for kind, label in _NOT_TEXT_LABELS.items() if kind in skipping)
        warnings.append(
            f"{_sentence(_joined(names))} will skip {_plural(len(skipping), 'file')} "
            f"that {'is' if len(skipping) == 1 else 'are'} not plain text ({which}). "
            f"Add --as text to force {'it' if len(names) == 1 else 'them'}."
        )
    return warnings


# --- reading files for display -----------------------------------------------


def read_text(path: Path, max_bytes: int) -> tuple[str, bool] | None:
    """Up to *max_bytes* of *path* decoded as UTF-8, and whether that was all of it.

    None when the file cannot be read or looks binary: offsets or a diff into
    a decoded ZIP or PNG would point at noise.
    """
    try:
        with path.open("rb") as source:
            data = source.read(max_bytes + 1)
    except OSError:
        return None
    head = data[:max_bytes]
    if looks_binary(head) is not None:
        return None
    return head.decode("utf-8", errors="replace"), len(data) <= max_bytes


#: Leading words the pipeline writes in lower case that are names, not words.
_ACRONYMS = {"svg": "SVG", "json-ld": "JSON-LD", "xmp": "XMP", "exif": "EXIF", "opf": "OPF"}


def _sentence(text: str) -> str:
    """*text* opening with a capital, for a line shown on its own.

    Only a leading plain word is capitalised: pipeline lines often open with a
    path or a key (``docProps/core.xml: ...``, ``ai:Claude``), and changing
    its case would name something that does not exist.
    """
    first, space, rest = text.partition(" ")
    word = first.removesuffix(":")
    if word in _ACRONYMS:
        return _ACRONYMS[word] + first[len(word) :] + space + rest
    if word.isalpha() and word.islower():
        return text[:1].upper() + text[1:]
    return text


def _capped(lines: list[str], limit: int, tail: str) -> list[str]:
    """At most *limit* lines; the last says how many were cut (``tail`` gets the count)."""
    if len(lines) <= limit:
        return lines
    return [*lines[: limit - 1], tail.format(len(lines) - (limit - 1))]


# --- inspection findings -----------------------------------------------------

#: Most lines a finding or a result carries.  A file with thousands of distinct
#: carriers would otherwise flood the panel; the counts still hold the totals.
MAX_FINDING_LINES = 12
_MORE_LINES = "{} more not shown."

#: ``text_unicode`` hit kinds, in the operator's words.
HIT_NOUNS: dict[str, str] = {
    "zwj_family": "zero-width carrier",
    "bidi": "bidi control",
    "tag_chars": "tag character",
    "variation_selector": "variation selector",
    "private_use": "private-use character",
    "space": "space homoglyph",
    "confusable": "confusable character",
    "strip": "invisible character",
    "other_cf": "format character",
}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _hits(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Layer A hits: ``hits`` for a text report, ``layer_a_hits`` for a container."""
    hits = report.get("hits") or report.get("layer_a_hits") or []
    return [hit for hit in hits if isinstance(hit, Mapping)]


def _metadata_findings(report: Mapping[str, Any]) -> list[str]:
    """Metadata findings, without the Layer A lines a container also lists there.

    Reports keep informational context in ``notes``, so every finding here is
    a mark (or a scan problem) and may be counted.  ``inspect_container`` also
    lists its Layer A hits among the findings, prefixed ``layer-a``, because the
    audits read them from there; they are counted from ``layer_a_hits`` instead.
    """
    findings = (str(item) for item in report.get("findings") or [])
    return [finding for finding in findings if not finding.startswith("layer-a")]


def _soft_binding(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    soft = (report.get("soft_binding") or {}).get("soft_binding")
    return soft if isinstance(soft, Mapping) else None


def finding_counts(report: Mapping[str, Any]) -> dict[str, int]:
    """Counts by class from an ``inspect_asset`` report.

    ``hidden`` is invisible-Unicode carriers (Layer A); ``metadata`` is
    metadata findings (Layer M).
    """
    hidden = sum(int(hit.get("count", 0)) for hit in _hits(report))
    if not hidden:
        hidden = int(report.get("suspicious_total") or 0)
    counts = {"hidden": hidden, "metadata": len(_metadata_findings(report))}
    if report.get("has_c2pa"):
        counts["c2pa"] = 1
    soft = _soft_binding(report)
    if soft is not None and soft.get("found"):
        counts["soft_binding"] = 1
    return counts


def _hit_line(hit: Mapping[str, Any]) -> str:
    noun = HIT_NOUNS.get(str(hit.get("kind")), "hidden character")
    return f"{_plural(int(hit.get('count', 0)), noun)} ({hit.get('codepoint', '?')})"


def finding_lines(report: Mapping[str, Any]) -> list[str]:
    """Short human labels for one report, at most ``MAX_FINDING_LINES``."""
    lines = [_hit_line(hit) for hit in _hits(report)]
    if report.get("has_c2pa"):
        lines.append("C2PA manifest present")
    lines.extend(_sentence(finding) for finding in _metadata_findings(report))
    soft = _soft_binding(report)
    if soft is not None:
        lines.append("Soft binding found" if soft.get("found") else "No soft binding")
    if report.get("note"):
        lines.append(_sentence(str(report["note"])))
    return _capped(lines, MAX_FINDING_LINES, _MORE_LINES)


def finding_from_report(report: Mapping[str, Any], path: Path, display: str) -> dict[str, Any]:
    """One PROTOCOL ``Finding`` from an ``inspect_asset`` report."""
    return {
        "path": str(path.resolve()),
        "display": display,
        "kind": str(report.get("kind", "unknown")),
        "suspicious": bool(report.get("suspicious")),
        "counts": finding_counts(report),
        "lines": finding_lines(report),
        "reveal": reveal_excerpts(path, report),
        "error": None,
    }


def failed_finding(path: Path, display: str, error: BaseException) -> dict[str, Any]:
    """The ``Finding`` for a file ``inspect_asset`` could not read."""
    return {
        "path": str(path.resolve()),
        "display": display,
        "kind": "unknown",
        "suspicious": False,
        "counts": {},
        "lines": [],
        "reveal": [],
        "error": f"{type(error).__name__}: {error}",
    }


# --- reveal: hidden characters shown in place --------------------------------

#: Most excerpts a finding carries, and the widest one.
MAX_REVEAL_EXCERPTS = 6
REVEAL_WIDTH = 100
#: Largest prefix read to build excerpts.  Past this the counts still hold.
REVEAL_MAX_BYTES = 2 * 1024 * 1024
#: One column per hidden codepoint, so a mark's columns are exact.
REVEAL_GLYPH = "◆"
ELLIPSIS = "…"

#: Short labels for the hidden codepoints an operator is likely to meet.
SHORT_NAMES: dict[int, str] = {
    0x00AD: "SHY",
    0x061C: "ALM",
    0x180E: "MVS",
    0x200B: "ZWSP",
    0x200C: "ZWNJ",
    0x200D: "ZWJ",
    0x200E: "LRM",
    0x200F: "RLM",
    0x202A: "LRE",
    0x202B: "RLE",
    0x202C: "PDF",
    0x202D: "LRO",
    0x202E: "RLO",
    0x2060: "WJ",
    0x2061: "FA",
    0x2062: "IT",
    0x2063: "IS",
    0x2064: "IP",
    0x2066: "LRI",
    0x2067: "RLI",
    0x2068: "FSI",
    0x2069: "PDI",
    0xFE0E: "VS15",
    0xFE0F: "VS16",
    0xFEFF: "BOM",
}


def short_name(codepoint: int) -> str:
    """A short label for a hidden codepoint, or its ``U+XXXX`` hex."""
    if codepoint in SHORT_NAMES:
        return SHORT_NAMES[codepoint]
    if 0xE0000 <= codepoint <= 0xE007F:
        return "TAG"
    if 0xFE00 <= codepoint <= 0xFE0D:
        return f"VS{codepoint - 0xFE00 + 1}"
    if 0xE0100 <= codepoint <= 0xE01EF:
        return f"VS{codepoint - 0xE0100 + 17}"
    return f"U+{codepoint:04X}"


def _is_hidden(char: str, flagged: frozenset[int] = frozenset()) -> bool:
    """Whether a character renders as nothing (or as a blank) where it sits.

    Format, separator and private-use characters are invisible by category,
    and so are variation selectors, whose category says "mark".  ``flagged``
    adds whatever the inspector itself reported, such as Hangul fillers or
    space look-alikes, whose categories say "letter" or "space".
    """
    codepoint = ord(char)
    if codepoint in flagged:
        return True
    if unicodedata.category(char) in ("Cf", "Zl", "Zp", "Co"):
        return True
    return 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF


def _hit_offsets(text: str, report: Mapping[str, Any]) -> tuple[list[int], frozenset[int]]:
    """Offsets of reported hits, each checked against the text it points into.

    The inspector decodes with ``surrogateescape`` and this reads with
    ``replace``; on a file with bad bytes the two can disagree about
    positions, so an offset is used only when the character there is the one
    reported, and a codepoint whose offsets all miss is found by search.
    """
    offsets: set[int] = set()
    flagged: set[int] = set()
    for hit in _hits(report):
        if hit.get("kind") == "confusable":
            continue
        try:
            codepoint = int(str(hit.get("codepoint", "")).removeprefix("U+"), 16)
        except ValueError:
            continue
        flagged.add(codepoint)
        char = chr(codepoint)
        valid = [
            offset
            for offset in hit.get("sample_offsets") or []
            if isinstance(offset, int) and 0 <= offset < len(text) and text[offset] == char
        ]
        if not valid:
            start = text.find(char)
            while start != -1 and len(valid) < 10:
                valid.append(start)
                start = text.find(char, start + 1)
        offsets.update(valid)
    return sorted(offsets), frozenset(flagged)


def _excerpt(line: str, first_mark: int, flagged: frozenset[int]) -> dict[str, Any]:
    """One line with hidden characters as glyphs, trimmed around its first mark."""
    columns: list[str] = []
    marks: list[list[Any]] = []
    for column, char in enumerate(line):
        if _is_hidden(char, flagged):
            columns.append(REVEAL_GLYPH)
            marks.append([column, column + 1, short_name(ord(char))])
        elif char == "\t" or unicodedata.category(char) == "Cc":
            columns.append(" ")
        else:
            columns.append(char)
    start, end = 0, len(columns)
    if len(columns) > REVEAL_WIDTH:
        window = REVEAL_WIDTH - 2  # room for an ellipsis at each cut edge
        start = min(max(0, first_mark - window // 2), len(columns) - window)
        end = start + window
    left = ELLIPSIS if start > 0 else ""
    right = ELLIPSIS if end < len(columns) else ""
    shift = len(left) - start
    return {
        "text": left + "".join(columns[start:end]) + right,
        "marks": [
            [begin + shift, finish + shift, name]
            for begin, finish, name in marks
            if start <= begin and finish <= end
        ],
    }


def reveal_excerpts(path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Up to six excerpts showing where the hidden characters sit.

    A count says "3 zero-width characters"; an excerpt says *which words* they
    are glued to, which is what an operator needs to judge whether a mark is a
    watermark or a legitimate joiner.  Only assets that read as text get
    excerpts.
    """
    read = read_text(path, REVEAL_MAX_BYTES)
    if read is None:
        return []
    text = read[0]
    offsets, flagged = _hit_offsets(text, report)
    excerpts: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for offset in offsets:
        line_start = text.rfind("\n", 0, offset) + 1
        if line_start in seen_lines:
            continue
        seen_lines.add(line_start)
        line_end = text.find("\n", offset)
        line = text[line_start : len(text) if line_end == -1 else line_end].removesuffix("\r")
        excerpt = _excerpt(line, offset - line_start, flagged)
        excerpts.append({"line": text.count("\n", 0, line_start) + 1, **excerpt})
        if len(excerpts) == MAX_REVEAL_EXCERPTS:
            break
    return excerpts


# --- what a clean did --------------------------------------------------------

#: Diff budget per ``file_done``.
DIFF_MAX_LINES = 200
#: Files above this are not diffed: the diff would be truncated to nothing
#: useful, and reading them twice costs more than it tells.
DIFF_MAX_BYTES = 1 << 20


def text_for_diff(path: Path) -> str | None:
    """The whole file as text, or None when it is binary, unreadable or too big."""
    read = read_text(path, DIFF_MAX_BYTES)
    return read[0] if read is not None and read[1] else None


def _visible(line: str) -> str:
    """Show invisible characters in a diff line.

    The whole point of a Layer A diff is a character nobody can see; a diff
    that renders it as nothing shows two identical lines.
    """
    return "".join(
        f"<U+{ord(char):04X}>"
        if _is_hidden(char) or (char != "\t" and unicodedata.category(char) == "Cc")
        else char
        for char in line
    )


def unified_diff(before: str | None, after: str | None, display: str) -> str | None:
    """A text-only diff, invisible characters made visible, at most 200 lines."""
    if before is None or after is None or before == after:
        return None
    lines = [
        _visible(line)
        for line in difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{display} (before)",
            tofile=f"{display} (after)",
            lineterm="",
            n=2,
        )
    ]
    lines = _capped(lines, DIFF_MAX_LINES, "... diff truncated ({} more lines)")
    return "\n".join(lines) if lines else None


def _count(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _layer_a_phrases(removed: int, replaced: int) -> list[str]:
    phrases = []
    if removed:
        phrases.append(f"removed {_plural(removed, 'hidden character')}")
    if replaced:
        phrases.append(f"replaced {_plural(replaced, 'look-alike character')}")
    return phrases


def _because(params: Mapping[str, Any]) -> str:
    return f" ({params['reason']})" if params.get("reason") else ""


def _failure(params: Mapping[str, Any]) -> str:
    """`` (exit code 1): detail`` for a tool failure, from whichever parts it has."""
    code = f" (exit code {params['returncode']})" if params.get("returncode") is not None else ""
    detail = f": {params['detail']}" if params.get("detail") else ""
    return code + detail


def _nested(params: Mapping[str, Any]) -> str:
    """``: dropped X; dropped Y`` for the changing steps of a nested clean."""
    inner = [
        phrase
        for step in params.get("actions") or ()
        if step.get("effect") == Effect.CHANGE.value
        for phrase in action_phrases(step)
    ]
    return f": {'; '.join(inner)}" if inner else ""


def _jpeg_segment(p: Mapping[str, Any]) -> str:
    if p["segment"] == "COM":
        return "dropped the JPEG comment"
    return f"dropped the JPEG {p['segment']} segment{_because(p)}"


def _tiff_tag(p: Mapping[str, Any]) -> str:
    if p.get("name"):
        return f"dropped TIFF tag {p['tag']} ({p['name']})"
    return f"dropped TIFF tag {p['tag']}, which carried AI markers"


def _pdf_metadata(p: Mapping[str, Any]) -> str:
    if p.get("pages"):
        return f"dropped page metadata from {_count(p['pages'], 'page')}"
    return f"dropped the PDF {p['target']}"


def _pdf_rewritten(p: Mapping[str, Any]) -> str:
    if p["tool"] == "qpdf":
        return "rebuilt the PDF structure with qpdf, so old metadata bytes are gone"
    return "rebuilt the PDF with pypdf, without its info dictionary or XMP"


def _gif_extension(p: Mapping[str, Any]) -> str:
    extension = p["extension"]
    if extension == "extension":
        return "dropped a GIF extension block"
    return f"dropped the GIF {extension} extension"


#: One entry per ``pipeline_actions.ActionCode``: the phrases a step reads as,
#: each opening with a lower-case verb written here, so capitalising the first
#: letter never re-cases a path, key or acronym.  An empty result means the
#: step says nothing worth a line (nothing removed, bookkeeping).
_ACTION_PHRASES: dict[str, Callable[[Mapping[str, Any]], Sequence[str]]] = {
    ActionCode.NOTHING_REMOVED: lambda p: (),
    ActionCode.DROP_PNG_CHUNK: lambda p: (f"dropped the PNG {p['chunk']} chunk{_because(p)}",),
    ActionCode.DROP_JPEG_SEGMENT: lambda p: (_jpeg_segment(p),),
    ActionCode.PRESERVE_JPEG_SCAN: lambda p: (),
    ActionCode.DROP_WEBP_CHUNK: lambda p: (f"dropped the WebP {p['chunk'].strip()} chunk",),
    ActionCode.DROP_GIF_EXTENSION: lambda p: (_gif_extension(p),),
    ActionCode.DROP_TIFF_TAG: lambda p: (_tiff_tag(p),),
    ActionCode.DROP_BMP_TRAILER: lambda p: (
        f"dropped {_count(p['bytes'], 'trailing byte')} after the BMP pixels"
        + (f" ({', '.join(p['markers'])})" if p.get("markers") else ""),
    ),
    ActionCode.KEEP_BMP_TRAILER: lambda p: (),
    ActionCode.BMP_UNPARSED: lambda p: (
        "left the BMP unchanged: its header could not be fully parsed",
    ),
    ActionCode.NEUTRALIZE_BOX: lambda p: (
        f"neutralized the '{p['box']}' C2PA box ({_count(p['bytes'], 'byte')} zeroed)",
    ),
    ActionCode.ZERO_PAYLOAD: lambda p: (
        f"zeroed the {p['target']} payload ({_count(p['bytes'], 'byte')})",
    ),
    ActionCode.NEUTRALIZE_TOKENS: lambda p: (
        f"neutralized AI tokens in the {p['target']} ({', '.join(p['tokens'])})",
    ),
    ActionCode.EXIFTOOL_STRIP: lambda p: ("stripped remaining metadata with exiftool",),
    ActionCode.SYNTHID_BAND_REMOVAL: lambda p: (
        f"suppressed the SynthID frequency band at strength {p['strength']}",
    ),
    ActionCode.WMCT_MARKER_WRITTEN: lambda p: (
        "wrote a wmCt marker recording that wm cleaned the file",
    ),
    ActionCode.WMCT_MARKER_SKIPPED: lambda p: (f"skipped the wmCt marker: {p['reason']}",),
    ActionCode.DROP_FRONTMATTER_KEY: lambda p: (
        f"dropped frontmatter key {p['key']}"
        + (" (its value names an AI tool)" if p.get("value_hit") else ""),
    ),
    ActionCode.DROP_EMPTY_FRONTMATTER: lambda p: ("removed the frontmatter block, now empty",),
    ActionCode.CLEAN_DATA_URI: lambda p: (f"cleaned an embedded {p['mime']} image{_nested(p)}",),
    ActionCode.DROP_HTML_META: lambda p: (f"dropped meta tag {p['tag']}",),
    ActionCode.DROP_JSON_LD: lambda p: ("dropped a JSON-LD provenance script",),
    ActionCode.DROP_DATA_AI_ATTRIBUTES: lambda p: (
        f"dropped {_count(p['count'], 'data-ai attribute')}",
    ),
    ActionCode.DROP_SVG_METADATA: lambda p: (
        f"dropped {_count(p['count'], 'SVG metadata block')}",
    ),
    ActionCode.DROP_SVG_XMP: lambda p: (f"dropped {_count(p['count'], 'XMP packet')}",),
    ActionCode.DROP_SVG_COMMENT: lambda p: ("dropped an SVG comment with AI markers",),
    ActionCode.DROP_SVG_GENERATOR_ATTRIBUTES: lambda p: (
        f"dropped {_count(p['count'], 'generator attribute')}",
    ),
    ActionCode.CLEAN_EMBEDDED_MEDIA: lambda p: (f"cleaned embedded image {p['part']}{_nested(p)}",),
    ActionCode.CLEAN_PART: lambda p: tuple(
        f"{phrase} in {p['part']}"
        for step in p.get("actions") or ()
        for phrase in action_phrases(step)
    ),
    ActionCode.DROP_PART: lambda p: (f"dropped part {p['part']}{_because(p)}",),
    ActionCode.SCRUB_FIELD: lambda p: (
        f"cleared {p['field']} in {p['part']}"
        if p.get("part")
        else f"cleared {p['field']}{_because(p)}",
    ),
    ActionCode.DROP_CONTENT_TYPE_OVERRIDES: lambda p: (
        f"dropped {_count(p['count'], p['target'] + ' content-type override')}",
    ),
    ActionCode.PRUNE_RELATIONSHIPS: lambda p: (
        f"pruned {_count(p['count'], 'dangling relationship')} in {p['part']}",
    ),
    ActionCode.PRUNE_MANIFEST: lambda p: (
        "pruned "
        + _count(
            p["count"],
            f"{p['manifest']} manifest entry",
            f"{p['manifest']} manifest entries",
        ),
    ),
    ActionCode.DROP_GENERATOR_META: lambda p: (f"dropped the {p['field']} field",),
    ActionCode.SCRUB_CREATOR: lambda p: (f"removed an AI {p['field']} field",),
    ActionCode.DROP_OPF_META: lambda p: ("dropped an AI-related OPF meta tag",),
    ActionCode.LAYER_A_TEXT: lambda p: _layer_a_phrases(p["removed"], p["replaced"]),
    ActionCode.EXIFTOOL_RUN: lambda p: (
        "stripped metadata with exiftool"
        if p["returncode"] == 0
        else f"ran exiftool, which exited with code {p['returncode']}",
    ),
    ActionCode.TOOL_FAILED: lambda p: (
        f"could not use {p['tool']}{_failure(p)}"
        + (f"; trying {p['fallback']} instead" if p.get("fallback") else ""),
    ),
    ActionCode.TOOL_MISSING: lambda p: (f"skipped {p['tool']}: not installed",),
    ActionCode.TRY_FALLBACK: lambda p: (f"fell back to {p['tool']}",),
    ActionCode.PDF_ENCRYPTED: lambda p: (
        "copied the PDF unchanged: it is encrypted and needs a password",
    ),
    ActionCode.PDF_DECRYPTED: lambda p: ("opened the encrypted PDF with an empty password",),
    ActionCode.DROP_PDF_METADATA: lambda p: (_pdf_metadata(p),),
    ActionCode.PDF_REWRITTEN: lambda p: (_pdf_rewritten(p),),
    ActionCode.PDF_REWRITE_FAILED: lambda p: (
        f"could not rebuild the PDF with qpdf{_failure(p)}; "
        "old metadata bytes may still be recoverable",
    ),
    ActionCode.PDF_COPIED_UNCHANGED: lambda p: (
        "copied the PDF unchanged: no structural cleaner succeeded",
    ),
    ActionCode.C2PATOOL_HINT: lambda p: (),
    ActionCode.VISIBLE_NEEDS_SOURCE: lambda p: (
        "found no visible mark to remove: give a mask, a box or a detector command",
    ),
    ActionCode.VISIBLE_PLAN: lambda p: (),
    ActionCode.MASK_SOURCE: lambda p: (),
    ActionCode.REFINE_MASK: lambda p: (
        f"filled holes and dilated the mask by {p['dilation_radius']} px "
        f"({p['pixels_before']} to {p['pixels_after']} pixels)",
    ),
    ActionCode.EFFECTIVE_MASK: lambda p: (
        (f"wrote the mask to {p['published']}",) if p.get("published") else ()
    ),
    ActionCode.INPAINT_SKIPPED: lambda p: (
        "ran no inpainting: the print-plan backend only builds the mask",
    ),
    ActionCode.INPAINT: lambda p: (f"inpainted the visible mark with the {p['backend']} backend",),
    ActionCode.PLAN_LOCALIZE: lambda p: (f"would find the visible mark from {p['source']}",),
    ActionCode.PLAN_REFINE_MASK: lambda p: (
        f"would fill holes and dilate the mask by {p['dilation_radius']} px",
    ),
    ActionCode.PLAN_INPAINT: lambda p: (f"would inpaint with the {p['backend']} backend",),
    ActionCode.PLAN_STRIP_METADATA: lambda p: ("would strip the requested metadata",),
    ActionCode.PLAN_DEGRADE: lambda p: (f"would apply {p['strategy']} degradation",),
    ActionCode.PLAN_PUBLISH: lambda p: (
        f"would write the mask to {p['mask']} and the image to {p['image']}",
    ),
    ActionCode.FAILED: lambda p: (f"failed: {p['error']}",),
}


def action_phrases(detail: Mapping[str, Any]) -> Sequence[str]:
    """The phrases one ``action_details`` entry reads as, verb first, no full stop."""
    return _ACTION_PHRASES[detail["code"]](detail.get("params") or {})


def _as_sentence(phrase: str) -> str:
    """A phrase from ``action_phrases`` as a line: its own verb capitalised, a full stop."""
    text = phrase[:1].upper() + phrase[1:]
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _rewrite_line(rewrite: Mapping[str, Any]) -> str | None:
    """The Layer B sentence from ``stats["tsapa"]``.

    That key holds ``rewrite()``'s info, except after a TSAPA run, where it
    holds only the search record (``generations``, ``population``, ...).
    """
    if "generations" in rewrite:
        how = f"a Layer B TSAPA search over {_plural(int(rewrite['generations']), 'generation')}"
    elif rewrite.get("mode") == "rewritten":
        model = rewrite.get("model")
        how = f"a Layer B {rewrite.get('strength')} rewrite" + (f" by {model}" if model else "")
    else:
        return None
    return f"Rewrote the text with {how}. Best-effort, no detector guarantee."


def result_lines(payload: Mapping[str, Any]) -> list[str]:
    """What a clean did to one file, in short sentences, from its payload.

    Built from the payload's numbers and its ``action_details`` codes, never
    from the human ``actions`` lines.  Steps that changed nothing say nothing.
    """
    if payload.get("error"):
        return [f"Failed: {payload['error']}"]
    phrases: list[str] = []
    stats = payload.get("stats")
    if isinstance(stats, Mapping):
        phrases.extend(
            _layer_a_phrases(
                int(stats.get("removed_count") or 0), int(stats.get("replaced_count") or 0)
            )
        )
    lines = [_as_sentence(phrase) for phrase in phrases]
    if isinstance(stats, Mapping):
        if stats.get("nfkc_changed"):
            lines.append("NFKC normalisation changed the text.")
        rewrite = stats.get("tsapa")
        if isinstance(rewrite, Mapping) and (line := _rewrite_line(rewrite)):
            lines.append(line)
    # A visible-mark clean runs first, then the metadata strip.
    visible = payload.get("visible")
    visible_steps = visible.get("action_details") or [] if isinstance(visible, Mapping) else []
    for detail in [*visible_steps, *(payload.get("action_details") or [])]:
        lines.extend(_as_sentence(phrase) for phrase in action_phrases(detail))
    skipped = payload.get("skipped_text_transforms")
    if skipped:
        lines.append(f"Skipped {', '.join(str(item) for item in skipped)}: not a text asset.")
    if payload.get("residual"):
        lines.append("Residual C2PA or AI signals may remain.")
    return _capped(lines, MAX_FINDING_LINES, _MORE_LINES)


def history_summary(total: int, errors: int, layer: str, *, cancelled: bool = False) -> str:
    """One history line: counts, whether it was cancelled, and the result class."""
    counts = f"{_plural(total, 'file')}, {_plural(errors, 'error')}"
    if cancelled:
        counts += ", cancelled"
    return f"{counts}. {result_class_for(layer)}."


# --- persisted setup ---------------------------------------------------------

#: Environment override for the settings file, so a test never touches the
#: real one and an operator can keep per-project setups side by side.
SETTINGS_ENV = "WATERMARKS_TUI_SETTINGS"


@dataclass(frozen=True)
class TuiSettings:
    """The setup wm-tui remembers between runs: a preset and an endpoint.

    Deliberately not routed through ``configuration``: that module is the
    shared CLI/server config seam with its own precedence rules, and this is
    one UI's memory of which endpoint you last pointed it at.  The generated
    command still carries every value explicitly, so a command copied out of
    the TUI runs the same way on a machine that has no settings file.

    There is no API key field, and there never will be one.  Nor is there a
    remote opt-in: sending text off-machine is agreed to per run, so a saved
    "yes" cannot silently apply to a later document.  Keys a file carries
    beyond these fields (``rewrite_reasoning_effort`` and
    ``rewrite_allow_remote`` in files written by earlier releases) are ignored
    on read and dropped on the next save.
    """

    preset: str | None = None
    rewrite_backend: str | None = None
    rewrite_base_url: str | None = None
    rewrite_model: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The PROTOCOL shape: ``{"preset", "endpoint": {backend, base_url, model}}``."""
        return {
            "preset": self.preset,
            "endpoint": {key: getattr(self, target) for key, target in ENDPOINT_FIELDS.items()},
        }


#: Every top-level key ``save_settings`` accepts.  An allow-list, so nothing a
#: client sends can widen what is written to disk.
SETTINGS_KEYS: tuple[str, ...] = ("preset", "endpoint")


def settings_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the setup file lives, honouring the usual per-platform roots."""
    env = os.environ if environ is None else environ
    override = env.get(SETTINGS_ENV)
    if override:
        return Path(override)
    base = env.get("XDG_CONFIG_HOME") or env.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    return root / "watermark-remover" / "tui.json"


def load_settings(path: Path | None = None) -> TuiSettings:
    """Read the setup file. Anything unreadable means "no saved setup".

    Fail-soft on purpose: a corrupt or hand-edited settings file must not stop
    the operator from starting the TUI, and every value in it is a convenience
    with a visible control behind it.  Every field is a string, and a value of
    any other type is dropped here rather than failing late, inside
    ``classify_endpoint``.
    """
    target = path or settings_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return TuiSettings()
    if not isinstance(raw, dict):
        return TuiSettings()
    known = {item.name for item in fields(TuiSettings)}
    loaded = {key: value for key, value in raw.items() if key in known and isinstance(value, str)}
    # hello echoes these into every state; a backend or preset this release
    # does not know would otherwise make every plan fail validation.
    if loaded.get("rewrite_backend") not in (None, *LIVE_REWRITE_BACKENDS):
        del loaded["rewrite_backend"]
    if loaded.get("preset") is not None and preset_for(loaded["preset"]) is None:
        del loaded["preset"]
    return TuiSettings(**loaded)


def apply_settings_update(current: TuiSettings, update: object) -> TuiSettings:
    """Validate a ``save_settings`` payload strictly and apply it to *current*.

    Unlike ``load_settings`` this refuses rather than drops: a client that
    sends an unknown key (anything named like a key, token or secret included)
    is wrong, and saying so beats silently writing less than it asked for.  A
    top-level key that is absent keeps its saved value; ``endpoint: null``
    clears the endpoint, and an empty string clears one field.
    """
    if not isinstance(update, Mapping):
        raise BadRequest("settings must be an object")
    unknown = sorted(str(key) for key in update if key not in SETTINGS_KEYS)
    if unknown:
        raise BadRequest(f"unknown settings key(s): {', '.join(unknown)}")
    changes: dict[str, str | None] = {}
    if "preset" in update:
        preset = update["preset"]
        if preset is not None and (not isinstance(preset, str) or preset_for(preset) is None):
            raise BadRequest(f"unknown preset: {preset!r}")
        changes["preset"] = preset
    if "endpoint" in update:
        endpoint = update["endpoint"]
        if endpoint is not None and not isinstance(endpoint, Mapping):
            raise BadRequest("settings.endpoint must be an object or null")
        try:
            changes.update(_endpoint_values(endpoint or {}, "settings.endpoint"))
        except InvalidOptions as error:
            raise BadRequest(str(error)) from error
    return replace(current, **changes)


def save_settings(settings: TuiSettings, path: Path | None = None) -> Path:
    """Write the setup file atomically and return where it went."""
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n")
    return target


# --- first-run setup ---------------------------------------------------------


def should_onboard(*, settings_exist: bool, force: bool = False, skip: bool = False) -> bool:
    """Whether wm-tui opens on the setup screen.

    Only a first run does: finishing *or* skipping the setup writes the
    settings file, and its existence is the "already set up" marker.  A file
    that exists but will not parse still counts as "set up": the fix for a
    corrupt file is the setup screen on demand (``--setup``), not a wizard that
    ambushes every start.
    """
    if force:
        return True
    if skip:
        return False
    return not settings_exist


#: Loopback servers worth asking, in the order a person would guess them.
#: Every one is loopback on purpose: detection runs before the operator has
#: agreed to anything, so it must never be what reaches another machine.
#: Base URLs carry no ``/v1``; ``rewrite_text`` and discovery both append
#: their own routes.
LOCAL_ENDPOINT_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("ollama", DEFAULT_BASE_URL, "Ollama"),
    ("openai-compatible", "http://127.0.0.1:1234", "LM Studio"),
    ("openai-compatible", "http://127.0.0.1:8080", "llama.cpp server"),
    ("openai-compatible", "http://127.0.0.1:8000", "vLLM / other OpenAI-compatible"),
)

#: Per-candidate timeout for detection.  A closed port refuses instantly; this
#: bounds the port that accepts and then never answers.  Probes run in
#: parallel, so the whole scan is one timeout, not one per candidate.
DETECT_TIMEOUT = 2.0


def endpoint_candidates(environ: Mapping[str, str] | None = None) -> list[tuple[str, str, str]]:
    """The candidates to probe, the environment's own endpoint first.

    An endpoint already configured through ``WATERMARKS_REWRITE_*`` is the
    likeliest answer, so it leads, but it goes through the same loopback-only
    probe as the rest, so a remote URL in the environment is reported as
    blocked rather than contacted.
    """
    env = os.environ if environ is None else environ
    candidates: list[tuple[str, str, str]] = []
    backend = env.get("WATERMARKS_REWRITE_BACKEND")
    base_url = env.get("WATERMARKS_REWRITE_BASE_URL")
    if backend in MODEL_ROUTES and base_url:
        candidates.append((backend, base_url.rstrip("/"), "from WATERMARKS_REWRITE_*"))
    for candidate in LOCAL_ENDPOINT_CANDIDATES:
        if all(candidate[:2] != known[:2] for known in candidates):
            candidates.append(candidate)
    return candidates


def detect_local_endpoints(
    candidates: Sequence[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Probe every candidate at once and report each, in candidate order.

    Never sends document text: ``probe_backend`` reads the model list only,
    refuses non-loopback hosts before any request (``allow_remote=False``),
    and turns every failure into an unreachable probe.  Parallel because four
    hung ports probed one after another is a frozen-looking screen for four
    timeouts.
    """
    targets = list(endpoint_candidates() if candidates is None else candidates)
    if not targets:
        return []

    def ask(target: tuple[str, str, str]) -> dict[str, Any]:
        backend, base_url, label = target
        probe = probe_backend(backend, base_url, allow_remote=False, timeout=DETECT_TIMEOUT)
        return {
            "label": label,
            "backend": probe.backend,
            "base_url": probe.base_url,
            "reachable": probe.reachable,
            "models": list(probe.models),
            "error": probe.error,
        }

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        return list(pool.map(ask, targets))


@dataclass(frozen=True)
class EnvironmentCheck:
    """One line of the setup screen's "what do I have" table."""

    name: str
    state: str
    detail: str
    #: The shell command that fixes it, or "" when there is nothing to do.
    fix: str = ""
    #: False for things a first run needs; True for opt-in capabilities.
    optional: bool = True
    #: Whether this line is in the state an operator wants.  Stated rather
    #: than inferred from ``state`` so the frontend never parses prose.
    good: bool = True


#: What each extra adds, in the operator's terms rather than the import's.
EXTRA_PURPOSE: dict[str, str] = {
    "visible": "Images: visible-mark cleaning and pixel work (numpy, Pillow, OpenCV).",
    "quality": "Image quality scoring (scikit-image).",
    "ai": "Torch-backed adapters such as CtrlRegen and the text detectors (torch).",
    "provenance": "C2PA manifest reading (c2pa-python).",
}


def environment_checks(
    environ: Mapping[str, str] | None = None,
    settings_file: Path | None = None,
) -> list[EnvironmentCheck]:
    """What this install can do, and the command for each thing it cannot.

    The core clean needs nothing beyond the standard library, so every extra
    is reported as optional.  The table's job on a first run is to say "you
    are ready" first and "here is what more you could add" second.  The API
    key line says whether a key is set, never what it is.
    """
    env = os.environ if environ is None else environ
    checks = [
        EnvironmentCheck(
            "Python",
            "OK",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. "
            "Hidden-mark and metadata cleaning work with nothing else installed.",
            optional=False,
        )
    ]
    for extra in KNOWN_EXTRAS:
        availability = check_optional(extra)
        checks.append(
            EnvironmentCheck(
                f"Extra: {extra}",
                "Installed" if availability.available else "Not installed",
                EXTRA_PURPOSE.get(extra, availability.hint),
                "" if availability.available else f'pip install "watermark-remover[{extra}]"',
                good=availability.available,
            )
        )
    key_set = bool(env.get(API_KEY_ENV))
    checks.append(
        EnvironmentCheck(
            "Layer B API key",
            "Set" if key_set else "Not set",
            f"Read from {API_KEY_ENV} at run time, never shown or saved. "
            "Local servers do not need one.",
            "" if key_set else f"export {API_KEY_ENV}=...   # only for a keyed server",
            good=key_set,
        )
    )
    target = settings_file or settings_path(env)
    checks.append(
        EnvironmentCheck(
            "Settings file",
            "Exists" if target.exists() else "Will be created",
            str(target),
        )
    )
    if env.get("TERM_PROGRAM") == "Apple_Terminal":
        # OSC 52 is ignored by Terminal.app, and nothing tells the app so.
        checks.append(
            EnvironmentCheck(
                "Clipboard",
                "Limited",
                "Terminal.app ignores OSC 52 copy. Select commands from the boxes instead.",
                good=False,
            )
        )
    return checks
