#!/usr/bin/env python3
"""Argparse-independent clean options, plan construction, and batch preflight.

``clean_file.py`` used to read ~30 attributes straight off an argparse
``Namespace`` while building ``CleanPlan``/``TextCleanPlan``/``RewritePlan``/
``VisiblePlan``.  That made the flag surface reachable only from the CLI: any
other front end (the TUI, a library caller) had to either fake a ``Namespace``
or rebuild plan construction and drift from it.

This module owns that seam instead:

``CleanRequest``
    A frozen, typed record of every clean option.  Field names mirror the CLI
    flags so ``from_args`` stays mechanical and any drift is visible in review.

``build_clean_plan(request, dest, kind)``
    ``CleanRequest`` -> ``CleanPlan``, including the Layer B and Layer V policy
    that used to live in ``clean_file._build_clean_plan``.

``plan_work(items, request, batch)``
    Resolves and validates every destination *before the first write*: output
    aliasing, batch collisions, size caps, and the binary-as-text guard.

The CLI and the TUI both go through here, so a refusal is a refusal on both.
"""

from __future__ import annotations

import math
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from asset_kind import AssetKind, classify_asset
from batch_inputs import InputItem, safe_output_path
from clean_asset import (
    CleanPlan,
    ImageDegradePlan,
    TextCleanPlan,
)
from common import (
    MAX_INPUT_BYTES,
    ROUTER_ADVICE,
    backup_path,
    cleaned_path,
    guard_binary,
    paths_alias,
    validate_output_path,
)
from morphomod import DEFAULT_DILATION_RADIUS, VisiblePlan
from perturb_text import MODES as PERTURB_MODES
from rewrite_text import REWRITE_STRENGTHS, RewritePlan

#: Layer B strengths that ``--rewrite`` accepts.  ``tsapa`` is included so
#: ``--rewrite tsapa`` and the older ``--tsapa`` flag mean the same thing.
REWRITE_CLI_CHOICES = REWRITE_STRENGTHS

QUALITY_PROFILES = ("fast", "balanced", "high")
FORCED_KINDS = ("auto", "text", "image", "container")

#: Defaults ``command_line`` compares against so it can omit a flag that is
#: already the CLI's own default.  Named here rather than repeated as literals
#: in ``clean_file``'s parser: the two must not drift, or a copied command
#: silently runs with different settings than the one that produced it.
DEFAULT_VISIBLE_PROMPT = "Remove watermark, fill with background"
DEFAULT_TIMEOUT = 1800.0


class CleanPlanPreflightError(RuntimeError):
    """A per-asset policy failed before batch execution."""

    def __init__(self, path: Path, output: Path, error: Exception) -> None:
        super().__init__(str(error))
        self.path = path
        self.output = output


@dataclass(frozen=True, slots=True)
class CleanRequest:
    """Every clean option, independent of how it was collected.

    Defaults match ``clean_file``'s argparse defaults exactly; a
    ``CleanRequest()`` is the same run as bare ``wm FILE``.
    """

    # --- input selection -------------------------------------------------
    paths: tuple[Path, ...] = ()
    output: Path | None = None
    in_place: bool = False
    recursive: bool = False
    glob: str = "*"
    extensions: str | None = None

    # --- routing ---------------------------------------------------------
    force_type: str = "auto"
    force_text: bool = False

    # --- Layer A: hidden Unicode ----------------------------------------
    nfkc: bool = False
    aggressive_homoglyphs: bool = False
    strip_semantic_format: bool = False

    # --- Layer M: metadata ----------------------------------------------
    keep_non_ai_metadata: bool = False
    soft_binding: bool = False

    # --- Layer B: LLM rewrite -------------------------------------------
    rewrite: str | None = None
    rewrite_backend: str | None = None
    rewrite_model: str | None = None
    rewrite_base_url: str | None = None
    rewrite_api_key: str | None = field(default=None, repr=False)
    rewrite_lang: str | None = None
    rewrite_original_lang: str | None = None
    rewrite_timeout: float | None = None
    rewrite_temperature: float | None = None
    rewrite_candidates: int | None = None
    rewrite_reasoning_effort: str | None = None
    rewrite_disable_thinking: bool | None = None
    # None means "not stated": the environment decides. The TUI's checkbox
    # is an explicit control and always sets True or False.
    rewrite_allow_remote: bool | None = None
    tsapa: bool = False
    tsapa_generations: int = 5
    tsapa_population: int = 12

    # --- character perturbation -----------------------------------------
    char_perturb: bool = False
    char_mode: str = "zero-width"
    char_strength: float = 0.1
    seed: int | None = None

    # --- Layer V: visible marks -----------------------------------------
    visible_mask: Path | None = None
    visible_box: tuple[int, int, int, int] | None = None
    detect_command: str | None = None
    dilate: int | None = None
    visible_backend: str = "texture"
    inpaint_command: str | None = None
    visible_prompt: str = DEFAULT_VISIBLE_PROMPT
    quality: str = "balanced"
    dry_run: bool = False
    timeout: float = DEFAULT_TIMEOUT

    # --- image degradation ----------------------------------------------
    degrade: str | None = None
    morpho: str | None = None
    degrade_strength: float = 0.6
    degrade_seed: int | None = None

    # --- SynthID ---------------------------------------------------------
    remove_synthid: bool = False
    synthid_strength: float = 0.6
    wmct_marker: bool = False

    # --- reporting -------------------------------------------------------
    keep_artifacts: bool = False
    quiet: bool = False
    json: bool = False
    audit: str | None = None

    def __post_init__(self) -> None:
        if self.force_type not in FORCED_KINDS:
            raise ValueError(f"unsupported forced asset kind: {self.force_type}")
        if self.quality not in QUALITY_PROFILES:
            raise ValueError(f"unknown quality profile: {self.quality}")
        if self.rewrite is not None and self.rewrite not in REWRITE_CLI_CHOICES:
            raise ValueError(f"unknown rewrite strength: {self.rewrite}")
        if self.char_mode not in PERTURB_MODES:
            raise ValueError(f"unknown perturb mode: {self.char_mode}")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")

    # -- construction ------------------------------------------------------

    @property
    def rewrite_strength(self) -> str | None:
        """The Layer B strength this request asks for, or None.

        ``--rewrite`` wins; bare ``--tsapa`` remains an alias for
        ``--rewrite tsapa`` so the older flag keeps working unchanged.
        """
        if self.rewrite is not None:
            return self.rewrite
        return "tsapa" if self.tsapa else None

    @classmethod
    def from_args(cls, args) -> CleanRequest:
        """Adapt an argparse ``Namespace`` from ``clean_file._build_parser``."""
        paths = tuple(getattr(args, "path", ()) or ())
        return cls(
            paths=paths,
            output=args.output,
            in_place=args.in_place,
            recursive=args.recursive,
            glob=args.glob,
            extensions=args.extensions,
            force_type=args.force_type,
            force_text=args.force_text,
            nfkc=args.nfkc,
            aggressive_homoglyphs=args.aggressive_homoglyphs,
            strip_semantic_format=args.strip_semantic_format,
            keep_non_ai_metadata=args.keep_non_ai_metadata,
            soft_binding=args.soft_binding,
            rewrite=args.rewrite,
            rewrite_backend=args.rewrite_backend,
            rewrite_model=args.rewrite_model,
            rewrite_base_url=args.rewrite_base_url,
            rewrite_lang=args.rewrite_lang,
            rewrite_original_lang=args.rewrite_original_lang,
            rewrite_timeout=args.rewrite_timeout,
            rewrite_temperature=args.rewrite_temperature,
            rewrite_candidates=args.rewrite_candidates,
            rewrite_reasoning_effort=args.rewrite_reasoning_effort,
            rewrite_disable_thinking=args.rewrite_disable_thinking,
            rewrite_allow_remote=args.rewrite_allow_remote,
            tsapa=args.tsapa,
            tsapa_generations=args.tsapa_generations,
            tsapa_population=args.tsapa_population,
            char_perturb=args.char_perturb,
            char_mode=args.char_mode,
            char_strength=args.char_strength,
            seed=args.seed,
            visible_mask=args.visible_mask,
            visible_box=args.visible_box,
            detect_command=args.detect_command,
            dilate=args.dilate,
            visible_backend=args.visible_backend,
            inpaint_command=args.inpaint_command,
            visible_prompt=args.visible_prompt,
            quality=args.quality,
            dry_run=args.dry_run,
            timeout=args.timeout,
            degrade=args.degrade,
            morpho=args.morpho,
            degrade_strength=args.degrade_strength,
            degrade_seed=args.degrade_seed,
            remove_synthid=args.remove_synthid,
            synthid_strength=args.synthid_strength,
            wmct_marker=args.wmct_marker,
            keep_artifacts=args.keep_artifacts,
            quiet=args.quiet,
            json=args.json,
            audit=args.audit,
        )

    def with_paths(self, paths: Sequence[Path]) -> CleanRequest:
        return replace(self, paths=tuple(paths))

    # -- derived views -----------------------------------------------------

    def allowed_extensions(self, default: set[str]) -> set[str]:
        """Resolve ``--extensions`` against the supported-extension default."""
        if not self.extensions:
            return default
        return {
            "." + extension.strip().lstrip(".").lower()
            for extension in self.extensions.split(",")
            if extension.strip()
        }

    def visible_requested(self) -> bool:
        return any(
            (
                self.visible_mask,
                self.visible_box,
                self.detect_command,
                self.dilate is not None,
                self.inpaint_command,
                self.visible_backend != "texture",
            )
        )

    def command_string(self) -> str:
        """``command_line`` as one shell-safe line, for display and copying.

        Quoting is not cosmetic here: file names routinely carry spaces and
        glob metacharacters, and a copyable command that silently splits
        ``report[1] draft.txt`` into two arguments is worse than no command at
        all.
        """
        return shlex.join(self.command_line())

    def command_line(self) -> list[str]:
        """The ``wm`` invocation equivalent to this request, as argv.

        Rendered from the request, never from a built ``RewritePlan``, so an
        API key can never reach the string (``--rewrite-api-key`` has no CLI
        form; the environment variable is named instead).
        """
        argv = ["wm", *(str(path) for path in self.paths)]
        if self.output is not None:
            argv += ["-o", str(self.output)]
        for flag, value in (
            ("--in-place", self.in_place),
            ("--recursive", self.recursive),
            ("--json", self.json),
            ("--nfkc", self.nfkc),
            ("--aggressive-homoglyphs", self.aggressive_homoglyphs),
            ("--strip-semantic-format", self.strip_semantic_format),
            ("--keep-non-ai-metadata", self.keep_non_ai_metadata),
            ("--soft-binding", self.soft_binding),
            ("--char-perturb", self.char_perturb),
            ("--remove-synthid", self.remove_synthid),
            ("--wmct-marker", self.wmct_marker),
            ("--keep-artifacts", self.keep_artifacts),
            ("--quiet", self.quiet),
            ("--dry-run", self.dry_run),
        ):
            if value:
                argv.append(flag)
        if self.glob != "*":
            argv += ["--glob", self.glob]
        if self.extensions:
            argv += ["--extensions", self.extensions]
        if self.force_type != "auto":
            argv += ["--as", self.force_type]
        if self.force_text:
            argv.append("--force-text")
        if self.rewrite is not None:
            argv += ["--rewrite", self.rewrite]
        if self.tsapa:
            argv.append("--tsapa")
        for flag, value in (
            ("--rewrite-backend", self.rewrite_backend),
            ("--rewrite-model", self.rewrite_model),
            ("--rewrite-base-url", self.rewrite_base_url),
            ("--rewrite-lang", self.rewrite_lang),
            ("--rewrite-original-lang", self.rewrite_original_lang),
            ("--rewrite-reasoning-effort", self.rewrite_reasoning_effort),
        ):
            if value:
                argv += [flag, str(value)]
        for flag, value in (
            ("--rewrite-timeout", self.rewrite_timeout),
            ("--rewrite-temperature", self.rewrite_temperature),
            ("--rewrite-candidates", self.rewrite_candidates),
        ):
            if value is not None:
                argv += [flag, str(value)]
        if self.rewrite_disable_thinking:
            argv.append("--rewrite-disable-thinking")
        if self.rewrite_allow_remote:
            argv.append("--rewrite-allow-remote")
        if self.rewrite_strength == "tsapa":
            argv += [
                "--tsapa-generations",
                str(self.tsapa_generations),
                "--tsapa-population",
                str(self.tsapa_population),
            ]
        if self.char_perturb:
            argv += ["--char-mode", self.char_mode, "--char-strength", str(self.char_strength)]
            if self.seed is not None:
                argv += ["--seed", str(self.seed)]
        if self.visible_mask is not None:
            argv += ["--visible-mask", str(self.visible_mask)]
        if self.visible_box is not None:
            argv += ["--visible-box", ",".join(str(v) for v in self.visible_box)]
        if self.detect_command:
            argv += ["--detect-command", self.detect_command]
        if self.dilate is not None:
            argv += ["--dilate", str(self.dilate)]
        if self.visible_backend != "texture":
            argv += ["--visible-backend", self.visible_backend]
        if self.inpaint_command:
            argv += ["--inpaint-command", self.inpaint_command]
        if self.quality != "balanced":
            argv += ["--quality", self.quality]
        if self.visible_prompt != DEFAULT_VISIBLE_PROMPT:
            argv += ["--visible-prompt", self.visible_prompt]
        if self.timeout != DEFAULT_TIMEOUT:
            argv += ["--timeout", str(self.timeout)]
        if self.degrade:
            argv += ["--degrade", self.degrade]
        if self.morpho:
            argv += ["--morpho", self.morpho]
        if (self.degrade or self.morpho) and self.degrade_strength != 0.6:
            argv += ["--degrade-strength", str(self.degrade_strength)]
        if self.degrade_seed is not None:
            argv += ["--degrade-seed", str(self.degrade_seed)]
        if self.remove_synthid and self.synthid_strength != 0.6:
            argv += ["--synthid-strength", str(self.synthid_strength)]
        if self.audit is not None:
            argv += ["--audit", self.audit]
        return argv


def build_rewrite_plan(request: CleanRequest) -> RewritePlan | None:
    """Build the Layer B plan for *request*, or None when no rewrite is asked for.

    Every strength — not just ``tsapa`` — routes through
    ``RewritePlan.live_from_environment`` so explicit settings win over the
    ``WATERMARKS_REWRITE_*`` environment and both surfaces hit the same checks.
    """
    strength = request.rewrite_strength
    if strength is None:
        return None
    label = "--tsapa" if (request.rewrite is None and request.tsapa) else f"--rewrite {strength}"
    return RewritePlan.live_from_environment(
        strength,
        label=label,
        backend=request.rewrite_backend,
        model=request.rewrite_model,
        base_url=request.rewrite_base_url,
        api_key=request.rewrite_api_key,
        generations=request.tsapa_generations,
        population=request.tsapa_population,
        lang=request.rewrite_lang,
        original_lang=request.rewrite_original_lang,
        timeout=request.rewrite_timeout,
        temperature=request.rewrite_temperature,
        candidates=request.rewrite_candidates,
        reasoning_effort=request.rewrite_reasoning_effort,
        disable_thinking=request.rewrite_disable_thinking,
        allow_remote=request.rewrite_allow_remote,
    )


def dropped_text_transforms(request: CleanRequest, kind: AssetKind) -> tuple[str, ...]:
    """Text-body transforms *request* asks for that ``kind`` cannot apply.

    Text-only options are deliberate no-ops on image and container assets — that
    is what lets a mixed batch carry ``--rewrite`` without failing on the images
    (``tests/test_batch.py`` pins the behaviour).  But a silent no-op lets a
    caller believe a Markdown body was rewritten when ``_clean_container_asset``
    never consulted ``plan.text`` at all.  Naming the drop keeps the no-op and
    removes the false belief; callers report these, they do not raise.
    """
    if kind == "text":
        return ()
    dropped: list[str] = []
    strength = request.rewrite_strength
    if strength is not None:
        dropped.append(f"Layer B rewrite ({strength})")
    if request.char_perturb:
        dropped.append("character perturbation")
    if request.nfkc:
        dropped.append("NFKC normalization")
    if request.aggressive_homoglyphs:
        dropped.append("aggressive homoglyph folding")
    if request.strip_semantic_format:
        dropped.append("semantic format stripping")
    return tuple(dropped)


def describe_dropped_text_transforms(
    request: CleanRequest, kind: AssetKind, path: Path
) -> str | None:
    """One actionable line naming what was skipped, or None."""
    dropped = dropped_text_transforms(request, kind)
    if not dropped:
        return None
    return (
        f"note: {path.name} routes to {kind}; skipped {', '.join(dropped)} "
        "(text-body transforms apply to text assets — use --as text to force)"
    )


def build_clean_plan(request: CleanRequest, dest: Path, kind: AssetKind) -> CleanPlan:
    """Compose the immutable ``CleanPlan`` for one asset."""
    if (request.degrade or request.morpho) and kind != "image":
        raise ValueError("--degrade/--morpho are only valid for image assets")
    if request.dry_run:
        if kind != "image":
            raise ValueError("--dry-run is only valid for image assets")
        if not any((request.visible_mask, request.visible_box, request.detect_command)):
            raise ValueError(
                "--dry-run requires --visible-mask, --visible-box, or --detect-command"
            )

    text_plan = TextCleanPlan()
    if kind == "text":
        text_plan = TextCleanPlan(
            nfkc=request.nfkc,
            aggressive_homoglyphs=request.aggressive_homoglyphs,
            preserve_semantic=not request.strip_semantic_format,
            rewrite_plan=build_rewrite_plan(request),
            perturb_mode=request.char_mode if request.char_perturb else None,
            perturb_strength=request.char_strength if request.char_perturb else 0.1,
            perturb_seed=request.seed if request.char_perturb else None,
        )

    visible_plan = None
    if kind == "image" and request.visible_requested():
        backend = request.visible_backend
        if request.quality == "fast" and backend in ("texture", "simple"):
            backend = "simple"
        elif request.quality == "high" and backend == "simple":
            backend = "texture"

        visible_plan = VisiblePlan(
            mask_path=request.visible_mask,
            box=request.visible_box,
            detect_command=request.detect_command,
            backend=backend,
            command=request.inpaint_command,
            dilation_radius=(
                request.dilate if request.dilate is not None else DEFAULT_DILATION_RADIUS
            ),
            # mask_output is always a *planned* path (needed for output-collision
            # checks and dry-run reporting); publication is gated by
            # publish_mask, so frictionless runs leave no .mask.pgm on disk.
            mask_output=dest.with_name(f"{dest.stem}.mask.pgm"),
            prompt=request.visible_prompt,
            timeout=request.timeout,
            publish_mask=request.keep_artifacts,
        )

    degrade_plan = None
    if kind == "image":
        strategy = request.degrade or request.morpho
        if strategy:
            degrade_plan = ImageDegradePlan(
                strategy=strategy,
                strength=request.degrade_strength,
                seed=request.degrade_seed,
            )

    return CleanPlan(
        forced_kind=kind,
        in_place=request.in_place,
        text=text_plan,
        strip_all_metadata=not request.keep_non_ai_metadata,
        visible=visible_plan,
        inspect_soft_binding=request.soft_binding,
        degrade=degrade_plan,
        remove_synthid=request.remove_synthid,
        synthid_strength=request.synthid_strength,
        wmct_marker=request.wmct_marker,
    )


def resolve_kind(path: Path, request: CleanRequest) -> AssetKind:
    """Classify *path*, applying the CLI's unknown-format and binary refusals."""
    kind = classify_asset(path, forced_kind=request.force_type)
    if kind == "unknown" and not (request.force_text or request.force_type == "text"):
        raise ValueError(
            f"refusing to classify {path}: unrecognized format\n" + "\n".join(ROUTER_ADVICE)
        )
    if kind == "unknown":
        kind = "text"
    if kind == "text" and not request.force_text:
        with path.open("rb") as source:
            head = source.read(8192)
        try:
            guard_binary(
                head,
                str(path),
                allow_binary=request.force_text,
                advice=ROUTER_ADVICE,
            )
        except SystemExit as error:
            # guard_binary raises SystemExit(2) directly, which would blow past
            # the preflight error handling in the caller and swallow the
            # structured JSON report in --json mode. Convert to ValueError so it
            # reports like every other preflight refusal while preserving the
            # usage-error exit mapping.
            raise ValueError(f"refusing to treat {path} as text: binary content") from error
    return kind


def plan_work(
    items: Sequence[InputItem],
    request: CleanRequest,
    batch: bool,
) -> list[tuple[InputItem, Path | None, CleanPlan]]:
    """Resolve and validate every destination and plan before the first write."""
    inputs = [item.path for item in items]
    ancillary_inputs = [candidate for candidate in (request.visible_mask,) if candidate is not None]
    for ancillary in ancillary_inputs:
        if not ancillary.is_file() or ancillary.is_symlink():
            raise ValueError(f"not a regular mask file: {ancillary}")
    all_inputs = [*inputs, *ancillary_inputs]
    destinations: list[Path] = []
    work: list[tuple[InputItem, Path | None, CleanPlan]] = []

    for item in items:
        if request.in_place:
            backup = backup_path(item.path)
            if backup.exists() or backup.is_symlink():
                raise ValueError(f"backup already exists: {backup}")
            output = None
            dest = item.path
        else:
            if request.output is None:
                output = cleaned_path(item.path)
            elif batch:
                output = safe_output_path(request.output, item.relative)
            else:
                output = request.output

            validate_output_path(item.path, output)
            for source in all_inputs:
                if paths_alias(output, source):
                    raise ValueError(f"output aliases an input: {output}")
            for existing in destinations:
                if paths_alias(output, existing):
                    raise ValueError(f"batch output collision: {output}")
            destinations.append(output)
            dest = output

        if item.path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError(f"refusing input larger than {MAX_INPUT_BYTES} bytes: {item.path}")
        kind = resolve_kind(item.path, request)
        try:
            plan = build_clean_plan(request, dest, kind)
        except Exception as error:
            raise CleanPlanPreflightError(item.path, dest, error) from error

        if plan.visible is not None:
            mask_output = plan.visible.mask_output
            if mask_output is None:
                raise ValueError("visible plan is missing a mask output path")
            if mask_output.is_symlink():
                raise ValueError(f"mask output is a symlink: {mask_output}")
            if any(paths_alias(mask_output, source) for source in all_inputs):
                raise ValueError(f"mask output aliases an input: {mask_output}")
            if any(paths_alias(mask_output, existing) for existing in destinations):
                raise ValueError(f"mask/output collision: {mask_output}")
            destinations.append(mask_output)

        work.append((item, output, plan))
    return work
