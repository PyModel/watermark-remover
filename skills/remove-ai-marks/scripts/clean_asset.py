"""Presentation-free cleaning for one validated asset."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from asset_kind import AssetKind, classify_asset
from common import (
    atomic_write_text,
    backup_path,
    create_backup,
    paths_alias,
    validate_output_path,
)
from container_meta import clean_container
from dct_frequency import degrade_image
from image_meta import clean_image
from inspect_soft_binding import inspect_soft_binding
from morphomod import VISIBLE_CLEAN_BACKENDS, VisiblePlan, remove_visible
from perturb_text import MODES as PERTURB_MODES
from perturb_text import perturb_text
from rewrite_text import RewritePlan, rewrite
from text_unicode import clean_text

_IMAGE_DEGRADE_STRATEGIES = frozenset(
    {
        "freq-dct",
        "blur",
        "median",
        "jpeg",
        "rotate",
        "two-stage",
        "grid",
        "diagonal",
        "noise",
        "quantize",
    }
)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("CleanResult detail keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("CleanResult details must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TextCleanPlan:
    nfkc: bool = False
    aggressive_homoglyphs: bool = False
    preserve_semantic: bool = True
    rewrite_plan: RewritePlan | None = None
    perturb_mode: str | None = None
    perturb_strength: float = 0.1
    perturb_seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("nfkc", "aggressive_homoglyphs", "preserve_semantic"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.rewrite_plan is not None and not isinstance(self.rewrite_plan, RewritePlan):
            raise TypeError("rewrite_plan must be a RewritePlan")
        if self.perturb_mode is not None and self.perturb_mode not in PERTURB_MODES:
            raise ValueError(f"unknown perturb mode: {self.perturb_mode}")
        if (
            isinstance(self.perturb_strength, bool)
            or not isinstance(self.perturb_strength, (int, float))
            or not math.isfinite(self.perturb_strength)
            or not 0.0 <= self.perturb_strength <= 1.0
        ):
            raise ValueError("perturb_strength must be a finite number in [0, 1]")
        if self.perturb_seed is not None and (
            isinstance(self.perturb_seed, bool) or not isinstance(self.perturb_seed, int)
        ):
            raise TypeError("perturb_seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class ImageDegradePlan:
    """Optional frequency/morphological degradation applied after visible clean."""

    strategy: str = "freq-dct"
    strength: float = 0.6
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in _IMAGE_DEGRADE_STRATEGIES:
            raise ValueError(f"unknown degrade strategy: {self.strategy}")
        if not isinstance(self.strength, (int, float)) or isinstance(self.strength, bool):
            raise TypeError("strength must be a number")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class CleanPlan:
    forced_kind: str = "auto"
    in_place: bool = False
    text: TextCleanPlan = field(default_factory=TextCleanPlan)
    strip_all_metadata: bool = True
    visible: VisiblePlan | None = None
    inspect_soft_binding: bool = False
    degrade: ImageDegradePlan | None = None

    def __post_init__(self) -> None:
        if self.forced_kind not in ("auto", "text", "image", "container"):
            raise ValueError(f"unsupported forced asset kind: {self.forced_kind}")
        for name in ("in_place", "strip_all_metadata", "inspect_soft_binding"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.text, TextCleanPlan):
            raise TypeError("text must be a TextCleanPlan")
        if self.visible is not None:
            if not isinstance(self.visible, VisiblePlan):
                raise TypeError("visible must be a VisiblePlan")
            if self.visible.backend not in VISIBLE_CLEAN_BACKENDS:
                raise ValueError("visible cleaning requires an inpainting backend")
        if self.degrade is not None and not isinstance(self.degrade, ImageDegradePlan):
            raise TypeError("degrade must be an ImageDegradePlan or None")


@dataclass(frozen=True, slots=True)
class CleanResult:
    kind: AssetKind
    input: Path
    output: Path
    residual: bool
    _details: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        details = dict(self._details)
        details.pop("kind", None)
        details.pop("input", None)
        details.pop("output", None)
        if "exit_code" in details:
            raise ValueError("CleanResult details must not contain exit_code")
        object.__setattr__(self, "_details", _freeze_json(details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "input": str(self.input),
            "output": str(self.output),
            **{key: _thaw_json(value) for key, value in self._details.items()},
        }


def _validate_operation(path: Path, dest: Path, plan: CleanPlan) -> None:
    if not isinstance(plan, CleanPlan):
        raise TypeError("plan must be a CleanPlan")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    if plan.in_place:
        if not paths_alias(path, dest):
            raise ValueError("in-place destination must alias input")
        backup = backup_path(path)
        if backup.exists() or backup.is_symlink():
            raise ValueError(f"backup already exists: {backup}")
        if (
            plan.visible is not None
            and plan.visible.mask_output is not None
            and paths_alias(plan.visible.mask_output, path)
        ):
            raise ValueError(f"mask output aliases input: {plan.visible.mask_output}")
        if (
            plan.visible is not None
            and plan.visible.mask_output is not None
            and paths_alias(plan.visible.mask_output, backup)
        ):
            raise ValueError(f"mask output aliases backup: {plan.visible.mask_output}")
    else:
        validate_output_path(path, dest)
        if (
            plan.visible is not None
            and plan.visible.mask_output is not None
            and paths_alias(plan.visible.mask_output, path)
        ):
            raise ValueError(f"mask output aliases input: {plan.visible.mask_output}")
        if (
            plan.visible is not None
            and plan.visible.mask_output is not None
            and paths_alias(plan.visible.mask_output, dest)
        ):
            raise ValueError(f"mask output aliases image output: {plan.visible.mask_output}")


def _clean_text_asset(path: Path, dest: Path, plan: CleanPlan) -> CleanResult:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    cleaned, stats = clean_text(
        text,
        nfkc=plan.text.nfkc,
        aggressive_homoglyphs=plan.text.aggressive_homoglyphs,
        preserve_semantic=plan.text.preserve_semantic,
    )
    if plan.text.rewrite_plan is not None:
        cleaned, rewrite_info = rewrite(cleaned, plan.text.rewrite_plan)
        stats["tsapa"] = rewrite_info.get("tsapa", rewrite_info)
    if plan.text.perturb_mode is not None:
        cleaned, perturb_stats = perturb_text(
            cleaned,
            mode=plan.text.perturb_mode,
            strength=plan.text.perturb_strength,
            seed=plan.text.perturb_seed,
        )
        stats["char_perturb"] = perturb_stats
    if plan.in_place:
        create_backup(path)
    atomic_write_text(dest, cleaned)
    return CleanResult("text", path, dest, False, {"stats": stats})


def _clean_image_asset(path: Path, dest: Path, plan: CleanPlan) -> CleanResult:
    from common import read_bytes_bounded

    soft = inspect_soft_binding(path) if plan.inspect_soft_binding else None
    visible_report = None

    # Step 1: visible mark removal (inpaint)
    if plan.visible is not None:
        with tempfile.TemporaryDirectory(prefix="wm-visible-") as temp_dir:
            visible_dest = Path(temp_dir) / f"visible{path.suffix}"
            visible_report = remove_visible(path, visible_dest, plan.visible)
            if visible_report["status"] != "completed":
                raise RuntimeError("visible pipeline did not produce an output")
            if plan.in_place:
                create_backup(path)
            report = clean_image(
                visible_dest,
                dest,
                strip_all_metadata=plan.strip_all_metadata,
            )
    else:
        source = create_backup(path) if plan.in_place else path
        report = clean_image(source, dest, strip_all_metadata=plan.strip_all_metadata)

    # Step 2: frequency/morphological degradation (optional)
    if plan.degrade is not None:
        cleaned_data = read_bytes_bounded(dest, limit=256 * 1024 * 1024, label="cleaned image")
        from morphomod import decode_png, encode_png

        raster = decode_png(bytes(cleaned_data))
        degrade_result = degrade_image(
            bytes(raster.data),
            raster.width,
            raster.height,
            raster.channels,
            strategy=plan.degrade.strategy,
            suppress=plan.degrade.strength,
            seed=plan.degrade.seed,
        )

        out_raster = type(
            "Raster",
            (),
            {
                "width": degrade_result.width,
                "height": degrade_result.height,
                "channels": degrade_result.channels,
                "data": degrade_result.data,
            },
        )()
        fd, temp_name = tempfile.mkstemp(prefix=".degraded-", dir=str(dest.parent))
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encode_png(out_raster))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, dest)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

        report["degrade"] = degrade_result.to_dict()

    report["input"] = str(path)
    if visible_report is not None:
        report["visible"] = visible_report
    if soft is not None:
        report["soft_binding"] = soft
    soft_found = bool(soft and soft["soft_binding"]["found"])
    residual = bool(
        report.get("still_has_c2pa", False)
        or report.get("still_has_ai_metadata", False)
        or soft_found
        or plan.degrade is not None
    )
    return CleanResult("image", path, dest, residual, report)


def _clean_container_asset(path: Path, dest: Path, plan: CleanPlan) -> CleanResult:
    source = create_backup(path) if plan.in_place else path
    report = clean_container(source, dest)
    report["input"] = str(path)
    residual = bool(report["still_has_c2pa"] or report["still_has_ai_metadata"])
    return CleanResult("container", path, dest, residual, report)


def clean_asset(path: Path, dest: Path, plan: CleanPlan) -> CleanResult:
    """Clean one asset; raise failures and leave presentation to the caller."""
    if plan.visible is not None and plan.visible.mask_output is None:
        plan = replace(
            plan,
            visible=replace(
                plan.visible,
                mask_output=dest.with_name(f"{dest.stem}.mask.pgm"),
            ),
        )
    _validate_operation(path, dest, plan)
    kind = classify_asset(path, forced_kind=plan.forced_kind)
    if plan.visible is not None and kind != "image":
        raise ValueError("visible cleaning is only valid for image assets")
    if kind == "text":
        return _clean_text_asset(path, dest, plan)
    if kind == "image":
        return _clean_image_asset(path, dest, plan)
    return _clean_container_asset(path, dest, plan)
