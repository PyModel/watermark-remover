"""Presentation-free cleaning for one validated asset."""

from __future__ import annotations

import math
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from asset_kind import AssetKind, classify_asset
from common import (
    atomic_write_bytes,
    atomic_write_text,
    backup_path,
    create_backup,
    paths_alias,
    validate_output_path,
)
from container_meta import clean_container
from dct_frequency import DEGRADE_STRATEGIES, degrade_image
from image_meta import clean_image, detect_format
from inspect_soft_binding import inspect_soft_binding
from morpho_perturb import MORPHO_STRATEGIES, MORPHO_STRATEGY_KWARGS, morpho_perturb
from morphomod import (
    VISIBLE_CLEAN_BACKENDS,
    Raster,
    VisiblePlan,
    decode_png,
    encode_png,
    remove_visible,
)
from perturb_text import MODES as PERTURB_MODES
from perturb_text import perturb_text
from rewrite_text import RewritePlan, rewrite
from text_unicode import clean_text

_IMAGE_DEGRADE_STRATEGIES = frozenset((*DEGRADE_STRATEGIES, *MORPHO_STRATEGIES))
DEGRADE_CLI_CHOICES: tuple[str, ...] = DEGRADE_STRATEGIES
MORPHO_CLI_CHOICES: tuple[str, ...] = MORPHO_STRATEGIES
_BOUNDARY_UNCERTAIN_THRESHOLD = 0.5
_HALO_UNCERTAIN_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class _ArtifactSnapshot:
    data: bytes | None
    mode: int | None


def _snapshot_artifact(path: Path) -> _ArtifactSnapshot:
    if path.is_symlink():
        raise ValueError(f"output path is a symlink: {path}")
    if not path.exists():
        return _ArtifactSnapshot(None, None)
    if not path.is_file():
        raise ValueError(f"output path is not a regular file: {path}")
    return _ArtifactSnapshot(path.read_bytes(), stat.S_IMODE(path.stat().st_mode))


def _publish_image_artifacts(
    dest: Path,
    image_data: bytes,
    *,
    mask_output: Path | None,
    mask_data: bytes | None,
    backup_output: Path | None,
) -> None:
    if (mask_output is None) != (mask_data is None):
        raise ValueError("mask output and data must be provided together")

    targets = [dest]
    if mask_output is not None:
        targets.append(mask_output)
    if backup_output is not None:
        targets.append(backup_output)
    snapshots = {target: _snapshot_artifact(target) for target in targets}

    source_snapshot: _ArtifactSnapshot | None = None
    if backup_output is not None:
        source_snapshot = snapshots[dest]
        if source_snapshot.data is None or source_snapshot.mode is None:
            raise RuntimeError("in-place publication requires an existing source")

    publications: list[tuple[Path, bytes]] = [(dest, image_data)]
    if mask_output is not None and mask_data is not None:
        publications.append((mask_output, mask_data))

    attempted: list[Path] = []
    backup_identity: tuple[int, int] | None = None
    backup_temp: Path | None = None
    try:
        if backup_output is not None and source_snapshot is not None:
            backup_fd, temp_name = tempfile.mkstemp(
                prefix=f".{backup_output.name}.", dir=backup_output.parent
            )
            backup_temp = Path(temp_name)
            try:
                backup_stat = os.fstat(backup_fd)
                backup_identity = (backup_stat.st_dev, backup_stat.st_ino)
                with os.fdopen(backup_fd, "wb", closefd=False) as backup_file:
                    backup_file.write(source_snapshot.data)
                    backup_file.flush()
                    os.fsync(backup_fd)
                os.fchmod(backup_fd, source_snapshot.mode)
            finally:
                os.close(backup_fd)

            os.link(backup_temp, backup_output, follow_symlinks=False)
            attempted.append(backup_output)
            backup_temp.unlink()
            backup_temp = None

        for target, data in publications:
            attempted.append(target)
            atomic_write_bytes(target, data)
    except BaseException as error:
        rollback_errors: list[str] = []
        if (
            backup_output is not None
            and backup_identity is not None
            and backup_output not in attempted
        ):
            try:
                current_stat = backup_output.lstat()
            except FileNotFoundError:
                pass
            except BaseException as rollback_error:
                rollback_errors.append(f"{backup_output}: {rollback_error}")
            else:
                current_identity = (current_stat.st_dev, current_stat.st_ino)
                if current_identity == backup_identity:
                    attempted.append(backup_output)

        for target in reversed(attempted):
            snapshot = snapshots[target]
            try:
                if snapshot.data is None:
                    if target == backup_output and backup_identity is not None:
                        try:
                            current_stat = target.lstat()
                        except FileNotFoundError:
                            pass
                        else:
                            current_identity = (current_stat.st_dev, current_stat.st_ino)
                            if current_identity == backup_identity:
                                target.unlink()
                    else:
                        target.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(target, snapshot.data)
                    if snapshot.mode is not None:
                        target.chmod(snapshot.mode)
            except BaseException as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        if backup_temp is not None:
            try:
                backup_temp.unlink(missing_ok=True)
            except BaseException as rollback_error:
                rollback_errors.append(f"{backup_temp}: {rollback_error}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"final publication failed and rollback failed: {details}"
            ) from error
        raise


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
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(self.strength)
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ValueError("strength must be a finite number in [0, 1]")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
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
    remove_synthid: bool = False
    synthid_strength: float = 0.6
    wmct_marker: bool = False

    def __post_init__(self) -> None:
        if self.forced_kind not in ("auto", "text", "image", "container"):
            raise ValueError(f"unsupported forced asset kind: {self.forced_kind}")
        for name in (
            "in_place",
            "strip_all_metadata",
            "inspect_soft_binding",
            "remove_synthid",
            "wmct_marker",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.text, TextCleanPlan):
            raise TypeError("text must be a TextCleanPlan")
        if (
            not isinstance(self.synthid_strength, (int, float))
            or not 0.0 <= self.synthid_strength <= 1.0
        ):
            raise ValueError("synthid_strength must be in [0, 1]")
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


def _apply_image_degrade(raster: Raster, degrade: ImageDegradePlan) -> Any:
    """Run one degradation strategy on a decoded raster.

    Morphological strategies dispatch to ``morpho_perturb``; the rest dispatch
    to ``dct_frequency.degrade_image``. ``strength`` only configures freq-dct
    (the documented CLI meaning); other strategies use their own conservative
    defaults. ``seed`` is forwarded only to strategies that accept one.
    """
    strategy = degrade.strategy
    if strategy in MORPHO_STRATEGIES:
        kwargs: dict[str, Any] = {}
        if "seed" in MORPHO_STRATEGY_KWARGS[strategy]:
            kwargs["seed"] = degrade.seed
        return morpho_perturb(
            bytes(raster.data),
            raster.width,
            raster.height,
            raster.channels,
            strategy=strategy,
            **kwargs,
        )
    kwargs = {}
    if strategy == "freq-dct":
        kwargs["suppress"] = degrade.strength
    return degrade_image(
        bytes(raster.data),
        raster.width,
        raster.height,
        raster.channels,
        strategy=strategy,
        **kwargs,
    )


def _clean_image_asset(path: Path, dest: Path, plan: CleanPlan) -> CleanResult:
    from common import read_bytes_bounded
    from domain_types import QualityResult, map_quality_to_status
    from verification import verify_boundary_seam, verify_halo, verify_outside_mask

    soft = inspect_soft_binding(path) if plan.inspect_soft_binding else None
    visible_report = None
    verification_status = "VERIFIED"
    final_mask_output = plan.visible.mask_output if plan.visible is not None else None

    with tempfile.TemporaryDirectory(prefix="wm-image-") as temp_dir:
        staging = Path(temp_dir)
        staged_visible = staging / f"visible{path.suffix}"
        staged_metadata = staging / f"metadata{path.suffix}"
        staged_final = staged_metadata
        staged_mask = staging / "effective-mask.pgm"
        mask_details: dict[str, Any] = {}

        if plan.visible is not None:
            staged_visible_plan = replace(plan.visible, mask_output=staged_mask)
            visible_report = remove_visible(
                path,
                staged_visible,
                staged_visible_plan,
                mask_details=mask_details,
            )
            if visible_report["status"] != "completed":
                raise RuntimeError("visible pipeline did not produce an output")
            metadata_source = staged_visible
        else:
            metadata_source = path

        report = clean_image(
            metadata_source,
            staged_metadata,
            strip_all_metadata=plan.strip_all_metadata,
            remove_synthid=plan.remove_synthid,
            synthid_strength=plan.synthid_strength,
            wmct_marker=plan.wmct_marker,
        )

        if plan.degrade is not None:
            cleaned_data = read_bytes_bounded(
                staged_metadata,
                limit=256 * 1024 * 1024,
                label="cleaned image",
            )
            if detect_format(bytes(cleaned_data)) != "png":
                raise ValueError(
                    "degradation requires PNG output; JPEG/HEIF/AVIF inputs are not supported"
                )
            raster = decode_png(bytes(cleaned_data))
            degrade_result = _apply_image_degrade(raster, plan.degrade)
            out_raster = Raster(
                degrade_result.width,
                degrade_result.height,
                degrade_result.channels,
                degrade_result.data,
            )
            staged_final = staging / "final.png"
            atomic_write_bytes(staged_final, encode_png(out_raster))
            report["degrade"] = degrade_result.to_dict()

        if plan.visible is not None:
            try:
                source_data = read_bytes_bounded(
                    path,
                    limit=256 * 1024 * 1024,
                    label="source image",
                )
                final_data = read_bytes_bounded(
                    staged_final,
                    limit=256 * 1024 * 1024,
                    label="final image",
                )
                if (
                    detect_format(bytes(source_data)) != "png"
                    or detect_format(bytes(final_data)) != "png"
                ):
                    raise ValueError("visible verification requires PNG source and output")
                source_raster = decode_png(bytes(source_data))
                final_raster = decode_png(bytes(final_data))
                original_mask = mask_details["original"]
                effective_mask = mask_details["effective"]
                outside_preserved, diff_count = verify_outside_mask(
                    source_raster,
                    final_raster,
                    effective_mask,
                )
                boundary_score = verify_boundary_seam(
                    source_raster,
                    final_raster,
                    effective_mask,
                )
                halo_score, halo_warnings = verify_halo(
                    source_raster,
                    final_raster,
                    original_mask,
                )
                quality = QualityResult(
                    outside_mask_preserved=outside_preserved,
                    outside_mask_difference_count=diff_count,
                    boundary_score=round(boundary_score, 6),
                    halo_score=round(halo_score, 6),
                    warnings=halo_warnings,
                )
                verification_status = map_quality_to_status(
                    outside_mask_modified=not outside_preserved,
                    quality_uncertain=(
                        bool(halo_warnings)
                        or boundary_score >= _BOUNDARY_UNCERTAIN_THRESHOLD
                        or halo_score >= _HALO_UNCERTAIN_THRESHOLD
                    ),
                )
                report["verification"] = {
                    **quality.to_dict(),
                    "status": verification_status,
                }
            except Exception as error:
                verification_status = map_quality_to_status(quality_uncertain=True)
                report["verification"] = {
                    "status": verification_status,
                    "error": str(error),
                }

        mask_data = (
            staged_mask.read_bytes()
            if staged_mask.is_file() and final_mask_output is not None
            else None
        )
        _publish_image_artifacts(
            dest,
            staged_final.read_bytes(),
            mask_output=final_mask_output if mask_data is not None else None,
            mask_data=mask_data,
            backup_output=backup_path(path) if plan.in_place else None,
        )

        report["bytes_out"] = staged_final.stat().st_size
        report["output"] = str(dest)
        if visible_report is not None:
            visible_report["output"] = str(dest)
            # Only rewrite the mask report to the published path when a mask
            # artifact was actually written.  In frictionless mode
            # (publish_mask=False) the external backend still needs a real mask
            # file, so publication is forced for it.
            published = (
                final_mask_output is not None
                and (plan.visible is not None)
                and (plan.visible.publish_mask or plan.visible.backend == "external")
            )
            if published:
                staged_mask_text = str(staged_mask)
                final_mask_text = str(final_mask_output)
                visible_report["mask"] = final_mask_text
                visible_report["actions"] = [
                    action.replace(staged_mask_text, final_mask_text)
                    for action in visible_report["actions"]
                ]

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
        or verification_status != "VERIFIED"
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
    if kind == "unknown":
        kind = "text"
    if plan.visible is not None and kind != "image":
        raise ValueError("visible cleaning is only valid for image assets")
    if plan.degrade is not None and kind != "image":
        raise ValueError("degradation is only valid for image assets")
    if kind == "text":
        return _clean_text_asset(path, dest, plan)
    if kind == "image":
        return _clean_image_asset(path, dest, plan)
    return _clean_container_asset(path, dest, plan)
