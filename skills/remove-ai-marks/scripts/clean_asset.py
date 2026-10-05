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

import humanizer
import tiling
from asset_kind import AssetKind, classify_asset
from common import (
    atomic_write_bytes,
    atomic_write_text,
    backup_path,
    create_backup,
    paths_alias,
    read_bytes_bounded,
    validate_output_path,
)
from container_meta import clean_container
from dct_frequency import DEGRADE_STRATEGIES, degrade_image
from image_io import PixelDependencyUnavailable, _import_pixel
from image_meta import clean_image, detect_format
from inspect_soft_binding import inspect_soft_binding
from morpho_perturb import MORPHO_STRATEGIES, MORPHO_STRATEGY_KWARGS, morpho_perturb
from morphomod import (
    VISIBLE_CLEAN_BACKENDS,
    Mask,
    Raster,
    VisiblePlan,
    decode_png,
    decode_to_raster,
    encode_png,
    remove_visible,
    write_pgm,
)
from perturb_text import MODES as PERTURB_MODES
from perturb_text import perturb_text
from pipeline_actions import Action, ActionCode, Effect, retarget_report
from rewrite_text import RewritePlan, TokenSink, rewrite
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
                if hasattr(os, "fchmod"):
                    os.fchmod(backup_fd, source_snapshot.mode)
                else:
                    os.chmod(backup_temp, source_snapshot.mode)
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
    humanize: bool = False
    humanize_seed: int | None = None

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
        if not isinstance(self.humanize, bool):
            raise TypeError("humanize must be a bool")
        if self.humanize_seed is not None and (
            isinstance(self.humanize_seed, bool) or not isinstance(self.humanize_seed, int)
        ):
            raise TypeError("humanize_seed must be an integer or None")


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


def _clean_text_asset(
    path: Path, dest: Path, plan: CleanPlan, on_token: TokenSink | None = None
) -> CleanResult:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    cleaned, stats = clean_text(
        text,
        nfkc=plan.text.nfkc,
        aggressive_homoglyphs=plan.text.aggressive_homoglyphs,
        preserve_semantic=plan.text.preserve_semantic,
    )
    if plan.text.rewrite_plan is not None:
        cleaned, rewrite_info = rewrite(cleaned, plan.text.rewrite_plan, on_token=on_token)
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


def _crop_mask(mask: Mask, tile: tiling.Tile) -> Mask:
    """The whole-image mask cropped to *tile* (pure row slicing)."""
    data = bytearray(tile.width * tile.height)
    for row in range(tile.height):
        source = (tile.y + row) * mask.width + tile.x
        data[row * tile.width : (row + 1) * tile.width] = mask.data[source : source + tile.width]
    return Mask(tile.width, tile.height, data)


def _probe_dimensions(path: Path) -> tuple[int, int] | None:
    """Pixel dimensions from the PNG/JPEG header, no decoder involved.

    The visible pipeline's raster decoder (``decode_to_raster``) accepts only
    PNG and JPEG, so a pure header parse covers every image that could take
    the tiled path. Returns None for anything else (or a malformed header);
    the untiled path then produces its own, real, error. Deliberately does
    not touch numpy/cv2: probing must stay available in pixel-less
    environments and must not add stub-cv2 surface to the CLI path.
    """
    data = read_bytes_bounded(path, limit=256 * 1024, label="image header")
    data = bytes(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                return None
            marker = data[offset + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # standalone markers
                offset += 2
                continue
            segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if offset + 9 > len(data):
                    return None
                height = int.from_bytes(data[offset + 5 : offset + 7], "big")
                width = int.from_bytes(data[offset + 7 : offset + 9], "big")
                return width, height
            offset += 2 + segment_length
    return None


def _run_tiled_visible(
    path: Path,
    staged_visible: Path,
    staged_visible_plan: VisiblePlan,
    mask_details: dict[str, Mask],
    staging: Path,
) -> dict[str, Any]:
    """Run the visible clean per tile above the auto-tile threshold.

    The mask is computed ONCE on the whole image -- through a ``print-plan``
    ``remove_visible`` call, so the mask-source logic, the ``mask_details``
    contract, and the mask actions are the untiled path's own -- then every
    tile is filled by a plain ``remove_visible`` call on the tile crop with the
    already-refined mask (radius-0 re-refine is idempotent) and the results are
    feather-blended (see ``tiling`` for the same-result guarantee). Tile
    reports are folded into one honest report: the whole-image mask actions
    plus a single tiled-inpaint action, and the engagement dict.
    """
    data = read_bytes_bounded(path, limit=256 * 1024 * 1024, label="source image")
    fmt = detect_format(bytes(data))
    raster = decode_to_raster(bytes(data), fmt)
    width, height, channels = raster.width, raster.height, raster.channels

    mask_report = remove_visible(
        path,
        None,
        replace(staged_visible_plan, backend="print-plan"),
        mask_details=mask_details,
    )
    if mask_report["status"] != "mask-ready":
        return mask_report
    refined = mask_details["effective"]

    tile_size = tiling.auto_tile_size()
    overlap = tiling.auto_tile_overlap()
    tiles = tiling.plan_tiles(width, height, tile_size, overlap)
    np = _import_pixel("numpy")
    arr = np.frombuffer(bytes(raster.data), dtype=np.uint8).reshape(height, width, channels)

    def clean_tile(tile: tiling.Tile, crop: Any) -> Any:
        tag = f"{tile.x}-{tile.y}"
        tile_in = staging / f"tile-{tag}-in.png"
        tile_out = staging / f"tile-{tag}-out.png"
        tile_mask = staging / f"tile-{tag}-mask.pgm"
        atomic_write_bytes(
            tile_in,
            encode_png(Raster(tile.width, tile.height, channels, bytearray(crop.tobytes()))),
        )
        tile_mask_crop = _crop_mask(refined, tile)
        if tile_mask_crop.marked == 0:
            # No part of the mark lives in this tile: pass the crop through
            # unchanged (an identity tile keeps the blend exact).
            return crop
        write_pgm(tile_mask_crop, tile_mask)
        # The whole-image mask is already refined; radius-0 re-refinement is
        # the identity on it, so tile fills see exactly the untiled mask.
        tile_plan = replace(
            staged_visible_plan,
            mask_path=tile_mask,
            box=None,
            detect_command=None,
            mask_output=staging / f"tile-{tag}-mask-out.pgm",
            dilation_radius=0,
        )
        tile_report = remove_visible(tile_in, tile_out, tile_plan)
        if tile_report["status"] != "completed":
            raise RuntimeError(
                f"tiled visible clean failed on tile ({tag}): {tile_report['status']}"
            )
        tile_raster = decode_png(tile_out.read_bytes())
        return np.frombuffer(bytes(tile_raster.data), dtype=np.uint8).reshape(
            tile_raster.height, tile_raster.width, tile_raster.channels
        )

    blended = tiling.run_tiled(
        clean_tile,
        arr,
        tile_size,
        overlap,
        set_progress=lambda line: None,
    )
    atomic_write_bytes(
        staged_visible,
        encode_png(Raster(width, height, channels, bytearray(blended.tobytes()))),
    )

    details = [
        detail for detail in mask_report["action_details"] if detail["code"] != "inpaint_skipped"
    ]
    inpaint_action = Action(
        ActionCode.INPAINT,
        Effect.CHANGE,
        f"tiled {staged_visible_plan.backend} inpaint over {len(tiles)} "
        f"{tiles[0].width}x{tiles[0].height} tiles (overlap {overlap}) + feather blend",
        {
            "backend": staged_visible_plan.backend,
            "tiles": len(tiles),
            "tile_size": tile_size,
            "overlap": overlap,
        },
    )
    details.append(inpaint_action.to_dict())
    return {
        "status": "completed",
        "input": str(path),
        "output": str(staged_visible),
        "format": fmt,
        "backend": staged_visible_plan.backend,
        "mask": mask_report["mask"],
        "initial_mask_pixels": mask_report["initial_mask_pixels"],
        "refined_mask_pixels": mask_report["refined_mask_pixels"],
        "dilation_radius": staged_visible_plan.dilation_radius,
        "actions": [detail["text"] for detail in details],
        "action_details": details,
        "tiling": {
            "engaged": True,
            "threshold": tiling.auto_tile_threshold(),
            "tiles": len(tiles),
            "tile_size": tile_size,
            "overlap": overlap,
            "note": (
                "per-tile fills feather-blended; identical to whole-image processing "
                "whenever the per-tile results agree across tile seams (see tiling docstring)"
            ),
        },
        "note": mask_report["note"],
    }


def _apply_humanize_stage(
    staged_final: Path, reference_path: Path, seed: int | None
) -> dict[str, Any]:
    """Apply the ``--humanize`` post-stage to the staged PNG output, in place.

    Runs after verification on purpose (upstream order): the verification
    speaks about the fill, and grain everywhere would fail outside-mask by
    construction. Requires PNG output (the degrade precedent) so the stage
    never silently re-encodes a lossy format.
    """
    try:
        np = _import_pixel("numpy")
        cv2 = _import_pixel("cv2")
    except PixelDependencyUnavailable as error:
        return {
            "status": "unavailable",
            "hint": f"Install watermark-remover[pixel] to enable the humanize stage. Reason: {error}",
        }
    final_data = read_bytes_bounded(staged_final, limit=256 * 1024 * 1024, label="final image")
    if detect_format(bytes(final_data)) != "png":
        raise ValueError("--humanize requires PNG output; JPEG/HEIF/AVIF inputs are not supported")
    raster = decode_png(bytes(final_data))
    if raster.channels != 3:
        raise ValueError("--humanize supports 3-channel output; alpha images are not supported")
    image = np.frombuffer(bytes(raster.data), dtype=np.uint8).reshape(
        raster.height, raster.width, 3
    )
    ref_data = read_bytes_bounded(reference_path, limit=256 * 1024 * 1024, label="source image")
    ref_raster = decode_to_raster(bytes(ref_data), detect_format(bytes(ref_data)))
    if ref_raster.channels != 3:
        raise ValueError("--humanize requires a 3-channel source image")
    reference = np.frombuffer(bytes(ref_raster.data), dtype=np.uint8).reshape(
        ref_raster.height, ref_raster.width, 3
    )

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
    if ref_bgr.shape[:2] != bgr.shape[:2]:
        ref_bgr = cv2.resize(
            ref_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )
    out_bgr, stage_report = humanizer.humanize_stage(bgr, ref_bgr, seed=seed)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    atomic_write_bytes(
        staged_final,
        encode_png(
            Raster(
                raster.width, raster.height, 3, bytearray(np.ascontiguousarray(out_rgb).tobytes())
            )
        ),
    )
    return {"status": "applied", **stage_report, "seed": seed}


def _clean_image_asset(path: Path, dest: Path, plan: CleanPlan) -> CleanResult:
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
            try:
                dims = _probe_dimensions(path)
                use_tiled = dims is not None and tiling.should_auto_tile(*dims)
            except PixelDependencyUnavailable:
                use_tiled = False
            if use_tiled:
                try:
                    visible_report = _run_tiled_visible(
                        path,
                        staged_visible,
                        staged_visible_plan,
                        mask_details,
                        staging,
                    )
                except PixelDependencyUnavailable as error:
                    # The tiled attempt is an optimization: degrade to the
                    # whole-image clean and say so, never crash.
                    visible_report = remove_visible(
                        path,
                        staged_visible,
                        staged_visible_plan,
                        mask_details=mask_details,
                    )
                    visible_report["tiling"] = {
                        "engaged": False,
                        "status": "unavailable",
                        "hint": (
                            "Install watermark-remover[pixel] to enable tiled "
                            f"processing. Reason: {error}"
                        ),
                    }
            else:
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

        if plan.humanize:
            report["humanize"] = _apply_humanize_stage(staged_final, path, plan.humanize_seed)

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
                retarget_report(visible_report, staged_mask_text, final_mask_text)

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


def clean_asset(
    path: Path, dest: Path, plan: CleanPlan, *, on_token: TokenSink | None = None
) -> CleanResult:
    """Clean one asset; raise failures and leave presentation to the caller.

    *on_token* is a display-only sink for Layer B generation fragments.  It is
    deliberately a call argument rather than a ``CleanPlan`` field: the plan is a
    frozen, serialisable description of *what* to do, and a live callback is
    neither frozen nor serialisable.
    """
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
        # Contract (asset_kind.classify_asset): unknown binaries are never
        # mangled as text. The CLI preflight refuses earlier; any direct
        # caller gets the same refusal here.
        raise ValueError(
            f"refusing to clean {path}: unrecognized format "
            "(pass --force-text to handle the bytes as text anyway, or --as to force a format)"
        )
    if plan.visible is not None and kind != "image":
        raise ValueError("visible cleaning is only valid for image assets")
    if plan.degrade is not None and kind != "image":
        raise ValueError("degradation is only valid for image assets")
    if plan.humanize and kind != "image":
        raise ValueError("humanize is only valid for image assets")
    if kind == "text":
        return _clean_text_asset(path, dest, plan, on_token)
    if kind == "image":
        return _clean_image_asset(path, dest, plan)
    return _clean_container_asset(path, dest, plan)
