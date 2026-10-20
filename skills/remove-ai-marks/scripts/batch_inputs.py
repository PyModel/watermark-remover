"""Shared batch input discovery for inspect/clean CLIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath


@dataclass(frozen=True, slots=True)
class InputItem:
    path: Path
    relative: Path  # stable output path below a batch output root


@dataclass(frozen=True, slots=True)
class InputSelection:
    items: tuple[InputItem, ...]
    batch: bool


#: The cleaning stages a batch mode selects. ``metadata`` is the default and
#: preserves today's behavior; ``visible`` runs the registry-driven visible-mark
#: pass without the metadata stage; ``all`` runs both.
BATCH_MODES = ("metadata", "visible", "all")
DEFAULT_BATCH_MODE = "metadata"


@dataclass(frozen=True, slots=True)
class BatchModePolicy:
    """Which stages a batch mode applies, as request-field overrides.

    ``visible`` selects the registry-driven visible-mark pass and
    ``strip_metadata`` selects the metadata stage; they are independent so
    ``visible`` is "visible marks only" and ``all`` is "both".
    """

    mode: str
    visible: bool
    strip_metadata: bool

    @property
    def stages(self) -> tuple[str, ...]:
        """The stage names this mode runs, in pipeline order."""
        return tuple(
            name
            for name, enabled in (("visible", self.visible), ("metadata", self.strip_metadata))
            if enabled
        )


_BATCH_MODE_POLICIES = {
    "metadata": BatchModePolicy("metadata", visible=False, strip_metadata=True),
    "visible": BatchModePolicy("visible", visible=True, strip_metadata=False),
    "all": BatchModePolicy("all", visible=True, strip_metadata=True),
}


def resolve_batch_mode(mode: str | None) -> BatchModePolicy:
    """The policy for *mode*; ``None`` (no flag) is the metadata default.

    An unknown mode raises rather than falling back to the default: a typo must
    never run a different stage set than the one that was asked for.
    """
    if mode is None:
        return _BATCH_MODE_POLICIES[DEFAULT_BATCH_MODE]
    try:
        return _BATCH_MODE_POLICIES[mode]
    except KeyError:
        expected = ", ".join(BATCH_MODES)
        raise ValueError(f"unknown batch mode {mode!r}; expected one of {expected}") from None


def batch_summary(results: Sequence[dict], *, mode: str | None = None) -> dict:
    """The batch JSON envelope: every file's outcome, the mode, and the counts.

    A file whose payload carries a non-zero (or absent) ``exit_code`` is counted
    failed and labelled so; every file is counted exactly once, so ``done`` plus
    ``failed`` always equals ``total`` even when one file fails and the rest
    succeed. A per-file ``status`` the pipeline already set (a dry-run) is kept
    rather than overwritten.
    """
    policy = resolve_batch_mode(mode)
    counted = {"done": 0, "failed": 0}
    decorated: list[dict] = []
    for result in results:
        outcome = "failed" if result.get("exit_code", 1) != 0 else "done"
        counted[outcome] += 1
        decorated.append({**result, "status": result.get("status", outcome)})
    return {
        "total": len(decorated),
        "mode": policy.mode,
        "stages": list(policy.stages),
        "counts": {**counted, "total": len(decorated)},
        "results": decorated,
    }


def is_generated(path: Path) -> bool:
    name = path.name
    return path.suffix == ".bak" or ".cleaned." in name or ".mask." in name


def _validate_pattern(pattern: str) -> None:
    parts = PurePath(pattern).parts
    if not pattern or Path(pattern).is_absolute() or ".." in parts:
        raise ValueError("glob must be a non-empty relative pattern without '..'")


def safe_output_path(root: Path, relative: Path) -> Path:
    """Build a batch destination without following symlinks below ``root``.

    Existing symlink components are rejected so a shared or pre-populated output
    tree cannot redirect a write outside the requested root. The caller remains
    responsible for creating directories after this check.
    """
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative output path: {relative}")

    candidate = root / relative
    current = root
    if current.is_symlink():
        raise ValueError(f"output root is a symlink: {root}")
    if current.exists() and not current.is_dir():
        raise ValueError(f"output root is not a directory: {root}")

    for part in relative.parts:
        if part in ("", "."):
            continue
        current /= part
        if current.is_symlink():
            raise ValueError(f"output path contains a symlink: {current}")
    return candidate


def select_inputs(
    sources: Iterable[Path],
    *,
    recursive: bool,
    pattern: str,
    extensions: set[str] | frozenset[str],
    excluded_roots: Iterable[Path] = (),
) -> InputSelection:
    """Validate sources, discover supported files, and determine batch mode."""
    _validate_pattern(pattern)
    roots = tuple(sources)
    if not roots:
        raise ValueError("no input sources")
    invalid = [
        source
        for source in roots
        if source.is_symlink() or not source.exists() or not (source.is_file() or source.is_dir())
    ]
    if invalid:
        rendered = ", ".join(str(source) for source in invalid)
        raise ValueError(f"not a regular file or directory: {rendered}")
    multiple_roots = len(roots) > 1
    allowed = {extension.lower() for extension in extensions}
    excluded = tuple(root.resolve() for root in excluded_roots)
    seen: set[Path] = set()
    items: list[InputItem] = []
    for source in roots:
        if source.is_dir():
            iterator = source.rglob(pattern) if recursive else source.glob(pattern)
            for path in sorted(iterator):
                resolved = path.resolve()
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.suffix.lower() not in allowed
                    or is_generated(path)
                    or resolved in seen
                    or any(resolved.is_relative_to(root) for root in excluded)
                ):
                    continue
                relative = path.relative_to(source)
                if multiple_roots:
                    relative = Path(source.name) / relative
                items.append(InputItem(path, relative))
                seen.add(resolved)
        else:
            resolved = source.resolve()
            if resolved not in seen:
                items.append(InputItem(source, Path(source.name)))
                seen.add(resolved)

    if not items:
        raise ValueError("no matching input files")
    return InputSelection(
        items=tuple(items),
        batch=len(items) > 1 or any(source.is_dir() for source in roots),
    )
