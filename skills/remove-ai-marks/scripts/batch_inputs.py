"""Shared batch input discovery for inspect/clean CLIs."""

from __future__ import annotations

from collections.abc import Iterable
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
    invalid = [
        source
        for source in roots
        if source.is_symlink() or not source.exists() or not (source.is_file() or source.is_dir())
    ]
    if invalid:
        rendered = ", ".join(str(source) for source in invalid)
        raise ValueError(f"not a regular file or directory: {rendered}")
    if not roots:
        raise ValueError("no input sources")

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
