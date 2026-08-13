"""Shared batch input discovery for inspect/clean CLIs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputItem:
    path: Path
    relative: Path  # stable output path below a batch output root


def is_generated(path: Path) -> bool:
    name = path.name
    return path.suffix == ".bak" or ".cleaned." in name or ".mask." in name


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


def collect_inputs(
    sources: Iterable[Path],
    *,
    recursive: bool,
    pattern: str,
    extensions: set[str] | frozenset[str],
) -> list[InputItem]:
    """Discover unique supported files and assign collision-resistant relative paths.

    A single directory keeps paths relative to that directory. Multiple roots
    are namespaced by each root's basename. Explicit files retain their name.
    Missing sources are ignored; the CLI reports them before calling here.
    """
    roots = list(sources)
    multiple_roots = len(roots) > 1
    allowed = {e.lower() for e in extensions}
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
                ):
                    continue
                relative = path.relative_to(source)
                if multiple_roots:
                    relative = Path(source.name) / relative
                items.append(InputItem(path, relative))
                seen.add(resolved)
        elif source.is_file() and not source.is_symlink():
            resolved = source.resolve()
            if resolved not in seen:
                items.append(InputItem(source, Path(source.name)))
                seen.add(resolved)
    return items
