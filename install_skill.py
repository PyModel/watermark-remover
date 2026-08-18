#!/usr/bin/env python3
"""Install the lightweight text skill for Cursor."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

SKILL_NAME = "clean-user-facing-text"
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "skills" / SKILL_NAME


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _stage(skills_dir: Path) -> tuple[Path, Path]:
    skills_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{SKILL_NAME}.staging.", dir=skills_dir)
    )
    staged_skill = staging_root / SKILL_NAME
    try:
        shutil.copytree(SOURCE, staged_skill)
        if not (staged_skill / "SKILL.md").is_file():
            raise RuntimeError("staged skill is missing SKILL.md")
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root, staged_skill


def _install(destination: Path, force: bool) -> tuple[bool, Path | None]:
    if _present(destination) and not force:
        print(f"already exists: {destination}", file=sys.stderr)
        print(
            "No changes made. Re-run with --force to back up and replace.",
            file=sys.stderr,
        )
        return False, None

    staging_root, staged_skill = _stage(destination.parent)
    backup: Path | None = None
    try:
        if _present(destination):
            backup = destination.with_name(
                f"{destination.name}.backup.{uuid.uuid4().hex[:12]}"
            )
            os.replace(destination, backup)
        try:
            os.replace(staged_skill, destination)
        except BaseException:
            if backup is not None and not _present(destination):
                os.replace(backup, destination)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return True, backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Back up and replace an existing Cursor installation",
    )
    parser.add_argument("--cursor-home", help="Override Cursor home (default: ~/.cursor)")
    args = parser.parse_args()

    cursor_home = Path(
        args.cursor_home
        or os.environ.get("CURSOR_HOME", Path.home() / ".cursor")
    ).expanduser()
    destination = cursor_home / "skills" / SKILL_NAME

    installed, backup = _install(destination, args.force)
    if not installed:
        return 1
    if backup is not None:
        print(f"Cursor: backed up existing skill to {backup}")
    print(f"Cursor: installed {destination}")
    print("Start a new Cursor session if the skill does not appear automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
