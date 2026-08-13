"""Shared helpers for remove-ai-marks scripts."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_HTTP_JSON_LIMIT = 16 * 1024 * 1024


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def read_bool_env(name: str, default: bool = False) -> bool:
    """Read an optional environment flag without treating arbitrary text as true."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def read_text_input(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="surrogateescape")


def write_text_output(text: str, path: str | None) -> None:
    if path is None or path == "-":
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    atomic_write_text(Path(path), text)


def paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths name the same file, including hard links."""
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def validate_output_path(source: Path, dest: Path) -> None:
    """Reject implicit in-place writes and destination symlinks."""
    if dest.is_symlink():
        raise ValueError(f"output path is a symlink: {dest}")
    if paths_alias(source, dest):
        raise ValueError("output aliases input; use --in-place for a guarded overwrite")
    if dest.exists() and not dest.is_file():
        raise ValueError(f"output path is not a regular file: {dest}")


def backup_path(source: Path) -> Path:
    return source.with_suffix(source.suffix + ".bak")


def create_backup(source: Path) -> Path:
    """Create a durable, byte-exact backup without following or replacing links."""
    dest = backup_path(source)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW

    source_fd = os.open(source, read_flags)
    dest_fd: int | None = None
    created = False
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"not a regular file: {source}")
        dest_fd = os.open(dest, flags, stat.S_IMODE(source_stat.st_mode))
        created = True
        with (
            os.fdopen(source_fd, "rb", closefd=False) as source_file,
            os.fdopen(dest_fd, "wb", closefd=False) as dest_file,
        ):
            while chunk := source_file.read(1024 * 1024):
                dest_file.write(chunk)
            dest_file.flush()
            os.fsync(dest_fd)
        os.fchmod(dest_fd, stat.S_IMODE(source_stat.st_mode))
    except Exception:
        if dest_fd is not None:
            os.close(dest_fd)
            dest_fd = None
        if created:
            dest.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
        if dest_fd is not None:
            os.close(dest_fd)
    return dest


def atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Publish bytes atomically without following a destination symlink."""
    if dest.is_symlink():
        raise ValueError(f"output path is a symlink: {dest}")
    existing_mode = stat.S_IMODE(dest.stat().st_mode) if dest.exists() else None
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if existing_mode is not None:
            os.chmod(temp, existing_mode)
        if dest.is_symlink():
            raise ValueError(f"output path became a symlink: {dest}")
        os.replace(temp, dest)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def read_bytes_bounded(path: Path, limit: int, *, label: str = "file") -> bytes:
    """Read a regular file through one descriptor, rejecting growth past ``limit``."""
    if limit < 0:
        raise ValueError("read limit must be non-negative")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if metadata.st_size > limit:
            raise ValueError(f"{label} exceeds safety limit of {limit:,} bytes")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"{label} exceeds safety limit of {limit:,} bytes")
        return data
    finally:
        os.close(fd)


def atomic_write_text(
    dest: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    errors: str = "surrogateescape",
) -> None:
    atomic_write_bytes(dest, text.encode(encoding, errors=errors))


def emit_json(data: Any) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def cleaned_path(src: Path, suffix: str = ".cleaned") -> Path:
    """path/to/file.ext -> path/to/file.cleaned.ext"""
    return src.with_name(f"{src.stem}{suffix}{src.suffix}")


def which(cmd: str) -> str | None:
    from shutil import which as _which

    return _which(cmd)
