"""Small, dependency-free helpers for the text-only skill."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = int(os.environ.get("WATERMARKS_MAX_INPUT_BYTES", str(256 << 20)))
MAX_STDIN_BYTES = int(os.environ.get("WATERMARKS_MAX_STDIN_BYTES", str(64 << 20)))
BINARY_SNIFF_BYTES = 8192

BINARY_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "a ZIP container such as DOCX or ODT"),
    (b"%PDF-", "a PDF"),
    (b"\x89PNG\r\n\x1a\n", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF87a", "a GIF image"),
    (b"GIF89a", "a GIF image"),
    (b"RIFF", "a RIFF container"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"7z\xbc\xaf\x27\x1c", "a 7-Zip archive"),
    (b"Rar!\x1a\x07", "a RAR archive"),
    (b"\x7fELF", "an ELF binary"),
)

_ALLOWED_CONTROLS = frozenset({0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1B})


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _configure_stdio() -> None:
    for stream, errors in (
        (sys.stdin, "surrogateescape"),
        (sys.stdout, "backslashreplace"),
        (sys.stderr, "backslashreplace"),
    ):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors=errors)
            except (OSError, ValueError):
                pass


_configure_stdio()


def looks_binary(data: bytes) -> str | None:
    if not data:
        return None
    for magic, label in BINARY_MAGIC:
        if data.startswith(magic):
            return label
    head = data[:BINARY_SNIFF_BYTES]
    if b"\x00" in head:
        return "binary data containing NUL bytes"
    controls = sum(1 for value in head if value < 0x20 and value not in _ALLOWED_CONTROLS)
    if controls / len(head) > 0.05:
        return "binary data dense in control bytes"
    return None


def guard_binary(data: bytes, origin: str, *, allow_binary: bool = False) -> None:
    if allow_binary:
        return
    kind = looks_binary(data)
    if kind is None:
        return
    eprint(f"refusing to treat {origin} as text: it looks like {kind}.")
    eprint("This lightweight skill accepts text files only.")
    raise SystemExit(2)


def _read_stdin_capped(*, allow_binary: bool = False) -> str:
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        text = sys.stdin.read()
        data = text.encode("utf-8", errors="surrogateescape")
        if len(data) > MAX_STDIN_BYTES:
            raise SystemExit(f"refusing stdin input larger than {MAX_STDIN_BYTES} bytes")
        guard_binary(data[:BINARY_SNIFF_BYTES], "stdin", allow_binary=allow_binary)
        return text

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(1 << 20)
        if not chunk:
            break
        if not chunks:
            guard_binary(
                chunk[:BINARY_SNIFF_BYTES],
                "stdin",
                allow_binary=allow_binary,
            )
        total += len(chunk)
        if total > MAX_STDIN_BYTES:
            raise SystemExit(f"refusing stdin input larger than {MAX_STDIN_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="surrogateescape")


def read_text_input(path: str | None, *, allow_binary: bool = False) -> str:
    if path is None or path == "-":
        return _read_stdin_capped(allow_binary=allow_binary)
    source = Path(path)
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise SystemExit(f"refusing input larger than {MAX_INPUT_BYTES} bytes: {path}")
    data = source.read_bytes()
    guard_binary(data, str(source), allow_binary=allow_binary)
    return data.decode("utf-8", errors="surrogateescape")


def _default_file_mode() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def safe_write_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if destination.is_symlink():
        raise OSError(f"refusing to write through symlink: {destination}")

    parent_fd: int | None = None
    if os.name != "nt":
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        parent_fd = os.open(parent, flags)

    parent_identity = os.fstat(parent_fd) if parent_fd is not None else os.lstat(parent)
    fd = -1
    temporary: Path | None = None

    def validate_paths(staged_identity: os.stat_result) -> None:
        current_parent = os.lstat(parent)
        if stat.S_ISLNK(current_parent.st_mode) or not os.path.samestat(
            parent_identity, current_parent
        ):
            raise OSError(f"destination directory changed while writing: {parent}")

        if parent_fd is not None:
            staged = os.stat(temporary.name, dir_fd=parent_fd, follow_symlinks=False)
            try:
                current_destination = os.stat(
                    destination.name, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                current_destination = None
        else:
            staged = os.lstat(temporary)
            try:
                current_destination = os.lstat(destination)
            except FileNotFoundError:
                current_destination = None

        if not os.path.samestat(staged_identity, staged):
            raise OSError(f"temporary path changed while writing: {temporary}")
        if current_destination is None:
            return
        if stat.S_ISLNK(current_destination.st_mode):
            raise OSError(f"refusing to write through symlink: {destination}")
        if os.path.samestat(staged_identity, current_destination):
            raise OSError(f"destination aliases temporary file: {destination}")

    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
        temporary = Path(temporary_name)
        staged_identity = os.fstat(fd)
        validate_paths(staged_identity)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, _default_file_mode())
        handle = os.fdopen(fd, "wb")
        fd = -1
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        validate_paths(staged_identity)
        if parent_fd is not None:
            os.replace(
                temporary.name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        else:
            os.replace(temporary, destination)
        temporary = None
    except BaseException:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)
        if temporary is not None:
            try:
                if parent_fd is not None:
                    os.unlink(temporary.name, dir_fd=parent_fd)
                else:
                    os.unlink(temporary)
            except OSError:
                pass
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def write_text_output(text: str, path: str | None) -> None:
    if path is None or path == "-":
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    safe_write_bytes(path, text.encode("utf-8", errors="surrogateescape"))


def backup_path(source: Path) -> Path:
    backup = source.with_suffix(source.suffix + ".bak")
    try:
        safe_write_bytes(backup, source.read_bytes())
    except OSError as error:
        eprint(f"cannot create backup {backup}: {error}")
        raise SystemExit(2) from error
    return backup


def emit_json(data: Any) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def cleaned_path(source: Path, suffix: str = ".cleaned") -> Path:
    return source.with_name(f"{source.stem}{suffix}{source.suffix}")
