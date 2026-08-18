"""Shared asset-kind routing for clean, inspect, and demo entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from container_meta import detect_container_format
from image_meta import detect_format as detect_image_format

AssetKind = Literal["text", "image", "container", "unknown"]

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif", ".bmp", ".gif", ".tiff", ".tif"}
)
_CONTAINER_EXTENSIONS = frozenset(
    {
        ".svg",
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".odt",
        ".epub",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".mdx",
    }
)
_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".text",
        ".css",
        ".js",
        ".py",
        ".rs",
        ".go",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
    }
)
SUPPORTED_EXTENSIONS = _IMAGE_EXTENSIONS | _CONTAINER_EXTENSIONS | _TEXT_EXTENSIONS

_ASSET_KINDS: tuple[AssetKind, ...] = ("text", "image", "container", "unknown")
_SNIFF_BYTES = 4096  # HEIF brand detection scans at most the first 4 KiB.

#: image_meta.detect_format values that route to the image pipeline. The
#: HEIF/HEIC family is reported as "heif" (see heif_meta.detect_heif), never
#: "heic" — keep one copy so the three sniffers below cannot drift apart.
_IMAGE_FORMAT_NAMES = frozenset({"png", "jpeg", "webp", "avif", "heif", "bmp", "gif", "tiff"})

#: Bytes read for header-only sniffing. Every supported image/container
#: magic lives in the prefix; zip-based containers (docx/odt/...) need the
#: full central directory, which sits at the end of the archive, so only a
#: PK header triggers a whole-file read.
CLASSIFY_HEADER_BYTES = 4096


def classify_bytes(data: bytes, suffix: str | None = None) -> AssetKind:
    """Classify *data* by extension first, then by magic bytes."""
    ext = (suffix or "").lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _CONTAINER_EXTENSIONS:
        return "container"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    if detect_image_format(data) in _IMAGE_FORMAT_NAMES:
        return "image"
    if data:
        sniff_path = Path("input") if not ext else Path(f"input{ext}")
        if detect_container_format(sniff_path, data) != "unknown":
            return "container"
    return "unknown"


def classify(path: Path) -> AssetKind:
    """Classify a file on disk by extension, then by its bytes."""
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _CONTAINER_EXTENSIONS:
        return "container"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    with path.open("rb") as fh:
        head = fh.read(CLASSIFY_HEADER_BYTES)
    if detect_image_format(head) in _IMAGE_FORMAT_NAMES:
        return "image"
    if head:
        data = path.read_bytes() if head[:4] == b"PK" else head
        sniff_path = Path("input") if not ext else Path(f"input{ext}")
        if detect_container_format(sniff_path, data) != "unknown":
            return "container"
    return "unknown"


def classify_asset(path: Path, *, forced_kind: str = "auto") -> AssetKind:
    """Return the processing family for a caller-validated regular file.

    Explicit selection wins, followed by known extension, bounded magic-byte
    detection, then the historical text fallback. Concrete format parsing stays
    in the image and container modules.
    """
    if forced_kind != "auto":
        if forced_kind not in _ASSET_KINDS:
            raise ValueError(f"unsupported forced asset kind: {forced_kind}")
        return cast(AssetKind, forced_kind)

    extension = path.suffix.lower()
    if extension in _IMAGE_EXTENSIONS:
        return "image"
    if extension in _CONTAINER_EXTENSIONS:
        return "container"
    if extension in _TEXT_EXTENSIONS:
        return "text"

    with path.open("rb") as source:
        prefix = source.read(_SNIFF_BYTES)
    if detect_image_format(prefix) in _IMAGE_FORMAT_NAMES:
        return "image"
    if detect_container_format(path, prefix) != "unknown":
        return "container"
    # No extension names a known format and no magic matched: refuse unless
    # the caller forces a kind, so unknown binaries are never mangled as text.
    return "unknown"
