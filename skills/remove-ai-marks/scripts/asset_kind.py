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
    if detect_image_format(prefix) in (
        "png",
        "jpeg",
        "webp",
        "avif",
        "heic",
        "heif",
        "bmp",
        "gif",
        "tiff",
    ):
        return "image"
    if detect_container_format(path, prefix) != "unknown":
        return "container"
    # No extension names a known format and no magic matched: refuse unless
    # the caller forces a kind, so unknown binaries are never mangled as text.
    return "unknown"
