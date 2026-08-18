"""Behavioral contract for shared asset-kind routing."""

from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _routing_module():
    return importlib.import_module("asset_kind")


def _ftyp(brand: bytes) -> bytes:
    payload = brand + b"\x00\x00\x00\x00" + brand
    return (len(payload) + 8).to_bytes(4, "big") + b"ftyp" + payload


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("known.txt", b"\x89PNG\r\n\x1a\n", "text"),
        ("known.PNG", b"plain text", "image"),
        ("known.pdf", b"plain text", "container"),
        ("unknown.bin", b"\x89PNG\r\n\x1a\n", "image"),
        ("unknown-jpeg.blob", b"\xff\xd8\xff\xe0", "image"),
        ("unknown-heif.blob", _ftyp(b"heic"), "image"),
        ("unknown-avif.blob", _ftyp(b"avif"), "image"),
        ("unknown-svg.blob", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', "container"),
        ("unknown.data", b"%PDF-1.7\n", "container"),
        ("unknown.payload", b"plain text", "unknown"),
    ],
)
def test_classify_asset_preserves_override_extension_sniffing_order(
    tmp_path: Path, name: str, content: bytes, expected: str
):
    path = tmp_path / name
    path.write_bytes(content)

    assert _routing_module().classify_asset(path) == expected


def test_classify_asset_forced_kind_wins_over_extension_and_bytes(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert _routing_module().classify_asset(path, forced_kind="container") == "container"


def test_classify_asset_rejects_unknown_forced_kind(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported forced asset kind"):
        _routing_module().classify_asset(path, forced_kind="audio")


@pytest.mark.parametrize(
    "members",
    [
        ("[Content_Types].xml", "word/document.xml"),
        ("content.xml", "meta.xml"),
    ],
)
def test_classify_asset_detects_unknown_suffix_zip_container(
    tmp_path: Path, members: tuple[str, str]
):
    path = tmp_path / "document.payload"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(members[0], "<root/>")
        archive.writestr("payload.bin", b"x" * 8_192)
        archive.writestr(members[1], "<root/>")

    assert _routing_module().classify_asset(path) == "container"


def test_classify_asset_avoids_unbounded_read_for_unknown_suffix(tmp_path: Path, monkeypatch):
    path = tmp_path / "extensionless"
    path.write_bytes(b"plain text" + b"x" * 8_192)

    def reject_unbounded_read(_path):
        raise AssertionError("classification must not read the whole file")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    assert _routing_module().classify_asset(path) == "unknown"
