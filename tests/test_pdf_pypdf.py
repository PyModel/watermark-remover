"""Structural PDF cleaning tests."""

from __future__ import annotations

import builtins
import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("pypdf")
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import container_meta


def _make_pdf(with_metadata: bool = True, with_features: bool = False) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    if with_features:
        writer.add_outline_item("First page", 0)
        writer.add_attachment("evidence.txt", b"preserve this attachment")
        writer.page_layout = "/TwoColumnLeft"
    if with_metadata:
        writer.add_metadata(
            {
                "/Producer": "OpenAI ChatGPT",
                "/Creator": "c2pa-contentcredentials test",
                "/Author": "Unit Test",
            }
        )
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _disable_exiftool(monkeypatch) -> None:
    monkeypatch.setattr(container_meta, "which", lambda _name: None)


def test_pypdf_clean_strips_docinfo(tmp_path: Path, monkeypatch):
    _disable_exiftool(monkeypatch)
    src = tmp_path / "gen.pdf"
    src.write_bytes(_make_pdf())
    dest = tmp_path / "gen.cleaned.pdf"
    result = container_meta.clean_container(src, dest)
    assert result["format"] == "pdf"
    assert result["meta"]["mode"] == "pypdf"
    out = dest.read_bytes()
    reader = PdfReader(io.BytesIO(out))
    assert len(reader.pages) == 2
    assert reader.metadata is None
    assert b"OpenAI ChatGPT" not in out
    assert b"contentcredentials" not in out


def test_pypdf_clone_preserves_catalog_features(tmp_path: Path, monkeypatch):
    _disable_exiftool(monkeypatch)
    src = tmp_path / "features.pdf"
    src.write_bytes(_make_pdf(with_features=True))
    dest = tmp_path / "features.cleaned.pdf"
    result = container_meta.clean_container(src, dest)
    assert result["meta"]["mode"] == "pypdf"
    reader = PdfReader(str(dest))
    assert len(reader.outline) == 1
    assert "evidence.txt" in reader.attachments
    assert reader.page_layout == "/TwoColumnLeft"
    assert reader.metadata is None


def test_pypdf_clean_preserves_page_count_without_metadata(tmp_path: Path, monkeypatch):
    _disable_exiftool(monkeypatch)
    src = tmp_path / "plain.pdf"
    src.write_bytes(_make_pdf(with_metadata=False))
    dest = tmp_path / "plain.cleaned.pdf"
    container_meta.clean_container(src, dest)
    assert len(PdfReader(str(dest)).pages) == 2


def test_no_structural_cleaner_copies_pdf_byte_exact(tmp_path: Path, monkeypatch):
    """The stdlib fallback must not shift PDF xref/object offsets."""
    _disable_exiftool(monkeypatch)
    src = tmp_path / "source.pdf"
    original = _make_pdf(with_features=True)
    src.write_bytes(original)
    dest = tmp_path / "copy.pdf"

    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    actions, meta = container_meta.clean_pdf_pypdf(src, dest)
    assert meta == {"mode": "copy", "degraded": True}
    assert dest.read_bytes() == original
    assert any("copied unchanged" in action for action in actions)
    # Still parseable with the already imported reader.
    assert len(PdfReader(str(dest)).pages) == 2


def test_clean_pdf_output_is_parseable(tmp_path: Path, monkeypatch):
    _disable_exiftool(monkeypatch)
    src = tmp_path / "gen.pdf"
    src.write_bytes(_make_pdf())
    dest = tmp_path / "gen.cleaned.pdf"
    container_meta.clean_container(src, dest)
    assert dest.read_bytes().startswith(b"%PDF")
    PdfReader(str(dest))
