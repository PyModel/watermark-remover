"""Tests for the qpdf structural rewrite that follows the exiftool PDF pass."""

from __future__ import annotations

import builtins
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import container_meta
from container_meta import clean_pdf


def _ai_pdf() -> bytes:
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << >> >>",
        b"<< /Producer (Claude Opus) /Creator (Anthropic Claude) >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 4 0 R >>\n" % (len(objs) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def _fake_tools(
    monkeypatch,
    *,
    qpdf: bool,
    qpdf_rc: int = 0,
    qpdf_writes: bool = True,
    exiftool_rc: int = 0,
):
    seen: list[list[str]] = []

    def fake_which(cmd: str):
        if cmd == "exiftool":
            return "/fake/bin/exiftool"
        if cmd == "qpdf":
            return "/fake/bin/qpdf" if qpdf else None
        return None

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        if cmd[0].endswith("qpdf"):
            if qpdf_writes:
                Path(cmd[-1]).write_bytes(b"%PDF-1.4\n% rebuilt\n%%EOF\n")
            return SimpleNamespace(returncode=qpdf_rc, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=exiftool_rc, stdout=b"", stderr=b"")

    monkeypatch.setattr(container_meta, "which", fake_which)
    monkeypatch.setattr(container_meta, "run_command", fake_run)
    return seen


def test_without_qpdf_the_incremental_leak_is_reported(monkeypatch, tmp_path: Path):
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    src.write_bytes(_ai_pdf())
    _fake_tools(monkeypatch, qpdf=False)
    actions, meta = clean_pdf(src, dest)
    assert meta["structural_rewrite"] is False
    assert any("incremental" in a for a in actions), actions
    assert any("qpdf" in a for a in actions), actions


def test_with_qpdf_the_document_is_rebuilt(monkeypatch, tmp_path: Path):
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    src.write_bytes(_ai_pdf())
    seen = _fake_tools(monkeypatch, qpdf=True)
    _actions, meta = clean_pdf(src, dest)
    assert meta["structural_rewrite"] is True
    qpdf_cmd = [c for c in seen if c[0].endswith("qpdf")]
    assert len(qpdf_cmd) == 1
    assert "--linearize" in qpdf_cmd[0]
    assert dest.read_bytes() == b"%PDF-1.4\n% rebuilt\n%%EOF\n"
    assert not list(tmp_path.glob("*.qpdf-tmp"))


def test_qpdf_warning_exit_code_still_counts(monkeypatch, tmp_path: Path):
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    src.write_bytes(_ai_pdf())
    _fake_tools(monkeypatch, qpdf=True, qpdf_rc=3)
    actions, meta = clean_pdf(src, dest)
    assert meta["structural_rewrite"] is True
    assert any("rc=3" in a for a in actions), actions


def test_qpdf_failure_keeps_exiftool_output_and_warns(monkeypatch, tmp_path: Path):
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    src.write_bytes(_ai_pdf())
    _fake_tools(monkeypatch, qpdf=True, qpdf_rc=2, qpdf_writes=False)
    actions, meta = clean_pdf(src, dest)
    assert meta["structural_rewrite"] is False
    assert dest.is_file()
    assert any("recoverable" in a for a in actions), actions
    assert not list(tmp_path.glob("*.qpdf-tmp"))


def test_exiftool_failure_falls_back_to_pypdf(monkeypatch, tmp_path: Path):
    # Regression: exiftool present but failing used to publish the original
    # bytes under mode "exiftool" with no degraded flag and no fallback.
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    src.write_bytes(_ai_pdf())
    seen = _fake_tools(monkeypatch, qpdf=True, exiftool_rc=1)

    actions, meta = clean_pdf(src, dest)

    assert meta == {"mode": "pypdf", "degraded": False}
    assert any("exiftool degraded (rc=1)" in a for a in actions), actions
    assert any("pypdf" in a for a in actions), actions
    assert dest.read_bytes().startswith(b"%PDF")
    # The qpdf structural rewrite belongs to the exiftool path only.
    assert not any(c[0].endswith("qpdf") for c in seen)


def test_exiftool_failure_without_pypdf_copies_degraded(monkeypatch, tmp_path: Path):
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    original = _ai_pdf()
    src.write_bytes(original)
    _fake_tools(monkeypatch, qpdf=True, exiftool_rc=1)

    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    actions, meta = clean_pdf(src, dest)

    assert meta == {"mode": "copy", "degraded": True}
    assert dest.read_bytes() == original
    assert any("copied unchanged" in a for a in actions), actions


@pytest.mark.skipif(
    shutil.which("exiftool") is None or shutil.which("qpdf") is None,
    reason="needs real exiftool and qpdf",
)
def test_end_to_end_no_recoverable_metadata_bytes(tmp_path: Path):
    src = tmp_path / "in.pdf"
    dest = tmp_path / "out.pdf"
    src.write_bytes(_ai_pdf())
    _actions, meta = clean_pdf(src, dest)
    body = dest.read_bytes()
    assert b"Claude" not in body
    assert b"Anthropic" not in body
    assert meta["structural_rewrite"] is True
