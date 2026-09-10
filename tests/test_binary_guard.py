"""Tests for the binary-input guard on the text-only tools."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import guard_binary, looks_binary

DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body><w:p><w:r><w:t>Table 1 holds the results.</w:t></w:r></w:p></w:body>"
    "</w:document>"
)


def make_docx(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", DOCX_XML)
    return path


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# --- looks_binary ----------------------------------------------------------


@pytest.mark.parametrize(
    "data,expected_fragment",
    [
        (b"PK\x03\x04rest", "ZIP"),
        (b"%PDF-1.7\n", "PDF"),
        (b"\x89PNG\r\n\x1a\nrest", "PNG"),
        (b"\xff\xd8\xff\xe0rest", "JPEG"),
        (b"\x7fELF\x02\x01", "ELF"),
        (b"SQLite format 3\x00", "SQLite"),
        (b"plain text\x00with a nul", "NUL"),
    ],
)
def test_flags_binary(data, expected_fragment):
    kind = looks_binary(data)
    assert kind is not None
    assert expected_fragment.lower() in kind.lower()


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"Just some prose.\n",
        b"# Markdown\n\n- bullet\n",
        "Accented prose: na\xefve caf\xe9 r\xe9sum\xe9\n".encode(),
        "Zero width\u200b and nbsp\u00a0 here\n".encode(),
        b"Latin-1 bytes: caf\xe9 na\xefve\n",  # not UTF-8, still text
        b"a\tb\r\nc\x0cd\x1b[0m\n",  # tabs, CRLF, form feed, ANSI escape
    ],
)
def test_allows_text(data):
    assert looks_binary(data) is None


def test_compressed_bytes_are_flagged(tmp_path):
    data = make_docx(tmp_path / "x.docx").read_bytes()
    assert looks_binary(data) is not None


def test_guard_binary_can_be_overridden():
    guard_binary(b"PK\x03\x04", "x.docx", allow_binary=True)  # must not raise
    with pytest.raises(SystemExit) as exc:
        guard_binary(b"PK\x03\x04", "x.docx")
    assert exc.value.code == 2


# --- CLI behaviour ---------------------------------------------------------


def test_inspect_text_refuses_docx(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    r = run("inspect_text.py", str(docx))
    assert r.returncode == 2
    assert "looks like" in r.stderr
    assert "inspect_file.py" in r.stderr
    assert "Suspicious:" not in r.stdout


def test_inspect_text_force_text_still_works(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    r = run("inspect_text.py", str(docx), "--force-text")
    assert r.returncode in (0, 1)
    assert "Length:" in r.stdout


def test_clean_text_refuses_docx_and_writes_nothing(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    before = docx.read_bytes()
    out = tmp_path / "doc.cleaned.docx"
    r = run("clean_text.py", str(docx), "-o", str(out))
    assert r.returncode == 2
    assert not out.exists()
    assert docx.read_bytes() == before


def test_clean_text_in_place_leaves_docx_intact(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    before = docx.read_bytes()
    r = run("clean_text.py", str(docx), "--in-place")
    assert r.returncode == 2
    assert docx.read_bytes() == before
    assert not (tmp_path / "doc.docx.bak").exists()


def test_clean_file_still_routes_docx_to_container(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    out = tmp_path / "out.docx"
    r = run("clean_file.py", str(docx), "-o", str(out), "--json")
    assert r.returncode == 0, r.stderr
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        assert "word/document.xml" in zf.namelist()


def test_clean_file_refuses_unknown_binary(tmp_path):
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    out = tmp_path / "out.bin"
    r = run("clean_file.py", str(blob), "-o", str(out))
    assert r.returncode == 2
    assert not out.exists()


def test_clean_file_in_place_refuses_before_taking_a_backup(tmp_path):
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    before = blob.read_bytes()
    r = run("clean_file.py", str(blob), "--in-place")
    assert r.returncode == 2
    assert blob.read_bytes() == before
    assert not (tmp_path / "mystery.bin.bak").exists()
    assert list(tmp_path.iterdir()) == [blob]


def test_clean_file_in_place_as_text_on_docx_leaves_no_backup(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    before = docx.read_bytes()
    r = run("clean_file.py", str(docx), "--in-place", "--as", "text")
    assert r.returncode == 2
    assert docx.read_bytes() == before
    assert not (tmp_path / "doc.docx.bak").exists()


def test_clean_file_json_emits_structured_error_for_binary_text(tmp_path):
    """guard_binary preflight refusals must surface as structured JSON in --json mode."""
    blob = tmp_path / "binary.txt"
    blob.write_bytes(make_docx(tmp_path / "x.docx").read_bytes())
    out = tmp_path / "out.txt"
    r = run("clean_file.py", str(blob), "-o", str(out), "--json")
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert "error" in payload
    assert "refusing to treat" in payload["error"]
    assert payload["exit_code"] == 2
    assert not out.exists()


def test_clean_file_binary_text_refusal_keeps_exit_2_human_mode(tmp_path):
    blob = tmp_path / "binary.txt"
    blob.write_bytes(make_docx(tmp_path / "x.docx").read_bytes())
    out = tmp_path / "out.txt"
    r = run("clean_file.py", str(blob), "-o", str(out))
    assert r.returncode == 2
    assert "refusing to treat" in r.stderr
    assert not out.exists()


def test_clean_file_json_batch_binary_text_error_is_structured(tmp_path):
    clean = tmp_path / "ok.txt"
    clean.write_text("plain text", encoding="utf-8")
    blob = tmp_path / "binary.txt"
    blob.write_bytes(make_docx(tmp_path / "x.docx").read_bytes())
    r = run("clean_file.py", str(clean), str(blob), "--json")
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["total"] == 1
    assert payload["results"][0]["exit_code"] == 2
    assert "refusing to treat" in payload["results"][0]["error"]


def test_clean_file_auto_refuses_unknown_text_like_bytes(tmp_path):
    blob = tmp_path / "no_extension"
    blob.write_text("just plain text, no extension, no magic\n", encoding="utf-8")
    before = blob.read_bytes()
    r = run("clean_file.py", str(blob), "--in-place")
    assert r.returncode == 2
    assert blob.read_bytes() == before
    assert not (tmp_path / "no_extension.bak").exists()


def test_clean_file_as_text_opt_in_cleans_unknown(tmp_path):
    blob = tmp_path / "no_extension"
    blob.write_text("Hidden\u200bmark here.\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    r = run("clean_file.py", str(blob), "-o", str(out), "--as", "text")
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == "Hiddenmark here.\n"


def test_clean_file_force_text_opt_in_on_unknown_binary(tmp_path):
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    out = tmp_path / "out.bin"
    r = run("clean_file.py", str(blob), "-o", str(out), "--force-text")
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_inspect_file_json_reports_unknown_kind(tmp_path):
    blob = tmp_path / "no_extension"
    blob.write_text("no magic, no extension\n", encoding="utf-8")
    r = run("inspect_file.py", str(blob), "--json")
    assert r.returncode == 3  # EXIT_PARTIAL: unrecognized input was not scanned
    import json

    payload = json.loads(r.stdout)
    assert payload["kind"] == "unknown"
    assert payload["unscanned"] is True
    assert "note" in payload


def test_router_advice_is_not_circular(tmp_path):
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(b"\x00\x01\x02\x03" * 64)
    r = run("clean_file.py", str(blob))
    assert r.returncode == 2
    assert "no supported text, image or container format" in r.stderr
    assert "Use inspect_file.py / clean_file.py" not in r.stderr
    assert "--force-text" in r.stderr
    r = run("inspect_file.py", str(blob))
    assert r.returncode == 3  # EXIT_PARTIAL: unrecognized input was not scanned
    assert "Kind: unknown" in r.stdout
    assert "--as text|image|container" in r.stdout


def test_text_only_scripts_keep_the_pointer_to_the_routers(tmp_path):
    docx = make_docx(tmp_path / "doc.docx")
    r = run("clean_text.py", str(docx))
    assert r.returncode == 2
    assert "Use inspect_file.py / clean_file.py" in r.stderr


def test_text_files_are_unaffected(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("Hidden\u200bmark here.\n", encoding="utf-8")
    out = tmp_path / "note.cleaned.txt"
    r = run("clean_text.py", str(src), "-o", str(out))
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == "Hiddenmark here.\n"


def test_stdin_binary_is_refused():
    docx = io.BytesIO()
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", DOCX_XML)
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_text.py")],
        input=docx.getvalue(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 2
    assert b"looks like" in r.stderr


PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 32
JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"A" * 64


@pytest.mark.parametrize("data,label", [(PNG_HEADER, "PNG"), (JPEG_HEADER, "JPEG")])
@pytest.mark.parametrize("io_encoding", [None, "cp1252", "latin-1"])
def test_stdin_non_ascii_magic_is_refused_whatever_the_codec(data, label, io_encoding):
    env = dict(os.environ)
    if io_encoding is None:
        env.pop("PYTHONIOENCODING", None)
    else:
        env["PYTHONIOENCODING"] = io_encoding
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_text.py")],
        input=data,
        capture_output=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert r.returncode == 2, (label, io_encoding, r.stderr)
    assert label.encode() in r.stderr, (label, io_encoding, r.stderr)


def test_stdin_text_still_flows_through():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "clean_text.py")],
        input="plain\u200btext\n".encode(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.replace(b"\r\n", b"\n") == b"plaintext\n"
