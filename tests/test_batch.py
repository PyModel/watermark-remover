"""Tests for batch mode on clean_file.py / inspect_file.py."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"

ZWSP = "hello​world clean me"


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _png(with_residual_marker: bool = False) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat_payload = b"c2pa literal" if with_residual_marker else zlib.compress(b"\x00\x00\x00")
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat_payload)
        + _png_chunk(b"IEND", b"")
    )


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
    )


def _populate(d: Path) -> None:
    (d / "note.txt").write_text(ZWSP, encoding="utf-8")
    (d / "img.png").write_bytes(_png())
    (d / "skipme.bin").write_bytes(b"\x00\x01\x02")  # unsupported: must be ignored
    (d / "old.cleaned.txt").write_text("stale", encoding="utf-8")  # own output: skip
    (d / "img.png.bak").write_bytes(b"bak")  # sidecar: skip


def test_clean_directory_batch(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    _populate(src)
    out = tmp_path / "out"
    r = _run("clean_file.py", src, "-o", out, "--json")
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["total"] == 2  # txt + png only
    assert (out / "note.txt").is_file()
    assert (out / "img.png").is_file()
    assert "​" not in (out / "note.txt").read_text(encoding="utf-8")


def test_clean_directory_recursive_and_extensions(tmp_path: Path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    _populate(src)
    (src / "sub" / "deep.txt").write_text(ZWSP, encoding="utf-8")
    r = _run("clean_file.py", src, "-o", tmp_path / "out", "--recursive", "--extensions", "txt")
    assert r.returncode == 0, r.stderr
    # Preserve relative paths: avoids collisions between same-named files.
    assert (tmp_path / "out" / "sub" / "deep.txt").is_file()
    assert not (tmp_path / "out" / "img.png").exists()


def test_batch_rejects_parent_traversal_glob(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (tmp_path / "outside.txt").write_text(ZWSP, encoding="utf-8")
    for script in ("clean_file.py", "inspect_file.py"):
        r = _run(script, src, "--glob", "../*.txt")
        assert r.returncode == 2
        assert "invalid input selection" in r.stderr


def test_explicit_symlink_input_is_rejected(tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text(ZWSP, encoding="utf-8")
    linked = tmp_path / "linked.txt"
    linked.symlink_to(outside)
    r = _run("clean_file.py", linked, "--in-place")
    assert r.returncode == 2
    assert outside.read_text(encoding="utf-8") == ZWSP


def test_clean_directory_glob(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    _populate(src)
    out = tmp_path / "out"
    r = _run("clean_file.py", src, "-o", out, "--glob", "*.txt")
    assert r.returncode == 0, r.stderr
    assert (out / "note.txt").is_file()
    assert not (out / "img.png").exists()


def test_clean_multi_file_and_single_json(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text(ZWSP, encoding="utf-8")
    b.write_text(ZWSP, encoding="utf-8")
    r = _run("clean_file.py", a, b, "-o", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "a.txt").is_file()
    assert (tmp_path / "out" / "b.txt").is_file()

    r1 = _run("clean_file.py", a, "--json")
    assert r1.returncode == 0, r1.stderr
    single = json.loads(r1.stdout)
    assert single["kind"] == "text"
    assert single["stats"]["removed_count"] >= 1


def test_single_output_cannot_alias_input(tmp_path: Path):
    src = tmp_path / "draft.txt"
    src.write_text(ZWSP, encoding="utf-8")
    original = src.read_bytes()
    r = _run("clean_file.py", src, "-o", src)
    assert r.returncode == 2
    assert "output aliases input" in r.stderr
    assert src.read_bytes() == original


def test_batch_preflight_prevents_output_from_overwriting_another_input(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    r = _run("clean_file.py", first, second, "-o", tmp_path, "--json")
    assert r.returncode == 2
    assert "output aliases input" in r.stderr
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"


def test_in_place_rejects_existing_backup_symlink(tmp_path: Path):
    src = tmp_path / "draft.txt"
    src.write_text(ZWSP, encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("do not overwrite", encoding="utf-8")
    src.with_suffix(".txt.bak").symlink_to(outside)
    r = _run("clean_file.py", src, "--in-place")
    assert r.returncode == 2
    assert "backup already exists" in r.stderr
    assert outside.read_text(encoding="utf-8") == "do not overwrite"
    assert src.read_text(encoding="utf-8") == ZWSP


def test_in_place_batch_does_not_follow_symlinked_files(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.txt").write_text(ZWSP, encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text(ZWSP, encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    original = outside.read_bytes()
    r = _run("clean_file.py", root, "--recursive", "--in-place")
    assert r.returncode == 0, r.stderr
    assert outside.read_bytes() == original
    assert not outside.with_suffix(".txt.bak").exists()
    assert "​" not in (root / "inside.txt").read_text(encoding="utf-8")


def test_batch_rejects_symlinked_output_component(tmp_path: Path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "note.txt").write_text(ZWSP, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "sub").symlink_to(outside, target_is_directory=True)

    r = _run("clean_file.py", src, "-o", out, "--recursive", "--json")

    assert r.returncode == 2
    assert "output path contains a symlink" in r.stderr
    assert not (outside / "note.txt").exists()


def test_recursive_batch_excludes_nested_output_tree(tmp_path: Path):
    src = tmp_path / "in"
    out = src / "cleaned"
    out.mkdir(parents=True)
    (src / "source.txt").write_text(ZWSP, encoding="utf-8")
    (out / "old.txt").write_text(ZWSP, encoding="utf-8")
    r = _run("clean_file.py", src, "-o", out, "--recursive", "--json")
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["total"] == 1
    assert (out / "source.txt").is_file()
    assert not (out / "cleaned" / "old.txt").exists()


def test_missing_input_is_usage_error_even_with_valid_input(tmp_path: Path):
    valid = tmp_path / "valid.txt"
    valid.write_text("ok", encoding="utf-8")
    r = _run("clean_file.py", valid, tmp_path / "missing.txt")
    assert r.returncode == 2
    assert "not a regular file or directory" in r.stderr


def test_multi_file_output_collision_fails_before_writes(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "same.txt").write_text("left", encoding="utf-8")
    (right / "same.txt").write_text("right", encoding="utf-8")
    r = _run(
        "clean_file.py", left / "same.txt", right / "same.txt", "-o", tmp_path / "out", "--json"
    )
    assert r.returncode == 2
    assert "batch output collision" in r.stderr
    assert not (tmp_path / "out" / "same.txt").exists()


def test_degraded_container_residual_is_failure(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import clean_file

    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(
        clean_file,
        "clean_container",
        lambda _src, dest: {
            "input": str(_src),
            "output": str(dest),
            "format": "pdf",
            "actions": ["copied unchanged"],
            "bytes_in": 15,
            "bytes_out": 15,
            "still_has_c2pa": True,
            "still_has_ai_metadata": False,
            "post_findings": ["residual"],
            "meta": {"degraded": True},
        },
    )
    args = SimpleNamespace(force_type="container", in_place=False, json=True)
    result = clean_file._clean_single_file(src, tmp_path / "out.pdf", args)
    assert result["exit_code"] == 1


def test_clean_residual_marker_exit_1(tmp_path: Path):
    f = tmp_path / "residual.png"
    f.write_bytes(_png(with_residual_marker=True))
    r = _run("clean_file.py", f)
    assert r.returncode == 1
    assert "residual" in r.stderr


def test_inspect_directory_batch(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    _populate(src)
    r = _run("inspect_file.py", src, "--json")
    assert r.returncode == 1  # zwsp text is suspicious
    report = json.loads(r.stdout)
    assert report["total"] == 2
    kinds = {res["kind"] for res in report["results"]}
    assert kinds == {"text", "image"}


def test_inspect_single_json(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text(ZWSP, encoding="utf-8")
    r = _run("inspect_file.py", f, "--json")
    assert r.returncode == 1
    report = json.loads(r.stdout)
    assert report["kind"] == "text"
    assert report["suspicious"] is True
