"""Tests for batch mode on clean_file.py / inspect_file.py."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

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


def test_clean_residual_marker_exit_1(tmp_path: Path):
    f = tmp_path / "residual.png"
    f.write_bytes(_png(with_residual_marker=True))
    r = _run("clean_file.py", f)
    assert r.returncode == 1
    assert "residual" in r.stderr


@pytest.mark.parametrize(("name", "remote"), [("unset", False), ("remote", True)])
def test_image_tsapa_flag_does_not_resolve_text_backend(tmp_path: Path, name: str, remote: bool):
    source = tmp_path / f"{name}.png"
    source.write_bytes(_png())
    destination = tmp_path / f"{name}.cleaned.png"
    env = os.environ.copy()
    for key in (
        "WATERMARKS_REWRITE_BACKEND",
        "WATERMARKS_REWRITE_MODEL",
        "WATERMARKS_REWRITE_BASE_URL",
        "WATERMARKS_REWRITE_API_KEY",
    ):
        env.pop(key, None)
    if remote:
        env.update(
            WATERMARKS_REWRITE_BACKEND="openai-compatible",
            WATERMARKS_REWRITE_MODEL="must-not-be-used",
            WATERMARKS_REWRITE_BASE_URL="https://example.test",
        )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(source),
            "-o",
            str(destination),
            "--tsapa",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert destination.is_file()
    assert "rewrite base URL" not in result.stderr


def test_image_ignores_invalid_text_perturb_policy(tmp_path: Path):
    source = tmp_path / "input.png"
    source.write_bytes(_png())
    destination = tmp_path / "output.png"

    result = _run(
        "clean_file.py",
        source,
        "-o",
        destination,
        "--char-perturb",
        "--char-strength",
        "2",
    )

    assert result.returncode == 0, result.stderr
    assert destination.is_file()


def test_mixed_batch_preflights_tsapa_plan_before_writes(tmp_path: Path):
    image = tmp_path / "first.png"
    image.write_bytes(_png())
    text = tmp_path / "second.txt"
    text.write_text(ZWSP, encoding="utf-8")
    output = tmp_path / "out"
    env = os.environ.copy()
    for key in (
        "WATERMARKS_REWRITE_BACKEND",
        "WATERMARKS_REWRITE_MODEL",
        "WATERMARKS_REWRITE_BASE_URL",
        "WATERMARKS_REWRITE_API_KEY",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(image),
            str(text),
            "-o",
            str(output),
            "--tsapa",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert "requires a live backend" in result.stderr
    assert not (output / image.name).exists()
    assert not (output / text.name).exists()


def test_mixed_batch_preflights_visible_plan_before_writes(tmp_path: Path):
    text = tmp_path / "first.txt"
    text.write_text(ZWSP, encoding="utf-8")
    image = tmp_path / "second.png"
    image.write_bytes(_png())
    output = tmp_path / "out"

    result = _run(
        "clean_file.py",
        text,
        image,
        "-o",
        output,
        "--visible-backend",
        "external",
    )

    assert result.returncode == 1
    assert "requires a localization source" in result.stderr
    assert not (output / text.name).exists()
    assert not (output / image.name).exists()


def test_single_plan_preflight_preserves_json_error(tmp_path: Path):
    image = tmp_path / "input.png"
    image.write_bytes(_png())
    output = tmp_path / "output.png"

    result = _run(
        "clean_file.py",
        image,
        "-o",
        output,
        "--visible-backend",
        "external",
        "--json",
    )

    assert result.returncode == 1
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["kind"] == "unknown"
    assert payload["input"] == str(image)
    assert payload["output"] == str(output)
    assert payload["exit_code"] == 1
    assert payload["actions"] == [f"error: {payload['error']}"]
    assert "requires a localization source" in payload["error"]
    assert not output.exists()


def test_batch_plan_preflight_preserves_json_envelope(tmp_path: Path):
    text = tmp_path / "first.txt"
    text.write_text(ZWSP, encoding="utf-8")
    image = tmp_path / "second.png"
    image.write_bytes(_png())
    output = tmp_path / "out"

    result = _run(
        "clean_file.py",
        text,
        image,
        "-o",
        output,
        "--visible-backend",
        "external",
        "--json",
    )

    assert result.returncode == 1
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["total"] == 1
    assert len(report["results"]) == 1
    payload = report["results"][0]
    assert payload["kind"] == "unknown"
    assert payload["input"] == str(image)
    assert payload["output"] == str(output / image.name)
    assert payload["exit_code"] == 1
    assert payload["actions"] == [f"error: {payload['error']}"]
    assert "requires a localization source" in payload["error"]
    assert not (output / text.name).exists()
    assert not (output / image.name).exists()


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
