"""Tests for batch mode on clean_file.py / inspect_file.py."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import types
import zlib
from pathlib import Path

import pytest
from conftest import fake_command_result

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


def _decodable_png(*, text_chunk: bool = False) -> bytes:
    """A PNG the visible pass can decode; optionally with a non-AI tEXt chunk.

    ``_png`` above is chunk-valid but not pixel-valid, so it only exercises the
    metadata path. The registry visible pass decodes the raster, so a mode that
    runs it needs one valid scanline.
    """
    chunks = [_png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))]
    if text_chunk:
        chunks.append(_png_chunk(b"tEXt", b"Comment\x00hello"))
    chunks.append(_png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")))
    chunks.append(_png_chunk(b"IEND", b""))
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _corrupt_png() -> bytes:
    """PNG magic (so routing says image) with a body no decoder can read."""
    return b"\x89PNG\r\n\x1a\n" + b"not a png body"


# --- synthetic Matroska + the faked video runtime ---------------------------

DOUBAO_USCC = "91110102MACQD9K640"
VIDEO_LABEL = (
    b'{"Label":"1","ContentProducer":"91110102MACQD9K640",'
    b'"ProduceID":"c6a75f11-3f4e-4b0a-9d21-77c0aa12e55f"}'
)


def _ebml_size(value: int) -> bytes:
    for length in range(1, 9):
        if value < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | value).to_bytes(length, "big")
    raise ValueError("EBML test value is too large")


def _ebml_element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _ebml_size(len(payload)) + payload


def _matroska(*values: bytes) -> bytes:
    """A minimal Matroska carrying one ``SimpleTag TagName=AIGC`` per value."""
    simple_tags = b"".join(
        _ebml_element(
            b"\x67\xc8",
            _ebml_element(b"\x45\xa3", b"AIGC") + _ebml_element(b"\x44\x87", value),
        )
        for value in values
    )
    tags = _ebml_element(b"\x12\x54\xc3\x67", _ebml_element(b"\x73\x73", simple_tags))
    segment = _ebml_element(b"\x18\x53\x80\x67", tags)
    return _ebml_element(b"\x1a\x45\xdf\xa3", b"") + segment


def _fake_video_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the video preflight without depending on the host's tools."""
    bindir = tmp_path / "tools"
    bindir.mkdir(exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        tool = bindir / name
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))


def _ffmpeg_writing(output: bytes):
    """A command-runner fake that publishes ``output`` at the staged path."""

    def fake_run(argv, **_kwargs):
        Path(argv[-1]).write_bytes(output)
        return fake_command_result(0)

    return fake_run


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def _populate(d: Path) -> None:
    (d / "note.txt").write_text(ZWSP, encoding="utf-8")
    (d / "img.png").write_bytes(_png())
    (d / "skipme.bin").write_bytes(b"\x00\x01\x02")  # unsupported: must be ignored
    (d / "old.cleaned.txt").write_text("stale", encoding="utf-8")  # own output: skip
    (d / "img.png.bak").write_bytes(b"bak")  # sidecar: skip


def _json(result: subprocess.CompletedProcess) -> dict:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_batch_rejects_an_unknown_mode(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "note.txt").write_text(ZWSP, encoding="utf-8")

    r = _run("clean_file.py", src, "-o", tmp_path / "out", "--recursive", "--mode", "everything")

    assert r.returncode == 2
    assert "invalid choice" in r.stderr
    assert not (tmp_path / "out").exists()


def test_batch_mode_metadata_matches_the_default_run(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "note.txt").write_text(ZWSP, encoding="utf-8")
    (src / "img.png").write_bytes(_png())
    plain = tmp_path / "plain"
    metadata = tmp_path / "metadata"

    _json(_run("clean_file.py", src, "-o", plain, "--recursive", "--json"))
    _json(_run("clean_file.py", src, "-o", metadata, "--recursive", "--json", "--mode", "metadata"))

    assert (metadata / "note.txt").read_bytes() == (plain / "note.txt").read_bytes()
    assert (metadata / "img.png").read_bytes() == (plain / "img.png").read_bytes()


def test_recursive_batch_with_no_mode_matches_the_single_file_output(tmp_path: Path):
    """The default batch is the pre-existing per-file path, file for file."""
    src = tmp_path / "in"
    src.mkdir()
    (src / "note.txt").write_text(ZWSP, encoding="utf-8")
    (src / "img.png").write_bytes(_png())
    out = tmp_path / "out"
    single = tmp_path / "single"
    single.mkdir()

    report = _json(_run("clean_file.py", src, "-o", out, "--recursive", "--json"))
    _json(_run("clean_file.py", src / "note.txt", "-o", single / "note.txt", "--json"))
    _json(_run("clean_file.py", src / "img.png", "-o", single / "img.png", "--json"))

    assert report["total"] == 2
    assert (out / "note.txt").read_bytes() == (single / "note.txt").read_bytes()
    assert (out / "img.png").read_bytes() == (single / "img.png").read_bytes()
    assert "\u200b" not in (out / "note.txt").read_text(encoding="utf-8")


def test_recursive_batch_with_no_mode_removes_video_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    import clean_file
    import video_encoding

    src = tmp_path / "in"
    src.mkdir()
    (src / "clip.mkv").write_bytes(_matroska(VIDEO_LABEL))
    out = tmp_path / "out"
    _fake_video_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(video_encoding, "run_command", _ffmpeg_writing(_matroska()))
    monkeypatch.setattr(
        sys, "argv", ["clean_file.py", str(src), "-o", str(out), "--recursive", "--json"]
    )

    code = clean_file.main()
    captured = capsys.readouterr()

    assert code == 0, captured.err
    result = json.loads(captured.out)["results"][0]
    assert result["kind"] == "video"
    assert [m["label"]["ContentProducer"] for m in result["markers_removed"]] == [DOUBAO_USCC]
    assert result["markers_remaining"] == []
    assert result["visible"] is None  # metadata only: no frame is decoded, no pixel changes
    assert result["audio_verified"] is False
    assert (out / "clip.mkv").read_bytes() == _matroska()  # the stripped remux, byte for byte


def test_batch_mode_metadata_runs_no_visible_pass(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "img.png").write_bytes(_decodable_png())

    report = _json(
        _run(
            "clean_file.py",
            src,
            "-o",
            tmp_path / "out",
            "--recursive",
            "--json",
            "--mode",
            "metadata",
        )
    )

    assert report["mode"] == "metadata"
    assert report["stages"] == ["metadata"]
    # No visible plan ran, so the image payload carries no visible report at all.
    assert "visible" not in report["results"][0]


def test_batch_mode_visible_runs_the_registry_pass(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "img.png").write_bytes(_decodable_png())

    report = _json(
        _run(
            "clean_file.py",
            src,
            "-o",
            tmp_path / "out",
            "--recursive",
            "--json",
            "--mode",
            "visible",
        )
    )

    assert report["mode"] == "visible"
    assert report["stages"] == ["visible"]
    assert report["counts"] == {"done": 1, "failed": 0, "total": 1}
    visible = report["results"][0]["visible"]
    assert visible["mark"] == "auto"
    assert visible["mark_status"] == "no_mark"


def test_batch_mode_all_runs_the_visible_and_metadata_stages(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "img.png").write_bytes(_decodable_png(text_chunk=True))
    out = tmp_path / "out"

    report = _json(_run("clean_file.py", src, "-o", out, "--recursive", "--json", "--mode", "all"))

    assert report["mode"] == "all"
    assert report["stages"] == ["visible", "metadata"]
    assert report["results"][0]["visible"]["mark_status"] == "no_mark"
    assert b"Comment" not in (out / "img.png").read_bytes()


def test_batch_summary_counts_one_bad_file_and_the_rest_done(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "note.txt").write_text(ZWSP, encoding="utf-8")
    (src / "broken.png").write_bytes(_corrupt_png())
    out = tmp_path / "out"

    result = _run("clean_file.py", src, "-o", out, "--recursive", "--json")

    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["total"] == 2
    assert report["counts"] == {"done": 1, "failed": 1, "total": 2}
    assert {Path(r["input"]).name: r["status"] for r in report["results"]} == {
        "note.txt": "done",
        "broken.png": "failed",
    }
    assert (out / "note.txt").is_file()


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
        check=False,
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
        check=False,
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
    assert "actions" not in payload
    assert payload["action_details"][0]["text"] == f"error: {payload['error']}"
    assert [d["code"] for d in payload["action_details"]] == ["failed"]
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
    assert "actions" not in payload
    assert payload["action_details"][0]["text"] == f"error: {payload['error']}"
    assert [d["code"] for d in payload["action_details"]] == ["failed"]
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
