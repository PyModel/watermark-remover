"""Contracts for presentation-free single-asset cleaning."""

from __future__ import annotations

import os
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import clean_asset as clean_asset_module
import pytest
import rewrite_text
import verification
from clean_asset import CleanPlan, ImageDegradePlan, TextCleanPlan, clean_asset
from morphomod import Raster, VisiblePlan, encode_png
from rewrite_text import RewritePlan


def test_clean_asset_returns_semantics_without_printing_or_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("hello\u200bworld", encoding="utf-8")
    destination = tmp_path / "output.txt"

    result = clean_asset(source, destination, CleanPlan())

    assert result.residual is False
    assert result.kind == "text"
    assert "exit_code" not in result.to_dict()
    assert "\u200b" not in destination.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_clean_asset_live_rewrite_stays_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr(
        rewrite_text.layer_b_http,
        "request_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": "rewritten"}}]},
    )
    rewrite_plan = RewritePlan(
        backend="openai-compatible",
        model="model",
        base_url="https://example.test",
        strength="paraphrase",
        allow_remote=True,
    )

    clean_asset(
        source,
        tmp_path / "output.txt",
        CleanPlan(text=TextCleanPlan(rewrite_plan=rewrite_plan)),
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    # remote endpoint emits the deny-by-default remote_warning hint on stderr
    assert "content will leave this machine" in captured.err


def test_clean_result_nested_details_are_immutable(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("hello\u200bworld", encoding="utf-8")
    result = clean_asset(source, tmp_path / "output.txt", CleanPlan())

    with pytest.raises(TypeError):
        result._details["stats"]["removed_count"] = 999

    payload = result.to_dict()
    payload["stats"]["removed_count"] = 999
    assert result.to_dict()["stats"]["removed_count"] != 999


def test_clean_asset_preserves_residual_as_typed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    destination = tmp_path / "output.pdf"

    def fake_clean_container(path: Path, dest: Path) -> dict:
        dest.write_bytes(path.read_bytes())
        return {
            "input": str(path),
            "output": str(dest),
            "format": "pdf",
            "actions": ["copied unchanged"],
            "bytes_in": path.stat().st_size,
            "bytes_out": dest.stat().st_size,
            "still_has_c2pa": True,
            "still_has_ai_metadata": False,
            "post_findings": ["residual"],
            "meta": {"degraded": True},
        }

    monkeypatch.setattr(clean_asset_module, "clean_container", fake_clean_container)

    result = clean_asset(
        source,
        destination,
        CleanPlan(forced_kind="container"),
    )

    assert result.residual is True
    assert result.to_dict()["post_findings"] == ["residual"]
    assert "exit_code" not in result.to_dict()


def test_clean_asset_raises_failures_without_serializing_them(tmp_path: Path) -> None:
    source = tmp_path / "missing.txt"
    destination = tmp_path / "output.txt"

    with pytest.raises(ValueError, match="not a regular file"):
        clean_asset(source, destination, CleanPlan())

    assert not destination.exists()


def test_clean_asset_preflights_derived_mask_output_alias(tmp_path: Path) -> None:
    source = tmp_path / "a.mask.pgm"
    source.write_bytes(b"preserve")

    with pytest.raises(ValueError, match="mask output aliases input"):
        clean_asset(
            source,
            tmp_path / "a.pgm",
            CleanPlan(
                forced_kind="image",
                visible=VisiblePlan(box=(0, 0, 1, 1), backend="texture"),
            ),
        )

    assert source.read_bytes() == b"preserve"


def test_late_image_failure_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64))))
    destination = tmp_path / "output.png"
    destination.write_bytes(b"existing destination")
    monkeypatch.setattr(
        clean_asset_module,
        "_apply_image_degrade",
        lambda raster, plan: (_ for _ in ()).throw(RuntimeError("late failure")),
    )

    with pytest.raises(RuntimeError, match="late failure"):
        clean_asset(
            source,
            destination,
            CleanPlan(forced_kind="image", degrade=ImageDegradePlan(strategy="jpeg")),
        )

    assert destination.read_bytes() == b"existing destination"


def test_late_in_place_failure_preserves_source_without_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    original = encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64)))
    source.write_bytes(original)
    monkeypatch.setattr(
        clean_asset_module,
        "_apply_image_degrade",
        lambda raster, plan: (_ for _ in ()).throw(RuntimeError("late failure")),
    )

    with pytest.raises(RuntimeError, match="late failure"):
        clean_asset(
            source,
            source,
            CleanPlan(
                forced_kind="image",
                in_place=True,
                degrade=ImageDegradePlan(strategy="jpeg"),
            ),
        )

    assert source.read_bytes() == original
    assert not source.with_suffix(".png.bak").exists()


def test_image_in_place_publication_without_fchmod_preserves_backup_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    original = encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64)))
    source.write_bytes(original)
    monkeypatch.delattr(clean_asset_module.os, "fchmod", raising=False)

    result = clean_asset(
        source,
        source,
        CleanPlan(
            forced_kind="image",
            in_place=True,
            degrade=ImageDegradePlan(strategy="jpeg"),
        ),
    )

    assert source.with_suffix(".png.bak").read_bytes() == original
    assert result.output == source
    assert result.to_dict()["degrade"]["strategy"] == "jpeg"
    assert source.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _inject_failure_after_atomic_publish(
    monkeypatch: pytest.MonkeyPatch,
    failure_target: Path,
) -> None:
    original_atomic_write = clean_asset_module.atomic_write_bytes
    failed = False

    def fail_once_after_write(target: Path, data: bytes) -> None:
        nonlocal failed
        original_atomic_write(target, data)
        if target == failure_target and not failed:
            failed = True
            raise OSError(f"injected publication failure: {target.name}")

    monkeypatch.setattr(clean_asset_module, "atomic_write_bytes", fail_once_after_write)


def _visible_plan(mask_output: Path) -> CleanPlan:
    return CleanPlan(
        forced_kind="image",
        visible=VisiblePlan(
            box=(3, 3, 1, 1),
            backend="simple",
            dilation_radius=0,
            mask_output=mask_output,
        ),
    )


def test_boundary_score_one_cannot_be_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64))))
    monkeypatch.setattr(verification, "verify_boundary_seam", lambda *_args: 1.0)
    monkeypatch.setattr(verification, "verify_halo", lambda *_args: (0.0, ()))

    result = clean_asset(source, tmp_path / "output.png", _visible_plan(tmp_path / "mask.pgm"))

    assert result.residual is True
    assert result.to_dict()["verification"]["boundary_score"] == 1.0
    assert result.to_dict()["verification"]["status"] != "VERIFIED"


def test_image_publication_failure_rolls_back_destination_and_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64))))
    destination = tmp_path / "output.png"
    destination.write_bytes(b"prior destination")
    destination.chmod(0o640)
    mask_output = tmp_path / "output.mask.pgm"
    mask_output.write_bytes(b"prior mask")
    mask_output.chmod(0o604)
    _inject_failure_after_atomic_publish(monkeypatch, destination)

    with pytest.raises(OSError, match="publication failure"):
        clean_asset(source, destination, _visible_plan(mask_output))

    assert destination.read_bytes() == b"prior destination"
    assert mask_output.read_bytes() == b"prior mask"
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o640
        assert stat.S_IMODE(mask_output.stat().st_mode) == 0o604


def test_mask_publication_failure_rolls_back_destination_and_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64))))
    destination = tmp_path / "output.png"
    destination.write_bytes(b"prior destination")
    destination.chmod(0o640)
    mask_output = tmp_path / "output.mask.pgm"
    mask_output.write_bytes(b"prior mask")
    mask_output.chmod(0o604)
    _inject_failure_after_atomic_publish(monkeypatch, mask_output)

    with pytest.raises(OSError, match="publication failure"):
        clean_asset(source, destination, _visible_plan(mask_output))

    assert destination.read_bytes() == b"prior destination"
    assert mask_output.read_bytes() == b"prior mask"
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o640
        assert stat.S_IMODE(mask_output.stat().st_mode) == 0o604


def test_backup_publication_failure_rolls_back_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    original = encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64)))
    source.write_bytes(original)
    source.chmod(0o640)
    mask_output = tmp_path / "output.mask.pgm"
    mask_output.write_bytes(b"prior mask")
    mask_output.chmod(0o604)
    backup = source.with_suffix(".png.bak")

    def fail_backup_link(_source: Path, target: Path, **_kwargs) -> None:
        if target == backup:
            raise OSError(f"injected publication failure: {backup.name}")

    monkeypatch.setattr(clean_asset_module.os, "link", fail_backup_link)
    plan = _visible_plan(mask_output)

    with pytest.raises(OSError, match="publication failure"):
        clean_asset(source, source, replace(plan, in_place=True))

    assert source.read_bytes() == original
    assert mask_output.read_bytes() == b"prior mask"
    assert not backup.exists()
    if os.name == "posix":
        assert stat.S_IMODE(source.stat().st_mode) == 0o640
        assert stat.S_IMODE(mask_output.stat().st_mode) == 0o604


def test_backup_fstat_failure_closes_fd_and_removes_partial_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    original = encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64)))
    source.write_bytes(original)
    source.chmod(0o640)
    backup = source.with_suffix(".png.bak")
    original_mkstemp = clean_asset_module.tempfile.mkstemp
    original_fstat = clean_asset_module.os.fstat
    original_close = clean_asset_module.os.close
    backup_fd: int | None = None
    fstat_failed = False
    fd_closed = False

    def track_backup_temp(*args, **kwargs) -> tuple[int, str]:
        nonlocal backup_fd
        fd, name = original_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == f".{backup.name}.":
            backup_fd = fd
        return fd, name

    def fail_backup_fstat(fd: int):
        nonlocal fstat_failed
        if fd == backup_fd and not fstat_failed:
            fstat_failed = True
            raise OSError("injected backup fstat failure")
        return original_fstat(fd)

    def track_backup_close(fd: int) -> None:
        nonlocal fd_closed
        if fd == backup_fd:
            fd_closed = True
        original_close(fd)

    monkeypatch.setattr(clean_asset_module.tempfile, "mkstemp", track_backup_temp)
    monkeypatch.setattr(clean_asset_module.os, "fstat", fail_backup_fstat)
    monkeypatch.setattr(clean_asset_module.os, "close", track_backup_close)

    with pytest.raises(OSError, match="backup fstat failure"):
        clean_asset(
            source,
            source,
            CleanPlan(
                forced_kind="image",
                in_place=True,
                degrade=ImageDegradePlan(strategy="jpeg"),
            ),
        )

    assert fstat_failed is True
    assert fd_closed is True
    assert source.read_bytes() == original
    assert not backup.exists()
    assert not list(tmp_path.glob(f".{backup.name}.*"))
    if os.name == "posix":
        assert stat.S_IMODE(source.stat().st_mode) == 0o640


def test_competing_backup_created_during_publication_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    original = encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64)))
    source.write_bytes(original)
    source.chmod(0o640)
    backup = source.with_suffix(".png.bak")
    competing_data = b"competing backup"
    original_link = clean_asset_module.os.link
    injected = False

    def create_competing_backup_before_link(source_path: Path, target_path: Path, **kwargs) -> None:
        nonlocal injected
        if target_path == backup and not injected:
            injected = True
            backup.write_bytes(competing_data)
        original_link(source_path, target_path, **kwargs)

    monkeypatch.setattr(clean_asset_module.os, "link", create_competing_backup_before_link)

    with pytest.raises(FileExistsError):
        clean_asset(
            source,
            source,
            CleanPlan(
                forced_kind="image",
                in_place=True,
                degrade=ImageDegradePlan(strategy="jpeg"),
            ),
        )

    assert source.read_bytes() == original
    assert backup.read_bytes() == competing_data
    if os.name == "posix":
        assert stat.S_IMODE(source.stat().st_mode) == 0o640


def test_keyboard_interrupt_after_in_place_source_publish_restores_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    original = encode_png(Raster(8, 8, 3, bytearray([10, 20, 30] * 64)))
    source.write_bytes(original)
    source.chmod(0o640)
    backup = source.with_suffix(".png.bak")
    original_link = clean_asset_module.os.link
    original_atomic_write = clean_asset_module.atomic_write_bytes
    interruption = KeyboardInterrupt("injected publication interruption")
    publication_targets: list[Path] = []
    interrupted = False

    def track_backup_link(source_path: Path, target_path: Path, **kwargs) -> None:
        original_link(source_path, target_path, **kwargs)
        if target_path == backup:
            publication_targets.append(backup)

    def interrupt_once_after_write(target: Path, data: bytes) -> None:
        nonlocal interrupted
        original_atomic_write(target, data)
        if target == source:
            publication_targets.append(source)
        if target == source and not interrupted:
            interrupted = True
            raise interruption

    monkeypatch.setattr(clean_asset_module.os, "link", track_backup_link)
    monkeypatch.setattr(clean_asset_module, "atomic_write_bytes", interrupt_once_after_write)

    with pytest.raises(KeyboardInterrupt) as caught:
        clean_asset(
            source,
            source,
            CleanPlan(
                forced_kind="image",
                in_place=True,
                degrade=ImageDegradePlan(strategy="jpeg"),
            ),
        )

    assert caught.value is interruption
    assert publication_targets[:2] == [backup, source]
    assert source.read_bytes() == original
    assert not backup.exists()
    if os.name == "posix":
        assert stat.S_IMODE(source.stat().st_mode) == 0o640


def test_final_bytes_and_outside_mask_failure_are_reported(tmp_path: Path) -> None:
    width = height = 16
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend(((x * 17) % 256, (y * 23) % 256, ((x + y) * 11) % 256))
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(width, height, 3, pixels)))
    destination = tmp_path / "output.png"

    result = clean_asset(
        source,
        destination,
        CleanPlan(
            forced_kind="image",
            visible=VisiblePlan(
                box=(7, 7, 2, 2),
                backend="simple",
                dilation_radius=0,
                mask_output=tmp_path / "output.mask.pgm",
            ),
            degrade=ImageDegradePlan(strategy="jpeg"),
        ),
    )
    payload = result.to_dict()

    assert result.residual is True
    assert payload["verification"]["status"] == "FAILED"
    assert payload["verification"]["outside_mask_difference_count"] > 0
    assert payload["bytes_out"] == destination.stat().st_size


def test_clean_plans_are_immutable() -> None:
    text_plan = TextCleanPlan()
    with pytest.raises(FrozenInstanceError):
        text_plan.nfkc = True

    clean_plan = CleanPlan()
    with pytest.raises(FrozenInstanceError):
        clean_plan.forced_kind = "image"
