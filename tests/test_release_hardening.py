"""Focused security regressions for release hardening."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
LIGHTWEIGHT_COMMON = ROOT / "skills" / "clean-user-facing-text" / "scripts" / "common.py"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import demo


def _load_lightweight_common():
    spec = importlib.util.spec_from_file_location(
        "lightweight_common_release_hardening", LIGHTWEIGHT_COMMON
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_rejects_upload_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hello\u200bworld", encoding="utf-8")
    source = tmp_path / "upload.txt"
    try:
        source.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    report, output, prompt = demo.clean_upload(str(source), False, False, "paraphrase")

    assert report.startswith("**Error cleaning upload.txt:**")
    assert "symlink" in report.lower()
    assert output is None
    assert prompt == ""
    assert target.read_text(encoding="utf-8") == "hello\u200bworld"


def test_demo_rejects_upload_through_symlinked_directory(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "upload.txt"
    source.write_text("hello\u200bworld", encoding="utf-8")
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    report, output, prompt = demo.clean_upload(
        str(linked / source.name), False, False, "paraphrase"
    )

    assert report.startswith("**Error cleaning upload.txt:**")
    assert "symlink" in report.lower()
    assert output is None
    assert prompt == ""


def test_demo_rejects_upload_outside_system_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_temp = tmp_path / "trusted"
    trusted_temp.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(demo.tempfile, "gettempdir", lambda: str(trusted_temp))

    report, output, prompt = demo.clean_upload(str(source), False, False, "paraphrase")

    assert report.startswith("**Error cleaning outside.txt:**")
    assert "temporary directory" in report
    assert output is None
    assert prompt == ""


def test_demo_rejects_non_regular_upload(tmp_path: Path) -> None:
    source = tmp_path / "folder.txt"
    source.mkdir()

    report, output, prompt = demo.clean_upload(str(source), False, False, "paraphrase")

    assert report.startswith("**Error cleaning folder.txt:**")
    assert "regular file" in report
    assert output is None
    assert prompt == ""


def test_demo_sanitizes_cleaned_output_name(tmp_path: Path) -> None:
    source = tmp_path / "report <bad>.txt"
    source.write_text("hello\u200bworld", encoding="utf-8")

    report, output, prompt = demo.clean_upload(str(source), False, False, "paraphrase")

    assert "**Kind:** `text`" in report
    assert output is not None
    cleaned = Path(output)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", cleaned.name)
    assert cleaned.read_text(encoding="utf-8") == "helloworld"
    assert prompt == ""


def test_safe_write_bytes_rejects_destination_symlink(tmp_path: Path) -> None:
    common = _load_lightweight_common()
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"preserve")
    destination = tmp_path / "output.txt"
    try:
        destination.symlink_to(victim)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(OSError, match="symlink"):
        common.safe_write_bytes(destination, b"replacement")

    assert destination.is_symlink()
    assert victim.read_bytes() == b"preserve"


def test_safe_write_bytes_preserves_parent_bytes_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _load_lightweight_common()
    if not hasattr(common.os, "fchmod"):
        pytest.skip("fchmod unavailable")
    destination = tmp_path / "created" / "output.txt"
    monkeypatch.setattr(common, "_default_file_mode", lambda: 0o640)
    payload = b"exact\x00bytes\xff"

    common.safe_write_bytes(destination, payload)

    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_safe_write_bytes_rejects_destination_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _load_lightweight_common()
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"preserve")
    destination = tmp_path / "output.txt"
    original_mkstemp = common.tempfile.mkstemp

    def replace_destination_with_symlink(*args, **kwargs):
        fd, temporary = original_mkstemp(*args, **kwargs)
        try:
            destination.symlink_to(victim)
        except OSError:
            os.close(fd)
            os.unlink(temporary)
            raise
        return fd, temporary

    monkeypatch.setattr(common.tempfile, "mkstemp", replace_destination_with_symlink)

    with pytest.raises(OSError, match="symlink"):
        common.safe_write_bytes(destination, b"replacement")

    assert destination.is_symlink()
    assert victim.read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".output.txt.*.tmp"))


def test_safe_write_bytes_rejects_temporary_destination_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _load_lightweight_common()
    destination = tmp_path / "output.txt"
    original_mkstemp = common.tempfile.mkstemp

    def alias_destination_to_temporary(*args, **kwargs):
        fd, temporary = original_mkstemp(*args, **kwargs)
        try:
            os.link(temporary, destination)
        except OSError:
            os.close(fd)
            os.unlink(temporary)
            raise
        return fd, temporary

    monkeypatch.setattr(common.tempfile, "mkstemp", alias_destination_to_temporary)

    with pytest.raises(OSError, match="aliases"):
        common.safe_write_bytes(destination, b"replacement")

    assert destination.read_bytes() == b""
    assert not list(tmp_path.glob(".output.txt.*.tmp"))


def test_safe_write_bytes_rejects_parent_directory_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX directory-descriptor hardening")
    common = _load_lightweight_common()
    parent = tmp_path / "output"
    parent.mkdir()
    moved_parent = tmp_path / "moved-output"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    destination = parent / "output.txt"
    original_mkstemp = common.tempfile.mkstemp

    def redirect_parent_after_staging(*args, **kwargs):
        fd, temporary = original_mkstemp(*args, **kwargs)
        parent.rename(moved_parent)
        parent.symlink_to(attacker, target_is_directory=True)
        return fd, temporary

    monkeypatch.setattr(common.tempfile, "mkstemp", redirect_parent_after_staging)

    with pytest.raises(OSError, match="directory changed"):
        common.safe_write_bytes(destination, b"replacement")

    assert not (attacker / destination.name).exists()
    assert not list(moved_parent.glob(".output.txt.*.tmp"))


def test_release_image_tags_validate_ref_from_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-images.yml").read_text(encoding="utf-8")

    assert workflow.count("REF_NAME: ${{ github.ref_name }}") == 2
    assert 'TAG="${{ github.ref_name }}"' not in workflow
    assert workflow.count('[[ ! "$REF_NAME" =~ ^v[0-9][0-9A-Za-z.+-]*$ ]]') == 2
    assert workflow.count('TAG="$REF_NAME"') == 2
    assert (
        "tags=ghcr.io/pymodel/watermark-remover:${TAG},ghcr.io/pymodel/watermark-remover:latest"
        in workflow
    )
    assert (
        "tags=ghcr.io/pymodel/watermark-remover:${{ matrix.tag }}-${TAG},"
        "ghcr.io/pymodel/watermark-remover:${{ matrix.tag }}-latest"
    ) in workflow
    assert "VERSION=${{ steps.tags.outputs.version }}" in workflow


@pytest.mark.parametrize(
    ("script_name", "extra_args", "sparse_command"),
    [
        (
            "setup_markllm.sh",
            [],
            "sparse-checkout set --no-cone /watermark/ /config/ /utils/ /exceptions/ "
            "/evaluation/dataset.py /LICENSE /README.md",
        ),
        ("setup_markdiffusion.sh", ["--checkout"], "sparse-checkout reapply"),
    ],
)
@pytest.mark.parametrize("head_matches", [True, False], ids=["matching-head", "stale-head"])
def test_existing_checkout_is_repinned_before_install(
    tmp_path: Path,
    script_name: str,
    extra_args: list[str],
    sparse_command: str,
    head_matches: bool,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    python = checkout / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    command_log = tmp_path / "commands.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    _write_executable(
        fake_bin / "git",
        """#!/bin/sh
printf 'git %s\n' "$*" >> "$COMMAND_LOG"
case " $* " in
  *" fetch "*) exit 1 ;;
  *" rev-parse HEAD "*) printf '%s\n' "$HEAD_REF" ;;
  *" rev-parse "*) printf '%s\n' "$EXPECTED_REF" ;;
esac
""",
    )
    _write_executable(
        fake_bin / "realpath",
        """#!/bin/sh
if [ "$1" = "-m" ]; then shift; fi
printf '%s\n' "$1"
""",
    )
    _write_executable(
        python,
        """#!/bin/sh
printf 'python %s\n' "$*" >> "$COMMAND_LOG"
""",
    )

    ref = "0123456789abcdef0123456789abcdef01234567"
    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(command_log),
            "EXPECTED_REF": ref,
            "HEAD_REF": ref if head_matches else "0000000000000000000000000000000000000000",
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
    )
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [
            bash,
            str(SCRIPTS / script_name),
            "--dir",
            str(checkout),
            "--ref",
            ref,
            *extra_args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=10,
    )

    commands = command_log.read_text(encoding="utf-8").splitlines()
    fetch_index = next(i for i, command in enumerate(commands) if " fetch " in command)
    checkout_index = next(
        i for i, command in enumerate(commands) if f"checkout --detach {ref}" in command
    )
    sparse_index = next(i for i, command in enumerate(commands) if sparse_command in command)
    verify_index = next(i for i, command in enumerate(commands) if "rev-parse HEAD" in command)
    assert fetch_index < checkout_index < sparse_index < verify_index
    if not head_matches:
        assert result.returncode == 1
        assert not any(command.startswith("python ") for command in commands)
        return

    assert result.returncode == 0, result.stderr
    install_index = next(i for i, command in enumerate(commands) if command.startswith("python "))
    assert verify_index < install_index


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
