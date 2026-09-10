"""clean-user-facing-text clean_text.py must reject --output aliases of the input.

Writing the cleaned text over the input itself (same path, hard link, or
symlink) destroys the source; the CLI must refuse before writing. --in-place
remains the sanctioned overwrite path (it keeps a .bak backup).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_TEXT = ROOT / "skills" / "clean-user-facing-text" / "scripts" / "clean_text.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLEAN_TEXT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_rejects_same_path_output(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")

    r = _run(str(src), "-o", str(src))

    assert r.returncode == 2
    assert "aliases" in r.stderr
    assert src.read_text(encoding="utf-8") == "a\u200bb"


def test_rejects_hardlink_output(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")
    out = tmp_path / "out.txt"
    os.link(src, out)

    r = _run(str(src), "-o", str(out))

    assert r.returncode == 2
    assert "aliases" in r.stderr
    assert src.read_text(encoding="utf-8") == "a\u200bb"
    assert out.read_text(encoding="utf-8") == "a\u200bb"


def test_rejects_symlink_output(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")
    out = tmp_path / "out.txt"
    try:
        out.symlink_to(src)
    except OSError as error:
        # Windows may require privileges for symlinks; safe_write_bytes still
        # protects the input there, so the CLI-level check is best-effort.
        if os.name == "nt":
            import pytest

            pytest.skip(f"symlinks unavailable: {error}")
        raise

    r = _run(str(src), "-o", str(out))

    assert r.returncode == 2
    assert "aliases" in r.stderr
    assert src.read_text(encoding="utf-8") == "a\u200bb"


def test_normal_output_still_works(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")
    out = tmp_path / "out.txt"

    r = _run(str(src), "-o", str(out))

    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == "ab"
    assert src.read_text(encoding="utf-8") == "a\u200bb"


def test_in_place_still_works_with_backup(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")

    r = _run(str(src), "--in-place")

    assert r.returncode == 0, r.stderr
    assert src.read_text(encoding="utf-8") == "ab"
    assert src.with_suffix(".txt.bak").is_file()
