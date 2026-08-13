"""Behavioral contracts for shared CLI input selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from batch_inputs import select_inputs


@pytest.mark.parametrize("source_kind", ["missing", "symlink"])
def test_select_inputs_rejects_invalid_sources(tmp_path: Path, source_kind: str) -> None:
    source = tmp_path / "input.txt"
    if source_kind == "symlink":
        target = tmp_path / "target.txt"
        target.write_text("text", encoding="utf-8")
        source.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file or directory"):
        select_inputs([source], recursive=False, pattern="*", extensions={".txt"})


def test_select_inputs_rejects_empty_directory_match(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "ignored.bin").write_bytes(b"ignored")

    with pytest.raises(ValueError, match="no matching input files"):
        select_inputs([source], recursive=False, pattern="*", extensions={".txt"})


def test_select_inputs_marks_multiple_explicit_files_as_batch(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    selection = select_inputs([first, second], recursive=False, pattern="*", extensions={".txt"})

    assert selection.batch is True
    assert [item.path for item in selection.items] == [first, second]
    assert [item.relative for item in selection.items] == [Path("first.txt"), Path("second.txt")]


def test_select_inputs_marks_single_directory_match_as_batch(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    child = source / "only.txt"
    child.write_text("text", encoding="utf-8")

    selection = select_inputs([source], recursive=False, pattern="*", extensions={".txt"})

    assert selection.batch is True
    assert len(selection.items) == 1
    assert selection.items[0].path == child
    assert selection.items[0].relative == Path("only.txt")


def test_select_inputs_excludes_roots(tmp_path: Path) -> None:
    source = tmp_path / "input"
    excluded = source / "excluded"
    excluded.mkdir(parents=True)
    included_file = source / "included.txt"
    excluded_file = excluded / "excluded.txt"
    included_file.write_text("included", encoding="utf-8")
    excluded_file.write_text("excluded", encoding="utf-8")

    selection = select_inputs(
        [source],
        recursive=True,
        pattern="*",
        extensions={".txt"},
        excluded_roots=[excluded],
    )

    assert included_file in [item.path for item in selection.items]
    assert excluded_file not in [item.path for item in selection.items]


def test_select_inputs_rejects_parent_traversal_pattern(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()

    with pytest.raises(
        ValueError, match="glob must be a non-empty relative pattern without '\\.\\.'"
    ):
        select_inputs([source], recursive=False, pattern="../*", extensions={".txt"})
