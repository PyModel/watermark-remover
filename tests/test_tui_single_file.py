"""Single-file TUI marking is enforced by the real Python bridge."""

from __future__ import annotations

import io
import shlex
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tui_bridge import Bridge, BridgeError, FrameWriter
from tui_core import BadRequest, InvalidOptions


def setup_files(tmp_path, monkeypatch):
    monkeypatch.setenv("WATERMARKS_TUI_SETTINGS", str(tmp_path / "settings.json"))
    folder = tmp_path / "inputs"
    folder.mkdir()
    files = [folder / "a.txt", folder / "b.txt"]
    for path in files:
        path.write_text("hello\u200b world\n", encoding="utf-8")
    bridge = Bridge(FrameWriter(io.BytesIO()))
    state = {
        "paths": [str(folder)],
        "preset": "hidden",
        "flags": [],
        "endpoint": {"backend": None, "base_url": None, "model": None},
        "clean_path": str(files[1]),
    }
    return bridge, state, files


def test_marked_clean_keeps_discovery_list_but_previews_and_writes_only_one_file(
    tmp_path, monkeypatch
):
    bridge, state, files = setup_files(tmp_path, monkeypatch)
    plan = bridge.plan(1, {"state": state})
    assert plan["files_total"] == 2
    assert {entry["path"] for entry in plan["files"]} == {str(path) for path in files}
    assert shlex.split(plan["command"]) == ["wm", str(files[1])]
    result = bridge.clean(2, {"state": state})
    assert result["total"] == 1 and result["errors"] == 0
    assert not files[0].with_name("a.cleaned.txt").exists()
    assert files[1].with_name("b.cleaned.txt").read_text() == "hello world\n"
    assert all("\u200b" in path.read_text() for path in files)


def test_no_mark_never_falls_back_to_cleaning_the_directory(tmp_path, monkeypatch):
    bridge, state, files = setup_files(tmp_path, monkeypatch)
    state["clean_path"] = None
    with pytest.raises(InvalidOptions, match="Mark one file"):
        bridge.clean(1, {"state": state})
    assert not list(files[0].parent.glob("*.cleaned.*"))


@pytest.mark.parametrize("target", ["outside.txt", "directory", "missing.txt"])
def test_mark_must_still_belong_to_the_discovered_regular_files(tmp_path, monkeypatch, target):
    bridge, state, files = setup_files(tmp_path, monkeypatch)
    path = tmp_path / target
    if target == "directory":
        path.mkdir()
    elif target == "outside.txt":
        path.write_text("outside\u200b")
    state["clean_path"] = str(path)
    with pytest.raises(InvalidOptions, match="no longer in"):
        bridge.clean(1, {"state": state})
    assert not list(files[0].parent.glob("*.cleaned.*"))


@pytest.mark.parametrize("target", ["", [], {}, 1, True, "\x00invalid"])
def test_mark_shape_is_validated(tmp_path, monkeypatch, target):
    bridge, state, _ = setup_files(tmp_path, monkeypatch)
    state["clean_path"] = target
    with pytest.raises(BadRequest, match="clean_path"):
        bridge.clean(1, {"state": state})


def test_only_the_marked_file_is_preflighted(tmp_path, monkeypatch):
    bridge, state, files = setup_files(tmp_path, monkeypatch)
    state["flags"] = ["--in-place"]
    files[0].with_name("a.txt.bak").write_text("an older backup")
    plan = bridge.plan(1, {"state": state})
    assert plan["preflight_error"] is None
    assert "1 file will be overwritten" in plan["confirm"][0]["message"]
    result = bridge.clean(2, {"state": state, "confirmed": ["in_place"]})
    assert result["total"] == 1 and result["errors"] == 0
    assert "\u200b" in files[0].read_text()
    assert "\u200b" not in files[1].read_text()


def test_disappearing_mark_does_not_keep_a_stale_file_list(tmp_path, monkeypatch):
    bridge, state, files = setup_files(tmp_path, monkeypatch)
    files[1].unlink()
    plan = bridge.plan(1, {"state": state})
    assert [entry["path"] for entry in plan["files"]] == [str(files[0])]
    assert "no longer in" in plan["preflight_error"]


def test_confirmations_count_only_the_marked_plain_text_file(tmp_path, monkeypatch):
    bridge, state, _ = setup_files(tmp_path, monkeypatch)
    state["preset"] = "rewrite"
    state["endpoint"] = {
        "backend": "openai-compatible",
        "base_url": "http://gpu.example.test:8000",
        "model": "own-model",
    }
    plan = bridge.plan(1, {"state": state})
    assert "full text of 1 file" in plan["confirm"][0]["message"]
    with pytest.raises(BridgeError) as error:
        bridge.clean(2, {"state": state})
    assert error.value.code == "needs_confirm"
