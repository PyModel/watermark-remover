"""Tests for how c2patool output is interpreted by run_optional_tools."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta  # noqa: E402

MANIFEST_OUTPUT = """{
  "active_manifest": "urn:c2pa:0000",
  "manifests": {
    "urn:c2pa:0000": {
      "claim_generator": "some_tool/1.0",
      "assertions": [{"label": "c2pa.actions"}]
    }
  }
}
"""


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_c2patool(monkeypatch, output: str, returncode: int, on_stderr: bool = False):
    """Pretend c2patool exists and emits `output`; hide every other tool."""

    def fake_which(cmd: str):
        return "/fake/bin/c2patool" if cmd == "c2patool" else None

    def fake_run(cmd, **kwargs):
        if on_stderr:
            return _Completed(returncode, stderr=output)
        return _Completed(returncode, stdout=output)

    monkeypatch.setattr(image_meta, "which", fake_which)
    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)


def test_no_claim_found_is_not_a_manifest(monkeypatch, tmp_path):
    """Absence of a manifest must not read as a hit.

    c2patool reports a missing manifest as "Error: No claim found", which
    contains the substring "claim"; a positive branch that is not vetoed by
    the negative markers flags every clean asset as carrying C2PA.
    """
    _fake_c2patool(monkeypatch, "Error: No claim found\n", 1, on_stderr=True)
    path = tmp_path / "clean.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    tools = image_meta.run_optional_tools(path)

    assert tools["c2patool"]["available"] is True
    assert tools["c2patool"]["has_manifest"] is False


def test_no_jumbf_data_is_not_a_manifest(monkeypatch, tmp_path):
    _fake_c2patool(monkeypatch, "No JUMBF data found\n", 1, on_stderr=True)
    path = tmp_path / "clean.jpg"
    path.write_bytes(b"\xff\xd8\xff")

    tools = image_meta.run_optional_tools(path)

    assert tools["c2patool"]["has_manifest"] is False


def test_real_manifest_is_detected(monkeypatch, tmp_path):
    """The negative guard must not suppress genuine manifests."""
    _fake_c2patool(monkeypatch, MANIFEST_OUTPUT, 0)
    path = tmp_path / "signed.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    tools = image_meta.run_optional_tools(path)

    assert tools["c2patool"]["has_manifest"] is True


def test_missing_c2patool_is_reported_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(image_meta, "which", lambda cmd: None)
    path = tmp_path / "any.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    tools = image_meta.run_optional_tools(path)

    assert tools["c2patool"] == {"available": False}
