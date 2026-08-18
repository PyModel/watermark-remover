"""Tests for the optional reverse-SynthID scorer adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import fake_command_result

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
from image_meta import ImageInspectReport, run_synthid_score

SCORE_SCRIPT = SCRIPTS / "score_synthid.py"


def test_score_synthid_cli_unavailable_without_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("REVERSE_SYNTHID_DIR", raising=False)
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"not really an image")

    r = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), str(dummy)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert r.returncode == 3
    assert "REVERSE_SYNTHID_DIR" in (r.stderr or "")


def test_run_synthid_score_unconfigured_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("REVERSE_SYNTHID_DIR", raising=False)
    assert run_synthid_score(Path("x.png")) is None


def test_run_synthid_score_unavailable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        image_meta.external_command,
        "run_command",
        lambda *args, **kwargs: fake_command_result(3, stderr="unavailable"),
    )
    assert run_synthid_score(Path("x.png"), upstream_dir="/tmp/upstream") is None  # noqa: S108


def test_run_synthid_score_parses_json(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "available": True,
        "is_watermarked": True,
        "confidence": 0.91,
        "phase_match": 0.65,
    }
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return fake_command_result(0, stdout=json.dumps(payload))

    monkeypatch.setattr(image_meta.external_command, "run_command", fake_run)
    result = run_synthid_score(Path("img.png"), upstream_dir="/tmp/upstream")  # noqa: S108

    assert result == payload
    assert "--json" in captured["cmd"]
    assert "--upstream-dir" in captured["cmd"]
    assert "/tmp/upstream" in captured["cmd"]  # noqa: S108


def test_run_synthid_score_runtime_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        image_meta.external_command,
        "run_command",
        lambda *args, **kwargs: fake_command_result(1, stderr="boom"),
    )
    result = run_synthid_score(Path("img.png"), upstream_dir="/tmp/upstream")  # noqa: S108

    assert result is not None
    assert result.get("available") is False
    assert "boom" in result.get("error", "")


def test_inspect_image_cli_prints_synthid_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "inspect_image_cli", str(SCRIPTS / "inspect_image.py")
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    report = ImageInspectReport(
        path="shot.png",
        format="png",
        has_c2pa=False,
        has_ai_metadata=False,
        synthid={
            "available": True,
            "is_watermarked": True,
            "confidence": 0.91,
        },
    )
    img = tmp_path / "shot.png"
    img.write_bytes(b"not really an image")
    monkeypatch.setattr(cli, "inspect_image", lambda path, synthid_dir=None: report)
    monkeypatch.setattr(sys, "argv", ["inspect_image.py", str(img)])

    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "SynthID score: confidence 0.910 (watermarked: yes)" in out


def test_inspect_report_to_dict_includes_synthid():
    report = ImageInspectReport(
        path="x.png",
        format="png",
        has_c2pa=False,
        has_ai_metadata=False,
        synthid={"available": True, "confidence": 0.8},
    )
    assert report.to_dict()["synthid"]["confidence"] == 0.8

    empty = ImageInspectReport(
        path="x.png",
        format="png",
        has_c2pa=False,
        has_ai_metadata=False,
    )
    assert empty.to_dict()["synthid"] is None
