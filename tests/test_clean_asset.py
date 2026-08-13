"""Contracts for presentation-free single-asset cleaning."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_asset as clean_asset_module
import rewrite_text
from clean_asset import CleanPlan, TextCleanPlan, clean_asset
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
    )

    clean_asset(
        source,
        tmp_path / "output.txt",
        CleanPlan(text=TextCleanPlan(rewrite_plan=rewrite_plan)),
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


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


def test_clean_plans_are_immutable() -> None:
    text_plan = TextCleanPlan()
    with pytest.raises(FrozenInstanceError):
        text_plan.nfkc = True

    clean_plan = CleanPlan()
    with pytest.raises(FrozenInstanceError):
        clean_plan.forced_kind = "image"
