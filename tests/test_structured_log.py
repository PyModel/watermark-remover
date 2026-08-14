"""Tests for the structured logger and its configuration wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from structured_log import (
    LogLevel,
    get_logger,
    init_logger,
    log_info,
)


@pytest.fixture(autouse=True)
def _clean_log_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WATERMARKS_LOG_LEVEL", raising=False)


def test_init_logger_defaults_to_info() -> None:
    assert init_logger()._level is LogLevel.INFO


def test_init_logger_honors_env_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATERMARKS_LOG_LEVEL", "DEBUG")
    assert init_logger()._level is LogLevel.DEBUG


def test_init_logger_accepts_string_level() -> None:
    assert init_logger("ERROR")._level is LogLevel.ERROR


def test_init_logger_accepts_enum_level() -> None:
    assert init_logger(LogLevel.WARNING)._level is LogLevel.WARNING


def test_init_logger_invalid_level_falls_back_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WATERMARKS_LOG_LEVEL", "LOUDER")
    assert init_logger()._level is LogLevel.INFO


def test_log_info_with_extras_emits_json_to_stderr_only(capsys) -> None:
    log_info("cleaned asset", module="clean_asset", path="/tmp/out.png")
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err.strip())
    assert payload["level"] == "INFO"
    assert payload["module"] == "clean_asset"
    assert payload["extra"]["path"] == "/tmp/out.png"


def test_logger_plain_entries_go_to_stderr(capsys) -> None:
    logger = init_logger("WARNING")
    logger.warning("residual signal", module="inspect")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[WARNING] inspect: residual signal\n"


def test_logger_respects_level_threshold(capsys) -> None:
    logger = init_logger("ERROR")
    logger.info("hidden", module="m")
    logger.debug("hidden too", module="m")
    captured = capsys.readouterr()
    assert captured.err == ""
    logger.error("visible", module="m")
    captured = capsys.readouterr()
    assert captured.err == "[ERROR] m: visible\n"


def test_get_logger_returns_singleton() -> None:
    assert get_logger() is get_logger()
