"""Tests for the layered configuration system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from configuration import (
    ConfigSource,
    _bool_from_str,
    get_config,
    load_configuration,
)

WATERMARKS_ENV = [
    "WATERMARKS_MAX_FILE_SIZE",
    "WATERMARKS_MAX_IMAGE_PIXELS",
    "WATERMARKS_MAX_CONCURRENCY",
    "WATERMARKS_LOG_LEVEL",
    "WATERMARKS_REWRITE_TIMEOUT",
    "WATERMARKS_MAX_REWRITE_GENERATIONS",
    "WATERMARKS_MAX_REWRITE_POPULATION",
    "WATERMARKS_REWRITE_BACKEND",
    "WATERMARKS_REWRITE_BASE_URL",
    "WATERMARKS_REWRITE_MODEL",
    "WATERMARKS_REWRITE_API_KEY",
    "WATERMARKS_REWRITE_DISABLE_THINKING",
    "WATERMARKS_PLL_MODEL",
    "WATERMARKS_EMBED_MODEL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in WATERMARKS_ENV:
        monkeypatch.delenv(name, raising=False)


def test_defaults_in_empty_project(tmp_path: Path) -> None:
    summary = load_configuration(tmp_path)
    settings = get_config(summary)
    assert settings["max_file_size"] == 100 * 1024 * 1024
    assert settings["max_image_pixels"] == 40_000_000
    assert settings["max_concurrency"] == 4
    assert settings["log_level"] == "INFO"
    assert summary.parse_errors == {}
    assert all(setting.source is ConfigSource.DEFAULT for setting in summary.settings.values())


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WATERMARKS_MAX_FILE_SIZE", "123")
    summary = load_configuration(tmp_path)
    setting = summary.settings["max_file_size"]
    assert setting.value == 123
    assert setting.source is ConfigSource.ENV_VAR


def test_dotenv_overrides_default(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("WATERMARKS_MAX_CONCURRENCY=9\n", encoding="utf-8")
    summary = load_configuration(tmp_path)
    setting = summary.settings["max_concurrency"]
    assert setting.value == 9
    assert setting.source is ConfigSource.ENV_FILE


def test_env_var_beats_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("WATERMARKS_MAX_CONCURRENCY=9\n", encoding="utf-8")
    monkeypatch.setenv("WATERMARKS_MAX_CONCURRENCY", "2")
    summary = load_configuration(tmp_path)
    assert summary.settings["max_concurrency"].value == 2
    assert summary.settings["max_concurrency"].source is ConfigSource.ENV_VAR


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
def test_pyproject_overrides_default(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.watermark]\nmax_image_pixels = 12345\n", encoding="utf-8"
    )
    summary = load_configuration(tmp_path)
    setting = summary.settings["max_image_pixels"]
    assert setting.value == 12345
    assert setting.source is ConfigSource.PYPROJECT


def test_bad_int_recorded_and_default_kept(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("WATERMARKS_MAX_CONCURRENCY=abc\n", encoding="utf-8")
    summary = load_configuration(tmp_path)
    assert summary.settings["max_concurrency"].value == 4
    assert summary.settings["max_concurrency"].source is ConfigSource.DEFAULT
    assert "max_concurrency" in summary.parse_errors


def test_bad_bool_recorded_and_default_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WATERMARKS_REWRITE_DISABLE_THINKING", "banana")
    summary = load_configuration(tmp_path)
    assert summary.settings["rewrite_disable_thinking"].value is False
    assert "rewrite_disable_thinking" in summary.parse_errors


def test_bool_parsing() -> None:
    assert _bool_from_str("true") is True
    assert _bool_from_str("OFF") is False
    assert _bool_from_str("1") is True
    assert _bool_from_str("no") is False
    with pytest.raises(ValueError, match="not a bool"):
        _bool_from_str("banana")


def test_to_dict_includes_parse_errors(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("WATERMARKS_MAX_CONCURRENCY=nope\n", encoding="utf-8")
    payload = load_configuration(tmp_path).to_dict()
    assert payload["max_concurrency"]["value"] == 4
    assert "max_concurrency" in payload["parse_errors"]


def test_missing_files_report_none_paths(tmp_path: Path) -> None:
    summary = load_configuration(tmp_path)
    assert summary.pyproject_path is None
    assert summary.env_file_path is None
