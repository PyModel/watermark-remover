"""Tests for optional_deps module."""

from __future__ import annotations

import importlib

import optional_deps
import pytest


def test_known_extra_uses_import_result(monkeypatch) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    result = optional_deps.check_optional("visible")
    assert result.available is True
    assert result.extra == "visible"


def test_missing_extra_is_reported_independent_of_environment(monkeypatch) -> None:
    def missing(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", missing)
    result = optional_deps.check_optional("ai")
    assert result.available is False
    assert result.extra == "ai"
    assert "missing torch" in result.reason
    assert "Install watermark-remover[ai]" in result.hint


def test_check_optional_reports_oserror(monkeypatch) -> None:
    def unavailable(_name: str):
        raise OSError("loader unavailable")

    monkeypatch.setattr(importlib, "import_module", unavailable)

    result = optional_deps.check_optional("visible")

    assert result.available is False
    assert result.reason == "loader unavailable"


def test_import_safe_returns_none_for_runtime_error(monkeypatch) -> None:
    def unavailable(_name: str):
        raise RuntimeError("broken optional module")

    monkeypatch.setattr(importlib, "import_module", unavailable)

    assert optional_deps._import_safe("optional.backend") is None


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_import_safe_propagates_base_exceptions(monkeypatch, error_type) -> None:
    def interrupted(_name: str):
        raise error_type

    monkeypatch.setattr(importlib, "import_module", interrupted)

    with pytest.raises(error_type):
        optional_deps._import_safe("optional.backend")


def test_unknown_and_unsupported_ocr_extra() -> None:
    for name in ("nonexistent", "ocr"):
        result = optional_deps.check_optional(name)
        assert result.available is False
        assert "unknown extra" in result.reason
    assert not hasattr(optional_deps, "has_ocr")
    assert not hasattr(optional_deps, "paddleocr")


def test_available_hint() -> None:
    result = optional_deps.BackendAvailability(available=True, extra="visible")
    assert result.hint == "visible extras are installed"


def test_convenience_functions_return_bool() -> None:
    assert isinstance(optional_deps.has_visible(), bool)
    assert isinstance(optional_deps.has_quality(), bool)
    assert isinstance(optional_deps.has_ai(), bool)
    assert isinstance(optional_deps.has_provenance(), bool)
