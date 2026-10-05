"""Wiring tests for the ``detect`` extra: declaration, checker, and HTTP surface.

The extra is the delivery contract for the open DWT-DCT detector: declared in
pyproject, answerable by ``check_optional`` (probing exactly numpy + cv2 +
pywt), listed in ``extras_status`` / ``/capabilities``, and carried by the
``/detect`` image path fail-soft — an absent extra is reported as
unavailable/unknown, never as "no watermark found".
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import optional_deps


def _pyproject() -> dict:
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def test_detect_extra_is_declared_and_checkable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`detect` joins the registry and probes exactly numpy + cv2 + pywt."""
    assert "detect" in optional_deps.KNOWN_EXTRAS

    probed: list[str] = []

    def recording_import(name: str):
        probed.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", recording_import)
    availability = optional_deps.check_optional("detect")
    assert availability.available is True
    assert availability.extra == "detect"
    assert probed == ["numpy", "cv2", "pywt"]


def test_detect_extra_missing_deps_report_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", missing)
    availability = optional_deps.check_optional("detect")
    assert availability.available is False
    assert "Install watermark-remover[detect]" in availability.hint


def test_aggregate_all_extra_joins_the_registry_with_a_real_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`all` stops being a pyproject-only name: it is checkable, honestly."""
    assert "all" in optional_deps.KNOWN_EXTRAS
    assert "all" in optional_deps.extras_status()

    probed: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: probed.append(name) or object())
    availability = optional_deps.check_optional("all")
    assert availability.available is True
    # The checker covers every aggregate member declared at this commit:
    # visible+quality+ai+provenance+detect (source-classify joins later).
    assert {"cv2", "skimage", "torch", "c2pa", "numpy", "pywt"} <= set(probed)


def test_pyproject_declares_detect_and_all_gains_it() -> None:
    """The manifest is the source of truth for what the checker probes."""
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]
    detect = extras["detect"]
    assert any(pkg.startswith("PyWavelets") or pkg == "pywavelets" for pkg in detect), detect
    assert any(pkg.startswith("opencv") for pkg in detect), detect
    assert any(pkg.lower() == "numpy" for pkg in detect), detect
    assert "watermark-remover[detect]" in extras["all"]


def test_the_dev_only_oracle_is_never_a_declared_dependency() -> None:
    """invisible-watermark stays a fixture-generation tool, never a dependency."""
    project = _pyproject()["project"]
    declared: list[str] = list(project.get("dependencies") or [])
    for extra_deps in (project.get("optional-dependencies") or {}).values():
        declared.extend(extra_deps)
    for requirement in declared:
        name = requirement.split(";")[0].split("[")[0].split("==")[0].split(">=")[0].strip().lower()
        assert name not in {"invisible-watermark", "imwatermark"}, requirement
    test_requirements = (ROOT / "requirements-test.txt").read_text(encoding="utf-8").lower()
    assert "invisible-watermark" not in test_requirements


def test_extras_status_reports_detect() -> None:
    status = optional_deps.extras_status()
    assert "detect" in status
    assert set(status["detect"]) == {"available", "hint"}


def test_capabilities_report_lists_detect() -> None:
    import server

    extras = server.capabilities()["extras"]
    assert "detect" in extras
