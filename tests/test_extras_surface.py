"""The classify / classify-onnx extras surface and the finished `all` sweep.

Seams (agreed in ticket 07): the extras registry (KNOWN_EXTRAS + checkers),
the pyproject manifest, and every reporting surface that lists extras
(extras_status, /capabilities, README). The checker tests probe through a
recording import — no test touches the network or the real ML stack.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from conftest import load_toml

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return load_toml("pyproject.toml")


def test_classify_and_classify_onnx_join_the_registry() -> None:
    import optional_deps

    assert "classify" in optional_deps.KNOWN_EXTRAS
    assert "classify-onnx" in optional_deps.KNOWN_EXTRAS
    status = optional_deps.extras_status()
    assert "classify" in status and "classify-onnx" in status
    assert set(status["classify"]) == {"available", "hint"}


def test_classify_checker_probes_its_exact_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[str] = []

    def recording_import(name: str):
        probed.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", recording_import)
    import optional_deps

    availability = optional_deps.check_optional("classify")
    assert availability.available is True
    assert availability.extra == "classify"
    assert probed == ["numpy", "cv2", "PIL", "torch", "transformers", "huggingface_hub"]


def test_classify_onnx_checker_probes_the_union_with_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[str] = []

    def recording_import(name: str):
        probed.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", recording_import)
    import optional_deps

    availability = optional_deps.check_optional("classify-onnx")
    assert availability.available is True
    assert probed == [
        "numpy",
        "cv2",
        "PIL",
        "torch",
        "transformers",
        "huggingface_hub",
        "onnxruntime",
    ]


@pytest.mark.parametrize("extra", ["classify", "classify-onnx"])
def test_missing_deps_report_the_install_hint(extra: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", missing)
    import optional_deps

    availability = optional_deps.check_optional(extra)
    assert availability.available is False
    assert f"Install watermark-remover[{extra}]" in availability.hint


def test_pyproject_declares_classify_and_classify_onnx() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    classify = extras["classify"]
    names = {req.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for req in classify}
    assert {"numpy", "opencv-python-headless", "pillow", "torch", "transformers"} <= names
    assert any(req.startswith("huggingface-hub") for req in classify)
    # classify-onnx = classify + the ONNX runtime, by reference like `all`.
    assert "watermark-remover[classify]" in extras["classify-onnx"]
    assert any(req == "onnxruntime" for req in extras["classify-onnx"])


def test_all_gains_source_classify_and_detect_and_no_new_torch_extra() -> None:
    """The aggregate sweep: the phase-5 delta joins `all` — detect +
    source-classify only. The NEW torch extras (trustmark, classify,
    classify-onnx) stay standalone: GQ1 ruling — torch is opt-in, never
    smuggled into the aggregate. (watermark-remover[ai] predates this run and
    is not part of this sweep.)
    """
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "watermark-remover[detect]" in extras["all"]
    assert "watermark-remover[source-classify]" in extras["all"]
    for torch_extra in (
        "watermark-remover[classify]",
        "watermark-remover[classify-onnx]",
        "watermark-remover[trustmark]",
    ):
        assert torch_extra not in extras["all"]


def test_all_checker_union_covers_its_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checker promises the union of the aggregate's members.

    visible+quality+ai+provenance+detect was the pre-07 set; source-classify
    joining adds PIL and huggingface_hub to the union.
    """
    probed: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: probed.append(name) or object())
    import optional_deps

    availability = optional_deps.check_optional("all")
    assert availability.available is True
    assert {
        "cv2",
        "skimage",
        "torch",
        "c2pa",
        "numpy",
        "pywt",
        "PIL",
        "huggingface_hub",
    } <= set(probed)


def test_capabilities_reports_every_known_extra() -> None:
    """/capabilities auto-surfaces extras_status(): the new extras must appear."""
    import optional_deps
    import server

    payload = server.capabilities()
    assert set(payload["extras"]) == set(optional_deps.KNOWN_EXTRAS)
    assert "classify" in payload["extras"]
    assert "classify-onnx" in payload["extras"]
    assert set(payload["extras"]["classify"]) == {"available", "hint"}


def test_readme_lists_the_new_extras() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for token in ("[classify]", "[classify-onnx]", "[detect]", "[source-classify]", "[trustmark]"):
        assert token in readme, f"README.md does not mention the {token} extra"
