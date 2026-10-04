"""Repo hygiene, wired into pytest (CI runs `pytest -q` and `make check`).

Replaces the orphan scripts `audit_placeholders.py` / `check_docs_honesty.py`,
which were never collected: their checks run here as real assertions.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"


def test_no_placeholder_markers_in_shipped_package():
    """No raise NotImplementedError and no TODO/FIXME/HACK/XXX markers ship."""
    raise_nie = re.compile(r"^\s*raise\s+NotImplementedError\b")
    marker = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
    violations: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if raise_nie.search(line) or marker.search(line):
                violations.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not violations, "\n".join(violations)


def test_gemini_slot_ships_no_dead_http_machinery():
    """The reserved detector slot must not carry an unreachable request layer.

    A placeholder that only looks wired (imports, constants, retry helpers,
    zero callers) is exactly what the no-placeholder rule forbids.
    """
    import text_detectors

    for name in ("_call_gemini", "request_json", "LayerBHTTPError", "_is_retryable_layer_b_error"):
        assert not hasattr(text_detectors, name), f"dead detector machinery: {name}"
    det = text_detectors.GeminiSynthIDTextDetector()
    assert det.available() is False
    report = det.detect("x")
    # available() and detect() must never disagree.
    assert report["available"] is False


def test_gemini_docstring_matches_code():
    import text_detectors

    doc = text_detectors.__doc__ or ""
    assert "RESERVED slot" in doc
    assert "never sends any" in doc, "docstring must state the no-network behavior"
    assert "WATERMARKS_GEMINI_API_KEY" in doc


def test_layer_a_text_unicode_copies_stay_identical():
    """The two skill copies are one file maintained by hand; byte drift is a
    behavior fork (the braille-blank fix must land in both)."""
    a = ROOT / "skills" / "remove-ai-marks" / "scripts" / "text_unicode.py"
    b = ROOT / "skills" / "clean-user-facing-text" / "scripts" / "text_unicode.py"
    assert a.read_bytes() == b.read_bytes(), "text_unicode.py copies drifted"


def test_context_routing_contract_matches_the_code(tmp_path):
    """Enforce routing through shipped code, not an optional local glossary."""
    from asset_kind import classify_asset
    from clean_asset import CleanPlan, clean_asset

    source = tmp_path / "unknown"
    source.write_bytes(b"\x00\x01\x02\x03" * 8)
    destination = tmp_path / "output"
    assert classify_asset(source) == "unknown"
    with pytest.raises(ValueError, match="unrecognized format"):
        clean_asset(source, destination, CleanPlan())
    assert not destination.exists()


@pytest.mark.parametrize(
    ("row", "must_contain"),
    [
        ("| PNG |", "tIME"),
        ("| TIFF / BigTIFF |", "Interoperability"),
        ("| HTML |", "comment"),
        ("| DOCX / XLSX / PPTX |", "thumbnail"),
    ],
)
def test_readme_format_table_names_current_capabilities(row, must_contain):
    """The per-format capability table is the user-facing contract; it must
    name what the cleaners actually drop."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    line = next((ln for ln in readme.splitlines() if ln.startswith(row)), "")
    assert line, f"README row missing: {row}"
    assert must_contain in line, f"{row} row does not mention {must_contain}: {line}"
