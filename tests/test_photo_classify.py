"""E1 photo classifier: the explicit classify verb, abstain-when-absent, and the
never-auto-run pin.

Seams (agreed in ticket 07):
- the classify CLI verb on morphomod.py — explicit invocation only, output
  always carries the absence-is-not-clean caveat, unavailable is labeled never
  silent;
- abstain-without-weights — missing weights or gate artifact never yields a
  label;
- a source scan pinning that inspect/clean/detect/server paths never import or
  call the classifier.

The learned runtime itself is never exercised here: torch is not a test
dependency, so every downstream path is tested through fakes of the module
contract (is_available / classify_pixels), the way the extra-absent world
actually looks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "skills" / "remove-ai-marks" / "scripts"

#: The caveat every classify output repeats. Absence of the classifier (or of
#: any signal) is unknown — never a clean bill.
CAVEAT_FRAGMENT = "never proves a file is clean"
NOT_PROVENANCE_FRAGMENT = "not provenance"


def _run_classify(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "morphomod.py"), "classify", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def photo(tmp_path: Path) -> Path:
    """A minimal valid PNG; the verb must accept it as an image path."""
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), (120, 120, 120)).save(path)
    return path


class TestClassifyVerbFailSoft:
    """`morphomod.py classify FILE` — explicit verb, honest unavailability."""

    def test_unavailable_without_the_extra_is_labeled_never_silent(
        self, photo: Path, tmp_path: Path
    ) -> None:
        proc = _run_classify(str(photo), "--json")
        # This venv has no torch, so the real probe must fail soft — no
        # traceback, an explicit unavailable status, exit 1.
        assert proc.returncode == 1, (proc.stdout, proc.stderr)
        assert "Traceback" not in proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["status"] == "unavailable"
        assert payload["result"] is None
        assert "watermark-remover[classify]" in payload["reason"]
        assert CAVEAT_FRAGMENT in payload["caveat"]
        assert NOT_PROVENANCE_FRAGMENT in payload["caveat"]

    def test_unavailable_human_output_carries_the_caveat(self, photo: Path) -> None:
        proc = _run_classify(str(photo))
        assert proc.returncode == 1
        assert "unavailable" in proc.stdout
        assert CAVEAT_FRAGMENT in proc.stdout
        assert "watermark-remover[classify]" in proc.stdout

    def test_missing_input_file_is_a_usage_error_not_a_classification(self, tmp_path: Path) -> None:
        proc = _run_classify(str(tmp_path / "absent.png"), "--json")
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr

    def test_weights_missing_reports_unavailable_and_never_a_label(
        self, photo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A present extra but absent weights: unavailable, caveat, no label.

        Runs in-process with a fake classifier module whose contract matches
        the real one (is_available True; classify_pixels raising the weights
        RuntimeError the module documents).
        """
        import morphomod

        def classify_pixels(path: Path, **kwargs: object):
            raise RuntimeError(
                "pixel classification weights are not installed. Set "
                "RAIW_CLASSIFY_WEIGHTS or download the pinned snapshot."
            )

        fake = types.ModuleType("classify")
        fake.is_available = lambda: True  # type: ignore[attr-defined]
        fake.classify_pixels = classify_pixels  # type: ignore[attr-defined]
        fake.CLASSIFY_EXTRA = "'watermark-remover[classify]'"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "classify", fake)
        code = morphomod.main(["classify", str(photo), "--json"])
        out = capsys.readouterr().out
        assert code == 1
        payload = json.loads(out)
        assert payload["status"] == "unavailable"
        assert payload["result"] is None
        assert "weights" in payload["reason"]
        assert CAVEAT_FRAGMENT in payload["caveat"]

    def test_success_path_emits_the_result_and_the_caveat(
        self, photo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The wiring only: a fake module result flows through, caveat attached."""
        import morphomod

        result = {
            "label": "unknown",
            "domain": "photo",
            "detector": "possibly",
            "provider": None,
        }
        fake = types.ModuleType("classify")
        fake.is_available = lambda: True  # type: ignore[attr-defined]
        fake.classify_pixels = lambda path, **kwargs: types.SimpleNamespace(  # type: ignore[attr-defined,misc]
            to_dict=lambda: result
        )
        fake.CLASSIFY_EXTRA = "'watermark-remover[classify]'"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "classify", fake)
        code = morphomod.main(["classify", str(photo), "--json"])
        out = capsys.readouterr().out
        assert code == 0
        payload = json.loads(out)
        assert payload["status"] == "ok"
        assert payload["result"] == result
        assert CAVEAT_FRAGMENT in payload["caveat"]
        assert NOT_PROVENANCE_FRAGMENT in payload["caveat"]


# ---------------------------------------------------------------------------
# The never-auto-run pin: a source scan over the always-on surfaces.
# ---------------------------------------------------------------------------

#: Files allowed to reference the classifier: the module itself, its embedder,
#: and the one CLI that hosts the explicit verb.
_CLASSIFY_SOURCE_ALLOWLIST = frozenset({"classify.py", "clip_l_ft.py", "morphomod.py"})

#: Patterns that would mean a photo-classifier call site outside the verb.
_MODULE_IMPORT = re.compile(r"^\s*(?:from\s+classify\s+import|import\s+classify)\b")
_CALL_TOKENS = ("classify_pixels", "clip_l_ft")


def _classify_call_sites(files: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _MODULE_IMPORT.match(line):
                violations.append(f"{path.name}:{lineno}: {line.strip()}")
        for token in _CALL_TOKENS:
            if token in source:
                violations.append(f"{path.name}: references {token}")
    return violations


def test_classify_is_never_called_from_inspect_clean_detect_server_paths() -> None:
    """inspect / clean / detect / server never import or run the classifier.

    The verb is opt-in by construction: the only sanctioned caller is the
    classify verb wiring on morphomod.py.
    """
    scripts = sorted(p for p in SCRIPTS.glob("*.py") if p.name not in _CLASSIFY_SOURCE_ALLOWLIST)
    # Non-vacuousness: the four honesty-contract surfaces are in the scan set.
    names = {p.name for p in scripts}
    assert {
        "inspect_file.py",
        "inspect_image.py",
        "clean_file.py",
        "server.py",
        "identify.py",
    } <= names
    assert "dwt_dct.py" in names and "invisible_watermark.py" in names
    violations = _classify_call_sites([*scripts, SCRIPTS.parents[2] / "demo.py"])
    assert violations == []


def test_the_scan_detects_a_planted_call_site(tmp_path: Path) -> None:
    """The guard is proved non-vacuous on a planted violation."""
    planted = tmp_path / "rogue.py"
    planted.write_text("from classify import classify_pixels\n", encoding="utf-8")
    assert _classify_call_sites([planted]) != []


def test_the_scan_ignores_unrelated_classify_names(tmp_path: Path) -> None:
    """asset_kind.classify / classify_asset / classify_finding_confidence stay legal."""
    legit = tmp_path / "asset_kind.py"
    legit.write_text(
        "from asset_kind import classify_asset\nkind = classify(path)\n",
        encoding="utf-8",
    )
    assert _classify_call_sites([legit]) == []
