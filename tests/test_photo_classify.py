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


# ---------------------------------------------------------------------------
# The classifier module: pure gates first (no torch anywhere).
# ---------------------------------------------------------------------------


class TestPureGates:
    """Model 1 AND/OR rule, labels, provider margin + tc260 veto — verbatim."""

    def test_detector_level_and_or_rule(self) -> None:
        import classify

        # Defaults: RIDGE_THRESHOLD ~0.306, MLP_THRESHOLD ~5.96 (2026-08-31
        # freeze operating point), so the MLP-side fixture score clears it.
        assert classify.detector_level(1.0, 6.0) == "definitely"
        assert classify.detector_level(1.0, 0.0) == "possibly"
        assert classify.detector_level(0.0, 6.0) == "possibly"
        assert classify.detector_level(0.0, 0.0) == "likely_human"

    def test_only_definitely_publishes_ai(self) -> None:
        import classify

        assert classify.label_for("definitely") == "ai"
        assert classify.label_for("likely_human") == "human"
        # POSSIBLY abstains — the honesty contract at the label level.
        assert classify.label_for("possibly") == "unknown"

    def test_provider_needs_margin_over_no_ai(self) -> None:
        import classify

        scores = {"openai": 1.0, "google": 0.0, "no_ai": 0.5}
        assert classify.provider_from_scores(scores) == "openai"
        # No class clears the margin: provider abstains.
        assert classify.provider_from_scores({"openai": 0.6, "no_ai": 0.5}) is None

    def test_tc260_is_a_veto_never_a_claim(self) -> None:
        import classify

        # The tc260 head winning abstains to None: it covers a whole
        # ecosystem, not one producer, so it never names a company.
        scores = {"openai": 0.4, "google": 0.4, "tc260": 3.0, "no_ai": 0.0}
        assert classify.provider_from_scores(scores) is None
        assert "tc260" not in classify.PUBLIC_PROVIDER or (
            classify.PUBLIC_PROVIDER["tc260"] is None
        )

    def test_provider_only_runs_for_ai_labels(self) -> None:
        import classify

        result = classify.classify_from_scores(0.0, 0.0, {"openai": 9.0, "no_ai": 0.0})
        assert result.label == "human"
        assert result.provider is None

    def test_definitely_receipt_hit_downgrades_to_unknown(self) -> None:
        import classify

        result = classify.classify_from_scores(
            1.0,
            1.0,
            {"openai": 9.0, "no_ai": 0.0},
            ridge_threshold=0.5,
            mlp_threshold=0.5,
            receipt_score=5.0,
            receipt_threshold=2.0,
        )
        assert result.label == "unknown"
        assert result.detector == "definitely"
        assert result.provider is None

    def test_definitely_receipt_miss_keeps_the_provider_path(self) -> None:
        import classify

        result = classify.classify_from_scores(
            1.0,
            1.0,
            {"openai": 9.0, "no_ai": 0.0},
            ridge_threshold=0.5,
            mlp_threshold=0.5,
            receipt_score=0.0,
            receipt_threshold=2.0,
        )
        assert result.label == "ai"
        assert result.provider == "openai"

    def test_receipt_score_and_threshold_are_passed_together(self) -> None:
        """One without the other is a caller error: head and threshold come
        from one artifact, so a bare score must never resolve a silent
        default threshold."""
        import classify

        with pytest.raises(ValueError, match="pass both or neither"):
            classify.classify_from_scores(
                1.0, 1.0, None, ridge_threshold=0.5, mlp_threshold=0.5, receipt_score=5.0
            )

    def test_to_dict_shape_is_the_four_field_contract(self) -> None:
        import classify

        result = classify.classify_from_scores(0.0, 0.0, None)
        assert set(result.to_dict()) == {"label", "domain", "detector", "provider"}


class TestWeightsArePinnedDownloadNeverBundled:
    """Gate npz + model weights: pinned download, absent = unavailable."""

    def test_receipt_gate_absent_from_weights_dir_raises_not_defaults(self, tmp_path: Path) -> None:
        """No bundled fallback: a weights dir without the gate artifact is a
        partial install and the gate refuses — it never silently substitutes a
        code-side threshold."""
        import classify

        with pytest.raises(RuntimeError, match="receipt-gate"):
            classify._load_receipt_gate(tmp_path)

    def test_receipt_gate_loads_stable_then_legacy_name(self, tmp_path: Path) -> None:
        import classify
        import numpy as np

        def write_gate(path: Path) -> None:
            np.savez(
                path,
                w=np.array([1.0, 0.0]),
                b=np.float64(0.5),
                mu=np.array([0.0, 0.0]),
                sd=np.array([1.0, 1.0]),
                threshold=np.float64(2.25),
            )

        stable = tmp_path / classify.RECEIPT_GATE_STABLE_FILE
        write_gate(stable)
        gate = classify._load_receipt_gate(tmp_path)
        assert gate["threshold"] == 2.25
        stable.unlink()
        legacy = tmp_path / classify.RECEIPT_GATE_FILE
        write_gate(legacy)
        assert classify._load_receipt_gate(tmp_path)["threshold"] == 2.25

    def test_receipt_gate_score_matches_the_linear_head(self, tmp_path: Path) -> None:
        import classify
        import numpy as np

        np.savez(
            tmp_path / classify.RECEIPT_GATE_STABLE_FILE,
            w=np.array([2.0, 0.0]),
            b=np.float64(1.0),
            mu=np.array([1.0, 2.0]),
            sd=np.array([2.0, 1.0]),
            threshold=np.float64(3.0),
        )
        gate = classify._load_receipt_gate(tmp_path)
        # Unit vector passes the normalizer unchanged: ((1-1)/2, (0-2)/1) @ [2,0] + 1.
        score = classify.receipt_gate_score(np.array([1.0, 0.0]), gate)
        assert score == pytest.approx(1.0)

    def test_weights_env_override_names_a_missing_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import classify

        monkeypatch.setenv(classify.WEIGHTS_ENV, str(tmp_path / "absent"))
        with pytest.raises(RuntimeError, match="RAIW_CLASSIFY_WEIGHTS"):
            classify._weights_dir()

    def test_require_files_names_a_missing_gate_even_when_models_present(
        self, tmp_path: Path
    ) -> None:
        import classify

        for name in (classify.CLIP_FILE, classify.DETECTOR_FILE, classify.PROBE_FILE):
            (tmp_path / name).write_bytes(b"x")
        with pytest.raises(RuntimeError, match="receipt-gate"):
            classify._require_files(tmp_path)

    def test_first_use_downloads_the_pinned_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No env override → snapshot_download with repo/revision/allow pins."""
        import types

        import classify

        calls: list[dict] = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return "/fake/snapshot"

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = fake_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        monkeypatch.delenv(classify.WEIGHTS_ENV, raising=False)
        assert classify._weights_dir() == Path("/fake/snapshot")
        assert calls == [
            {
                "repo_id": classify.WEIGHTS_REPO,
                "revision": classify.WEIGHTS_REVISION,
                "allow_patterns": list(classify._backend_allow_patterns("torch")),
            }
        ]
        # The pins are real content addresses, never a moving branch.
        assert classify.WEIGHTS_REVISION != "main"
        assert len(classify.WEIGHTS_REVISION) == 40
        int(classify.WEIGHTS_REVISION, 16)
        # The download asks for exactly the runtime file set, gate included.
        assert set(classify.WEIGHTS_ALLOW_PATTERNS) >= {
            classify.RECEIPT_GATE_STABLE_FILE,
            classify.RECEIPT_GATE_FILE,
            classify.PROBE_FILE,
            classify.DETECTOR_FILE,
            classify.PROVIDER_FILE,
            classify.CLIP_FILE,
            classify.OPERATING_POINT_FILE,
        }

    def test_weights_download_failure_is_a_runtime_error_not_a_network_trip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types

        import classify

        def failing_download(**kwargs):
            raise OSError("offline")

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = failing_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        monkeypatch.delenv(classify.WEIGHTS_ENV, raising=False)
        with pytest.raises(RuntimeError, match="watermark-remover\\[classify\\]"):
            classify._weights_dir()

    def test_is_available_probes_the_exact_extra_stack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import classify
        import optional_deps

        probed: list[str] = []
        monkeypatch.setattr(
            optional_deps,
            "module_available",
            lambda *names: probed.extend(names) or True,
        )
        assert classify.is_available() is True
        assert probed == ["torch", "transformers", "numpy", "cv2", "PIL"]

    def test_provenance_records_pin_the_photo_classify_weight_source(self) -> None:
        """The PROVENANCE sweep names the pinned snapshot: repo, revision, files,
        offline override — the download pins stay auditable in the tree."""
        import classify

        provenance = (SCRIPTS / "assets" / "PROVENANCE.md").read_text(encoding="utf-8")
        for needle in (
            classify.WEIGHTS_REPO,
            classify.WEIGHTS_REVISION,
            classify.CLIP_FILE,
            classify.DETECTOR_FILE,
            classify.PROVIDER_FILE,
            classify.PROBE_FILE,
            classify.RECEIPT_GATE_STABLE_FILE,
            classify.WEIGHTS_ENV,
        ):
            assert needle in provenance, needle

    def test_no_gate_npz_is_bundled_in_the_tree(self) -> None:
        """R3: the receipt-gate npz never ships in-tree. The forbidden-fragment
        guard pins asset names; this pins that the assets directory carries no
        gate file and the loader has no package-asset fallback default."""
        import inspect as inspect_module

        import classify

        assets = Path(classify.__file__).parent / "assets"
        if assets.is_dir():
            assert not list(assets.glob("*receipt-gate*"))
        signature = inspect_module.signature(classify._load_receipt_gate)
        assert all(
            param.default is inspect_module.Parameter.empty
            for param in signature.parameters.values()
        ), "_load_receipt_gate must not have a package-asset fallback default"
