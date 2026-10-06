# Tests for the E4 spectral internals (image_math.py, source_spectral.py) and
# the E2 source-export classifier (source_classify.py), plus the identify
# signal adapters and the extra-gated identify collector. No network: the HF
# download path is monkeypatched and weights are synthetic in-repo npz files
# built to the modules' own schemas.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import load_toml

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# E4a: image_math.dct_matrix — the pure orthonormal DCT-II basis factory
# ---------------------------------------------------------------------------


def test_dct_matrix_size_two_is_the_known_hadamard_like_basis() -> None:
    """A hand-computed literal: the 2-point orthonormal DCT-II basis."""
    import image_math
    import numpy as np

    matrix = image_math.dct_matrix(np, 2)
    inverse_root_two = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            [inverse_root_two, inverse_root_two],
            [inverse_root_two, inverse_root_two * -1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_dct_matrix_is_orthonormal_for_the_classifier_geometry() -> None:
    """M @ M.T is the identity at the 8-point and 16-point sizes."""
    import image_math
    import numpy as np

    for size in (8, 16):
        matrix = image_math.dct_matrix(np, size)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(size), atol=1e-12)


def test_dct_matrix_diagonalizes_a_constant_signal_into_dc_only() -> None:
    """A constant input has energy only in the DC (first) coefficient."""
    import image_math
    import numpy as np

    matrix = image_math.dct_matrix(np, 8)
    coefficients = matrix @ np.full(8, 3.0)
    np.testing.assert_allclose(coefficients[1:], 0.0, atol=1e-12)
    assert abs(coefficients[0]) > 0.0


# ---------------------------------------------------------------------------
# E4b: source_spectral — phase-free log-spectrum features
# ---------------------------------------------------------------------------


def test_make_geometry_covers_every_pixel_and_validates_arguments() -> None:

    import source_spectral

    geometry = source_spectral.make_geometry(size=256, grid=16)
    assert geometry is not None
    assert geometry.size == 256 and geometry.grid == 16
    assert int(geometry.radius_counts.sum()) == 256 * 256
    assert geometry.passband.any()
    with pytest.raises(ValueError):
        source_spectral.make_geometry(size=255, grid=16)
    with pytest.raises(ValueError):
        source_spectral.make_geometry(size=8, grid=2)
    with pytest.raises(ValueError):
        source_spectral.make_geometry(size=256, grid=1)


def test_phase_free_log_spectrum_is_invariant_to_sign_and_centered() -> None:
    import numpy as np
    import source_spectral

    rng = np.random.default_rng(20260922)
    channel = rng.standard_normal((32, 32))
    spectrum = source_spectral.phase_free_log_spectrum(channel)
    assert spectrum.shape == (32, 32)
    # Phase-free: negating the input flips every FFT phase, not any magnitude.
    np.testing.assert_allclose(
        source_spectral.phase_free_log_spectrum(-channel), spectrum, atol=1e-12
    )
    # A constant channel centers to zero, so the log magnitude is exactly zero.
    np.testing.assert_allclose(
        source_spectral.phase_free_log_spectrum(np.full((32, 32), 5.0)), 0.0, atol=1e-12
    )
    with pytest.raises(ValueError):
        source_spectral.phase_free_log_spectrum(np.zeros((2, 2, 2)))


def test_spectral_feature_shape_and_constant_image_zero_vector() -> None:
    import numpy as np
    import source_spectral

    geometry = source_spectral.make_geometry(size=64, grid=16)
    assert geometry is not None
    constant = np.full((64, 64, 3), 0.5)
    features = source_spectral.spectral_feature(constant, geometry)
    assert features.shape == (3 * 16 * 16,)
    np.testing.assert_allclose(features, 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        source_spectral.spectral_feature(np.zeros((8, 8, 3)), geometry)


def test_feature_from_image_is_deterministic_for_a_fixed_synthetic() -> None:
    import numpy as np
    import source_spectral
    from PIL import Image

    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    geometry = source_spectral.make_geometry()
    assert geometry is not None
    first = source_spectral.feature_from_image(image, geometry)
    assert first is not None and first.shape == (768,)
    second = source_spectral.feature_from_image(image, geometry)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_spectral_stack_abstains_when_numpy_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dep-absence is abstention (None), never zeros and never a raise."""
    import source_spectral
    from PIL import Image

    # make_geometry is process-wide cached; drop entries from earlier tests so
    # the absent-stack path is actually exercised, and again on the way out.
    source_spectral.make_geometry.cache_clear()
    try:
        monkeypatch.setitem(sys.modules, "numpy", None)
        assert not source_spectral.is_available()
        assert source_spectral.make_geometry() is None
        image = Image.new("RGB", (64, 64))
        # No geometry (and no numeric stack) means no feature: abstain, never zeros.
        assert source_spectral.feature_from_image(image, None) is None
    finally:
        source_spectral.make_geometry.cache_clear()


# ---------------------------------------------------------------------------
# E2: source_classify — the abstaining source-export-pipeline classifier
# ---------------------------------------------------------------------------


def _write_synthetic_mlp(path: Path, *, thresholds: tuple[float, float] = (0.5, 0.5)) -> str:
    """Build a source_classify-schema npz whose scores are constants.

    layer0/layer1 weights are zero, so both hidden layers collapse to
    gelu(0)=0 and the raw scores equal output_bias exactly:
    openai=1.0, google=0.0, unknown=0.0 — fully deterministic, no network.
    Returns the synthetic file's SHA-256 (the loader pin, substituted in tests).
    """
    import hashlib

    import numpy as np
    import source_classify

    payload = {
        "classes": np.array(source_classify.CLASSES),
        "kind": np.array("fused"),
        "activation": np.array("gelu_exact"),
        "mean": np.zeros(source_classify.FEATURE_WIDTH, dtype=np.float64),
        "scale": np.ones(source_classify.FEATURE_WIDTH, dtype=np.float64),
        "margins": np.array(thresholds, dtype=np.float64),
        "layer0_weight": np.zeros((128, source_classify.FEATURE_WIDTH), dtype=np.float32),
        "layer0_bias": np.zeros(128, dtype=np.float32),
        "layer1_weight": np.zeros((64, 128), dtype=np.float32),
        "layer1_bias": np.zeros(64, dtype=np.float32),
        "output_weight": np.zeros((3, 64), dtype=np.float32),
        "output_bias": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    with open(path, "wb") as sink:
        np.savez(sink, **payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _noise_png(path: Path, size: int = 512, seed: int = 11) -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture()
def synthetic_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point WEIGHTS_ENV at a directory holding the synthetic npz."""

    import source_classify

    weights = tmp_path / "sub" / source_classify.MODEL_FILE
    weights.parent.mkdir()
    digest = _write_synthetic_mlp(weights)
    monkeypatch.setenv(source_classify.WEIGHTS_ENV, str(weights.parent))
    monkeypatch.setattr(source_classify, "MODEL_SHA256", digest)
    # the loader is lru_cached on (path, sha); drop entries from earlier tests
    source_classify._load_model.cache_clear()
    yield weights
    source_classify._load_model.cache_clear()


def test_classify_source_labels_openai_when_the_margin_clears(
    tmp_path: Path, synthetic_weights: Path
) -> None:
    import source_classify

    image = _noise_png(tmp_path / "image.png")
    result = source_classify.classify_source(str(image))
    assert result.label == "openai"
    assert result.reason == "classified"
    assert result.scores["openai"] == pytest.approx(1.0)
    assert result.scores["google"] == pytest.approx(0.0)
    assert result.margins["openai"] == pytest.approx(1.0)
    assert result.thresholds["openai"] == pytest.approx(0.5)
    assert result.claim == source_classify.CLAIM
    as_dict = result.to_dict()
    assert as_dict["model_revision"] == source_classify.WEIGHTS_REVISION
    assert as_dict["label"] == "openai"


def test_classify_source_abstains_to_unknown_out_of_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_weights: Path
) -> None:
    """Margins that clear nothing report unknown — abstention, never a verdict."""
    import hashlib

    import source_classify

    # rebuild with thresholds nothing clears (the model's stored margins double
    # as the calibrated thresholds), then re-point the fixture seam at it
    digest = _write_synthetic_mlp(synthetic_weights, thresholds=(1e6, 1e6))
    monkeypatch.setattr(source_classify, "MODEL_SHA256", digest)
    source_classify._load_model.cache_clear()
    image = _noise_png(tmp_path / "image.png")
    result = source_classify.classify_source(str(image))
    assert result.label == "unknown"
    assert result.reason == "abstained"
    assert result.scores  # evidence-only scores are still reported
    assert result.to_dict()["label"] == "unknown"
    assert hashlib.sha256(synthetic_weights.read_bytes()).hexdigest() == digest


def test_classify_source_reports_feature_unavailable_below_the_geometry(
    tmp_path: Path, synthetic_weights: Path
) -> None:
    import source_classify

    image = _noise_png(tmp_path / "tiny.png", size=64)
    result = source_classify.classify_source(str(image))
    assert result.label == "unknown"
    assert result.reason == "feature_unavailable"
    assert result.scores == {}


def test_model_loader_rejects_a_sha_or_schema_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import source_classify

    weights = tmp_path / source_classify.MODEL_FILE
    _write_synthetic_mlp(weights)
    with pytest.raises(ValueError, match="SHA-256"):
        source_classify._load_model(weights, "0" * 64)
    source_classify._load_model.cache_clear()


def test_weights_env_override_names_a_missing_file_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import source_classify

    monkeypatch.setenv(source_classify.WEIGHTS_ENV, str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match=source_classify.WEIGHTS_ENV):
        source_classify._model_path()


def test_first_use_downloads_the_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env override → hf_hub_download with repo/file/revision pins; no 'main'."""
    import types

    import source_classify

    weights = tmp_path / source_classify.MODEL_FILE
    digest = _write_synthetic_mlp(weights)
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(weights)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv(source_classify.WEIGHTS_ENV, raising=False)
    monkeypatch.setattr(source_classify, "MODEL_SHA256", digest)
    source_classify._load_model.cache_clear()
    try:
        assert source_classify._model_path() == weights
    finally:
        source_classify._load_model.cache_clear()
    assert calls == [
        {
            "repo_id": source_classify.WEIGHTS_REPO,
            "filename": source_classify.MODEL_FILE,
            "revision": source_classify.WEIGHTS_REVISION,
        }
    ]
    # The pins are real content addresses, never a moving branch.
    assert source_classify.WEIGHTS_REVISION != "main"
    assert len(source_classify.WEIGHTS_REVISION) == 40
    assert len(source_classify.MODEL_SHA256) == 64
    int(source_classify.WEIGHTS_REVISION, 16)
    int(source_classify.MODEL_SHA256, 16)


def test_weights_download_failure_is_a_runtime_error_not_a_network_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import source_classify

    def failing_download(**kwargs):
        raise OSError("offline")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = failing_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv(source_classify.WEIGHTS_ENV, raising=False)
    with pytest.raises(RuntimeError):
        source_classify._model_path()


def test_model_loader_rejects_wrong_shapes_and_nonfinite_fields(tmp_path: Path) -> None:
    import numpy as np
    import source_classify

    payload = {
        "classes": np.array(source_classify.CLASSES),
        "kind": np.array("fused"),
        "activation": np.array("gelu_exact"),
        "mean": np.zeros(source_classify.FEATURE_WIDTH - 1, dtype=np.float64),
        "scale": np.ones(source_classify.FEATURE_WIDTH, dtype=np.float64),
        "margins": np.array([0.5, 0.5], dtype=np.float64),
        "layer0_weight": np.zeros((128, source_classify.FEATURE_WIDTH), dtype=np.float32),
        "layer0_bias": np.zeros(128, dtype=np.float32),
        "layer1_weight": np.zeros((64, 128), dtype=np.float32),
        "layer1_bias": np.zeros(64, dtype=np.float32),
        "output_weight": np.zeros((3, 64), dtype=np.float32),
        "output_bias": np.zeros(3, dtype=np.float32),
    }
    bad = tmp_path / "bad.npz"
    with open(bad, "wb") as sink:
        np.savez(sink, **payload)
    import hashlib

    with pytest.raises(ValueError, match="mean"):
        source_classify._load_model(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
    source_classify._load_model.cache_clear()


# ---------------------------------------------------------------------------
# source-classify extra: registry + manifest declaration
# ---------------------------------------------------------------------------


def test_source_classify_extra_joins_the_registry_and_probes_its_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checker probes exactly the module set classify_source imports."""
    import importlib

    import optional_deps

    assert "source-classify" in optional_deps.KNOWN_EXTRAS
    assert "source-classify" in optional_deps.extras_status()

    probed: list[str] = []

    def recording_import(name: str):
        probed.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", recording_import)
    availability = optional_deps.check_optional("source-classify")
    assert availability.available is True
    assert probed == ["numpy", "cv2", "PIL", "huggingface_hub"]


def test_source_classify_extra_missing_deps_report_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import optional_deps

    def missing(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", missing)
    availability = optional_deps.check_optional("source-classify")
    assert availability.available is False
    assert "Install watermark-remover[source-classify]" in availability.hint


def test_pyproject_declares_the_source_classify_extra() -> None:
    """The manifest carries the pixel stack + the hub client, never a weight."""
    extras = load_toml("pyproject.toml")["project"]["optional-dependencies"]
    stack = extras["source-classify"]
    assert any(pkg.lower() == "numpy" for pkg in stack), stack
    assert any(pkg.startswith("opencv") for pkg in stack), stack
    assert any(pkg.lower() == "pillow" for pkg in stack), stack
    assert any(pkg.lower().startswith("huggingface") for pkg in stack), stack
    # weights download at first use — nothing bundled under any name
    assert all("npz" not in pkg.lower() for pkg in stack)


# ---------------------------------------------------------------------------
# identify signals: evidence-only mapping + the extra-gated collector
# ---------------------------------------------------------------------------


def test_identify_source_classify_signal_maps_the_vocabulary() -> None:
    import identify
    import source_classify

    classified = source_classify.SourceClassification(
        "openai",
        "classified",
        {"openai": 1.0, "google": 0.0, "unknown": 0.0},
        {"openai": 1.0, "google": -1.0},
        {"openai": 0.5, "google": 0.5},
    )
    signal = identify.source_classify_signal(classified)
    assert signal["label"] == "openai"
    assert signal["reason"] == "classified"
    assert signal["asserts_ai"] is False  # an export pattern is never an AI verdict
    assert "openai" in signal["finding"]
    abstained = identify.source_classify_signal(
        source_classify.SourceClassification("unknown", "abstained", {"openai": 0.1}, {}, {})
    )
    assert abstained["label"] == "unknown"
    assert abstained["finding"] is None
    for junk in (None, 42, "nope", {}, object()):
        signal = identify.source_classify_signal(junk)
        assert signal["label"] is None
        assert signal["asserts_ai"] is False
        assert signal["finding"] is None


def test_collector_abstains_without_the_extra(tmp_path: Path) -> None:
    """huggingface_hub absent → explicit unavailable entries, never findings."""
    import source_classify

    image = _noise_png(tmp_path / "image.png")
    report = source_classify.source_classify_signals(image)
    assert set(report) == {"source_pipeline", "image_source"}
    for entry in report.values():
        assert entry["available"] is False
        assert "source-classify" in entry["error"]
        assert "finding" not in entry


def test_identify_collector_runs_both_classifiers_when_the_extra_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the extra present, both signals run and abstain on OOD pixels."""
    import types

    import source_classify
    import source_classify_v2

    weights_v1 = tmp_path / "v1" / source_classify.MODEL_FILE
    weights_v1.parent.mkdir()
    digest_v1 = _write_synthetic_mlp(weights_v1, thresholds=(1e6, 1e6))
    from test_source_classify_v2 import _write_synthetic_v2

    weights_v2 = tmp_path / "v2" / source_classify_v2.MODEL_FILE
    weights_v2.parent.mkdir()
    digest_v2 = _write_synthetic_v2(weights_v2)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: ""  # never called: env overrides set
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv(source_classify.WEIGHTS_ENV, str(weights_v1.parent))
    monkeypatch.setenv(source_classify_v2.WEIGHTS_ENV, str(weights_v2.parent))
    monkeypatch.setattr(source_classify, "MODEL_SHA256", digest_v1)
    monkeypatch.setattr(source_classify_v2, "MODEL_SHA256", digest_v2)
    source_classify._load_model.cache_clear()
    source_classify_v2._load_model.cache_clear()
    try:
        image = _noise_png(tmp_path / "image.png")
        report = source_classify.source_classify_signals(image)
    finally:
        source_classify._load_model.cache_clear()
        source_classify_v2._load_model.cache_clear()
    pipeline = report["source_pipeline"]
    assert pipeline["available"] is True
    assert pipeline["label"] == "unknown" and pipeline["reason"] == "abstained"
    assert pipeline["asserts_ai"] is False
    assert pipeline["classification"]["model_revision"] == source_classify.WEIGHTS_REVISION
    image_source = report["image_source"]
    assert image_source["available"] is True
    assert image_source["label"] == "unknown" and image_source["reason"] == "abstained"
    assert image_source["watermark_truth"] == "unknown"


def test_collector_skips_oversized_decodes_with_an_honest_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the decode bound the signal abstains explicitly (never a verdict)."""
    import types

    import source_classify

    class FakeSize:
        size = (source_classify.MAX_SOURCE_CLASSIFY_PIXELS // 100 + 1, 100)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(*args, **kwargs):
        # a header-only read: the fake returns the size without decoding pixels
        return FakeSize()

    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = types.SimpleNamespace(open=fake_open)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setattr(source_classify, "is_available", lambda: True)
    monkeypatch.setattr(
        source_classify,
        "classify_source",
        lambda source: (_ for _ in ()).throw(AssertionError("oversized decode must be skipped")),
    )
    report = source_classify.source_classify_signals("fake.png")
    entry = report["source_pipeline"]
    assert entry["available"] is True
    assert entry["label"] is None
    assert "decode bound" in entry["note"]


def test_provenance_records_pin_the_e2_weight_source() -> None:
    import source_classify

    provenance = (SCRIPTS / "assets" / "PROVENANCE.md").read_text(encoding="utf-8")
    for needle in (
        source_classify.WEIGHTS_REPO,
        source_classify.WEIGHTS_REVISION,
        source_classify.MODEL_SHA256,
        source_classify.MODEL_FILE,
        source_classify.WEIGHTS_ENV,
    ):
        assert needle in provenance, needle


def test_identify_source_classify_signal_accepts_the_to_dict_shape() -> None:
    """The docstring promises the serialized form maps identically."""
    import identify
    import source_classify

    as_dict = source_classify.SourceClassification(
        "google",
        "classified",
        {"openai": 0.0, "google": 1.0, "unknown": 0.0},
        {"openai": -1.0, "google": 1.0},
        {"openai": 0.5, "google": 0.5},
    ).to_dict()
    signal = identify.source_classify_signal(as_dict)
    assert signal["label"] == "google"
    assert signal["reason"] == "classified"
    assert signal["asserts_ai"] is False
    assert signal["scores"]["google"] == pytest.approx(1.0)
