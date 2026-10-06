# Tests for the E3 image-source classifier (source_classify_v2.py) and its
# identify signal adapter. No network: the HF download path is monkeypatched
# and the weights are a synthetic in-repo npz built to the module's own schema.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _noise_png(path: Path, width: int = 512, height: int = 512, seed: int = 23) -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def _write_synthetic_v2(path: Path, **threshold_overrides: float) -> str:
    """Build a source_classify_v2-schema npz whose template scores are all 0.0.

    Zero means/precision filters make every _template_score exactly 0.0 (a
    zero numerator over the 1e-12 floor), so with the default 1.0 thresholds
    nothing fires — a deterministic out-of-distribution abstention. Individual
    thresholds can be lowered to fire exactly one route deterministically.
    Returns the synthetic file's SHA-256 (the loader pin, substituted in tests).
    """
    import hashlib

    import numpy as np
    import source_classify_v2

    payload: dict[str, object] = {"schema": np.array(source_classify_v2.MODEL_SCHEMA)}
    for name in source_classify_v2._FILTER_NAMES:
        payload[f"{name}_mean"] = np.zeros((256, 256, 3), dtype=np.complex64)
        payload[f"{name}_precision"] = np.zeros((256, 256, 3), dtype=np.float32)
    for route in source_classify_v2.ROUTES:
        # a lone 1.0 gives a valid unit-norm template the 1024x1024 spatial
        # route (never engaged by these test sizes) cannot fire on
        unit = np.zeros((992, 992), dtype=np.float32)
        unit[0, 0] = np.float32(1.0)
        payload[f"openai_{route}_unit"] = unit
    for name in source_classify_v2._THRESHOLD_NAMES:
        payload[f"threshold_{name}"] = np.float64(1.0)
    for name, value in threshold_overrides.items():
        payload[f"threshold_{name}"] = np.float64(value)
    with open(path, "wb") as sink:
        np.savez_compressed(sink, **payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def synthetic_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point WEIGHTS_ENV at a directory holding the synthetic npz."""
    import source_classify_v2

    weights = tmp_path / "sub" / source_classify_v2.MODEL_FILE
    weights.parent.mkdir()
    digest = _write_synthetic_v2(weights)
    monkeypatch.setenv(source_classify_v2.WEIGHTS_ENV, str(weights.parent))
    monkeypatch.setattr(source_classify_v2, "MODEL_SHA256", digest)
    source_classify_v2._load_model.cache_clear()  # test fixture seam
    yield weights
    source_classify_v2._load_model.cache_clear()


def test_classify_image_source_abstains_out_of_distribution(
    tmp_path: Path, synthetic_v2: Path
) -> None:
    import source_classify_v2

    image = _noise_png(tmp_path / "image.png")
    result = source_classify_v2.classify_image_source(str(image))
    assert result.label == "unknown"
    assert result.reason == "abstained"
    assert result.watermark_truth == "unknown"
    assert result.scores  # evidence-only scores still reported
    assert result.claim == source_classify_v2.CLAIM
    as_dict = result.to_dict()
    assert as_dict["watermark_truth"] == "unknown"
    assert as_dict["model_revision"] == source_classify_v2.WEIGHTS_REVISION


def test_classify_image_source_labels_google_when_a_route_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_v2: Path
) -> None:
    import source_classify_v2

    digest = _write_synthetic_v2(synthetic_v2, r152_google=-0.5)
    monkeypatch.setattr(source_classify_v2, "MODEL_SHA256", digest)
    source_classify_v2._load_model.cache_clear()  # test fixture seam
    image = _noise_png(tmp_path / "image.png")
    result = source_classify_v2.classify_image_source(str(image))
    assert result.label == "google"
    assert result.reason == "classified"
    assert result.scores["r152_google"] == pytest.approx(0.0)


def test_classify_image_source_reports_conflict_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_v2: Path
) -> None:
    import source_classify_v2

    digest = _write_synthetic_v2(synthetic_v2, r152_google=-0.5, r152_openai=-0.5)
    monkeypatch.setattr(source_classify_v2, "MODEL_SHA256", digest)
    source_classify_v2._load_model.cache_clear()  # test fixture seam
    image = _noise_png(tmp_path / "image.png")
    result = source_classify_v2.classify_image_source(str(image))
    assert result.label == "unknown"
    assert result.reason == "conflict"


def test_classify_image_source_feature_unavailable_below_256(
    tmp_path: Path, synthetic_v2: Path
) -> None:
    import source_classify_v2

    image = _noise_png(tmp_path / "tiny.png", width=128, height=128)
    result = source_classify_v2.classify_image_source(str(image))
    assert result.label == "unknown"
    assert result.reason == "feature_unavailable"
    assert result.scores == {}


def test_classify_image_source_uses_the_aspect_group_for_the_r163_route(
    tmp_path: Path, synthetic_v2: Path
) -> None:
    import source_classify_v2

    # 640x1024 is "portrait"; not 1024x1024, so only the r163 route is scored
    image = _noise_png(tmp_path / "portrait.png", width=640, height=1024)
    result = source_classify_v2.classify_image_source(str(image))
    assert "r163_openai" in result.scores
    assert not any(name.startswith("openai_") for name in result.scores)


def test_first_use_downloads_the_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env override → hf_hub_download with repo/file/revision pins; no 'main'."""
    import types

    import source_classify_v2

    weights = tmp_path / source_classify_v2.MODEL_FILE
    digest = _write_synthetic_v2(weights)
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(weights)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv(source_classify_v2.WEIGHTS_ENV, raising=False)
    monkeypatch.setattr(source_classify_v2, "MODEL_SHA256", digest)
    source_classify_v2._load_model.cache_clear()  # test fixture seam
    try:
        assert source_classify_v2._model_path() == weights  # download seam
    finally:
        source_classify_v2._load_model.cache_clear()
    assert calls == [
        {
            "repo_id": source_classify_v2.WEIGHTS_REPO,
            "filename": source_classify_v2.MODEL_FILE,
            "revision": source_classify_v2.WEIGHTS_REVISION,
        }
    ]
    assert source_classify_v2.WEIGHTS_REVISION != "main"
    assert len(source_classify_v2.WEIGHTS_REVISION) == 40
    assert len(source_classify_v2.MODEL_SHA256) == 64
    int(source_classify_v2.WEIGHTS_REVISION, 16)
    int(source_classify_v2.MODEL_SHA256, 16)


def test_model_loader_rejects_a_sha_or_schema_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import numpy as np
    import source_classify_v2

    weights = tmp_path / source_classify_v2.MODEL_FILE
    _write_synthetic_v2(weights)
    with pytest.raises(ValueError, match="SHA-256"):
        source_classify_v2._load_model(weights, "0" * 64)  # loader pin seam
    source_classify_v2._load_model.cache_clear()
    # wrong schema string
    bad = tmp_path / "bad.npz"
    payload = {"schema": np.array("some-other-schema")}
    with open(bad, "wb") as sink:
        np.savez(sink, **payload)
    with pytest.raises(ValueError, match="schema"):
        source_classify_v2._load_model(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
    source_classify_v2._load_model.cache_clear()


def test_model_loader_rejects_a_nonunit_template(tmp_path: Path) -> None:
    import hashlib

    import numpy as np
    import source_classify_v2

    path = tmp_path / "nonunit.npz"
    payload: dict[str, object] = {"schema": np.array(source_classify_v2.MODEL_SCHEMA)}
    for name in source_classify_v2._FILTER_NAMES:  # schema names are the contract
        payload[f"{name}_mean"] = np.zeros((256, 256, 3), dtype=np.complex64)
        payload[f"{name}_precision"] = np.ones((256, 256, 3), dtype=np.float32)
    for route in source_classify_v2.ROUTES:
        payload[f"openai_{route}_unit"] = np.full((992, 992), 2.0, dtype=np.float32)
    for name in source_classify_v2._THRESHOLD_NAMES:
        payload[f"threshold_{name}"] = np.float64(1.0)
    with open(path, "wb") as sink:
        np.savez_compressed(sink, **payload)
    with pytest.raises(ValueError, match="nonunit"):
        source_classify_v2._load_model(path, hashlib.sha256(path.read_bytes()).hexdigest())
    source_classify_v2._load_model.cache_clear()


def test_identify_image_source_signal_maps_the_vocabulary() -> None:
    import identify
    import source_classify_v2

    classified = source_classify_v2.ImageSourceClassification(
        "google", "classified", {"r152_google": 0.4}
    )
    signal = identify.image_source_signal(classified)
    assert signal["label"] == "google"
    assert signal["reason"] == "classified"
    assert signal["asserts_ai"] is False
    assert "google" in signal["finding"]
    assert signal["scores"] == {"r152_google": 0.4}
    assert signal["watermark_truth"] == "unknown"
    abstained = identify.image_source_signal(
        source_classify_v2.ImageSourceClassification("unknown", "abstained", {})
    )
    assert abstained["label"] == "unknown"
    assert abstained["finding"] is None
    junk = identify.image_source_signal(None)
    assert junk["label"] is None
    assert junk["asserts_ai"] is False


def test_provenance_records_pin_the_e3_weight_source() -> None:
    import source_classify_v2

    provenance = (SCRIPTS / "assets" / "PROVENANCE.md").read_text(encoding="utf-8")
    for needle in (
        source_classify_v2.WEIGHTS_REPO,
        source_classify_v2.WEIGHTS_REVISION,
        source_classify_v2.MODEL_SHA256,
        source_classify_v2.MODEL_FILE,
        source_classify_v2.WEIGHTS_ENV,
    ):
        assert needle in provenance, needle
