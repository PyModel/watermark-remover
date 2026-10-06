"""Tests for the pixel-forensics collector (``pixel_evidence``).

Ported from wiltodelta/remove-ai-watermarks tests/test_pixel_evidence.py; the
adapted vectors keep the upstream pins that guard the service contract a
consumer cannot infer safely (families stay empty rather than wrong when the
image is too small for them, one failing family does not lose the other five,
image-identifying artifacts stay behind their opt-in, and the transport schema
is versioned, JSON-safe, and pathless) and add the wiring seams this repo
agrees for ticket 05: the identify signal adapter and the ``/detect`` image
entries (fail-soft, bounded, honest — absent evidence is unknown, never clean).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

FAMILIES = ("dct", "fft", "noise", "ela", "gradient", "color")


def _textured(path: Path, size: tuple[int, int] = (192, 160), *, seed: int = 0) -> Path:
    """A textured image. Flat color would make several families degenerate (zero
    residual, empty gradient histogram) and hide a real break."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    ramp = np.linspace(0, 255, size[0], dtype=np.uint8)[None, :, None]
    Image.fromarray(np.clip(base // 2 + ramp // 2, 0, 255).astype(np.uint8)).save(path)
    return path


class TestTransportContract:
    def test_versioned_json_pathless_to_dict_is_stable_across_calls(self, tmp_path: Path) -> None:
        """The seam every consumer codes against: one schema_version, no local
        path anywhere in the payload, and byte-identical output on repeat calls
        (timing is opt-in, so repeated collection is value-deterministic)."""
        from pixel_evidence import PIXEL_EVIDENCE_SCHEMA_VERSION, extract_pixel_evidence

        path = _textured(tmp_path / "textured.png")
        first = extract_pixel_evidence(path).to_dict()
        second = extract_pixel_evidence(path).to_dict()

        assert first == second
        assert first["schema_version"] == PIXEL_EVIDENCE_SCHEMA_VERSION == 1
        assert first["status"] == "complete"
        assert "path" not in first
        assert json.loads(json.dumps(first, allow_nan=False)) == first

    @pytest.mark.parametrize("schema_version", [2, True, 1.0])
    def test_transport_rejects_unsupported_output_schema(
        self, tmp_path: Path, schema_version: object
    ) -> None:
        from pixel_evidence import extract_pixel_evidence

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))

        with pytest.raises(ValueError, match="unsupported pixel evidence schema"):
            evidence.to_dict(schema_version=schema_version)  # type: ignore[arg-type]

    def test_decode_error_transport_does_not_leak_the_local_path(self, tmp_path: Path) -> None:
        from pixel_evidence import extract_pixel_evidence

        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")

        payload = extract_pixel_evidence(path).to_dict()

        assert payload["status"] == "error"
        assert payload["decode"]["error"] == "UnidentifiedImageError"
        assert str(path) not in json.dumps(payload)


class TestFamilies:
    def test_every_family_is_measured_on_a_textured_image(self, tmp_path: Path) -> None:
        from pixel_evidence import extract_pixel_evidence

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))

        assert evidence.decoded
        for family in FAMILIES:
            assert getattr(evidence, family), family

    def test_the_same_image_measures_the_same_twice(self, tmp_path: Path) -> None:
        """Determinism is what makes these comparable across runs and machines; the
        residual is computed in row chunks, which is exactly the kind of optimization
        that can perturb the last bits."""
        from pixel_evidence import extract_pixel_evidence

        path = _textured(tmp_path / "textured.png")

        first, second = extract_pixel_evidence(path), extract_pixel_evidence(path)

        assert dataclasses.asdict(first) == dataclasses.asdict(second)

    def test_source_dimensions_survive_the_downscale(self, tmp_path: Path) -> None:
        """The recorded size is the SOURCE size, read before the 2048px cap. Recording
        the analysed size instead would still pass every statistic check, because
        those run on the downscaled array either way."""
        from pixel_evidence import extract_pixel_evidence

        path = tmp_path / "oversize.png"
        Image.fromarray(np.zeros((80, 3000, 3), dtype=np.uint8)).save(path)

        assert extract_pixel_evidence(path).decode == {"width": 3000, "height": 80}

    def test_selected_dct_coefficients_match_the_full_transform(self) -> None:
        from pixel_evidence import AC_POSITIONS, _dct_matrix, dct_features

        gray = np.random.default_rng(7).uniform(0, 255, (24, 32)).astype(np.float32)
        basis = _dct_matrix(np)
        blocks = gray.reshape(3, 8, 4, 8).swapaxes(1, 2)
        full = np.einsum("ij,abjk,lk->abil", basis, blocks, basis)
        bins = np.linspace(-20.5, 20.5, 22)
        expected = [
            np.histogram(full[:, :, row, column].ravel(), bins=bins)[0].tolist()
            for row, column in AC_POSITIONS
        ]

        assert dct_features(gray)["dct_ac_hist"] == expected

    def test_perceptual_hash_matches_the_full_transform(self) -> None:
        from pixel_evidence import _dct_matrix, perceptual_hash

        gray = np.random.default_rng(8).uniform(0, 255, (32, 32)).astype(np.float32)
        basis = _dct_matrix(np, 32)
        coefficients = basis @ gray @ basis.T
        low = coefficients[:8, :8].ravel()[1:]
        bits = low > np.median(low)
        expected = f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"

        assert perceptual_hash(gray) == expected


class TestScaleRobustness:
    """Family-appropriate spot checks: the feature SHAPE must not depend on the
    input size (that is what makes cross-scale comparison possible), and the
    perceptual hash must track the same picture across a resize."""

    @pytest.mark.parametrize("size", [(64, 64), (192, 160), (1024, 700)])
    def test_feature_shapes_are_fixed_regardless_of_input_size(
        self, tmp_path: Path, size: tuple[int, int]
    ) -> None:
        from pixel_evidence import extract_pixel_evidence

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png", size))

        assert len(evidence.dct["dct_ac_hist"]) == 8
        assert all(len(hist) == 21 for hist in evidence.dct["dct_ac_hist"])
        assert len(evidence.fft["fft_band_energy"]) == 8
        assert len(evidence.fft["cfa_peaks"]) == 2
        assert len(evidence.color["color_hist_4x4x4"]) == 64
        assert len(evidence.gradient["gradient_hist"]) == 10

    def test_oversized_input_stays_bounded_and_complete(self, tmp_path: Path) -> None:
        """The 2048px cap is the bounded-compute promise: a 3000px input collects
        cleanly (the cap, not the caller, bounds the arrays) and still records
        the true source dimensions."""
        from pixel_evidence import MAX_SIDE, extract_pixel_evidence

        rng = np.random.default_rng(3)
        path = tmp_path / "wide.png"
        Image.fromarray(rng.integers(0, 255, (600, 3000, 3), dtype=np.uint8)).save(path)

        evidence = extract_pixel_evidence(path, artifacts=True)

        assert evidence.status == "complete"
        assert evidence.decode == {"width": 3000, "height": 600}
        assert evidence.artifacts["phash"]
        assert MAX_SIDE == 2048  # the cap itself is pinned

    def test_perceptual_hash_tracks_the_picture_across_a_resize(self, tmp_path: Path) -> None:
        """Same smooth picture at two scales: the low-frequency DCT bits mostly
        agree. An exact match is not required (resampling perturbs the tail
        bits); near-agreement is what identifies the same source."""
        from pixel_evidence import extract_pixel_evidence

        rng = np.random.default_rng(5)
        small = np.zeros((128, 128, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:128, 0:128]
        shade = ((xx * 1.7 + yy * 0.9) % 255).astype(np.uint8)
        small[..., 0] = shade
        small[..., 1] = np.roll(shade, 20, axis=1)
        small[..., 2] = rng.integers(0, 60, (128, 128), dtype=np.uint8) + 100
        big = np.asarray(Image.fromarray(small).resize((512, 512), Image.Resampling.LANCZOS))
        one = extract_pixel_evidence(_dump(tmp_path / "small.png", small), artifacts=True)
        two = extract_pixel_evidence(_dump(tmp_path / "big.png", big), artifacts=True)

        def hamming(a: str, b: str) -> int:
            return bin(int(a, 16) ^ int(b, 16)).count("1")

        assert hamming(one.artifacts["phash"], two.artifacts["phash"]) <= 16


class TestDegenerateInputs:
    def test_undecodable_file_reports_the_error_and_stays_empty(self, tmp_path: Path) -> None:
        from pixel_evidence import extract_pixel_evidence

        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")

        evidence = extract_pixel_evidence(path, artifacts=True)

        assert evidence.decoded is False
        assert "error" in evidence.decode
        assert all(getattr(evidence, family) == {} for family in FAMILIES)
        assert evidence.artifacts == {}

    def test_image_too_small_for_a_family_leaves_it_empty(self, tmp_path: Path) -> None:
        """8x8 is below the FFT's 32px floor but at the block DCT's. A consumer must
        not assume a fixed feature width, so the narrow case is pinned."""
        from pixel_evidence import extract_pixel_evidence

        path = tmp_path / "tiny.png"
        Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)).save(path)

        evidence = extract_pixel_evidence(path)

        assert evidence.decoded is True
        assert evidence.fft == {}
        assert evidence.color != {}

    def test_a_failing_family_does_not_lose_the_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pixel_evidence import extract_pixel_evidence

        path = _textured(tmp_path / "textured.png")

        def boom(*args: object, **kwargs: object) -> dict[str, Any]:
            raise ValueError("family failed")

        monkeypatch.setattr("pixel_evidence.color_features", boom)
        evidence = extract_pixel_evidence(path)

        assert evidence.color == {"error": "ValueError"}
        assert evidence.status == "partial"
        assert evidence.to_dict()["status"] == "partial"
        assert evidence.dct != {}
        assert evidence.gradient != {}


class TestArtifactsAreOptIn:
    """The artifacts identify the source image -- a thumbnail is a picture, a
    perceptual hash matches one. Everything else is a scalar or a fixed-length
    histogram. That difference in kind is the reason for the flag, so the flag is
    what these guard."""

    def test_off_by_default(self, tmp_path: Path) -> None:
        from pixel_evidence import extract_pixel_evidence

        assert extract_pixel_evidence(_textured(tmp_path / "textured.png")).artifacts == {}

    def test_on_request_it_returns_the_spatial_layer(self, tmp_path: Path) -> None:
        from pixel_evidence import extract_pixel_evidence

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"), artifacts=True)

        assert set(evidence.artifacts) == {
            "phash",
            "thumbnail_jpeg_b64",
            "ela_map",
            "noise_residual",
            "fft_phase",
        }
        assert len(evidence.artifacts["phash"]) == 16

    def test_the_thumbnail_is_a_readable_image_of_the_source(self, tmp_path: Path) -> None:
        """Stated plainly because it is the privacy claim: this field reconstructs
        the picture, at 128px."""
        import base64
        import io

        from pixel_evidence import extract_pixel_evidence

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"), artifacts=True)

        with Image.open(
            io.BytesIO(base64.b64decode(evidence.artifacts["thumbnail_jpeg_b64"]))
        ) as thumb:
            assert max(thumb.size) <= 128

    def test_a_different_image_hashes_differently(self, tmp_path: Path) -> None:
        from pixel_evidence import extract_pixel_evidence

        one = extract_pixel_evidence(_textured(tmp_path / "a.png", seed=1), artifacts=True)
        two = extract_pixel_evidence(_textured(tmp_path / "b.png", seed=2), artifacts=True)

        assert one.artifacts["phash"] != two.artifacts["phash"]

    def test_the_statistics_carry_no_array_payloads(self, tmp_path: Path) -> None:
        """Without the flag nothing array-shaped may appear: that is what makes the
        default set aggregates rather than content."""
        from pixel_evidence import extract_pixel_evidence

        evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))

        for family in FAMILIES:
            for key, value in getattr(evidence, family).items():
                assert isinstance(value, (int, float, str, list)), (family, key)
                if isinstance(value, list):
                    for item in value:
                        assert isinstance(item, (int, float, list)), (family, key)


def test_is_available_reports_the_optional_dependency() -> None:
    from pixel_evidence import is_available

    assert is_available() is True  # the test environment installs the pixel extra


def test_evidence_is_frozen(tmp_path: Path) -> None:
    from pixel_evidence import PixelEvidence, extract_pixel_evidence

    evidence = extract_pixel_evidence(_textured(tmp_path / "textured.png"))
    assert isinstance(evidence, PixelEvidence)
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.decode = {}  # type: ignore[misc]


def test_timings_are_opt_in(tmp_path: Path) -> None:
    from pixel_evidence import extract_pixel_evidence

    path = _textured(tmp_path / "textured.png")

    assert extract_pixel_evidence(path).timing_ms == {}
    assert set(extract_pixel_evidence(path, artifacts=True, timings=True).timing_ms) == {
        "decode",
        "noise",
        "fft",
        "ela",
        "dct",
        "gradient",
        "color",
        "full_artifacts",
        "total",
    }


def test_pixel_decode_registers_optional_heif_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = False

    def mark_registered() -> None:
        nonlocal registered
        registered = True

    import image_io

    monkeypatch.setattr(image_io, "_register_heif", mark_registered)

    from pixel_evidence import extract_pixel_evidence

    extract_pixel_evidence(_textured(tmp_path / "textured.png"))

    assert registered is True


def _dump(path: Path, pixels: np.ndarray) -> Path:
    Image.fromarray(pixels).save(path)
    return path


class TestIdentifySignal:
    """The identify-side seam: pure adapters that map collected evidence onto
    the finding vocabulary. They never run a collector, never raise, and never
    turn absent evidence into a clean verdict."""

    def test_a_collected_payload_maps_to_a_structured_finding(self, tmp_path: Path) -> None:
        from identify import pixel_evidence_signal
        from pixel_evidence import extract_pixel_evidence

        payload = extract_pixel_evidence(_textured(tmp_path / "textured.png")).to_dict()

        signal = pixel_evidence_signal(payload)

        assert signal["asserts_ai"] is False  # statistics never assert provenance
        assert signal["finding"] is not None
        assert set(signal["families"]) == set(FAMILIES)
        assert signal["status"] == "complete"
        assert "artifacts" not in json.dumps(signal)  # image content never crosses
        assert "never" in signal["note"]

    def test_absence_is_unknown_never_clean(self) -> None:
        from identify import pixel_evidence_signal

        for absent in (None, {}, "nonsense", 42):
            signal = pixel_evidence_signal(absent)

            assert signal["asserts_ai"] is False
            assert signal["finding"] is None
            assert signal["status"] == "unknown"
            assert "never" in signal["note"]

    def test_a_decode_failure_is_reported_not_hidden(self, tmp_path: Path) -> None:
        from identify import pixel_evidence_signal
        from pixel_evidence import extract_pixel_evidence

        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")

        signal = pixel_evidence_signal(extract_pixel_evidence(path).to_dict())

        assert signal["status"] == "error"
        assert signal["families"] == []
        assert signal["finding"] is None  # nothing measured: nothing claimed

    def test_forensic_metadata_record_maps_to_a_read_only_summary(self) -> None:
        from identify import forensic_metadata_signal

        record = {
            "record_type": "forensic_metadata",
            "schema_version": 1,
            "png_chunks": [],
            "exif": {},
            "c2pa_store": {},
            "sha256": "ab" * 32,
        }

        signal = forensic_metadata_signal(record)

        assert signal["present"] is True
        assert signal["record_type"] == "forensic_metadata"
        assert set(signal["sections"]) == {"c2pa_store", "exif", "png_chunks"}
        assert signal["finding"] is not None
        assert "read-only" in signal["note"]

    def test_forensic_metadata_absence_is_unknown_never_clean(self) -> None:
        from identify import forensic_metadata_signal

        for absent in (None, {}, "junk"):
            signal = forensic_metadata_signal(absent)

            assert signal["present"] is False
            assert signal["finding"] is None
            assert signal["sections"] == []
            assert "never" in signal["note"]
