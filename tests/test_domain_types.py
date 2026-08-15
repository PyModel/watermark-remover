"""Tests for immutable inpainting and verification domain payloads."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from domain_types import (
    InpaintRequest,
    InpaintResult,
    MaskResult,
    QualityResult,
    map_quality_to_status,
)


class TestMapQualityToStatus:
    def test_status_precedence(self) -> None:
        assert map_quality_to_status() == "VERIFIED"
        assert map_quality_to_status(quality_uncertain=True) == "BEST_EFFORT"
        assert map_quality_to_status(backend_unavailable=True) == "UNSUPPORTED"
        assert map_quality_to_status(outside_mask_modified=True) == "FAILED"
        assert (
            map_quality_to_status(
                backend_failed=True,
                backend_unavailable=True,
                outside_mask_modified=True,
                quality_uncertain=True,
            )
            == "FAILED"
        )


class TestQualityResult:
    def test_to_dict_and_tuple_copy(self) -> None:
        warnings = ["test warning"]
        result = QualityResult(
            outside_mask_preserved=True,
            boundary_score=0.1,
            halo_score=0.2,
            warnings=warnings,
        )
        warnings.append("late mutation")

        assert result.warnings == ("test warning",)
        assert result.to_dict() == {
            "outside_mask_preserved": True,
            "outside_mask_difference_count": 0,
            "boundary_score": 0.1,
            "halo_score": 0.2,
            "warnings": ["test warning"],
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"outside_mask_preserved": 1},
            {"outside_mask_preserved": True, "outside_mask_difference_count": -1},
            {"outside_mask_preserved": True, "boundary_score": float("nan")},
            {"outside_mask_preserved": True, "halo_score": 1.1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs) -> None:
        with pytest.raises((TypeError, ValueError)):
            QualityResult(**kwargs)

    def test_frozen(self) -> None:
        result = QualityResult(outside_mask_preserved=True)
        with pytest.raises(FrozenInstanceError):
            result.outside_mask_preserved = False


class TestImmutableMappings:
    def test_request_copies_and_freezes_params(self) -> None:
        params = {"radius": 3}
        request = InpaintRequest(source=None, mask=None, params=params)
        params["radius"] = 9

        assert request.params["radius"] == 3
        with pytest.raises(TypeError):
            request.params["radius"] = 5

    def test_result_copies_and_freezes_diagnostics(self) -> None:
        diagnostics = {"score": 0.5}
        result = InpaintResult(inpainted=None, backend_name="test", diagnostics=diagnostics)
        diagnostics["score"] = 0.9

        assert result.diagnostics["score"] == 0.5
        with pytest.raises(TypeError):
            result.diagnostics["score"] = 0.1

    @pytest.mark.parametrize("quality", [-0.1, 1.1, float("inf"), float("nan"), True])
    def test_result_rejects_invalid_quality(self, quality) -> None:
        with pytest.raises(ValueError, match="quality_estimate"):
            InpaintResult(inpainted=None, backend_name="test", quality_estimate=quality)

    def test_result_is_frozen(self) -> None:
        result = InpaintResult(inpainted=None, backend_name="test")
        with pytest.raises(FrozenInstanceError):
            result.backend_name = "other"


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_mask_result_rejects_invalid_confidence(confidence) -> None:
    with pytest.raises(ValueError, match="confidence"):
        MaskResult(None, None, "test", confidence=confidence)
