"""Tests for frequency-domain watermark degradation attacks (DCT/FFT)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from dct_frequency import (
    degrade_image,
    frequency_suppress,
    frequency_suppress_from_bytes,
    gaussian_blur_2d,
    jpeg_compress_sim,
    rotate_image,
    two_stage_attack,
)
from morpho_perturb import morpho_perturb


class TestFrequencySuppress:
    """Test DCT-based frequency suppression."""

    def test_suppress_keeps_dimensions(self) -> None:
        width, height, channels = 16, 16, 3
        pixels: list[list[float]] = [[128.0] * channels for _ in range(width * height)]
        result = frequency_suppress(pixels, width, height, channels, suppress=0.5)
        assert len(result) == width * height
        assert len(result[0]) == channels

    def test_suppress_no_effect_at_zero(self) -> None:
        width, height, channels = 8, 8, 3
        pixels: list[list[float]] = [[255.0, 0.0, 128.0]] * (width * height)
        result = frequency_suppress(pixels, width, height, channels, suppress=0.0)
        for _i, row in enumerate(result):
            assert row[0] == pytest.approx(255.0, abs=0.1)
            assert row[1] == pytest.approx(0.0, abs=0.1)

    def test_suppress_aggressive_changes_values(self) -> None:
        width, height, channels = 16, 16, 3
        pixels: list[list[float]] = [[255.0, 0.0, 128.0]] * (width * height)
        result = frequency_suppress(pixels, width, height, channels, suppress=0.9)
        # Aggressive suppression should change at least some values
        changed = any(r[0] != 255.0 or r[1] != 0.0 for r in result)
        assert changed

    def test_from_bytes(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = frequency_suppress_from_bytes(raw, width, height, channels, suppress=0.5)
        assert len(result) == width * height * channels
        assert isinstance(result, bytearray)


class TestDegradationStrategies:
    """Test all degrade_image strategies."""

    def test_freq_dct(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = degrade_image(raw, width, height, channels, strategy="freq-dct", suppress=0.5)
        assert result.strategy == "freq-dct"
        assert result.width == width
        assert result.height == height
        assert result.channels == channels
        assert len(result.data) == width * height * channels

    def test_blur(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = degrade_image(raw, width, height, channels, strategy="blur", sigma=1.0)
        assert result.strategy == "blur"
        assert len(result.data) == width * height * channels

    def test_median(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = degrade_image(raw, width, height, channels, strategy="median", kernel_size=3)
        assert result.strategy == "median"
        assert len(result.data) == width * height * channels

    def test_jpeg(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = degrade_image(raw, width, height, channels, strategy="jpeg", quality=40)
        assert result.strategy == "jpeg"
        assert len(result.data) == width * height * channels

    def test_rotate(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = degrade_image(raw, width, height, channels, strategy="rotate", angle_deg=3.0)
        assert result.strategy == "rotate"
        assert len(result.data) == width * height * channels

    def test_two_stage(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        result = degrade_image(
            raw, width, height, channels, strategy="two-stage", blur_sigma=1.0, quality=40
        )
        assert result.strategy == "two-stage"
        assert len(result.data) == width * height * channels

    def test_unknown_strategy(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        with pytest.raises(ValueError, match="unknown strategy"):
            degrade_image(raw, width, height, channels, strategy="banana")

    def test_wrong_raw_length(self) -> None:
        raw = b"\x00\x01\x02"
        with pytest.raises(ValueError, match="raw length"):
            degrade_image(raw, 2, 2, 3, strategy="blur")


class TestTwoStageAttack:
    """Test the combined two-stage degradation."""

    def test_two_stage_preserves_bytes(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([200, 50, 100] * (width * height))
        result = two_stage_attack(raw, width, height, channels)
        assert len(result.data) == len(raw)
        assert all(0 <= v <= 255 for v in result.data)

    def test_two_stage_different_from_input(self) -> None:
        width, height, channels = 16, 16, 3
        raw = bytes([i % 256 for i in range(width * height * channels)])
        result = two_stage_attack(raw, width, height, channels)
        assert result.data != bytearray(raw)


class TestMorphologicalPerturb:
    """Test morphological Layer C perturbations."""

    def test_grid(self) -> None:
        width, height, channels = 16, 16, 3
        raw = bytes([128] * (width * height * channels))
        result = morpho_perturb(raw, width, height, channels, strategy="grid", spacing=4)
        assert result.strategy == "grid"
        assert len(result.data) == width * height * channels

    def test_diagonal(self) -> None:
        width, height, channels = 16, 16, 3
        raw = bytes([128] * (width * height * channels))
        result = morpho_perturb(raw, width, height, channels, strategy="diagonal", opacity=0.05)
        assert result.strategy == "diagonal"
        assert len(result.data) == width * height * channels

    def test_noise(self) -> None:
        width, height, channels = 16, 16, 3
        raw = bytes([128] * (width * height * channels))
        result = morpho_perturb(raw, width, height, channels, strategy="noise", sigma=5.0)
        assert result.strategy == "noise"
        assert len(result.data) == width * height * channels

    def test_quantize(self) -> None:
        width, height, channels = 16, 16, 3
        raw = bytes([128] * (width * height * channels))
        result = morpho_perturb(raw, width, height, channels, strategy="quantize", levels=16)
        assert result.strategy == "quantize"
        assert len(result.data) == width * height * channels

    def test_unknown_strategy(self) -> None:
        width, height, channels = 8, 8, 3
        raw = bytes([128] * (width * height * channels))
        with pytest.raises(ValueError, match="unknown morphological strategy"):
            morpho_perturb(raw, width, height, channels, strategy="lava")

    def test_grid_preserves_bounds(self) -> None:
        width, height, channels = 32, 32, 3
        raw = bytes([200] * (width * height * channels))
        result = morpho_perturb(raw, width, height, channels, strategy="grid", opacity=0.1)
        assert all(0 <= v <= 255 for v in result.data)


class TestGaussianBlur:
    """Test Gaussian blur implementation."""

    def test_blur_smoother(self) -> None:
        width, height, channels = 16, 16, 3
        # Create an image with a bright spot
        pixels: list[list[float]] = [[0.0] * channels for _ in range(width * height)]
        pixels[width * 8 + 8] = [255.0, 0.0, 0.0]
        result = gaussian_blur_2d(pixels, width, height, channels, sigma=2.0)
        # Blur should spread the bright pixel
        assert result[width * 8 + 8][0] > 0.0


class TestJPEGCompress:
    """Test simulated JPEG compression."""

    def test_jpeg_keeps_bounds(self) -> None:
        width, height, channels = 16, 16, 3
        pixels: list[list[float]] = [[float(i % 256)] * channels for i in range(width * height)]
        result = jpeg_compress_sim(pixels, width, height, channels, quality=50)
        assert len(result) == width * height
        for row in result:
            assert all(0 <= v <= 255 for v in row)


class TestRotate:
    """Test rotation implementation."""

    def test_zero_rotation_preserves(self) -> None:
        width, height, channels = 8, 8, 3
        pixels: list[list[float]] = [[float(i % 256)] * channels for i in range(width * height)]
        result = rotate_image(pixels, width, height, channels, angle_deg=0.0)
        assert result == pixels

    def test_rotation_keeps_bounds(self) -> None:
        width, height, channels = 16, 16, 3
        pixels: list[list[float]] = [[float(i % 256)] * channels for i in range(width * height)]
        result = rotate_image(pixels, width, height, channels, angle_deg=3.0)
        for row in result:
            assert all(0 <= v <= 255 for v in row)


"""Validation and dispatch regressions for the degradation strategies."""


from dct_frequency import (
    MAX_FREQ_DCT_PIXELS,
    median_filter_2d,
)


class TestStrictKeywordDispatch:
    """degrade_image rejects keywords the selected strategy does not accept."""

    def test_unexpected_keyword_rejected(self) -> None:
        raw = bytes([128] * (8 * 8 * 3))
        with pytest.raises(TypeError, match="unexpected keyword"):
            degrade_image(raw, 8, 8, 3, strategy="blur", suppress=0.5)

    def test_seed_rejected_for_freq_dct(self) -> None:
        raw = bytes([128] * (8 * 8 * 3))
        with pytest.raises(TypeError, match="unexpected keyword"):
            degrade_image(raw, 8, 8, 3, strategy="freq-dct", seed=3)

    def test_typo_keyword_rejected(self) -> None:
        raw = bytes([128] * (8 * 8 * 3))
        with pytest.raises(TypeError, match="unexpected keyword"):
            degrade_image(raw, 8, 8, 3, strategy="blur", sigm=2.0)


class TestInputValidation:
    def test_suppress_out_of_range(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="suppress"):
            frequency_suppress(pixels, 8, 8, 3, suppress=1.5)

    def test_block_size_too_small(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="block_size"):
            frequency_suppress(pixels, 8, 8, 3, block_size=1)

    def test_overlap_at_least_block_size(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="overlap"):
            frequency_suppress(pixels, 8, 8, 3, overlap=8)

    def test_negative_overlap(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="overlap"):
            frequency_suppress(pixels, 8, 8, 3, overlap=-1)

    def test_invalid_channels(self) -> None:
        raw = bytes(8 * 8 * 2)
        with pytest.raises(ValueError, match="channels"):
            degrade_image(raw, 8, 8, 2, strategy="blur")

    def test_negative_sigma(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="sigma"):
            gaussian_blur_2d(pixels, 8, 8, 3, sigma=-1.0)

    def test_even_kernel_size(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="kernel_size"):
            median_filter_2d(pixels, 8, 8, 3, kernel_size=4)

    def test_jpeg_quality_out_of_range(self) -> None:
        pixels = [[128.0] * 3 for _ in range(64)]
        with pytest.raises(ValueError, match="quality"):
            jpeg_compress_sim(pixels, 8, 8, 3, quality=101)


class TestPixelCaps:
    """DCT-based strategies fail loudly above their pixel caps."""

    def test_freq_dct_rejects_oversized_image(self) -> None:
        size = 257  # 66_049 pixels > MAX_FREQ_DCT_PIXELS
        raw = bytes([128] * (size * size * 3))
        with pytest.raises(ValueError, match="downscale"):
            degrade_image(raw, size, size, 3, strategy="freq-dct")

    def test_freq_dct_accepts_boundary_size(self) -> None:
        size = 256
        assert size * size == MAX_FREQ_DCT_PIXELS
        raw = bytes([128] * (size * size))
        result = degrade_image(raw, size, size, 1, strategy="freq-dct")
        assert len(result.data) == len(raw)

    def test_jpeg_rejects_oversized_image(self) -> None:
        size = 513  # 263_169 pixels > MAX_JPEG_PIXELS
        raw = bytes([128] * (size * size * 3))
        with pytest.raises(ValueError, match="downscale"):
            degrade_image(raw, size, size, 3, strategy="jpeg")

    def test_two_stage_rejects_oversized_image(self) -> None:
        size = 513
        raw = bytes([128] * (size * size * 3))
        with pytest.raises(ValueError, match="downscale"):
            degrade_image(raw, size, size, 3, strategy="two-stage")
