"""Regression tests for the SynthID-class spectral removal (F1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/remove-ai-marks/scripts"
sys.path.insert(0, str(SCRIPTS))

from synthid_remove import (
    DEFAULT_REMOVE_STRENGTH,
    detect_synthid_pattern,
    embed_synthid_pattern,
    remove_synthid_from_bytes,
)


def _flat(width: int, height: int, channels: int, value: int = 128) -> bytes:
    return bytes([value]) * (width * height * channels)


def _flat_rgb(width: int, height: int) -> bytes:
    data = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            base = 120 + ((x * 7 + y * 3) % 40)
            data[idx] = base
            data[idx + 1] = base - 10
            data[idx + 2] = base - 20
    return bytes(data)


class TestEmbedDetectRemove:
    def test_embed_then_detect_is_watermarked(self):
        raw = _flat_rgb(64, 48)
        embedded = embed_synthid_pattern(raw, 64, 48, 3, seed=42, strength=0.25)
        detection = detect_synthid_pattern(bytes(embedded), 64, 48, 3, seed=42)
        assert detection.is_watermarked
        # Calibrated: strength 0.25 reads ~0.245, threshold is 0.1.
        assert detection.confidence >= 0.2

    def test_plain_image_is_not_watermarked(self):
        raw = _flat_rgb(64, 48)
        detection = detect_synthid_pattern(raw, 64, 48, 3, seed=42)
        assert not detection.is_watermarked

    def test_wrong_seed_does_not_detect(self):
        raw = _flat_rgb(64, 48)
        embedded = embed_synthid_pattern(raw, 64, 48, 3, seed=42, strength=0.25)
        detection = detect_synthid_pattern(bytes(embedded), 64, 48, 3, seed=99)
        assert not detection.is_watermarked

    def test_remove_destroys_signal(self):
        raw = _flat_rgb(64, 48)
        embedded = embed_synthid_pattern(raw, 64, 48, 3, seed=42, strength=0.25)
        before = detect_synthid_pattern(bytes(embedded), 64, 48, 3, seed=42)
        removed = remove_synthid_from_bytes(
            bytes(embedded), 64, 48, 3, strength=DEFAULT_REMOVE_STRENGTH
        )
        after = detect_synthid_pattern(bytes(removed), 64, 48, 3, seed=42)
        assert before.is_watermarked
        assert not after.is_watermarked
        assert after.confidence < before.confidence

    def test_remove_preserves_alpha(self):
        raw = bytearray([100, 110, 120, 200] * (32 * 32))
        embedded = embed_synthid_pattern(bytes(raw), 32, 32, 4, seed=7, strength=0.3)
        removed = remove_synthid_from_bytes(bytes(embedded), 32, 32, 4, strength=0.6)
        for i in range(32 * 32):
            assert removed[i * 4 + 3] == 200

    def test_removal_output_close_to_original(self):
        raw = _flat_rgb(64, 48)
        embedded = embed_synthid_pattern(raw, 64, 48, 3, seed=42, strength=0.25)
        removed = remove_synthid_from_bytes(bytes(embedded), 64, 48, 3, strength=0.6)
        # Band suppression must not wreck the flat image.
        mean_delta = sum(abs(a - b) for a, b in zip(raw, removed, strict=True)) / len(raw)
        assert mean_delta < 30

    def test_validation_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            embed_synthid_pattern(_flat(8, 8, 3), 8, 8, 3, seed=1, strength=2.0)
        with pytest.raises(ValueError):
            detect_synthid_pattern(_flat(8, 8, 3), 8, 8, 3, seed=1, block_size=0)
        with pytest.raises(ValueError):
            embed_synthid_pattern(_flat(8, 8, 3), 8, 8, 3, seed=1, block_size=0)
        with pytest.raises(ValueError):
            remove_synthid_from_bytes(_flat(8, 8, 3), 8, 8, 3, strength=1.5)
        with pytest.raises(ValueError):
            remove_synthid_from_bytes(b"x" * 10, 8, 8, 3, strength=0.5)

    def test_import_path_is_clean(self):
        # Importing the removal module must not pull optional deps into the
        # fresh process (stdlib-only core requirement).
        proc = subprocess.run(
            [sys.executable, "-c", "import sys; import synthid_remove; print(sorted(sys.modules))"],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS),
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "cv2" not in proc.stdout
        assert "skimage" not in proc.stdout
        assert "torch" not in proc.stdout
