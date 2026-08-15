"""Regressions for deterministic synthetic benchmark generation."""

from __future__ import annotations

import random

import benchmark
import pytest
from benchmark import BenchmarkConfig, _apply_synthetic_watermark, generate_synthetic_benchmark

PIL_IMAGE = pytest.importorskip("PIL.Image")


def test_watermark_opacity_changes_composited_pixels() -> None:
    source = PIL_IMAGE.new("RGB", (80, 80), (20, 40, 60))
    transparent, transparent_mask = _apply_synthetic_watermark(
        source,
        80,
        0,
        random.Random(7),
        (0.0,),
    )
    opaque, opaque_mask = _apply_synthetic_watermark(
        source,
        80,
        0,
        random.Random(7),
        (1.0,),
    )

    assert transparent.tobytes() == source.tobytes()
    assert opaque.tobytes() != source.tobytes()
    assert transparent_mask == opaque_mask


def test_generator_persists_jpeg_bytes_at_advertised_path(tmp_path) -> None:
    items = generate_synthetic_benchmark(
        BenchmarkConfig(
            output_dir=tmp_path,
            count=1,
            seed=3,
            sizes=(80,),
            watermark_opacities=(0.5,),
            jpeg_qualities=(75,),
        )
    )

    assert len(items) == 1
    assert items[0].watermark_path.suffix == ".jpg"
    assert items[0].watermark_path.read_bytes().startswith(b"\xff\xd8\xff")
    assert PIL_IMAGE.open(items[0].watermark_path).format == "JPEG"


def test_changed_pixels_are_contained_by_ground_truth_mask() -> None:
    source = PIL_IMAGE.new("RGB", (80, 80), (20, 40, 60))
    watermarked, mask = _apply_synthetic_watermark(
        source,
        80,
        0,
        random.Random(11),
        (1.0,),
    )

    source_bytes = source.tobytes()
    watermarked_bytes = watermarked.tobytes()
    changed = [
        index
        for index in range(80 * 80)
        if source_bytes[index * 3 : index * 3 + 3] != watermarked_bytes[index * 3 : index * 3 + 3]
    ]
    assert changed
    assert all(mask[index] == 255 for index in changed)


def test_missing_pillow_has_supported_extra_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark.optional_deps, "_import_safe", lambda name: None)
    with pytest.raises(RuntimeError, match=r"watermark-remover\[visible\]"):
        generate_synthetic_benchmark(BenchmarkConfig(output_dir=tmp_path))
