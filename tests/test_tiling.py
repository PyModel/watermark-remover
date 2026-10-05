"""Vectors for the ported tiling geometry (plan_tiles / feather_weights / run_tiled).

Ported from wiltodelta/remove-ai-watermarks ``_internal/tiling.py``, Apache-2.0;
modified: numpy-tier instead of PIL tiles, generate_tile receives the Tile,
rint-exact blending, auto-tile threshold policy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts")
)

import tiling
from tiling import Tile, plan_tiles

# A deterministic 48x64 three-channel ramp: every pixel distinct, no RNG.
_RAMP_48x64 = bytes(
    byte
    for y in range(64)
    for x in range(48)
    for byte in ((x * 5) % 256, (y * 7) % 256, ((x + y) * 3) % 256)
)


def _raster():
    np = pytest.importorskip("numpy")
    return np.frombuffer(_RAMP_48x64, dtype=np.uint8).reshape(64, 48, 3)


# --- plan_tiles: pure geometry -------------------------------------------------


def test_plan_tiles_covers_image_in_uniform_row_major_tiles():
    tiles = plan_tiles(3000, 2000, 1024, 128)
    # Every tile uniform 1024x1024; last column/row pulled flush to the edge.
    assert all(tile.width == 1024 and tile.height == 1024 for tile in tiles)
    assert tiles[0].x == 0 and tiles[0].y == 0
    assert tiles[-1].x + tiles[-1].width == 3000
    assert tiles[-1].y + tiles[-1].height == 2000
    # Row-major order: x advances fastest.
    assert [tile.x for tile in tiles[:3]] == sorted(tile.x for tile in tiles[:3])
    # Every pixel covered at least once.
    np = pytest.importorskip("numpy")
    coverage = np.zeros((2000, 3000), dtype=np.int32)
    for tile in tiles:
        coverage[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += 1
    assert bool((coverage >= 1).all())


def test_plan_tiles_small_image_is_one_tile_at_origin():
    assert plan_tiles(500, 400, 1024, 128) == [Tile(0, 0, 500, 400)]


def test_plan_tiles_rejects_nonpositive_tile_size():
    with pytest.raises(ValueError):
        plan_tiles(100, 100, 0, 8)


def test_plan_tiles_survives_pathological_overlap():
    # overlap >= tile clamps to tile-1: forward progress, full coverage.
    tiles = plan_tiles(300, 100, 64, 999)
    assert tiles
    assert tiles[-1].x + tiles[-1].width == 300


# --- feather_weights -----------------------------------------------------------


def test_feather_weights_interior_unity_edges_tapered_strictly_positive():
    pytest.importorskip("numpy")
    weights = tiling.feather_weights(64, 48, 8)
    assert weights.shape == (48, 64)
    assert weights[24, 32] == 1.0
    assert 0.0 < weights[0, 32] < 1.0 and 0.0 < weights[47, 32] < 1.0
    assert 0.0 < weights[24, 0] < 1.0 and 0.0 < weights[24, 63] < 1.0
    assert bool((weights > 0).all())


# --- run_tiled: the same-result guarantee ---------------------------------------


def test_run_tiled_identity_transform_reproduces_the_image_exactly():
    """The stated guarantee: agreeing per-tile results blend back byte-exact."""
    np = pytest.importorskip("numpy")
    image = _raster()
    out = tiling.run_tiled(lambda tile, crop: crop, image, tile_size=32, overlap=8)
    assert np.array_equal(out, image)


def test_run_tiled_constant_transform_is_exact_everywhere():
    np = pytest.importorskip("numpy")
    image = _raster()

    def paint_black(tile, crop):
        return np.zeros_like(crop)

    out = tiling.run_tiled(paint_black, image, tile_size=32, overlap=8)
    assert not out.any()


def test_run_tiled_reports_progress_per_tile():
    pytest.importorskip("numpy")
    image = _raster()
    lines: list[str] = []
    tiling.run_tiled(lambda tile, crop: crop, image, 32, 8, set_progress=lines.append)
    assert len(lines) == len(plan_tiles(48, 64, 32, 8))
    assert "1/" in lines[0]


# --- auto-tile threshold policy --------------------------------------------------


def test_should_auto_tile_default_threshold_2048():
    assert tiling.AUTO_TILE_MAX_DIMENSION == 2048
    assert tiling.should_auto_tile(2048, 100) is False
    assert tiling.should_auto_tile(2049, 100) is True
    assert tiling.should_auto_tile(100, 4000) is True


def test_auto_tile_threshold_env_override(monkeypatch):
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "64")
    assert tiling.auto_tile_threshold() == 64
    assert tiling.should_auto_tile(64, 10) is False
    assert tiling.should_auto_tile(65, 10) is True


# --- pixel extra labeled-unavailable ---------------------------------------------


def _run_blocked(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run *source* with numpy/cv2/PIL import-blocked via stub modules.

    The t01/t02 boundary technique: stub modules on PYTHONPATH that raise
    ModuleNotFoundError at import body, mirroring a truly absent wheel.
    """
    stubs = tmp_path / "blocked-stubs"
    stubs.mkdir()
    for name in ("numpy", "cv2", "PIL"):
        (stubs / f"{name}.py").write_text(
            f"raise ModuleNotFoundError('stub-blocked: {name}')\n", encoding="utf-8"
        )
    scripts = str(Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts")
    return subprocess.run(
        [sys.executable, "-c", source],
        input="",
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PYTHONPATH": f"{stubs}{os.pathsep}{scripts}", "PATH": "/usr/bin:/bin"},
        check=False,
    )


def test_tiling_without_numpy_is_labeled_unavailable(tmp_path):
    probe = (
        "import tiling\n"
        "try:\n"
        "    tiling.feather_weights(8, 8, 2)\n"
        "except tiling.PixelDependencyUnavailable as error:\n"
        "    print('LABELED:', error)\n"
        "else:\n"
        "    print('NO-RAISE')\n"
    )
    result = _run_blocked(probe, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "LABELED:" in result.stdout and "watermark-remover[pixel]" in result.stdout


def test_run_tiled_without_numpy_is_labeled_unavailable(tmp_path):
    # numpy itself is blocked, so run_tiled must raise the labeled error
    # before touching the (unusable) image argument.
    probe = (
        "import tiling\n"
        "try:\n"
        "    tiling.run_tiled(lambda tile, crop: crop, object(), 32, 8)\n"
        "except tiling.PixelDependencyUnavailable as error:\n"
        "    print('LABELED:', error)\n"
        "else:\n"
        "    print('NO-RAISE')\n"
    )
    result = _run_blocked(probe, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "LABELED:" in result.stdout and "watermark-remover[pixel]" in result.stdout


# --- wiring: the auto-tile threshold on the visible clean path ---------------------


def _tiled_fixture(tmp_path):
    """240x64 ramp with a box mark fully inside the first tile's interior."""
    from morphomod import Raster, encode_png

    w, h = 240, 64
    px = bytearray(
        byte
        for y in range(h)
        for x in range(w)
        for byte in ((x * 3) % 256, (y * 9) % 256, ((x + 2 * y) * 5) % 256)
    )
    source = tmp_path / "input.png"
    source.write_bytes(encode_png(Raster(w, h, 3, px)))
    return source


def _visible_plan(tmp_path):
    from clean_asset import CleanPlan
    from morphomod import VisiblePlan

    return CleanPlan(
        forced_kind="image",
        visible=VisiblePlan(
            box=(8, 8, 12, 10),
            backend="texture",
            mask_output=tmp_path / "mask.pgm",
            dilation_radius=1,
        ),
    )


def _visible_plan(tmp_path, backend="texture"):
    from clean_asset import CleanPlan
    from morphomod import VisiblePlan

    return CleanPlan(
        forced_kind="image",
        visible=VisiblePlan(
            box=(8, 8, 12, 10),
            backend=backend,
            mask_output=tmp_path / "mask.pgm",
            dilation_radius=1,
        ),
    )


def test_large_image_cleans_identically_tiled_and_untiled(tmp_path, monkeypatch):
    """The same-result guarantee, end to end: the simple fill is tile-local
    (nearest-boundary), so the tiled path reproduces the whole-image clean
    byte-for-byte when the mark sits inside one tile."""
    from clean_asset import clean_asset

    source = _tiled_fixture(tmp_path)
    untiled = tmp_path / "untiled.png"
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "1000000")
    clean_asset(source, untiled, _visible_plan(tmp_path, backend="simple"))

    tiled = tmp_path / "tiled.png"
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "96")
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_SIZE", "96")
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_OVERLAP", "24")
    result = clean_asset(source, tiled, _visible_plan(tmp_path, backend="simple")).to_dict()

    assert tiled.read_bytes() == untiled.read_bytes()
    assert result["visible"]["tiling"]["engaged"] is True


def test_tiling_engagement_appears_in_the_report_with_the_grid(tmp_path, monkeypatch):
    from clean_asset import clean_asset

    source = _tiled_fixture(tmp_path)
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "96")
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_SIZE", "96")
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_OVERLAP", "24")
    report = clean_asset(source, tmp_path / "out.png", _visible_plan(tmp_path)).to_dict()
    tiling_report = report["visible"]["tiling"]
    assert tiling_report["engaged"] is True
    assert tiling_report["threshold"] == 96
    assert tiling_report["tiles"] > 1
    assert tiling_report["tile_size"] == 96  # the env override, not the default
    inpaint_details = [
        detail for detail in report["visible"]["action_details"] if detail["code"] == "inpaint"
    ]
    assert inpaint_details and inpaint_details[0]["params"].get("tiles") == tiling_report["tiles"]


def test_below_threshold_no_tiling_engagement(tmp_path, monkeypatch):
    from clean_asset import clean_asset

    source = _tiled_fixture(tmp_path)
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "1000000")
    report = clean_asset(source, tmp_path / "out.png", _visible_plan(tmp_path)).to_dict()
    assert "tiling" not in report["visible"]


def test_tiling_without_pixel_deps_falls_back_labeled(tmp_path, monkeypatch):
    """numpy blocked: the tiled attempt degrades to the whole-image clean with a
    labeled-unavailable report, never a crash."""
    import clean_asset as clean_asset_module
    from clean_asset import clean_asset

    calls = {"tiled": 0}

    def _explode(*args, **kwargs):
        calls["tiled"] += 1
        raise clean_asset_module.PixelDependencyUnavailable("stub-blocked: numpy")

    monkeypatch.setattr(clean_asset_module, "_run_tiled_visible", _explode)
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "96")
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_SIZE", "96")
    monkeypatch.setenv("WATERMARKS_AUTO_TILE_OVERLAP", "24")
    source = _tiled_fixture(tmp_path)
    report = clean_asset(source, tmp_path / "out.png", _visible_plan(tmp_path)).to_dict()
    assert calls["tiled"] == 1
    assert report["visible"]["tiling"]["engaged"] is False
    assert report["visible"]["tiling"]["status"] == "unavailable"
    assert "watermark-remover[pixel]" in report["visible"]["tiling"]["hint"]
    assert Path(tmp_path / "out.png").is_file()
