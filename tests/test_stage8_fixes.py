"""Stage 8 fix-pass 2 regression tests (findings F1-F11).

Each test is the red/green seam for one cited finding from the stage-8 review;
commit messages reference the finding IDs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# F1: `--mark` + auto-tile crash
# ---------------------------------------------------------------------------


class TestF1MarkAutoTile:
    """--mark plans must never route through the tiled mask pass.

    The tiled pass calls ``remove_visible`` with ``dest=None`` for the
    whole-image mask computation; the mark path asserted ``dest is not None``,
    so any mark run over an above-threshold image crashed with an
    AssertionError instead of cleaning.
    """

    def _planted_fixture(self, path: Path) -> None:
        """A bottom-right Doubao-class mark at the engine's measured geometry."""
        import cv2
        import numpy as np
        from morphomod import Raster, encode_png

        asset = cv2.imread(
            str(SCRIPTS / "assets" / "doubao_outline_alpha.png"), cv2.IMREAD_GRAYSCALE
        )
        assert asset is not None
        at = asset.astype(np.float32) / 255.0
        w = h = 1024
        gw = int(0.1636 * w)
        gh = int(gw * 77 / 346)
        ax = w - int(0.015 * w) - gw
        ay = h - int(0.015 * h) - gh
        img = np.full((h, w, 3), 100, np.uint8)
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
        a3 = amap[:, :, None]
        img = (a3 * 255 + (1 - a3) * img).clip(0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        path.write_bytes(encode_png(Raster(w, h, 3, bytearray(rgb.tobytes()))))

    def test_mark_auto_on_tiled_image_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The >2048px vector, with the auto-tile threshold moved to this
        fixture via its policy env: a mark plan must complete untiled and
        report tiling as skipped-for-mark, never crash with an assert."""
        pytest.importorskip("cv2")
        from clean_asset import CleanPlan, clean_asset
        from morphomod import VisiblePlan

        src = tmp_path / "marked.png"
        self._planted_fixture(src)
        monkeypatch.setenv("WATERMARKS_AUTO_TILE_MAX_DIMENSION", "64")

        result = clean_asset(
            src,
            tmp_path / "out.png",
            CleanPlan(
                forced_kind="image",
                visible=VisiblePlan(mark="auto", backend="texture"),
            ),
        ).to_dict()

        visible = result["visible"]
        assert visible["status"] == "completed"
        tiling_report = visible.get("tiling")
        assert tiling_report is not None
        assert tiling_report["engaged"] is False
        assert tiling_report["status"] == "skipped-for-mark"
        assert (tmp_path / "out.png").is_file()

    def test_mark_plan_without_dest_raises_labeled_error_never_assert(self, tmp_path: Path):
        """Direct unit pin: a dest-less mark plan raises the labeled ValueError,
        not an AssertionError (the tiled mask pass must stay dest-less-legal to
        the type system)."""
        from morphomod import Raster, VisiblePlan, encode_png, remove_visible

        src = tmp_path / "in.png"
        src.write_bytes(encode_png(Raster(8, 8, 3, bytearray([9, 8, 7] * 64))))
        with pytest.raises(ValueError, match="output required"):
            remove_visible(src, None, VisiblePlan(mark="auto"))


# ---------------------------------------------------------------------------
# F2: migan/lama extras undeclared
# ---------------------------------------------------------------------------


class TestF2LearnedInpaintExtras:
    """The ``migan``/``lama`` extras the runtime hints name must exist."""

    @pytest.mark.parametrize("extra", ["migan", "lama"])
    def test_extras_status_lists_the_learned_backend_extras(self, extra: str):
        import optional_deps

        assert extra in optional_deps.KNOWN_EXTRAS
        status = optional_deps.extras_status()
        assert extra in status, f"extras_status() does not list {extra}"
        assert set(status[extra]) == {"available", "hint"}

    @pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
    @pytest.mark.parametrize("extra", ["migan", "lama"])
    def test_pyproject_declares_the_extra_with_runtime_requirements(self, extra: str):
        import tomllib

        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        assert extra in extras, f"pyproject.toml does not declare the {extra} extra"
        declared = {
            req.split(";")[0].split(">=")[0].split("==")[0].strip().lower() for req in extras[extra]
        }
        assert {"onnxruntime", "huggingface-hub"} <= declared

    @pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
    def test_uv_lock_carries_the_learned_backend_extras(self):
        """The regenerated lock must contain the declared extras."""
        import tomllib

        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        root = next(pkg for pkg in lock["package"] if pkg["name"] == "watermark-remover")
        extras = root.get("optional-dependencies", {})
        assert {"migan", "lama"} <= set(extras)
