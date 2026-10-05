"""Renderer-synthetic silhouette asset tests (ticket 06).

The shipped assets under skills/remove-ai-marks/scripts/assets/ must be
reproducible from the in-tree renderers alone: every test here regenerates the
assets into a temp directory and byte-compares against the committed files.
Provenance is test-enforced too: the table in assets/PROVENANCE.md must cover
exactly the shipped set, and the tree must never contain the capture-derived
asset names (upstream visible_alpha_solve outputs, gemini backgrounds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import render_pill_silhouette
import render_vendor_silhouettes

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
ASSETS = SCRIPTS / "assets"


def test_pill_asset_regenerates_deterministically(tmp_path: Path) -> None:
    """The committed jimeng_pill.png reproduces byte-for-byte from the renderer."""
    out = render_pill_silhouette.render_to_dir(tmp_path)
    assert out == tmp_path / render_pill_silhouette.ASSET_NAME
    assert out.read_bytes() == (ASSETS / render_pill_silhouette.ASSET_NAME).read_bytes()


@pytest.mark.parametrize(
    "spec",
    list(render_vendor_silhouettes.ASSET_SPECS),
    ids=lambda spec: spec.name,
)
def test_vendor_asset_regenerates_deterministically(spec: Any, tmp_path: Path) -> None:
    """Every committed vendor silhouette reproduces byte-for-byte in isolation."""
    paths = render_vendor_silhouettes.render_to_dir(tmp_path, names=[spec.name])
    assert paths == [tmp_path / spec.name]
    assert (tmp_path / spec.name).read_bytes() == (ASSETS / spec.name).read_bytes()
