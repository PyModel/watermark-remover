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

import render_pill_silhouette

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
ASSETS = SCRIPTS / "assets"


def test_pill_asset_regenerates_deterministically(tmp_path: Path) -> None:
    """The committed jimeng_pill.png reproduces byte-for-byte from the renderer."""
    out = render_pill_silhouette.render_to_dir(tmp_path)
    assert out == tmp_path / render_pill_silhouette.ASSET_NAME
    assert out.read_bytes() == (ASSETS / render_pill_silhouette.ASSET_NAME).read_bytes()
