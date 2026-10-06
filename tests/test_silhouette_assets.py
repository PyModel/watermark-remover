"""Renderer-synthetic silhouette asset tests (ticket 06).

The shipped assets under skills/remove-ai-marks/scripts/assets/ must be
reproducible from the in-tree renderers alone: every test here regenerates the
assets into a temp directory and byte-compares against the committed files.
Provenance is test-enforced too: the table in assets/PROVENANCE.md must cover
exactly the shipped set, and the tree must never contain the capture-derived
asset names (upstream visible_alpha_solve outputs, gemini backgrounds).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import render_pill_silhouette
import render_vendor_silhouettes
from conftest import repo_files
from render_vendor_silhouettes import AssetSpec

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
ASSETS = SCRIPTS / "assets"
_REPO = Path(__file__).resolve().parents[1]
_PROVENANCE = ASSETS / "PROVENANCE.md"

#: File-name fragments that must never appear anywhere in the tree: the gemini
#: backgrounds and every ``scripts/visible_alpha_solve.py`` output are derived
#: from controlled captures of real watermark carriers, and this repo ships
#: renderer-synthetic silhouettes only. Names, not content: documentation may
#: mention WHY an asset was excluded; the file itself must not exist.
_FORBIDDEN_NAME_FRAGMENTS = (
    "gemini_bg",  # gemini sparkle-on-black capture crops
    "visible_alpha_solve",  # the capture-solve generator itself
    "doubao_alpha.png",  # capture solve (synthetic twin: doubao_outline_alpha.png)
    "jimeng_alpha.png",  # capture solve (synthetic twins: jimeng_label_alpha/pill)
    "samsung_alpha.png",  # Italian capture solve (synthetic twins: samsung_ko/en)
    "qwen_symbol_alpha",  # fixture-residual solve
    "receipt-gate",  # captured receipt npz
)


#: PROVENANCE.md column header -> canonical row key.
CANONICAL_COLUMNS = {
    "Asset": "asset",
    "Generator": "generator",
    "Upstream source": "upstream_source",
    "License": "license",
    "Engines (wave)": "engines",
    "Notes": "notes",
}


def _forbidden_asset_hits(paths: Iterable[Path]) -> list[Path]:
    """The paths whose file name carries a forbidden fragment."""
    return [p for p in paths if any(f in p.name for f in _FORBIDDEN_NAME_FRAGMENTS)]


def _parse_provenance_table(path: Path) -> dict[str, dict[str, str]]:
    """Parse the PROVENANCE.md markdown table into {asset: {column: cell}}."""
    header: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "Asset":
            header = cells
            assert header[:4] == ["Asset", "Generator", "Upstream source", "License"], (
                f"bad header: {header}"
            )
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        assert header, "table row before header"
        row = {
            CANONICAL_COLUMNS[h]: cell
            for h, cell in zip(header, cells, strict=True)
            if h in CANONICAL_COLUMNS
        }
        rows[row["asset"]] = row
    return rows


def _skip_unless_renderable(*, raqm: bool) -> None:
    """Skip a regen test where the committed bytes cannot be reproduced.

    The renderers draw with macOS system fonts; the vendor set also needs
    Pillow's Raqm layout (FriBiDi loaded at runtime). Elsewhere, Linux CI
    included, the bytes are not expected to match, so the test says why.
    """
    if not Path(render_vendor_silhouettes._FONT).is_file():
        pytest.skip("the renderers draw with macOS system fonts, which are absent here")
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        from PIL import features
    except ImportError as exc:
        pytest.skip(f"the pixel extra is not installed ({exc})")
    if raqm and not features.check_feature("raqm"):
        pytest.skip("Pillow cannot load FriBiDi here, so Raqm text layout is unavailable")


def test_pill_asset_regenerates_deterministically(tmp_path: Path) -> None:
    """The committed jimeng_pill.png reproduces byte-for-byte from the renderer."""
    _skip_unless_renderable(raqm=False)
    out = render_pill_silhouette.render_to_dir(tmp_path)
    assert out == tmp_path / render_pill_silhouette.ASSET_NAME
    assert out.read_bytes() == (ASSETS / render_pill_silhouette.ASSET_NAME).read_bytes()


@pytest.mark.parametrize(
    "spec",
    list(render_vendor_silhouettes.ASSET_SPECS),
    ids=lambda spec: spec.name,
)
def test_vendor_asset_regenerates_deterministically(spec: AssetSpec, tmp_path: Path) -> None:
    """Every committed vendor silhouette reproduces byte-for-byte in isolation."""
    _skip_unless_renderable(raqm=True)
    paths = render_vendor_silhouettes.render_to_dir(tmp_path, names=[spec.name])
    assert paths == [tmp_path / spec.name]
    assert (tmp_path / spec.name).read_bytes() == (ASSETS / spec.name).read_bytes()


# --- provenance table + forbidden capture-derived assets -------------------


def test_forbidden_capture_derived_assets_absent_from_tree(tmp_path: Path) -> None:
    """No capture-derived asset name exists anywhere in the repo tree.

    The detector is proved non-vacuous first: the same scan must flag a
    planted forbidden name in a scratch directory (the plant lives in
    tmp_path, never in the tree), then the repo itself must scan clean.
    """
    planted_dir = tmp_path / "carrier"
    planted_dir.mkdir()
    (planted_dir / "gemini_bg_96.png").write_bytes(b"PNG")
    hits = _forbidden_asset_hits(tmp_path.rglob("*"))
    assert [p.name for p in hits] == ["gemini_bg_96.png"]
    assert _forbidden_asset_hits(repo_files()) == []


#: The 17 registry marks tickets 07-09 ship (gemini defers: pending-provenance).
_TICKET_MARKS = (
    "baidu",
    "dola",
    "doubao",
    "generic_ai_label",
    "jimeng",
    "jimeng_pill",
    "kling",
    "liblib",
    "liblib_pill",
    "microsoft",
    "openart",
    "qwen",
    "runninghub",
    "samsung",
    "wan",
    "workbuddy",
    "yuanbao",
)


def test_asset_set_covers_the_engine_ticket_marks() -> None:
    """Every mark the wave-1/2/3 engine tickets register has a shipped asset.

    Coverage is read from the provenance table's Engines column (the same rows
    the completeness test pins to the shipped files), so a deleted asset or a
    deleted row drops a mark and fails this gate. Gemini is absent on purpose:
    its slot reports pending-provenance.
    """
    rows = _parse_provenance_table(_PROVENANCE)
    covered: set[str] = set()
    for row in rows.values():
        for entry in row["engines"].split("(")[0].split(","):
            if entry.strip():
                covered.add(entry.strip())
    missing = set(_TICKET_MARKS) - covered
    assert not missing, f"marks without a shipped silhouette asset: {sorted(missing)}"


def test_provenance_table_covers_every_shipped_asset() -> None:
    """assets/PROVENANCE.md documents exactly the shipped set, row per asset.

    Every row names its generator script, upstream source, and license; the
    row set must equal the shipped PNG set and cover every ASSET_SPECS entry
    plus the pill asset -- an asset without a row (or a row without an asset)
    fails the gate.
    """
    rows = _parse_provenance_table(_PROVENANCE)
    shipped = {p.name for p in ASSETS.glob("*.png")}
    spec_names = {spec.name for spec in render_vendor_silhouettes.ASSET_SPECS} | {
        render_pill_silhouette.ASSET_NAME
    }
    assert set(rows) == shipped, f"table/asset mismatch: {set(rows) ^ shipped}"
    assert spec_names <= set(rows), f"specs missing from table: {spec_names - set(rows)}"
    for name, row in rows.items():
        assert "render_" in row["generator"], f"{name}: generator must be an in-tree renderer"
        assert row["upstream_source"].strip(), f"{name}: upstream source missing"
        assert "Apache-2.0" in row["license"], f"{name}: license must name Apache-2.0"
        assert row["engines"].strip(), f"{name}: engine wave column empty"
