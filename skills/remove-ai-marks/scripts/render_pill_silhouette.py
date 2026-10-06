# Ported from wiltodelta/remove-ai-watermarks scripts/render_pill_silhouette.py, Apache-2.0; modified: flattened into this repo's stdlib script package (no package-internal imports), the asset path moves to skills/remove-ai-marks/scripts/assets/, a labeled pixel-extra dependency guard replaces the bare imports, and a render_to_dir() seam plus an --out flag were added so the committed asset can be regenerated to an arbitrary directory (the in-tree determinism test byte-compares against it).
"""Render the SYNTHETIC 'AI生成' pill silhouette asset (data-safe, font-rendered).

The Jimeng-basic TC260 visible label is a rounded pill with 'AI生成' in the TOP-LEFT
corner (upstream issue #54). Unlike the reverse-alpha marks, this is a CAPTURE-LESS
mark: the committed asset is a font-rendered binary SILHOUETTE (mark shape only, zero
photo content), used ONLY to (a) detect the pill by edge-NCC in the top-left corner
and (b) build the inpaint mask. It is NOT an alpha map -- removal quality comes from
the inpaint backend (MI-GAN/cv2), so the silhouette need not be pixel-accurate, and
the synthetic render keeps the asset deterministic and reproducible.

The calibrated edge-NCC threshold is 0.22 in the top-left ROI. Re-run to regenerate
the asset:  .venv/bin/python skills/remove-ai-marks/scripts/render_pill_silhouette.py

Requires a CJK font (macOS STHeiti by default). The font renders the glyphs but is
never shipped; the committed PNG is pure synthetic output. See assets/PROVENANCE.md
for the per-asset provenance table.

The pixel extra (numpy + Pillow) is imported lazily: importing this module never
loads it, and a call without it raises the labeled
:class:`image_io.PixelDependencyUnavailable` error naming ``watermark-remover[pixel]``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SCRIPTS = Path(__file__).resolve().parent
#: The committed asset (written by render_to_dir's default output directory).
ASSET_NAME = "jimeng_pill.png"
ASSETS_DIR = _SCRIPTS / "assets"
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"


def _import_pixel(name: str) -> Any:
    """Import a pixel-extra module lazily, or raise the labeled guard error."""
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise image_io.PixelDependencyUnavailable(
            f"render_pill_silhouette requires the pixel extra; install "
            f"watermark-remover[pixel] (numpy + opencv-python-headless + Pillow). Reason: {exc}"
        ) from exc


def render_silhouette(w: int = 320) -> NDArray[Any]:
    """Rounded-pill outline + 'AI生成' text as a binary silhouette (255 = mark)."""
    Image = _import_pixel("PIL.Image")
    ImageDraw = _import_pixel("PIL.ImageDraw")
    ImageFont = _import_pixel("PIL.ImageFont")
    np = _import_pixel("numpy")

    h = int(w * 0.5)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    pad = int(w * 0.03)
    r = (h - 2 * pad) // 3
    d.rounded_rectangle([pad, pad, w - pad, h - pad], radius=r, outline=255, width=max(2, w // 90))
    fsz = int(h * 0.42)
    font = ImageFont.truetype(_FONT, fsz)
    txt = "AI生成"
    tb = d.textbbox((0, 0), txt, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.text(((w - tw) // 2 - tb[0], (h - th) // 2 - tb[1]), txt, font=font, fill=255)
    return np.asarray(im)


def _write(sil: NDArray[Any], out_dir: Path) -> Path:
    """Write a rendered silhouette as ``ASSET_NAME`` into ``out_dir`` (created)."""
    Image = _import_pixel("PIL.Image")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ASSET_NAME
    Image.fromarray(sil).save(path)
    return path


def render_to_dir(out_dir: Path | str | None = None) -> Path:
    """Render the pill silhouette and write it into ``out_dir`` (default: in-tree assets).

    Returns the written path. The PNG bytes are deterministic given the same font
    file and Pillow build, which is the property the in-tree determinism test pins.
    """
    return _write(render_silhouette(), Path(out_dir) if out_dir is not None else ASSETS_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default: the in-tree assets dir)"
    )
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else ASSETS_DIR
    try:
        sil = render_silhouette()
        path = _write(sil, out)
    except (image_io.PixelDependencyUnavailable, OSError) as e:
        print(
            f"Cannot render ({e}); install the pixel extra and a CJK font or edit _FONT.",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {path}  ({sil.shape[1]}x{sil.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
