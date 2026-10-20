# watermark-remover brand kit

Open `index.html` in a browser to see every asset. All files are plain SVG.

## The mark

Four rows of signal with a scrub line across them. The part the line has crossed is residue; the part ahead of it is still lit. The last row is short, so the mark never reads as a finished table.

| File | Use |
| --- | --- |
| `mark.svg` | Primary mark on its navy tile: app icon, avatars, the web UI header |
| `mark-light.svg` | Mark without a tile, for light backgrounds |
| `mark-dark.svg` | Mark without a tile, for navy or other dark backgrounds |
| `mark-mono.svg` | One color via `currentColor`, for single-ink print or monochrome UI |
| `favicon.svg` | Three-row reduction that stays legible at 16px |
| `wordmark.svg`, `wordmark-dark.svg` | Tile mark plus name, for light and dark backgrounds |
| `wordmark-flat.svg` | Flat mark plus name, for light backgrounds |
| `social-card.svg` | 1200×630 link preview |
| `banner.svg` | 1200×300 README banner (same file as `../banner.svg`) |
| `brand-sheet.svg` | Marks, palette and type on one page |
| `icons/` | The 25 interface icons from the web UI |

Keep clear space of a quarter of the mark's width on every side. Don't recolor the rows, rotate the mark, or put the light mark on a busy photo.

The wordmark text is live SVG text in the system UI font with a fixed `textLength`, so its width is stable on any platform. Convert it to outlines in a vector editor before print work.

## Color

| Name | Hex | Role |
| --- | --- | --- |
| Navy | `#0a2026` | Brand tile, primary buttons, active state |
| Ink | `#0d151b` | Headings and body text |
| Muted | `#5f6a74` | Secondary text; 4.5:1 on the shell |
| Shell | `#eef2f4` | Page background |
| Mark | `#a3176f` | Hidden marks on light surfaces, and nothing else |
| Mark lit | `#f05aa8` | Live marks on navy |
| Residue | `#234049` | Cleaned residue on navy |
| Blue | `#2589ff` | Progress bars and focus rings, not text |
| Green | `#2ab46c` | Removed, ready, verifiable |

Magenta means "a mark is here". Don't use it for decoration, links or buttons, or it stops meaning that.

## Type

System UI sans (SF Pro, Segoe UI, Roboto) at weights 520 to 700 with tight tracking for headings. Monospace only for file content and code points.

## Icons

24px grid, 1.7 stroke, round caps and joins, `stroke="currentColor"`. They come from `skills/remove-ai-marks/web/src/icons.tsx`; regenerate this folder from there when an icon changes.
