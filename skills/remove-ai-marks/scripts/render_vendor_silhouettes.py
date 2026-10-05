# Ported from wiltodelta/remove-ai-watermarks scripts/render_vendor_silhouettes.py, Apache-2.0; modified: flattened into this repo's stdlib script package (the upstream `mark_keys()` registry import is replaced by an explicit ASSET_SPECS table — the registry is a later ticket here), the upstream renderer set is narrowed to the silhouettes the wave-1/2/3 engine tickets load (unregistered upstream candidates are not ported), the companion generators render_overlay_variants.py (workbuddy/samsung_ko/doubao_outline/microsoft_text/jimeng_label/yuanbao_compact) and build_openart_alpha.py (openart; upstream's OpenCV-4.x-only refusal is dropped because this tree regenerates against the pinned pixel-extra OpenCV 5) are folded in, a labeled pixel-extra dependency guard replaces the bare imports, and render_to_dir()/--out were added so the committed assets can be regenerated to an arbitrary directory (the in-tree determinism test byte-compares against them).
"""Render synthetic detection silhouettes for the vendor text marks (data-safe).

Committed assets must be font/primitive-rendered and contain no source-image
pixels: every PNG this script writes is a synthetic glyph silhouette (255 =
mark) drawn from installed system fonts or primitive geometry, never from a
capture. Local evaluation inputs may be used only to learn glyph weight,
layout, and detector thresholds -- never to produce pixels.

The shipped set is exactly the silhouettes the engine tickets load (see
assets/PROVENANCE.md for the per-asset provenance table): the upstream
unregistered candidates (catlogo, hailuo, notebooklm, ...) stay unported -- no
engine here registers them. The system fonts render the glyphs but are never
shipped; STHeiti Medium approximates the semibold CJK sans these marks are set
in (the exact family is unpublished for every vendor; GB 45438-2025 only
requires a legible face).

Regenerate with:
    .venv/bin/python skills/remove-ai-marks/scripts/render_vendor_silhouettes.py

Pixel-extra modules (numpy/OpenCV/Pillow) are imported lazily: importing this
module never loads them, and a call without them raises the labeled
:class:`image_io.PixelDependencyUnavailable` error naming ``watermark-remover[pixel]``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SCRIPTS = Path(__file__).resolve().parent
ASSETS_DIR = _SCRIPTS / "assets"

#: Shared upstream attribution for every asset in the table (see ASSET_SPECS).
UPSTREAM_REPO = "wiltodelta/remove-ai-watermarks"
UPSTREAM_LICENSE = "Apache-2.0"
#: Snapshot the port was made from (upstream tree without git metadata; the
#: version field of its pyproject is the only revision marker it carries).
UPSTREAM_SNAPSHOT = "upstream snapshot remove-ai-watermarks-main v0.44.0 (2026-10)"

# System fonts (render-only, never shipped). Paths and face indices mirror the
# upstream generators.
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
_CJK_HIRAGINO: dict[str, Any] = {
    "font": "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "font_index": 2,
}


@dataclass(frozen=True)
class AssetSpec:
    """One committed silhouette: how it renders and why it ships.

    ``wave`` names the engine batch that loads the asset (1 = calibrated CJK,
    2 = wave-2 vendors, 3 = provisional); ``engines`` are the registry marks
    that read it. ``source`` is the upstream generator script the renderer was
    ported from; every asset shares UPSTREAM_LICENSE and renders synthetically.
    """

    name: str
    wave: str
    engines: tuple[str, ...]
    source: str
    text: str = ""
    opts: dict[str, Any] = field(default_factory=dict)
    note: str = ""


ASSET_SPECS: tuple[AssetSpec, ...] = (
    # --- wave 1: calibrated CJK set ---------------------------------------
    AssetSpec(
        name="generic_ai_label_alpha.png",
        wave="1",
        engines=("generic_ai_label",),
        source="scripts/render_vendor_silhouettes.py",
        text="AI生成",
        note="Brand-less TC260 fallback; bare glyph, no vendor wordmark.",
    ),
    AssetSpec(
        name="jimeng_label_alpha.png",
        wave="1",
        engines=("jimeng", "jimeng_pill"),
        source="scripts/render_overlay_variants.py",
        text="AI生成",
        note="Jimeng's text-form label; upstream ships it byte-identical to generic_ai_label's render.",
    ),
    AssetSpec(
        name="doubao_outline_alpha.png",
        wave="1",
        engines=("doubao",),
        source="scripts/render_overlay_variants.py",
        text="豆包AI生成",
        opts=dict(_CJK_HIRAGINO),
        note="Synthetic outline of the doubao strip; upstream's primary doubao_alpha.png is a visible_alpha_solve capture (forbidden here).",
    ),
    AssetSpec(
        name="qwen_alpha.png",
        wave="1",
        engines=("qwen",),
        source="scripts/render_vendor_silhouettes.py",
        text="千问AI生成",
        note="Upstream's qwen_symbol_alpha.png is a capture solve (forbidden here) and is not ported.",
    ),
    AssetSpec(
        name="kling_alpha.png",
        wave="1",
        engines=("kling",),
        source="scripts/render_vendor_silhouettes.py",
        text="可灵AI 3.0",
        note="Thin light-gray one-line stamp, bottom-right; the leading spiral logo is not rendered.",
    ),
    AssetSpec(
        name="kling_latin_alpha.png",
        wave="1",
        engines=("kling",),
        source="scripts/render_vendor_silhouettes.py",
        text="KlingAI 3.0",
        opts={"font": "/System/Library/Fonts/HelveticaNeue.ttc", "font_index": 1},
        note="Variant of the registered kling row, not a second mark key.",
    ),
    AssetSpec(
        name="baidu_alpha.png",
        wave="1",
        engines=("baidu",),
        source="scripts/render_vendor_silhouettes.py",
        text="百度",
    ),
    AssetSpec(
        name="yuanbao_alpha.png",
        wave="1",
        engines=("yuanbao",),
        source="scripts/render_vendor_silhouettes.py",
        text="元宝\nAI生成",
        opts={"gap_frac": 0.05, "dilate": 2, "shear": -0.60, **_CJK_HIRAGINO},
        note="TWO-LINE stamp (元宝 over AI生成), tightly stacked and italic-slanted.",
    ),
    AssetSpec(
        name="yuanbao_compact_alpha.png",
        wave="1",
        engines=("yuanbao",),
        source="scripts/render_overlay_variants.py",
        text="元宝\nAI生成",
        opts={"top_scale": 0.65, "shear": -0.15},
        note="Compact centered two-line variant.",
    ),
    AssetSpec(
        name="yuanbao_compact_gray_alpha.png",
        wave="1",
        engines=("yuanbao",),
        source="scripts/render_overlay_variants.py",
        text="元宝\nAI生成",
        opts={"top_scale": 0.65, "shear": -0.15},
        note="Gray-frontend twin of the compact variant (identical pixels, distinct asset).",
    ),
    # --- wave 2 ------------------------------------------------------------
    AssetSpec(
        name="samsung_ko_alpha.png",
        wave="2",
        engines=("samsung",),
        source="scripts/render_overlay_variants.py",
        text="AI로 생성한 콘텐츠",
        opts={"font": "/System/Library/Fonts/AppleSDGothicNeo.ttc"},
        note="Korean Galaxy AI label; upstream's registered Italian samsung_alpha.png is a capture solve (forbidden here).",
    ),
    AssetSpec(
        name="samsung_en_alpha.png",
        wave="2",
        engines=("samsung",),
        source="scripts/render_vendor_silhouettes.py",
        note="English Galaxy AI label with the leading four-point sparkle; upstream registered it only as a candidate.",
    ),
    AssetSpec(
        name="liblib_alpha.png",
        wave="2",
        engines=("liblib",),
        source="scripts/render_vendor_silhouettes.py",
        text="LibLibAI",
        opts={"font": "/System/Library/Fonts/Supplemental/Arial.ttf"},
        note="Bottom-center wordmark; the liblib_pill mark reuses jimeng_pill.png.",
    ),
    AssetSpec(
        name="microsoft_alpha.png",
        wave="2",
        engines=("microsoft",),
        source="scripts/render_vendor_silhouettes.py",
        note="Top-right white pill silhouette with dark-text holes and the sparkle cutout.",
    ),
    AssetSpec(
        name="microsoft_text_alpha.png",
        wave="2",
        engines=("microsoft",),
        source="scripts/render_overlay_variants.py",
        text="Made with AI",
        opts={"font": "/System/Library/Fonts/Supplemental/Arial.ttf"},
    ),
    AssetSpec(
        name="runninghub_alpha.png",
        wave="2",
        engines=("runninghub",),
        source="scripts/render_vendor_silhouettes.py",
        text="RunningHub AI生成",
        note="Top-left text mark; faint mid-gray class detected on the raw-grayscale frontend.",
    ),
    AssetSpec(
        name="dola_alpha.png",
        wave="2",
        engines=("dola",),
        source="scripts/render_vendor_silhouettes.py",
        text="Dola AI",
        opts={"font": "/System/Library/Fonts/Supplemental/Arial Bold.ttf"},
        note="Image-form wordmark; upstream's video-form Hershey templates are not ported (video engines out of scope).",
    ),
    AssetSpec(
        name="workbuddy_alpha.png",
        wave="2",
        engines=("workbuddy",),
        source="scripts/render_overlay_variants.py",
        note="Centered two-line block: AI生成 over a compact WORKBUDDY>_ run.",
    ),
    # --- wave 3 (provisional) ---------------------------------------------
    AssetSpec(
        name="wan_alpha.png",
        wave="3",
        engines=("wan",),
        source="scripts/render_vendor_silhouettes.py",
        note="Hexagonal logo approximation plus 'Wan' wordmark, bottom-right; provisional mark.",
    ),
    AssetSpec(
        name="openart_alpha.png",
        wave="3",
        engines=("openart",),
        source="scripts/build_openart_alpha.py",
        note="Procedural reconstruction of the OpenArt bowtie icon + Hershey wordmark; provisional, semi-transparent-overlay class (peak 173, not binary).",
    ),
)


def _import_pixel(name: str) -> Any:
    """Import a pixel-extra module lazily, or raise the labeled guard error."""
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise image_io.PixelDependencyUnavailable(
            f"render_vendor_silhouettes requires the pixel extra; install "
            f"watermark-remover[pixel] (numpy + opencv-python-headless + Pillow). Reason: {exc}"
        ) from exc


def _fit_font(
    font_path: str,
    reaches_target: Callable[[Any], bool],
    *,
    index: int = 0,
) -> Any:
    """Return the smallest 8-200 px font that reaches a render target."""
    ImageFont = _import_pixel("PIL.ImageFont")
    low, high = 8, 200
    while low < high:
        size = (low + high) // 2
        font = ImageFont.truetype(font_path, size, index=index)
        if reaches_target(font):
            high = size
        else:
            low = size + 1
    return ImageFont.truetype(font_path, low, index=index)


def render(text: str, width: int = 335, opts: dict[str, Any] | None = None) -> NDArray[Any]:
    """Binary glyph silhouette (255 = glyph), sized to the doubao asset's convention.

    Matching doubao's 335px asset width keeps the ``alpha_*_frac`` numbers
    transferable, since these marks are the same house style and scale. A "\n"
    in ``text`` renders a multi-line block: lines drawn left-aligned at one
    shared font size with a tight gap, then optional stroke dilation and an
    italic shear (see the per-asset opts in ASSET_SPECS).
    """
    Image = _import_pixel("PIL.Image")
    ImageDraw = _import_pixel("PIL.ImageDraw")
    np = _import_pixel("numpy")
    opts = opts or {}
    gap_frac = float(opts.get("gap_frac", 0.15))
    dilate = int(opts.get("dilate", 0))
    shear_k = float(opts.get("shear", 0.0))
    font_path = str(opts.get("font", _FONT))
    font_index = int(opts.get("font_index", 0))
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    lines = text.split("\n")
    font = _fit_font(
        font_path,
        lambda f: max(d0.textbbox((0, 0), ln, font=f)[2] for ln in lines) >= width * 0.98,
        index=font_index,
    )
    boxes = [d0.textbbox((0, 0), ln, font=font) for ln in lines]
    line_h = max(bb[3] - bb[1] for bb in boxes)
    gap = max(1, int(line_h * gap_frac))
    w = max(bb[2] - bb[0] for bb in boxes)
    h = line_h * len(lines) + gap * (len(lines) - 1)
    pad = max(2, int(line_h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    draw = ImageDraw.Draw(im)
    y = pad
    for ln, bb in zip(lines, boxes, strict=True):
        draw.text((pad - bb[0], y - bb[1]), ln, font=font, fill=255)
        y += line_h + gap
    sil = np.asarray(im)
    if dilate or shear_k:
        cv2 = _import_pixel("cv2")
        if dilate:
            sil = cv2.dilate(sil, np.ones((dilate, dilate), np.uint8))
        if shear_k:
            hh, ww = sil.shape
            extra = int(abs(shear_k) * hh)
            offset = extra if shear_k < 0 else 0
            sil = cv2.warpAffine(
                sil,
                np.float32([[1, shear_k, offset], [0, 1, 0]]),
                (ww + extra, hh),
            )
            ys, xs = np.where(sil > 0)
            if xs.size:
                sil = sil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return sil


def centered_lines(
    top: str,
    bottom: str,
    *,
    top_scale: float,
    bottom_height: float = 1.0,
    shear: float = 0.0,
) -> NDArray[Any]:
    """Stack two rendered lines centered (the AI-disclosure block layout).

    From upstream render_overlay_variants.py: the top line may be rescaled
    against the bottom one, and the whole block may take an italic shear.
    """
    cv2 = _import_pixel("cv2")
    np = _import_pixel("numpy")
    upper, lower = render(top, opts=dict(_CJK_HIRAGINO)), render(bottom, opts=dict(_CJK_HIRAGINO))
    if bottom_height != 1.0:
        lower = cv2.resize(lower, (upper.shape[1], round(upper.shape[0] * bottom_height)))
    else:
        upper = cv2.resize(upper, (round(lower.shape[1] * top_scale), round(lower.shape[0] * 0.95)))
    lines = [upper, lower]
    result = np.zeros(
        (sum(line.shape[0] for line in lines) + 2, max(line.shape[1] for line in lines)), np.uint8
    )
    y = 0
    for line in lines:
        x = (result.shape[1] - line.shape[1]) // 2
        result[y : y + line.shape[0], x : x + line.shape[1]] = line
        y += line.shape[0] + 2
    if shear:
        h, w = result.shape
        extra = round(abs(shear) * h)
        result = cv2.warpAffine(result, np.float32([[1, shear, extra], [0, 1, 0]]), (w + extra, h))
    return result


def _star_pts(cx: float, cy: float, r: float, waist: float) -> list[tuple[float, float]]:
    return [
        (cx, cy - r),
        (cx + r * waist, cy - r * waist),
        (cx + r, cy),
        (cx + r * waist, cy + r * waist),
        (cx, cy + r),
        (cx - r * waist, cy + r * waist),
        (cx - r, cy),
        (cx - r * waist, cy - r * waist),
    ]


def _sparkle(draw: Any, cx: float, cy: float, r: float) -> None:
    """Draw the four-point cutout used by synthetic candidate silhouettes."""
    draw.polygon(_star_pts(cx, cy, r, 0.22), fill=0)


def draw_workbuddy() -> NDArray[Any]:
    """Centered two-line WORKBUDDY block (AI生成 over a compact WORKBUDDY>_ run)."""
    return centered_lines("AI生成", "WORKBUDDY>_", top_scale=1.0, bottom_height=0.45)


def draw_yuanbao_compact() -> NDArray[Any]:
    """Compact centered 元宝/AI生成 block with a slight italic shear."""
    return centered_lines("元宝", "AI生成", top_scale=0.65, shear=-0.15)


def draw_msbadge(width: int = 335) -> NDArray[Any]:
    """Synthetic silhouette for the measured Microsoft top-right white pill.

    The top-hat front-end sees the bright pill with dark-text holes, so the
    template carries the same holes -- that is what discriminates this pill
    from any other white rounded element in the top-right corner. The text and
    four-point cutout approximate the measured internal shape; they do not
    assert one universal Microsoft icon or wording. Geometry measured on 17
    visually confirmed carriers on 2026-08-27 (upstream): pill 0.152W x 0.040W,
    margins ~0.010W right / ~0.007W top, glyph height ~0.39 of pill height.
    """
    Image = _import_pixel("PIL.Image")
    ImageDraw = _import_pixel("PIL.ImageDraw")
    np = _import_pixel("numpy")
    h = round(width / 3.78)
    im = Image.new("L", (width, h), 0)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, width - 1, h - 1], radius=h // 2, fill=255)
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    text = "Made with AI"
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    font = _fit_font(font_path, lambda f: d0.textbbox((0, 0), text, font=f)[3] >= h * 0.39)
    bb = d0.textbbox((0, 0), text, font=font)
    th = bb[3] - bb[1]
    r = h * 0.20  # sparkle radius, ~half the text height
    pad_l = h * 0.22
    cx = pad_l + r
    cy = h / 2 - 1
    tx = int(pad_l + 2 * r + h * 0.22)
    d.text((tx - bb[0], (h - th) // 2 - bb[1]), text, font=font, fill=0)
    _sparkle(d, cx, cy, r)
    return np.asarray(im)


def draw_samsung_en(width: int = 335) -> NDArray[Any]:
    """Samsung Galaxy AI English label: "AI-generated content" with the leading
    4-point sparkle, light-gray glyphs (same class as the registered Italian asset)."""
    Image = _import_pixel("PIL.Image")
    ImageDraw = _import_pixel("PIL.ImageDraw")
    np = _import_pixel("numpy")
    text = "AI-generated content"
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    font = _fit_font(font_path, lambda f: d0.textbbox((0, 0), text, font=f)[2] >= width * 0.80)
    bb = d0.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    r = th * 0.55
    gap = th * 0.45
    im = Image.new("L", (int(tw + gap + 2 * r + 8), th + 8), 0)
    d = ImageDraw.Draw(im)
    # sparkle as bright glyph (this silhouette is light-glyph class, not a pill)
    d.polygon(_star_pts(4 + r, 4 + th / 2, r, 0.22), fill=255)
    d.text((4 + 2 * r + gap - bb[0], 4 - bb[1]), text, font=font, fill=255)
    arr = np.asarray(im)
    ys, xs = np.where(arr > 0)
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def draw_wan(width: int = 335) -> NDArray[Any]:
    """Alibaba Wan logo plus "Wan" wordmark, bottom-right, light-glyph class.

    Geometry measured on the one real Wan 2.7 Pro export (2048 px square,
    2026-09-23, upstream): logo 102 x 103 px, a 19 px gap, then "Wan" 147 px
    wide with a 51 px cap height centered on the logo; 268 x 103 overall. The
    logo is approximated as a hexagon with a central triangular hole and three
    pinwheel cuts; the wordmark is Avenir Next Demi Bold, the nearest installed
    geometric sans. Hole size, cut width and weight were chosen by the best
    match to the real mark over a small grid.
    """
    import math

    Image = _import_pixel("PIL.Image")
    ImageDraw = _import_pixel("PIL.ImageDraw")
    np = _import_pixel("numpy")
    k = width / 268
    h = round(103 * k)
    im = Image.new("L", (width, h), 0)
    d = ImageDraw.Draw(im)
    logo = round(102 * k)
    cx, cy, radius = logo / 2, h / 2, logo / 2
    d.polygon(
        [
            (
                cx + radius * math.cos(math.radians(60 * i)),
                cy + radius * 0.98 * math.sin(math.radians(60 * i)),
            )
            for i in range(6)
        ],
        fill=255,
    )
    inner = [
        (
            cx + radius * 0.42 * math.cos(math.radians(90 + 120 * i)),
            cy + radius * 0.42 * math.sin(math.radians(90 + 120 * i)),
        )
        for i in range(3)
    ]
    d.polygon(inner, fill=0)
    cut = max(2, round(logo * 0.05))
    for i, (px, py) in enumerate(inner):
        angle = math.radians(90 + 120 * i + 90)
        d.line(
            [(px, py), (px + radius * 1.2 * math.cos(angle), py + radius * 1.2 * math.sin(angle))],
            fill=0,
            width=cut,
        )
    text = "Wan"
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    font = _fit_font(
        "/System/Library/Fonts/Avenir Next.ttc",
        lambda f: (
            probe.textbbox((0, 0), text, font=f)[3] - probe.textbbox((0, 0), text, font=f)[1]
            >= round(51 * k)
        ),
        index=2,
    )
    bb = probe.textbbox((0, 0), text, font=font)
    d.text(
        (round((102 + 19) * k) - bb[0], round(h / 2 - (bb[3] - bb[1]) / 2) - bb[1]),
        text,
        font=font,
        fill=255,
    )
    arr = np.asarray(im)
    ys, xs = np.where(arr > 0)
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


#: Peak alpha of the semi-transparent-overlay class upstream solved for doubao
#: (~68% opacity); the openart reconstruction matches it.
_OPENART_ALPHA_PEAK = 173


def draw_openart(width: int = 360, height: int = 90) -> NDArray[Any]:
    """Procedural reconstruction of the OpenArt mark (bowtie icon + wordmark).

    Ported from upstream build_openart_alpha.py: OpenArt's committed asset is a
    reconstruction from its public logo, not a recovered watermark alpha map
    (creating the account needed to capture one was outside upstream's
    allowances). Upstream pins OpenCV 4.x because Hershey glyphs differ in
    OpenCV 5; this tree regenerates against the pixel extra's OpenCV 5 and
    ships the OpenCV-5 rendering, so the asset and the detector see the same
    glyphs. Drawn as the semi-transparent-overlay class (peak 173), matching
    the general rendering style of the other bundled text-mark assets.
    """
    cv2 = _import_pixel("cv2")
    np = _import_pixel("numpy")

    def draw_infinity_icon(canvas: NDArray[Any], cx: int, cy: int, size: int) -> None:
        """Draw the bowtie/infinity glyph left of the wordmark (two mirrored lobes)."""
        half = size // 2
        # Two overlapping filled triangles meeting at the center, echoing the
        # angular "infinity" mark on openart.ai's own site header.
        left_lobe = np.array(
            [[cx - size, cy - half], [cx - size, cy + half], [cx, cy]], dtype=np.int32
        )
        right_lobe = np.array(
            [[cx + size, cy - half], [cx + size, cy + half], [cx, cy]], dtype=np.int32
        )
        cv2.fillConvexPoly(canvas, left_lobe, (_OPENART_ALPHA_PEAK,), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(canvas, right_lobe, (_OPENART_ALPHA_PEAK,), lineType=cv2.LINE_AA)

    canvas = np.zeros((height, width), np.uint8)
    icon_cx, icon_cy, icon_size = 34, height // 2, 22
    draw_infinity_icon(canvas, icon_cx, icon_cy, icon_size)
    text_x = icon_cx + icon_size + 18
    cv2.putText(
        canvas,
        "OpenArt",
        (text_x, height // 2 + 16),
        cv2.FONT_HERSHEY_DUPLEX,
        1.3,
        (_OPENART_ALPHA_PEAK,),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _resolve_draw(spec: AssetSpec) -> Callable[[], NDArray[Any]]:
    """The sentinel renderer for a draw-class spec (text-less), by asset name."""
    drawn: dict[str, Callable[[], NDArray[Any]]] = {
        "samsung_en_alpha.png": draw_samsung_en,
        "microsoft_alpha.png": draw_msbadge,
        "workbuddy_alpha.png": draw_workbuddy,
        "yuanbao_compact_alpha.png": draw_yuanbao_compact,
        "yuanbao_compact_gray_alpha.png": draw_yuanbao_compact,
        "wan_alpha.png": draw_wan,
        "openart_alpha.png": draw_openart,
    }
    fn = drawn.get(spec.name)
    if fn is None:
        raise ValueError(f"asset spec {spec.name} has no text or draw renderer")
    return fn


def render_spec(spec: AssetSpec) -> NDArray[Any]:
    """Render one asset's silhouette from its spec."""
    if spec.text:
        return render(spec.text, opts=dict(spec.opts))
    return _resolve_draw(spec)()


def render_to_dir(
    out_dir: Path | str | None = None,
    names: list[str] | None = None,
) -> list[Path]:
    """Render assets into ``out_dir`` (default: the in-tree assets dir).

    ``names`` selects a subset of ASSET_SPECS by file name (default: all, in
    table order). Returns the written paths in the same order. PNG bytes are
    deterministic given the same fonts/OpenCV/Pillow build -- the property the
    in-tree determinism test pins.
    """
    selected = list(ASSET_SPECS)
    if names is not None:
        by_name = {spec.name: spec for spec in ASSET_SPECS}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise ValueError(f"unknown asset name(s) {unknown}; known: {sorted(by_name)}")
        selected = [by_name[n] for n in names]
    out = Path(out_dir) if out_dir is not None else ASSETS_DIR
    Image = _import_pixel("PIL.Image")
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in selected:
        path = out / spec.name
        Image.fromarray(render_spec(spec)).save(path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default: the in-tree assets dir)"
    )
    parser.add_argument(
        "--list", action="store_true", help="list the asset table (name, wave, engines) and exit"
    )
    args = parser.parse_args(argv)
    if args.list:
        for spec in ASSET_SPECS:
            print(f"{spec.name}\twave {spec.wave}\t{','.join(spec.engines)}\t{spec.source}")
        return 0
    try:
        paths = render_to_dir(args.out)
    except (image_io.PixelDependencyUnavailable, OSError) as e:
        print(
            f"Cannot render ({e}); install the pixel extra and the CJK fonts or edit the font table.",
            file=sys.stderr,
        )
        return 1
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
