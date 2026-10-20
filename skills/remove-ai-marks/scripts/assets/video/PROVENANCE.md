# Video template provenance

These five PNGs are **renderer-synthetic**: each is a frozen
``cv2.putText`` Hershey-glyph render (white glyph on black, single channel),
carrying zero pixels from any captured frame or exported video.

They are frozen rather than rendered at import time on purpose. The video
detectors were calibrated against OpenCV 4.x Hershey glyph shapes
(byte-identical 4.8 through 4.14); OpenCV 5 reworked text rendering, so the
same ``cv2.putText`` call in this tree draws smaller, heavier glyphs and a
calibration would silently drift. The recipe that produced these files lives
upstream at ``scripts/render_video_hershey_templates.py`` (Apache-2.0, ported
portions), which draws every template at scale 2.2, thickness 3, anti-aliased,
baseline origin ``(2, 72)`` on a 100 px canvas, then crops to the nonzero
footprint.

| Asset | Glyph | Face | Upstream consumer |
| --- | --- | --- | --- |
| video_dola_hershey_duplex.png | ``Dola AI`` | ``FONT_HERSHEY_DUPLEX`` | dola video detector |
| video_kling_ai_hershey_simplex.png | ``KLING AI`` | ``FONT_HERSHEY_SIMPLEX`` | kling video detector |
| video_kling_ai_hershey_duplex.png | ``KLING AI`` | ``FONT_HERSHEY_DUPLEX`` | kling video detector |
| video_klingai_hershey_simplex.png | ``KlingAI`` | ``FONT_HERSHEY_SIMPLEX`` | kling video detector |
| video_klingai_hershey_duplex.png | ``KlingAI`` | ``FONT_HERSHEY_DUPLEX`` | kling video detector |

The Doubao video mark does **not** add an asset here: it reuses this tree's
renderer-synthetic ``assets/doubao_outline_alpha.png`` (upstream's primary
Doubao capture-solve asset is excluded from this repo by
``tests/test_silhouette_assets.py``).

Sora, veo, seedance, hailuo and vidu templates are drawn from primitive
geometry and Pillow's bundled font inside ``video_visible.py``; they ship as
code, not as files.
