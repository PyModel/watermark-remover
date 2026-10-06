# The video extra requires Python 3.11 while the core stays on 3.10

The `video` extra depends on PyAV (`av>=18,<19`), and av 18.x requires Python
>=3.11. The core floor stays >=3.10. The extra declares its own floor; the core
never imports av, numpy, or cv2 at module load.

Dropping av was the alternative: ffprobe's `bits_per_raw_sample` already covers
the one thing av is used for (component bit depth). It was rejected because the
component-depth probe is more accurate than the ffprobe field for aliased pixel
formats, and an extra-scoped floor is ordinary packaging. The decision worth
recording is the one a later reader will want to "fix": the core does not move
to 3.11 just because one extra needs it.
