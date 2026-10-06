"""Tests for the Unicode-safe cv2 image IO port (ticket 02).

Vectors adapted from upstream research/remove-ai-watermarks-main/tests/test_image_io.py
(wiltodelta/remove-ai-watermarks, Apache-2.0): the Unicode round-trip, failure
contract, alpha separation, and display-tag carry behaviours, re-pointed at this
repo's stdlib script package and its dependency set (no piexif here — orientation
bytes are verified through Pillow's independent EXIF reader; no isobmff module —
HEIF container-transform overrides are not ported).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import cv2
import image_io
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"

# Non-ASCII filenames that break cv2.imread/imwrite on Windows (upstream issue #17).
# Emoji + spaces join the ticket's list alongside CJK/Cyrillic/accented Latin.
_UNICODE_NAMES = [
    "jimeng-2026-05-27-一面白色的墙.png",  # Chinese
    "тест-изображение.png",  # Cyrillic
    "café-señor.png",  # accented Latin
    "quiz 🎨 copy .png",  # emoji + spaces
]


def _make_bgr() -> np.ndarray:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = (10, 120, 240)  # a BGR block so the round-trip is checkable
    return img


class TestUnicodeRoundTrip:
    def test_write_then_read_preserves_pixels(self, tmp_path: Path) -> None:
        for name in _UNICODE_NAMES:
            path = tmp_path / name
            src = _make_bgr()
            assert image_io.imwrite(path, src) is True
            assert path.exists()
            out = image_io.imread(path)
            assert out is not None, name
            # PNG is lossless: pixels must match exactly.
            assert np.array_equal(out, src), name

    def test_reads_file_written_by_raw_cv2(self, tmp_path: Path) -> None:
        # An ASCII file written by plain cv2 must read back identically through
        # the wrapper (decode path is byte-compatible with cv2.imread).
        path = tmp_path / "ascii.png"
        src = _make_bgr()
        cv2.imwrite(str(path), src)
        out = image_io.imread(path)
        assert out is not None
        assert np.array_equal(out, src)


class TestFailureSemantics:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert image_io.imread(tmp_path / "does-not-exist-不存在.png") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        assert image_io.imread(path) is None

    def test_undecodable_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.png"
        path.write_bytes(b"not an image")
        assert image_io.imread(path) is None

    def test_truncated_png_returns_none(self, tmp_path: Path) -> None:
        # A valid signature with the body cut off: stable None, never a cv2 error.
        path = tmp_path / "truncated.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 7)
        assert image_io.imread(path) is None

    def test_imwrite_to_missing_directory_returns_false(self, tmp_path: Path) -> None:
        # An unwritable path must return False (cv2.imwrite contract), not raise.
        path = tmp_path / "no-such-dir" / "out.png"
        assert image_io.imwrite(path, _make_bgr()) is False


class TestQualityPreservingWrite:
    """The removal only touches the mark footprint, so the container re-encode must
    not degrade the untouched pixels: JPEG/WebP are written at max quality."""

    def test_jpeg_written_near_lossless(self, tmp_path: Path) -> None:
        # a smooth gradient -> JPEG at quality 100 / 4:4:4 is near-lossless
        g = np.tile(np.linspace(30, 210, 64, dtype=np.uint8), (48, 1))
        img = np.dstack([g, g, g])
        p = tmp_path / "x.jpg"
        assert image_io.imwrite(p, img) is True
        back = image_io.imread(p)
        assert back is not None
        assert float(np.abs(img.astype(int) - back.astype(int)).mean()) < 1.0

    def test_webp_written_lossless(self, tmp_path: Path) -> None:
        # Regression (upstream): cv2 WebP quality 1-100 is LOSSY; lossless needs
        # > 100. A full-frame round-trip of random data must be bit-identical.
        img = np.random.default_rng(0).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        p = tmp_path / "x.webp"
        assert image_io.imwrite(p, img) is True
        back = image_io.imread(p)
        assert back is not None
        assert np.array_equal(back, img), "WebP re-encode was lossy"

    def test_unencodable_ext_returns_false_not_raises(self, tmp_path: Path) -> None:
        # a bogus extension cv2 can't encode returns False (never raises the cv2.error).
        assert image_io.imwrite(tmp_path / "x.zzz", _make_bgr()) is False


def test_jpeg_aliases_are_supported_and_written_as_jpeg(tmp_path: Path) -> None:
    # Upstream checks its format registry here; this port asserts the observable
    # contract instead: .jpe/.jfif encode through the JPEG codec (SOI marker).
    for suffix in (".jpe", ".jfif"):
        path = tmp_path / f"marked{suffix}"
        source = np.full((32, 48, 3), (10, 120, 240), dtype=np.uint8)
        assert image_io.imwrite(path, source), suffix
        assert path.read_bytes().startswith(b"\xff\xd8"), suffix
        decoded = image_io.imread(path)
        assert decoded is not None
        assert decoded.shape == source.shape, suffix


class TestAlphaPreservation:
    """RGBA sources survive the read/write pair with the alpha plane intact
    (upstream issue #30: the fill reconstructs real pixels, so preserving the
    input alpha keeps genuinely transparent backgrounds without new holes)."""

    def test_alpha_round_trip_with_unchanged_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "豆包-alpha.png"
        bgra = np.zeros((8, 8, 4), dtype=np.uint8)
        bgra[..., 3] = 128
        assert image_io.imwrite(path, bgra) is True
        out = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        assert out is not None
        assert out.shape[2] == 4
        assert np.array_equal(out, bgra)

    def test_read_bgr_and_alpha_splits_an_rgba_source(self, tmp_path: Path) -> None:
        path = tmp_path / "semi-透明.png"
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., :3] = (30, 140, 200)
        rgba[..., 3] = 200
        assert image_io.imwrite(path, rgba) is True
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        assert bgr is not None and alpha is not None
        assert bgr.shape == (8, 8, 3)
        assert alpha.shape == (8, 8)
        assert np.array_equal(alpha, rgba[..., 3])

    def test_read_bgr_and_alpha_of_an_opaque_source_has_no_alpha(self, tmp_path: Path) -> None:
        path = tmp_path / "opaque.png"
        assert image_io.imwrite(path, _make_bgr()) is True
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        assert alpha is None
        assert bgr is not None
        assert np.array_equal(bgr, _make_bgr())

    def test_read_bgr_and_alpha_promotes_grayscale(self, tmp_path: Path) -> None:
        path = tmp_path / "gray.png"
        gray = np.full((8, 8), 120, dtype=np.uint8)
        assert image_io.imwrite(path, gray) is True
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        assert alpha is None
        assert bgr is not None
        assert bgr.shape == (8, 8, 3)

    def test_read_bgr_and_alpha_of_garbage_is_none_none(self, tmp_path: Path) -> None:
        bgr, alpha = image_io.read_bgr_and_alpha("does-not-exist-画像.png")
        assert bgr is None and alpha is None

    def test_write_bgr_with_alpha_rejoins_the_plane(self, tmp_path: Path) -> None:
        bgra = np.zeros((8, 8, 4), dtype=np.uint8)
        bgra[..., :3] = (10, 120, 240)
        bgra[..., 3] = 77
        path = tmp_path / "rebuilt.png"
        assert image_io.write_bgr_with_alpha(path, bgra[..., :3], bgra[..., 3]) is True
        out = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        assert out is not None
        assert np.array_equal(out, bgra)

    def test_write_bgr_with_alpha_carries_display_tags(self, tmp_path: Path) -> None:
        # The alpha write must thread the tag carry through like imwrite does:
        # an RGBA source with ICC + orientation re-encodes with both intact.
        src = TestDisplayTagCarry._source(tmp_path / "src.png")
        source_icc, _ = TestDisplayTagCarry._tags(src)
        bgr, alpha = image_io.read_bgr_and_alpha(src)
        assert bgr is not None
        path = tmp_path / "rebuilt-tagged.png"
        assert image_io.write_bgr_with_alpha(path, bgr, alpha, display_tags_from=src) is True
        icc, orient = TestDisplayTagCarry._tags(path)
        assert icc == source_icc
        assert orient == TestDisplayTagCarry.ORIENT

    def test_write_bgr_with_alpha_drops_alpha_on_a_flat_ext(self, tmp_path: Path) -> None:
        # JPEG has no alpha plane: the write must still succeed as plain BGR.
        bgra = np.zeros((8, 8, 4), dtype=np.uint8)
        bgra[..., 3] = 77
        path = tmp_path / "flat.jpg"
        assert image_io.write_bgr_with_alpha(path, bgra[..., :3], bgra[..., 3]) is True
        out = image_io.imread(path)
        assert out is not None
        assert out.shape == (8, 8, 3)

    def test_failed_write_reports_false(self, tmp_path: Path) -> None:
        # The flag is load-bearing (upstream issue: a failed write used to run on
        # to a bare FileNotFoundError): no directory, no file, False back.
        path = tmp_path / "no-such-dir" / "out.png"
        assert image_io.write_bgr_with_alpha(path, _make_bgr(), None) is False


class TestToBgr:
    def test_grayscale_2d_promoted_to_bgr(self) -> None:
        gray = np.full((4, 5), 120, dtype=np.uint8)
        out = image_io.to_bgr(gray)
        assert out.shape == (4, 5, 3)
        # GRAY2BGR replicates the channel, so all three match the source.
        assert np.array_equal(out[..., 0], gray)
        assert np.array_equal(out[..., 0], out[..., 2])

    def test_single_channel_3d_promoted(self) -> None:
        gray = np.full((4, 5, 1), 7, dtype=np.uint8)
        assert image_io.to_bgr(gray).shape == (4, 5, 3)

    def test_bgra_dropped_to_bgr(self) -> None:
        bgra = np.zeros((4, 5, 4), dtype=np.uint8)
        bgra[..., :3] = (10, 120, 240)
        out = image_io.to_bgr(bgra)
        assert out.shape == (4, 5, 3)
        assert np.array_equal(out, bgra[..., :3])

    def test_bgr_returned_unchanged(self) -> None:
        bgr = _make_bgr()
        assert image_io.to_bgr(bgr) is bgr  # 3-channel: no copy


class TestLoadImageBgr:
    def test_reads_successfully(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.png"
        assert image_io.imwrite(path, _make_bgr()) is True
        assert np.array_equal(image_io.load_image_bgr(path), _make_bgr())

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        # imread keeps cv2's None contract; scripts want the loud failure.
        with pytest.raises(FileNotFoundError):
            image_io.load_image_bgr(tmp_path / "gone-不存在.png")


class TestDisplayTagCarry:
    """A re-encode must not eat the source's ICC profile and EXIF orientation
    (upstream issue #98): cv2's encoders write no container metadata, so pixel-path
    outputs used to come back colourless and, on a build that leaves the raster
    stored-orientation, sideways. The pixel paths decode with IMREAD_UNCHANGED
    (which never rotates), so the raster keeps the source's stored orientation and
    the tag must ride along to the output. Fixtures are built with Pillow's own
    EXIF writer; the assertions read tags back through Pillow's parser, which is
    the independent truth (no piexif in this repo).
    """

    ORIENT = 6  # Rotate 90 CW: a portrait display of a landscape-stored raster.

    @staticmethod
    def _icc() -> bytes:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    @classmethod
    def _source(cls, path: Path, *, orient: int | None = 6, icc: bool = True) -> Path:
        """A 96x64 stored raster carrying an sRGB profile and Orientation=6."""
        from PIL import Image

        exif = Image.Exif()
        if orient is not None:
            exif[0x0112] = orient
        kwargs: dict[str, object] = {}
        if icc:
            kwargs["icc_profile"] = cls._icc()
        if orient is not None:
            kwargs["exif"] = exif.tobytes()
        Image.new("RGB", (96, 64), "red").save(path, **kwargs)
        return path

    @staticmethod
    def _tags(path: Path) -> tuple[object, object]:
        from PIL import Image

        with Image.open(path) as im:
            return im.info.get("icc_profile"), im.getexif().get(0x0112)

    def _stored_raster(self, path: Path) -> np.ndarray:
        # What the pixel paths hold after reading the source: UNCHANGED never rotates.
        raster = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        assert raster is not None
        return raster

    def test_png_output_keeps_profile_and_orientation(self, tmp_path: Path) -> None:
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert icc == source_icc
        assert orient == self.ORIENT
        # The splice must not touch pixels: Pillow decodes without rotating.
        from PIL import Image

        with Image.open(out) as im:
            assert np.array_equal(np.asarray(im)[:, :, ::-1], raster)

    def test_jpeg_output_carries_both_tags(self, tmp_path: Path) -> None:
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.jpg"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert icc == source_icc
        assert orient == self.ORIENT

    def test_webp_output_carries_both_tags_losslessly(self, tmp_path: Path) -> None:
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.webp"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert icc == source_icc
        assert orient == self.ORIENT
        # The lossless Pillow re-save must stay pixel-identical.
        back = self._stored_raster(out)
        assert back is not None
        assert np.array_equal(back, raster)

    def test_webp_source_orientation_travels(self, tmp_path: Path) -> None:
        # IMREAD_UNCHANGED preserves stored pixels, so even a fresh PNG output must
        # be tagged, or the portrait source would display sideways.
        src = self._source(tmp_path / "src.webp")
        raster = self._stored_raster(src)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        assert self._tags(out)[1] == self.ORIENT

    def test_an_upright_raster_is_not_tagged_into_a_second_rotation(self, tmp_path: Path) -> None:
        # A raster already turned upright must not receive the tag again. Stored
        # 96x64 with Orientation 6 displays as 64x96, so the upright raster is 64 wide.
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        upright = np.zeros((96, 64, 3), np.uint8)  # 64 cols x 96 rows: stored dims swapped
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, upright, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert orient is None
        assert icc == source_icc

    def test_a_raster_of_unexplained_shape_is_not_tagged(self, tmp_path: Path) -> None:
        # A resized pipeline output matches neither the stored nor the upright
        # dimensions; carrying the tag there would be a guess, so it is dropped.
        src = self._source(tmp_path / "src.png")
        resized = np.zeros((31, 17, 3), np.uint8)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, resized, display_tags_from=src) is True
        assert self._tags(out)[1] is None

    def test_an_untagged_source_changes_nothing(self, tmp_path: Path) -> None:
        # No tags to carry: the output must be byte-identical to a plain write.
        src = self._source(tmp_path / "src.png", orient=None, icc=False)
        raster = self._stored_raster(src)
        tagged, plain = tmp_path / "tagged.png", tmp_path / "plain.png"
        assert image_io.imwrite(tagged, raster, display_tags_from=src) is True
        assert image_io.imwrite(plain, raster) is True
        assert tagged.read_bytes() == plain.read_bytes()

    def test_a_missing_decode_source_still_writes(self, tmp_path: Path) -> None:
        raster = np.zeros((8, 8, 3), np.uint8)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=tmp_path / "gone.png") is True
        assert out.exists()
        assert self._tags(out) == (None, None)

    def test_in_place_rewrite_keeps_the_tags(self, tmp_path: Path) -> None:
        # Tags are read BEFORE the write, so rewriting a file over itself cannot
        # lose what it is about to restore.
        path = self._source(tmp_path / "in.png")
        source_icc, _ = self._tags(path)
        raster = self._stored_raster(path)
        assert image_io.imwrite(path, raster, display_tags_from=path) is True
        icc, orient = self._tags(path)
        assert icc == source_icc
        assert orient == self.ORIENT

    def test_rewrite_of_a_tagged_output_replaces_rather_than_stacks(self, tmp_path: Path) -> None:
        # Upstream asserts its private PNG splicer directly; this port holds the
        # same invariant at the public seam: rewriting an already-tagged PNG (with
        # itself as the display-tag source) must not duplicate the chunks.
        path = self._source(tmp_path / "in.png")
        raster = self._stored_raster(path)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=path) is True
        assert image_io.imwrite(out, raster, display_tags_from=out) is True

        kinds: list[bytes] = []
        data = out.read_bytes()
        i = 8
        while i + 8 <= len(data):
            length = int.from_bytes(data[i : i + 4], "big")
            kinds.append(data[i + 4 : i + 8])
            i += 12 + length
        assert kinds.count(b"iCCP") == 1
        assert kinds.count(b"eXIf") == 1


def _avif_writable() -> bool:
    """True when the installed Pillow can ENCODE AVIF (needed to synthesize a fixture);
    read support is what we test, but we need a writer to make the sample."""
    from PIL import Image, features

    if not (hasattr(features, "check") and features.check("avif")):
        return False
    try:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(io.BytesIO(), format="AVIF")
        return True
    except Exception:
        return False


_HAS_AVIF_WRITE = _avif_writable()


def _heif_writable() -> bool:
    try:
        image_io._register_heif()
        from PIL import Image

        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(
            io.BytesIO(), format="HEIF", quality=100
        )
        return True
    except Exception:
        return False


_HAS_HEIF_WRITE = _heif_writable()


class TestPillowFallback:
    """OpenCV cannot decode HEIC/AVIF; :func:`imread` falls back to Pillow (with the
    pillow-heif plugin for HEIF) so the pixel/removal path reads them. Uses synthetic
    AVIF (no corpus files); HEIC is exercised the same way in production."""

    @pytest.mark.skipif(not _HAS_AVIF_WRITE, reason="Pillow AVIF encoder unavailable")
    def test_avif_decodes_via_fallback(self, tmp_path: Path) -> None:
        from PIL import Image

        path = tmp_path / "x.avif"
        Image.fromarray(np.full((32, 48, 3), (10, 20, 200), np.uint8), "RGB").save(path)
        img = image_io.imread(path)  # cv2 fails on AVIF -> Pillow fallback
        assert img is not None
        assert img.shape == (32, 48, 3)
        # Source RGB (R=10, G=20, B=200) -> BGR, so blue [...,0] is high, red [...,2] low
        # (thresholds loose for AVIF's lossy compression).
        assert int(img[0, 0, 0]) > 150
        assert int(img[0, 0, 2]) < 70

    @pytest.mark.skipif(not _HAS_AVIF_WRITE, reason="Pillow AVIF encoder unavailable")
    def test_avif_alpha_survives_read_bgr_and_alpha(self, tmp_path: Path) -> None:
        from PIL import Image

        path = tmp_path / "x.avif"
        rgba = np.dstack([np.full((20, 20, 3), 80, np.uint8), np.full((20, 20), 128, np.uint8)])
        Image.fromarray(rgba, "RGBA").save(path)
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        assert bgr is not None
        assert bgr.shape == (20, 20, 3)
        assert alpha is not None  # the source alpha plane survives the fallback
        assert alpha.shape == (20, 20)

    def test_pillow_fallback_engages_when_cv2_cannot_decode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # cv2 5.x decodes AVIF natively here, so the fallback is forced directly:
        # with cv2's decoder returning None (the HEIC situation on every build),
        # imread must still produce pixels through Pillow.
        monkeypatch.setattr(cv2, "imdecode", lambda *args, **kwargs: None)
        path = tmp_path / "forced.png"
        src = _make_bgr()
        assert image_io.imwrite(path, src) is True
        out = image_io.imread(path)
        assert out is not None
        assert out.shape == (8, 8, 3)
        # Pillow decodes RGB; the wrapper flips to BGR, so the blue block reads back.
        assert np.array_equal(out, src)

    def test_pillow_fallback_survives_undecodable_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fallback must not turn a data failure into a crash: garbage stays None
        # even when cv2's decoder is out of the picture.
        monkeypatch.setattr(cv2, "imdecode", lambda *args, **kwargs: None)
        path = tmp_path / "garbage.png"
        path.write_bytes(b"still not an image")
        assert image_io.imread(path) is None

    @pytest.mark.skipif(not _HAS_HEIF_WRITE, reason="no HEIC encoder in this env")
    def test_heic_write_roundtrips(self, tmp_path: Path) -> None:
        # cv2 cannot encode HEIC (used to raise); imwrite must route through Pillow.
        img = np.full((32, 48, 3), (30, 140, 200), np.uint8)
        p = tmp_path / "x.heic"
        assert image_io.imwrite(p, img) is True
        back = image_io.imread(p)
        assert back is not None
        assert back.shape == (32, 48, 3)


class TestLabeledPixelGuard:
    """Ticket criterion: importing the module without numpy/cv2 installed is a
    labeled guard and no pixel import happens at stdlib-core load. Verified in a
    subprocess with import-blocked stubs shadowing numpy/cv2 on PYTHONPATH (the
    same boundary-blocking technique ticket 01 used for cv2), because the pooled
    test venv always has the pixel extra installed.
    """

    @staticmethod
    def _blocked_env(tmp_path: Path) -> dict[str, str]:
        stubs = tmp_path / "blocked-pixel"
        stubs.mkdir()
        for name in ("numpy", "cv2"):
            (stubs / f"{name}.py").write_text(
                f"raise ModuleNotFoundError('{name} is not installed (blocked by test)')\n",
                encoding="utf-8",
            )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(stubs), str(SCRIPTS)))}
        env.pop("PYTHONSTARTUP", None)
        return env

    def _run(self, tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=self._blocked_env(tmp_path),
            cwd=tmp_path,
        )

    def test_import_loads_with_no_pixel_module(self, tmp_path: Path) -> None:
        out = self._run(
            tmp_path,
            "import sys; import image_io; "
            "print('PIXEL_LOADED', 'numpy' in sys.modules, 'cv2' in sys.modules, 'PIL' in sys.modules)",
        )
        assert out.returncode == 0, out.stderr
        assert "PIXEL_LOADED False False False" in out.stdout

    def test_imread_raises_the_labeled_guard(self, tmp_path: Path) -> None:
        out = self._run(
            tmp_path,
            "import image_io\n"
            "try:\n"
            "    image_io.imread('anything.png')\n"
            "except ImportError as e:\n"
            "    print('LABELED', type(e).__name__, 'watermark-remover[pixel]' in str(e))\n"
            "else:\n"
            "    print('NO_ERROR')\n",
        )
        assert out.returncode == 0, out.stderr
        assert "LABELED PixelDependencyUnavailable True" in out.stdout

    def test_imwrite_raises_the_labeled_guard(self, tmp_path: Path) -> None:
        out = self._run(
            tmp_path,
            "import image_io\n"
            "try:\n"
            "    image_io.imwrite('out.png', None)\n"
            "except ImportError as e:\n"
            "    print('LABELED', type(e).__name__, 'watermark-remover[pixel]' in str(e))\n"
            "else:\n"
            "    print('NO_ERROR')\n",
        )
        assert out.returncode == 0, out.stderr
        assert "LABELED PixelDependencyUnavailable True" in out.stdout
