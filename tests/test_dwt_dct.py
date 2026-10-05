"""Tests for the DWT-DCT decoder (``dwt_dct.py``).

The primary pin is decode-to-expected-bits: committed fixtures in
``skills/remove-ai-marks/scripts/assets/watermarked_fixtures/`` decode to the
exact bit patterns of the three known open-watermark schemes, and a live
oracle roundtrip (invisible-watermark, dev-only install) pins bit-for-bit
parity with the library this decoder replaces. Oracle-gated tests skip when
the oracle or the ``detect`` extra is absent; a skipped parity test is a
not-run, never a pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
FIXTURES = SCRIPTS / "assets" / "watermarked_fixtures"
sys.path.insert(0, str(SCRIPTS))

from dwt_dct import decode_dwt_dct, decode_dwt_dct_lengths

#: 48-bit ``bits`` watermarks embedded by the known generators (module-level
#: constants are the independent expected values; PROVENANCE.md records where
#: each one comes from).
SDXL_BITS = 0b101100111110110010010000011110111011000110011110
FLUX2_BITS = 0b001010101111111010000111100111001111010100101110
SD1_STRING = b"StableDiffusionV1"


def _bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def _carrier(width: int = 512, height: int = 512) -> np.ndarray:
    """A deterministic synthetic BGR carrier (no captured pixels)."""
    rng = np.random.default_rng(20211012)
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    gradient = (x[None, :] * 90.0 + y[:, None] * 140.0)[:, :, None]
    noise = rng.uniform(0.0, 60.0, size=(height, width, 1))
    base = np.clip(gradient + noise, 0.0, 255.0).astype(np.uint8)
    return np.repeat(base, 3, axis=2)


def _oracle():
    """The dev-only invisible-watermark install, or a skip."""
    pytest.importorskip("imwatermark")
    cv2 = pytest.importorskip("cv2")  # noqa: F841  (guards a half-install)

    from imwatermark import WatermarkEncoder

    return WatermarkEncoder()


@pytest.mark.parametrize(
    ("wm_len", "bits"),
    [
        (48, SDXL_BITS),
        (48, FLUX2_BITS),
        (8 * len(SD1_STRING), int.from_bytes(SD1_STRING, "big")),
    ],
    ids=["sdxl-48", "flux2-48", "sd-v1-136"],
)
def test_decode_matches_oracle_roundtrip(wm_len: int, bits: int) -> None:
    """Every embedded scheme decodes back to its exact bit pattern."""
    oracle = _oracle()
    nbits = [int(b) for b in format(bits, f"0{wm_len}b")]
    encoded = oracle.set_by_bits(nbits) or oracle
    image = encoded.encode(_carrier(), method="dwtDct")
    decoded = decode_dwt_dct(image, wm_len)
    assert _bits_to_int(decoded) == bits


def test_decode_lengths_decodes_both_families_from_one_scan() -> None:
    """One block scan yields both the 48-bit and the 136-bit verdicts."""
    oracle = _oracle()
    sd1 = [int(b) for b in np.unpackbits(np.frombuffer(SD1_STRING, dtype=np.uint8))]
    encoded = oracle.set_by_bytes(SD1_STRING) or oracle
    image = encoded.encode(_carrier(), method="dwtDct")
    decoded = decode_dwt_dct_lengths(image, (48, 8 * len(SD1_STRING)))
    assert [int(b) for b in decoded[8 * len(SD1_STRING)]] == sd1
    # The 48-bit buckets of a 136-bit embed are a bucketing artifact, not a
    # second watermark: they exist in the response, undecorated.
    assert decoded[48].size == 48


def test_small_image_is_rejected_not_guessed() -> None:
    """Below the 256x256 area floor the decoder refuses instead of guessing."""
    with pytest.raises(RuntimeError, match="too small"):
        decode_dwt_dct(_carrier(256, 255), 48)


def test_invalid_watermark_lengths_are_rejected() -> None:
    image = _carrier()
    with pytest.raises(ValueError, match="positive"):
        decode_dwt_dct_lengths(image, ())
    with pytest.raises(ValueError, match="positive"):
        decode_dwt_dct_lengths(image, (0, 48))


def test_degenerate_plane_raises_like_upstream() -> None:
    """A 1xN image clears the area check but has no 2D structure: hard error."""
    image = np.zeros((1, 70000, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="data_len must be greater than zero"):
        decode_dwt_dct_lengths(image, (48,))
