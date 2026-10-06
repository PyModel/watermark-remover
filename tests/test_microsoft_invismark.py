"""Tests for the Paint invismark reader/remover (``microsoft_invismark.py``).

Clean-room pins for the observed ``Watermarker.dll`` 144-bit DCT payload. The
primary vectors are produced by an *independent* synthetic writer (its own
DCT matrix, its own payload assembly — deliberately not calling production
helpers), so a decode-to-expected-UUID pin can never pass by construction.
The honesty contract under test: the repeated message is validated before the
remover touches pixels, and every refusal leaves the output path untouched.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("numpy")

from microsoft_invismark import detect_local_invismark

#: The GUID the synthetic writer encodes (independent expected value).
_WATERMARK_ID = uuid.UUID("83424621-03cb-40e3-9808-a9fae837156d")


def _dct_matrix() -> np.ndarray:
    matrix = np.empty((4, 4), dtype=np.float64)
    for frequency in range(4):
        scale = np.sqrt(1 / 4) if frequency == 0 else np.sqrt(2 / 4)
        for sample in range(4):
            matrix[frequency, sample] = scale * np.cos(np.pi * (2 * sample + 1) * frequency / 8)
    return matrix


def _synthetic_writer(
    image: np.ndarray, watermark_id: uuid.UUID, *, valid_checksum: bool = True
) -> np.ndarray:
    """Independent format producer; it deliberately does not call production helpers."""
    guid = watermark_id.bytes_le
    checksum = sum(guid) & 0xFF
    if not valid_checksum:
        checksum ^= 1
    payload = bytes((0x4C,)) + guid + bytes((checksum,))
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

    rgb = image.astype(np.float64)
    luminance = rgb @ np.array((0.299, 0.587, 0.114), dtype=np.float64)
    ll = (
        luminance[0::2, 0::2]
        + luminance[0::2, 1::2]
        + luminance[1::2, 0::2]
        + luminance[1::2, 1::2]
    ) * 0.5
    original_ll = ll.copy()
    dct = _dct_matrix()

    for block_y in range(ll.shape[0] // 4):
        for block_x in range(ll.shape[1] // 4):
            top, left = block_y * 4, block_x * 4
            block = ll[top : top + 4, left : left + 4]
            coefficients = dct @ block @ dct.T
            ac = coefficients.reshape(-1)[1:].reshape(3, 5)
            u, singular, vh = np.linalg.svd(ac, full_matrices=False)
            bit = int(bits[(block_y % 12) * 12 + (block_x % 12)])
            singular[0] = (np.floor(singular[0] / 24) + 0.25 + 0.5 * bit) * 24
            rewritten = coefficients.reshape(-1).copy()
            rewritten[1:] = (u @ np.diag(singular) @ vh).reshape(-1)
            ll[top : top + 4, left : left + 4] = dct.T @ rewritten.reshape(4, 4) @ dct

    delta = (ll - original_ll) * 0.5
    pixel_delta = np.repeat(np.repeat(delta, 2, axis=0), 2, axis=1)
    return np.clip(np.rint(rgb + pixel_delta[..., None]), 0, 255).astype(np.uint8)


def _write_carrier(path: Path, *, valid_checksum: bool = True) -> np.ndarray:
    original = np.random.default_rng(20260913).integers(24, 232, size=(192, 192, 3), dtype=np.uint8)
    encoded = _synthetic_writer(original, _WATERMARK_ID, valid_checksum=valid_checksum)
    Image.fromarray(encoded, "RGB").save(path)
    return original


def test_detects_repeated_payload_and_windows_guid_order(tmp_path: Path) -> None:
    source = tmp_path / "paint.png"
    _write_carrier(source)

    detection = detect_local_invismark(source)

    assert detection is not None
    assert detection.watermark_id == _WATERMARK_ID
    expected_payload = (
        b"\x4c" + _WATERMARK_ID.bytes_le + bytes((sum(_WATERMARK_ID.bytes_le) & 0xFF,))
    )
    assert detection.payload == expected_payload
    assert detection.minimum_support >= 3


def test_rejects_clean_and_malformed_payloads(tmp_path: Path) -> None:
    clean = tmp_path / "clean.png"
    malformed = tmp_path / "malformed.png"
    Image.fromarray(
        np.random.default_rng(20260913).integers(24, 232, size=(192, 192, 3), dtype=np.uint8),
        "RGB",
    ).save(clean)
    _write_carrier(malformed, valid_checksum=False)

    assert detect_local_invismark(clean) is None
    assert detect_local_invismark(malformed) is None


def test_rejects_the_zero_uuid_and_tiny_carriers(tmp_path: Path) -> None:
    zero = tmp_path / "zero.png"
    tiny = tmp_path / "tiny.png"
    plain = np.random.default_rng(20260913).integers(24, 232, size=(192, 192, 3), dtype=np.uint8)
    Image.fromarray(_synthetic_writer(plain, uuid.UUID(int=0)), "RGB").save(zero)
    # 24px tall: the LL plane is 12px -> 3 block rows, so map cells cannot reach
    # the minimum support; the reader must answer None, not crash.
    Image.fromarray(
        np.random.default_rng(7).integers(24, 232, size=(24, 192, 3), dtype=np.uint8), "RGB"
    ).save(tiny)

    assert detect_local_invismark(zero) is None
    assert detect_local_invismark(tiny) is None


def test_disruption_requires_expected_guid_and_invalidates_payload(tmp_path: Path) -> None:
    from microsoft_invismark import disrupt_local_invismark

    source = tmp_path / "paint.png"
    wrong_output = tmp_path / "wrong.png"
    output = tmp_path / "clean.png"
    original = _write_carrier(source)

    assert disrupt_local_invismark(source, wrong_output, expected_watermark_id=uuid.uuid4()) is None
    assert not wrong_output.exists()

    result = disrupt_local_invismark(source, output, expected_watermark_id=_WATERMARK_ID)

    assert result is not None
    assert result.watermark_id == _WATERMARK_ID
    assert result.changed_blocks >= 432
    assert output.exists()
    assert detect_local_invismark(output) is None
    cleaned = np.asarray(Image.open(output).convert("RGB"))
    assert np.mean(np.abs(cleaned.astype(np.int16) - original.astype(np.int16))) < 2.0


def test_disruption_refuses_images_the_reader_cannot_validate(tmp_path: Path) -> None:
    """No pixel mutation path runs when the repeated message fails validation."""
    from microsoft_invismark import disrupt_local_invismark

    clean = tmp_path / "clean.png"
    malformed = tmp_path / "malformed.png"
    refused = tmp_path / "refused.png"
    Image.fromarray(
        np.random.default_rng(20260913).integers(24, 232, size=(192, 192, 3), dtype=np.uint8),
        "RGB",
    ).save(clean)
    _write_carrier(malformed, valid_checksum=False)

    assert disrupt_local_invismark(clean, refused) is None
    assert disrupt_local_invismark(malformed, refused) is None
    assert not refused.exists()
    # The refusal must not touch the source either.
    assert detect_local_invismark(malformed) is None
