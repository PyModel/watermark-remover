"""Tests for the Paint invismark reader/remover (``microsoft_invismark.py``).

Clean-room pins for the observed ``Watermarker.dll`` 144-bit DCT payload. The
primary vectors are produced by an *independent* synthetic writer (its own
DCT matrix, its own payload assembly — deliberately not calling production
helpers), so a decode-to-expected-UUID pin can never pass by construction.
The honesty contract under test: the repeated message is validated before the
remover touches pixels, and every refusal leaves the output path untouched.
"""

from __future__ import annotations

import struct
import sys
import uuid
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("numpy")

from microsoft_invismark import (
    detect_local_invismark,
    invismark_id_from_c2pa_info,
)

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


# ---------------------------------------------------------------------------
# C2PA watermark-id helpers (wheel-optional, stdlib caBX fallback)
# ---------------------------------------------------------------------------


def test_c2pa_uuid_requires_exact_algorithm_and_one_valid_value() -> None:
    info = {
        "soft_binding_algorithm": "com.microsoft.invismark.1",
        "soft_binding_value": str(_WATERMARK_ID),
    }

    assert invismark_id_from_c2pa_info(info) == _WATERMARK_ID
    assert invismark_id_from_c2pa_info({**info, "soft_binding_algorithm": "io.iscc.v0"}) is None
    assert (
        invismark_id_from_c2pa_info(
            {**info, "soft_binding_value": f"{_WATERMARK_ID}, {uuid.uuid4()}"}
        )
        is None
    )
    assert invismark_id_from_c2pa_info({**info, "soft_binding_value": "not-a-uuid"}) is None
    assert invismark_id_from_c2pa_info({}) is None


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _cabx_png(path: Path, cabx_payload: bytes) -> Path:
    """Write a minimal PNG with one caBX chunk before IDAT (1x1 RGBA)."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_idat = zlib.compress(b"\x00\x00\x00\x00")
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"caBX", cabx_payload)
        + _png_chunk(b"IDAT", raw_idat)
        + _png_chunk(b"IEND", b"")
    )
    return path


def test_c2pa_file_fallback_scans_a_cabx_chunk_and_stays_fail_soft(tmp_path: Path) -> None:
    from microsoft_invismark import invismark_id_from_c2pa

    # Real caBX payloads are c2pa/JUMBF structures; the token keeps the chunk
    # recognizable to the same scanner the provenance record uses.
    signed = _cabx_png(
        tmp_path / "signed.png",
        (
            '{"c2pa.soft-binding": {"alg": "com.microsoft.invismark.1",'
            f' "value": "{_WATERMARK_ID}"}}}}'
        ).encode(),
    )
    other = _cabx_png(
        tmp_path / "other.png",
        f'{{"c2pa.soft-binding": {{"alg": "io.iscc.v0", "value": "{_WATERMARK_ID}"}}}}'.encode(),
    )
    ambiguous = _cabx_png(
        tmp_path / "ambiguous.png",
        (
            '{"c2pa.soft-binding": {"alg": "com.microsoft.invismark.1",'
            f' "value": "{_WATERMARK_ID}, {uuid.uuid4()}"}}}}'
        ).encode(),
    )
    bare = tmp_path / "bare.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB").save(bare)
    unreadable = tmp_path / "unreadable.png"
    unreadable.write_bytes(b"not a png")

    assert invismark_id_from_c2pa(signed) == _WATERMARK_ID
    assert invismark_id_from_c2pa(other) is None
    assert invismark_id_from_c2pa(ambiguous) is None
    assert invismark_id_from_c2pa(bare) is None
    assert invismark_id_from_c2pa(unreadable) is None


def test_c2pa_wheel_path_parses_a_manifest_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optional c2pa wheel is authoritative when it answers; garbage is None."""
    import json
    import types

    from microsoft_invismark import invismark_id_from_c2pa

    source = tmp_path / "signed.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB").save(source)
    store = {
        "manifests": {
            "paint": {
                "assertions": [
                    {
                        "label": "c2pa.soft-binding",
                        "data": {
                            "alg": "com.microsoft.invismark.1",
                            "blocks": [{"value": str(_WATERMARK_ID)}],
                        },
                    }
                ]
            }
        }
    }

    class FakeReader:
        def __init__(self, payload: bytes | None) -> None:
            self._payload = payload

        def __enter__(self) -> FakeReader:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def json(self) -> str:
            return "" if self._payload is None else self._payload.decode("utf-8")

    def make_module(payload: bytes | None) -> types.ModuleType:
        module = types.ModuleType("c2pa")
        module.Reader = types.SimpleNamespace(try_create=lambda _path: FakeReader(payload))
        return module

    monkeypatch.setitem(sys.modules, "c2pa", make_module(json.dumps(store).encode("utf-8")))
    assert invismark_id_from_c2pa(source) == _WATERMARK_ID

    monkeypatch.setitem(sys.modules, "c2pa", make_module(None))
    assert invismark_id_from_c2pa(source) is None

    monkeypatch.setitem(sys.modules, "c2pa", make_module(b"{not json"))
    assert invismark_id_from_c2pa(source) is None


# ---------------------------------------------------------------------------
# identify signal (pure evidence-vocabulary mapping)
# ---------------------------------------------------------------------------


def test_identify_signal_maps_a_validated_payload_into_the_evidence_vocabulary() -> None:
    import identify

    signal = identify.microsoft_invismark_signal(_WATERMARK_ID)

    assert signal["watermark_id"] == str(_WATERMARK_ID)
    assert signal["asserts_ai"] is True
    assert "InvisMark" in signal["finding"]
    assert str(_WATERMARK_ID) in signal["finding"]
    # Origin-neutral: the mapper cannot know whether the id came from pixels
    # or metadata, so the finding must never claim one over the other.
    assert "C2PA soft binding" in signal["finding"]
    # Honesty contract: the fragility caveat rides along — absence is never clean.
    assert "absence is never" in signal["note"]


def test_identify_signal_absence_and_junk_are_unknown_never_clean() -> None:
    import identify

    for missing in (None, "not-a-uuid", 42):
        signal = identify.microsoft_invismark_signal(missing)
        assert signal["watermark_id"] is None
        assert signal["asserts_ai"] is False
        assert signal["finding"] is None
        assert "absence is never" in signal["note"]


def test_identify_signal_accepts_the_canonical_string_form() -> None:
    import identify

    signal = identify.microsoft_invismark_signal(str(_WATERMARK_ID).upper())
    assert signal["watermark_id"] == str(_WATERMARK_ID)
    assert signal["asserts_ai"] is True


# ---------------------------------------------------------------------------
# Repo-contract pins: numpy-only, provenance header
# ---------------------------------------------------------------------------


def test_module_is_numpy_only_no_cv2_or_pywt_imports() -> None:
    """The invismark transform stack is numpy-only; no cv2/pywt import anywhere."""
    source = (SCRIPTS / "microsoft_invismark.py").read_text(encoding="utf-8")
    for forbidden in ("import cv2", "from cv2", "import pywt", "from pywt"):
        assert forbidden not in source, forbidden


def test_module_carries_the_upstream_provenance_header() -> None:
    source = (SCRIPTS / "microsoft_invismark.py").read_text(encoding="utf-8")
    assert "Ported from wiltodelta/remove-ai-watermarks" in source
    assert "Apache-2.0" in source
    # The clean-room honesty statement is load-bearing contract, not prose.
    assert "clean-room" in source
    assert "validates the repeated 144-bit message before the remover touches pixels" in source


def test_importing_the_module_pulls_in_no_cv2_or_pywt() -> None:
    """Top-level import stays numpy-only: cv2/pywt must not enter sys.modules."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, r'{SCRIPTS}'); import microsoft_invismark; "
            "print('cv2' in sys.modules, 'pywt' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["False", "False"]


def test_disruption_rewrites_large_carriers_across_rewind_batches(tmp_path: Path) -> None:
    """A 640px carrier has >4096 selected blocks, exercising the rewrite batching."""
    from microsoft_invismark import disrupt_local_invismark

    source = tmp_path / "large.png"
    output = tmp_path / "large-clean.png"
    original = np.random.default_rng(20260913).integers(24, 232, size=(640, 640, 3), dtype=np.uint8)
    Image.fromarray(_synthetic_writer(original, _WATERMARK_ID), "RGB").save(source)

    result = disrupt_local_invismark(source, output, expected_watermark_id=_WATERMARK_ID)

    assert result is not None
    assert result.changed_blocks > 4096
    assert output.exists()
    assert detect_local_invismark(output) is None
