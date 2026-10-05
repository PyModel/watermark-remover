"""Tests for open invisible-watermark scheme detection (``invisible_watermark.py``).

Scheme detection is the evidence signal: a decoded payload matching a known
fixed pattern (SDXL 48-bit, FLUX.2 48-bit, SD 1.x/2.x 136-bit string) names the
generator; anything else is ``None`` — a finding of absence here is never a
clean bill, only "no known open watermark decoded".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import invisible_watermark as iw


def _carrier(width: int = 512, height: int = 512) -> np.ndarray:
    rng = np.random.default_rng(20211012)
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    gradient = (x[None, :] * 90.0 + y[:, None] * 140.0)[:, :, None]
    noise = rng.uniform(0.0, 60.0, size=(height, width, 1))
    base = np.clip(gradient + noise, 0.0, 255.0).astype(np.uint8)
    return np.repeat(base, 3, axis=2)


def _embedded(bits_or_bytes, *, as_bits: bool) -> np.ndarray:
    pytest.importorskip("imwatermark")
    pytest.importorskip("cv2")
    from imwatermark import WatermarkEncoder

    encoder = WatermarkEncoder()
    if as_bits:
        encoder.set_by_bits(list(bits_or_bytes))
    else:
        encoder.set_by_bytes(bits_or_bytes)
    return encoder.encode(_carrier(), method="dwtDct")


@pytest.mark.skipif(not iw.is_available(), reason="detect extra (numpy/cv2/pywt) not installed")
@pytest.mark.parametrize(
    ("payload", "as_bits", "scheme"),
    [
        (
            [int(b) for b in format(iw._BITS_48["Stable Diffusion XL"], "048b")],
            True,
            "Stable Diffusion XL",
        ),
        (
            [int(b) for b in format(iw._BITS_48["FLUX.2 (Black Forest Labs)"], "048b")],
            True,
            "FLUX.2 (Black Forest Labs)",
        ),
        (iw._SD1_STRING, False, "Stable Diffusion 1.x / 2.x"),
    ],
    ids=["sdxl", "flux2", "sd-v1"],
)
def test_known_payloads_are_detected(payload, as_bits: bool, scheme: str) -> None:
    assert (
        iw.detect_invisible_watermark(Path("unused.png"), image=_embedded(payload, as_bits=as_bits))
        == scheme
    )


@pytest.mark.skipif(not iw.is_available(), reason="detect extra (numpy/cv2/pywt) not installed")
def test_plain_synthetic_image_yields_no_finding() -> None:
    assert iw.detect_invisible_watermark(Path("unused.png"), image=_carrier()) is None


@pytest.mark.skipif(not iw.is_available(), reason="detect extra (numpy/cv2/pywt) not installed")
def test_undecodable_image_yields_no_finding_not_a_crash() -> None:
    # A 24x24 array passes imread-shaped input checks and fails the decoder's
    # area floor: fail-soft None, never an exception.
    assert iw.detect_invisible_watermark(Path("unused.png"), image=_carrier(24, 24)) is None


def test_missing_file_yields_no_finding(tmp_path: Path) -> None:
    assert iw.detect_invisible_watermark(tmp_path / "nope.png") is None


def test_is_available_follows_the_numeric_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    assert iw.is_available() is True
    import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *names: False)
    assert iw.is_available() is False


@pytest.mark.skipif(not iw.is_available(), reason="detect extra (numpy/cv2/pywt) not installed")
def test_unavailable_decoder_yields_no_finding_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    import optional_deps

    monkeypatch.setattr(optional_deps, "module_available", lambda *names: False)
    assert iw.detect_invisible_watermark(Path("unused.png"), image=_carrier()) is None
