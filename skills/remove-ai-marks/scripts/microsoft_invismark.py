# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/microsoft_invismark.py,
# Apache-2.0; modified: attribution is adapted to this repo (the upstream file is a clean-room
# implementation with no third-party dependency, so no NOTICE entry is added), the module is
# relocated next to ``synthid_remove`` in this repo's scripts directory, image I/O goes through
# this repo's ``image_io`` (numpy + the pixel extra) instead of upstream's internal package,
# and ``invismark_id_from_c2pa`` reads C2PA evidence through this repo's layering (the optional
# ``c2pa`` wheel first, then the bounded stdlib caBX byte scan from ``c2pa_constants``) instead
# of upstream's ``_internal.c2pa.extract_c2pa_info``. The reader/remover logic itself is verbatim.
"""Detect and disrupt the pixel payload written by local Microsoft Paint.

This is a clean-room implementation of the observed ``Watermarker.dll`` wire
format.  It carries no Microsoft binary, symbols, or writer-quality model.  The
reader validates the repeated 144-bit message before the remover touches pixels;
callers with pristine C2PA should also pass its signed watermark UUID.

Honesty contract: ``None`` from the reader means *uncertain* — a clean-looking
decode is never a clean bill, because re-encoding or resizing destroys the
payload and the external Microsoft detector is not available here.  The remover
invalidates the observed local message; it is not a claim about any watermark
class this module cannot read.

Usage
-----
    from microsoft_invismark import detect_local_invismark, disrupt_local_invismark
    detection = detect_local_invismark("paint.png")          # None when uncertain
    result = disrupt_local_invismark("paint.png", "clean.png")
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

if TYPE_CHECKING:
    from numpy.typing import NDArray

_BLOCK_SIZE = 4
_MAP_SIZE = 12
_PAYLOAD_BITS = _MAP_SIZE * _MAP_SIZE
_QUANTUM = 24.0
_CENTERS = (6.0, 18.0)
_MAX_CENTER_ERROR = 2.0
_MIN_SUPPORT = 3

#: Rewriting 4x4 carriers is batched so peak temporary memory stays bounded on
#: large images; the batch size is a memory knob only, never a quality one.
_REWRITE_BATCH = 4096

#: The soft-binding algorithm name that anchors the C2PA fallback byte scan.
_INVISMARK_ALGORITHM = "com.microsoft.invismark.1"

_UUID_TEXT = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


@dataclass(frozen=True, slots=True)
class LocalInvisMarkDetection:
    """A validated local Paint payload recovered from repeated pixel carriers."""

    watermark_id: uuid.UUID
    payload: bytes
    minimum_support: int


@dataclass(frozen=True, slots=True)
class LocalInvisMarkRemoval(LocalInvisMarkDetection):
    """A disrupted local Paint payload and the number of rewritten carriers."""

    changed_blocks: int


@dataclass(frozen=True, slots=True)
class _Analysis:
    ll: NDArray[Any]
    bits: NDArray[Any]
    eligible: NDArray[Any]
    bit_indices: NDArray[Any]


def _dct_matrix() -> NDArray[Any]:
    """The orthonormal 4x4 DCT-II basis (numpy-only; no cv2/pywt dependency)."""
    import numpy as np

    frequencies = np.arange(_BLOCK_SIZE)[:, None]
    samples = np.arange(_BLOCK_SIZE)[None, :]
    matrix = np.cos(np.pi * (2 * samples + 1) * frequencies / (2 * _BLOCK_SIZE))
    matrix[0, :] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / _BLOCK_SIZE)


def _luminance(rgb: NDArray[Any]) -> NDArray[Any]:
    import numpy as np

    return rgb.astype(np.float64) @ np.array((0.299, 0.587, 0.114), dtype=np.float64)


def _ll_plane(luminance: NDArray[Any]) -> NDArray[Any]:
    height = luminance.shape[0] - luminance.shape[0] % 2
    width = luminance.shape[1] - luminance.shape[1] % 2
    plane = luminance[:height, :width]
    return (plane[0::2, 0::2] + plane[0::2, 1::2] + plane[1::2, 0::2] + plane[1::2, 1::2]) * 0.5


def _analyze(rgb: NDArray[Any]) -> _Analysis:
    import numpy as np

    ll = _ll_plane(_luminance(rgb))
    block_rows = ll.shape[0] // _BLOCK_SIZE
    block_columns = ll.shape[1] // _BLOCK_SIZE
    blocks = (
        ll[: block_rows * _BLOCK_SIZE, : block_columns * _BLOCK_SIZE]
        .reshape(block_rows, _BLOCK_SIZE, block_columns, _BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
    )
    transform = _dct_matrix()
    coefficients = transform @ blocks @ transform.T
    matrices = coefficients.reshape(block_rows, block_columns, -1)[..., 1:].reshape(
        block_rows, block_columns, 3, 5
    )
    singular_values = np.linalg.svd(matrices, compute_uv=False)[..., 0]
    residue = singular_values % _QUANTUM
    center_errors = np.stack(
        [
            np.minimum(abs(residue - center), _QUANTUM - abs(residue - center))
            for center in _CENTERS
        ],
        axis=-1,
    )
    bits = np.argmin(center_errors, axis=-1).astype(np.uint8)
    eligible = np.min(center_errors, axis=-1) <= _MAX_CENTER_ERROR
    block_y, block_x = np.indices((block_rows, block_columns))
    bit_indices = (block_y % _MAP_SIZE) * _MAP_SIZE + block_x % _MAP_SIZE
    return _Analysis(ll=ll, bits=bits, eligible=eligible, bit_indices=bit_indices)


def _validated_detection(rgb: NDArray[Any]) -> tuple[LocalInvisMarkDetection, _Analysis] | None:
    """Decode and validate the repeated 144-bit message, or return ``None``.

    This is the ONLY gate in front of pixel mutation: a payload is accepted
    only when every map cell reaches :data:`_MIN_SUPPORT` without a tie, the
    packed bytes carry the 0x4C header, a correct checksum, and a nonzero
    watermark UUID. Anything else is uncertain — never clean, never removable.
    """
    import numpy as np

    analysis = _analyze(rgb)
    support = np.zeros((_PAYLOAD_BITS, 2), dtype=np.int32)
    np.add.at(
        support, (analysis.bit_indices[analysis.eligible], analysis.bits[analysis.eligible]), 1
    )
    winning_support = support.max(axis=1)
    if np.any(winning_support < _MIN_SUPPORT) or np.any(support[:, 0] == support[:, 1]):
        return None

    bits = (support[:, 1] > support[:, 0]).astype(np.uint8)
    payload = np.packbits(bits).tobytes()
    if len(payload) != 18 or payload[0] != 0x4C or payload[-1] != sum(payload[1:17]) & 0xFF:
        return None
    watermark_id = uuid.UUID(bytes_le=payload[1:17])
    if watermark_id.int == 0:
        return None
    detection = LocalInvisMarkDetection(
        watermark_id=watermark_id,
        payload=payload,
        minimum_support=int(winning_support.min()),
    )
    return detection, analysis


def _read_rgb(source: str | Path) -> tuple[NDArray[Any], NDArray[Any] | None]:
    from image_io import read_bgr_and_alpha

    bgr, alpha = read_bgr_and_alpha(source)
    if bgr is None:
        raise ValueError(f"Could not read image: {source}")
    return bgr[..., ::-1], alpha


def detect_local_invismark(source: str | Path) -> LocalInvisMarkDetection | None:
    """Return a validated local Paint payload, or ``None`` for an uncertain image."""
    rgb, _alpha = _read_rgb(source)
    recovered = _validated_detection(rgb)
    return recovered[0] if recovered is not None else None


def invismark_id_from_c2pa_info(info: dict[str, Any]) -> uuid.UUID | None:
    """Return the one signed UUID for a Microsoft InvisMark soft binding.

    ``info`` follows the collected-C2PA shape: ``soft_binding_algorithm`` and
    ``soft_binding_value`` (comma-separated lists allowed, as multiple
    assertions may concatenate). Refuses anything but the exact algorithm
    string and exactly one unique parseable UUID — a guess is worse than
    ``None``.
    """
    algorithms = {
        value.strip()
        for value in str(info.get("soft_binding_algorithm") or "").split(",")
        if value.strip()
    }
    if _INVISMARK_ALGORITHM not in algorithms:
        return None
    values = [
        value.strip()
        for value in str(info.get("soft_binding_value") or "").split(",")
        if value.strip()
    ]
    parsed: list[uuid.UUID] = []
    for value in values:
        try:
            parsed.append(uuid.UUID(value))
        except ValueError:
            continue
    unique = tuple(dict.fromkeys(parsed))
    return unique[0] if len(unique) == 1 else None


def _c2pa_info_from_manifest_store(store: dict[str, Any]) -> dict[str, Any]:
    """Collect ``(soft_binding_algorithm, soft_binding_value)`` pairs from a store."""
    info: dict[str, Any] = {}
    algorithms: list[str] = []
    values: list[str] = []
    for manifest in (store.get("manifests") or {}).values():
        for assertion in manifest.get("assertions") or []:
            if "soft" not in str(assertion.get("label", "")).lower():
                continue
            data = assertion.get("data") or {}
            algorithm = data.get("alg")
            if algorithm:
                algorithms.append(str(algorithm))
            for block in data.get("blocks") or []:
                if isinstance(block, dict) and block.get("value"):
                    values.append(str(block["value"]))
    if algorithms:
        info["soft_binding_algorithm"] = ",".join(algorithms)
    if values:
        info["soft_binding_value"] = ",".join(values)
    return info


def _invismark_id_via_wheel(source: Path) -> uuid.UUID | None:
    """Read the signed UUID through the optional ``c2pa`` wheel; ``None`` if unusable."""
    try:
        import c2pa  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        reader = c2pa.Reader.try_create(source)
        if reader is None:
            return None
        with reader:
            try:
                store = json.loads(reader.json())
            except (TypeError, ValueError):
                return None
    except Exception:
        return None
    if not isinstance(store, dict):
        return None
    return invismark_id_from_c2pa_info(_c2pa_info_from_manifest_store(store))


def _invismark_id_via_cabx_scan(source: Path) -> uuid.UUID | None:
    """Bounded stdlib fallback: one UUID in a caBX chunk carrying the algorithm token.

    The full CBOR structure is not decoded here; the heuristic is deliberately
    narrow — the exact algorithm string must appear in the chunk payload and
    exactly one unique UUID-format text may exist in it. Anything ambiguous is
    ``None``, never a guess.
    """
    from c2pa_constants import find_c2pa_chunk

    payload, _reason = find_c2pa_chunk(source)
    if not payload:
        return None
    lowered = payload.lower()
    if _INVISMARK_ALGORITHM.encode("ascii") not in lowered:
        return None
    found = payload.decode("utf-8", errors="replace")
    unique = tuple(dict.fromkeys(match.group(0) for match in _UUID_TEXT.finditer(found)))
    if len(unique) != 1:
        return None
    try:
        return uuid.UUID(unique[0])
    except ValueError:
        return None


def invismark_id_from_c2pa(source: str | Path) -> uuid.UUID | None:
    """Read the signed Microsoft InvisMark UUID from a pristine image.

    Fail-soft by contract: an unreadable file, an absent wheel, a malformed
    manifest, or ambiguous evidence all return ``None`` (unknown), never a
    guessed identifier.
    """
    source_path = Path(source)
    try:
        via_wheel = _invismark_id_via_wheel(source_path)
        if via_wheel is not None:
            return via_wheel
        return _invismark_id_via_cabx_scan(source_path)
    except Exception:
        return None


def disrupt_local_invismark(
    source: str | Path,
    output: str | Path,
    *,
    expected_watermark_id: str | uuid.UUID | None = None,
) -> LocalInvisMarkRemoval | None:
    """Invert confirmed local Paint carriers and atomically write the result.

    No output is written unless the payload validates, an optional signed UUID
    agrees, and the encoded output no longer contains a valid local payload.
    This invalidates the observed message; it is not a claim about an unavailable
    external Microsoft detector.
    """
    import numpy as np
    from image_io import write_bgr_with_alpha

    source_path, output_path = Path(source), Path(output)
    rgb, alpha = _read_rgb(source_path)
    recovered = _validated_detection(rgb)
    if recovered is None:
        return None
    detection, analysis = recovered
    if expected_watermark_id is not None:
        try:
            expected = uuid.UUID(str(expected_watermark_id))
        except ValueError:
            return None
        if detection.watermark_id != expected:
            return None

    luminance = _luminance(rgb)
    ll = analysis.ll
    original_ll = ll.copy()
    transform = _dct_matrix()
    payload_bits = np.unpackbits(np.frombuffer(detection.payload, dtype=np.uint8))
    selected = analysis.eligible & (analysis.bits == payload_bits[analysis.bit_indices])
    selected_y, selected_x = np.nonzero(selected)
    changed_blocks = len(selected_y)
    block_rows, block_columns = analysis.bits.shape
    block_grid = (
        ll[: block_rows * _BLOCK_SIZE, : block_columns * _BLOCK_SIZE]
        .reshape(block_rows, _BLOCK_SIZE, block_columns, _BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .copy()
    )
    for start in range(0, changed_blocks, _REWRITE_BATCH):
        batch_y = selected_y[start : start + _REWRITE_BATCH]
        batch_x = selected_x[start : start + _REWRITE_BATCH]
        coefficients = transform @ block_grid[batch_y, batch_x] @ transform.T
        matrices = coefficients.reshape(-1, 16)[:, 1:].reshape(-1, 3, 5)
        target_bits = 1 - analysis.bits[batch_y, batch_x]
        u, singular, vh = np.linalg.svd(matrices, full_matrices=False)
        singular[:, 0] = (np.floor(singular[:, 0] / _QUANTUM) + 0.25 + 0.5 * target_bits) * _QUANTUM
        rewritten = coefficients.reshape(-1, 16).copy()
        rewritten[:, 1:] = ((u * singular[:, None, :]) @ vh).reshape(-1, 15)
        block_grid[batch_y, batch_x] = transform.T @ rewritten.reshape(-1, 4, 4) @ transform
    ll[: block_rows * _BLOCK_SIZE, : block_columns * _BLOCK_SIZE] = block_grid.transpose(
        0, 2, 1, 3
    ).reshape(block_rows * _BLOCK_SIZE, block_columns * _BLOCK_SIZE)

    height, width = original_ll.shape
    delta = (ll - original_ll) * 0.5
    pixel_delta = np.zeros(luminance.shape, dtype=np.float64)
    pixel_delta[: height * 2 : 2, : width * 2 : 2] = delta
    pixel_delta[: height * 2 : 2, 1 : width * 2 : 2] = delta
    pixel_delta[1 : height * 2 : 2, : width * 2 : 2] = delta
    pixel_delta[1 : height * 2 : 2, 1 : width * 2 : 2] = delta
    cleaned_rgb = np.clip(np.rint(rgb.astype(np.float64) + pixel_delta[..., None]), 0, 255).astype(
        np.uint8
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if not write_bgr_with_alpha(
            temporary_path,
            cleaned_rgb[..., ::-1],
            alpha,
            display_tags_from=source_path,
            orientation_applied=False,
        ):
            raise OSError(f"failed to write output (is the destination writable?): {output_path}")
        if detect_local_invismark(temporary_path) is not None:
            return None
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return LocalInvisMarkRemoval(
        watermark_id=detection.watermark_id,
        payload=detection.payload,
        minimum_support=detection.minimum_support,
        changed_blocks=changed_blocks,
    )
