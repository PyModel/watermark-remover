# Ported from wiltodelta/remove-ai-watermarks
# src/remove_ai_watermarks/_internal/image_math.py; modified: relocated to this
# repo's flat scripts directory under the claimed image_math.py path (the
# upstream module lives in its _internal package). The transform itself is
# verbatim: pure, dependency-injected (the numpy module is a parameter), so it
# stays importable — and testable — without numpy installed.
"""Small shared numerical transforms used by independent pixel analyzers."""

from __future__ import annotations

from typing import Any


def dct_matrix(np: Any, size: int = 8) -> Any:
    """Return an orthonormal ``size`` by ``size`` DCT-II basis."""
    frequencies = np.arange(size)[:, None]
    samples = np.arange(size)[None, :]
    matrix = np.cos(np.pi * (2 * samples + 1) * frequencies / (2 * size))
    matrix[0, :] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / size)
