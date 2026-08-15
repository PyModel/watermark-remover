"""Dependency-independent regressions for the optional OpenCV adapters."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import opencv_inpaint
from morphomod import Raster


class _FakeArray:
    def __init__(self, data: bytes, shape: tuple[int, ...]) -> None:
        self._data = bytes(data)
        self.shape = shape

    def reshape(self, *shape: int) -> _FakeArray:
        return _FakeArray(self._data, tuple(shape))

    def __getitem__(self, key):
        rows, columns, channels = key
        assert rows == slice(None) and columns == slice(None)
        start = channels.start or 0
        stop = channels.stop or self.shape[2]
        selected = bytearray()
        for offset in range(0, len(self._data), self.shape[2]):
            selected.extend(self._data[offset + start : offset + stop])
        return _FakeArray(bytes(selected), (self.shape[0], self.shape[1], stop - start))

    def tobytes(self) -> bytes:
        return self._data


def _concatenate(arrays: list[_FakeArray], axis: int) -> _FakeArray:
    assert axis == -1
    left, right = arrays
    channels = left.shape[2] + right.shape[2]
    combined = bytearray()
    pixels = left.shape[0] * left.shape[1]
    for index in range(pixels):
        left_start = index * left.shape[2]
        right_start = index * right.shape[2]
        combined.extend(left._data[left_start : left_start + left.shape[2]])
        combined.extend(right._data[right_start : right_start + right.shape[2]])
    return _FakeArray(bytes(combined), (left.shape[0], left.shape[1], channels))


def test_cv2_conversion_restores_exact_original_rgba_alpha(monkeypatch) -> None:
    fake_numpy = SimpleNamespace(
        uint8=object(),
        frombuffer=lambda data, dtype: _FakeArray(bytes(data), (len(data),)),
        concatenate=_concatenate,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    original = Raster(2, 1, 4, bytearray([1, 2, 3, 17, 4, 5, 6, 231]))
    restored_rgb = _FakeArray(bytes([90, 91, 92, 93, 94, 95]), (1, 2, 3))

    result = opencv_inpaint._cv2_to_raster(restored_rgb, original)

    assert result.data == bytearray([90, 91, 92, 17, 93, 94, 95, 231])
