"""Focused dispatch regression for combined morphological perturbations."""

from __future__ import annotations

import pytest
from morpho_perturb import combined_morpho


def test_seeded_combined_unknown_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown morphological strategy: missing"):
        combined_morpho(
            bytes([128] * (4 * 4 * 3)),
            4,
            4,
            3,
            strategies=("grid", "missing"),
            seed=7,
        )
