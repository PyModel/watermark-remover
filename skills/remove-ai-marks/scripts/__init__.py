"""Skills remove-ai-marks scripts package.

Exports core types for use by CLI entry points and demo.
"""

from __future__ import annotations

# Re-export core types for convenience
from operation import (
    BEST_EFFORT,
    CONFIRMED_CLEAN,
    EXIT_CODE_MAPPING,
    FAILED,
    OPERATION_STATUS_MAP,
    RESIDUAL_RISK,
    UNSUPPORTED,
    VERIFIED,
    VerificationStatus,
)

__all__ = [
    "BEST_EFFORT",
    "CONFIRMED_CLEAN",
    "EXIT_CODE_MAPPING",
    "FAILED",
    "OPERATION_STATUS_MAP",
    "RESIDUAL_RISK",
    "UNSUPPORTED",
    "VERIFIED",
    "VerificationStatus",
]
