"""Skills remove-ai-marks scripts package.

Exports core types for use by CLI entry points and demo.

The script modules use flat sibling imports (``from operation import ...``),
which resolve for direct script execution because the script directory is on
sys.path. Keep the same contract when this package is imported from an
installed wheel by putting this package's directory on sys.path first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

