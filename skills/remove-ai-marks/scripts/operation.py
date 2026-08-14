"""Operation status, exit codes, and verification types for the watermark-remover tool.

Defines the canonical lifecycle states for any operation and a stable exit-code
contract so callers (CLI, CI, batch engine) can reason about results without
parsing presentation strings.

Exit codes
----------
    0  — All requested operations completed with verified status.
    1  — Processing error or residual signal detected (data exists but may not
         be fully clean).
    2  — Usage / input-selection error (bad flags, missing files, unsupported
         format).

Operation status (enum)
-----------------------
    verified        — The post-operation inspection confirms the desired state.
    best_effort     — The operation ran successfully, but we cannot confirm the
                      desired state (e.g., neural inpainting, LLM rewrite).
    residual_risk   — Residual evidence was detected (e.g. C2PA still present).
    unsupported     — The input or feature is not supported.
    failed          — The operation raised an error.

Verification status (enum)
--------------------------
    confirmed_clean — A post-operation re-inspection confirms removal.
    residual_detected — Post-operation inspection found remaining signal.
    verified_partial — Some aspects verified, others best-effort.
    not_verified — No post-operation re-inspection was performed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class OperationStatus(Enum):
    """Post-operation confidence level."""

    #: A deterministic post-inspection confirmed the desired state.
    VERIFIED = auto()
    #: The operation ran, but the result is probabilistic (e.g. neural inpaint).
    BEST_EFFORT = auto()
    #: Residual evidence was detected after the operation.
    RESIDUAL_RISK = auto()
    #: The input or feature is not supported by any pipeline.
    UNSUPPORTED = auto()
    #: The operation raised an error.
    FAILED = auto()

    def is_terminal(self) -> bool:
        return self in (OperationStatus.FAILED, OperationStatus.UNSUPPORTED)


# Module-level convenience aliases for backward compatibility
VERIFIED = OperationStatus.VERIFIED
BEST_EFFORT = OperationStatus.BEST_EFFORT
RESIDUAL_RISK = OperationStatus.RESIDUAL_RISK
UNSUPPORTED = OperationStatus.UNSUPPORTED
FAILED = OperationStatus.FAILED


class VerificationStatus(Enum):
    """Post-operation re-inspection result."""

    #: Post-inspection confirms clean.
    CONFIRMED_CLEAN = auto()
    #: Post-inspection found remaining signal.
    RESIDUAL_DETECTED = auto()
    #: Some aspects verified, others are best-effort.
    VERIFIED_PARTIAL = auto()
    #: No post-inspection was performed.
    NOT_VERIFIED = auto()


# Module-level convenience alias
CONFIRMED_CLEAN = VerificationStatus.CONFIRMED_CLEAN


class ExitCode(Enum):
    """Stable CLI / program exit codes.

    - 0: success (verified).
    - 1: residual signal or processing error.
    - 2: usage or input-selection error.
    """

    SUCCESS = 0
    RESIDUAL_OR_ERROR = 1
    USAGE_ERROR = 2


# Mapping of operation status → exit code
OPERATION_STATUS_MAP: dict[OperationStatus, int] = {
    OperationStatus.VERIFIED: ExitCode.SUCCESS.value,
    OperationStatus.BEST_EFFORT: ExitCode.RESIDUAL_OR_ERROR.value,
    OperationStatus.RESIDUAL_RISK: ExitCode.RESIDUAL_OR_ERROR.value,
    OperationStatus.UNSUPPORTED: ExitCode.USAGE_ERROR.value,
    OperationStatus.FAILED: ExitCode.RESIDUAL_OR_ERROR.value,
}

# Mapping of verification status → outcome
EXIT_CODE_MAPPING: dict[VerificationStatus, int] = {
    VerificationStatus.CONFIRMED_CLEAN: ExitCode.SUCCESS.value,
    VerificationStatus.RESIDUAL_DETECTED: ExitCode.RESIDUAL_OR_ERROR.value,
    VerificationStatus.VERIFIED_PARTIAL: ExitCode.RESIDUAL_OR_ERROR.value,
    VerificationStatus.NOT_VERIFIED: ExitCode.RESIDUAL_OR_ERROR.value,
}


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Result of a single named operation within the clean pipeline."""

    operation_name: str
    status: OperationStatus
    details: dict[str, Any] = field(default_factory=dict)
    verified_after: bool = False
    verification: VerificationStatus = VerificationStatus.NOT_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation_name,
            "status": self.status.name.lower().replace("_", "-"),
            "details": self.details,
            "verified_after": self.verified_after,
            "verification": self.verification.name.lower().replace("_", "-"),
        }


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

OPERATION_STATUS_ORDER: list[OperationStatus] = [
    OperationStatus.VERIFIED,
    OperationStatus.BEST_EFFORT,
    OperationStatus.RESIDUAL_RISK,
    OperationStatus.UNSUPPORTED,
    OperationStatus.FAILED,
]


def worst_status(*statuses: OperationStatus) -> OperationStatus:
    """Return the worst status among the ones provided.

    Ordering (best → worst): verified < best_effort < residual_risk < unsupported < failed.
    """
    if not statuses:
        return OperationStatus.VERIFIED
    rank = {s: i for i, s in enumerate(OPERATION_STATUS_ORDER)}
    return max(statuses, key=lambda s: rank[s])


def status_to_exit_code(status: OperationStatus) -> int:
    """Map a single operation status to the stable exit code."""
    if status is OperationStatus.VERIFIED:
        return ExitCode.SUCCESS.value
    if status is OperationStatus.UNSUPPORTED:
        return ExitCode.USAGE_ERROR.value
    return ExitCode.RESIDUAL_OR_ERROR.value


def print_status_line(operation_name: str, status: OperationStatus) -> None:
    """Print a one-line status to stderr for backward-compatible logging."""
    sys.stderr.write(f"[{operation_name}] {status.name.replace('_', ' ')}\n")
    sys.stderr.flush()
