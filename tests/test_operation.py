"""Tests for the operation status and exit-code contract."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from operation import (
    EXIT_CODE_MAPPING,
    OPERATION_STATUS_MAP,
    ExitCode,
    OperationResult,
    OperationStatus,
    VerificationStatus,
    status_to_exit_code,
    worst_status,
)


def test_exit_code_values() -> None:
    assert ExitCode.SUCCESS.value == 0
    assert ExitCode.RESIDUAL_OR_ERROR.value == 1
    assert ExitCode.USAGE_ERROR.value == 2


def test_status_to_exit_code() -> None:
    assert status_to_exit_code(OperationStatus.VERIFIED) == 0
    assert status_to_exit_code(OperationStatus.BEST_EFFORT) == 1
    assert status_to_exit_code(OperationStatus.RESIDUAL_RISK) == 1
    assert status_to_exit_code(OperationStatus.UNSUPPORTED) == 2
    assert status_to_exit_code(OperationStatus.FAILED) == 1


def test_operation_status_map_agrees_with_status_to_exit_code() -> None:
    for status, code in OPERATION_STATUS_MAP.items():
        assert status_to_exit_code(status) == code


def test_worst_status_ordering() -> None:
    assert worst_status() is OperationStatus.VERIFIED
    assert worst_status(OperationStatus.VERIFIED) is OperationStatus.VERIFIED
    assert (
        worst_status(OperationStatus.VERIFIED, OperationStatus.BEST_EFFORT)
        is OperationStatus.BEST_EFFORT
    )
    assert (
        worst_status(
            OperationStatus.VERIFIED,
            OperationStatus.RESIDUAL_RISK,
            OperationStatus.BEST_EFFORT,
        )
        is OperationStatus.RESIDUAL_RISK
    )
    assert (
        worst_status(OperationStatus.UNSUPPORTED, OperationStatus.FAILED) is OperationStatus.FAILED
    )


def test_terminal_statuses() -> None:
    assert OperationStatus.FAILED.is_terminal()
    assert OperationStatus.UNSUPPORTED.is_terminal()
    assert not OperationStatus.VERIFIED.is_terminal()
    assert not OperationStatus.BEST_EFFORT.is_terminal()


def test_operation_result_to_dict() -> None:
    result = OperationResult(
        operation_name="visible-restoration",
        status=OperationStatus.BEST_EFFORT,
        details={"backend": "texture"},
        verified_after=False,
        verification=VerificationStatus.NOT_VERIFIED,
    )
    payload = result.to_dict()
    assert payload == {
        "operation": "visible-restoration",
        "status": "best-effort",
        "details": {"backend": "texture"},
        "verified_after": False,
        "verification": "not-verified",
    }


def test_verification_status_exit_mapping() -> None:
    assert EXIT_CODE_MAPPING[VerificationStatus.CONFIRMED_CLEAN] == 0
    for status in (
        VerificationStatus.RESIDUAL_DETECTED,
        VerificationStatus.VERIFIED_PARTIAL,
        VerificationStatus.NOT_VERIFIED,
    ):
        assert EXIT_CODE_MAPPING[status] == 1


def test_status_enum_membership() -> None:
    names = {status.name for status in OperationStatus}
    assert names == {"VERIFIED", "BEST_EFFORT", "RESIDUAL_RISK", "UNSUPPORTED", "FAILED"}
