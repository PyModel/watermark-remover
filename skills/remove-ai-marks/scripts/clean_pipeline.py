"""Canonical planning and execution seam for cleaning requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pipeline_actions as act
from clean_asset import CleanPlan, clean_asset
from clean_request import CleanRequest, dropped_text_transforms, plan_work, select_request_inputs
from common import cleaned_path
from operation import OperationStatus, status_to_exit_code
from rewrite_text import TokenSink, remote_warning


@dataclass(frozen=True, slots=True)
class ItemResult:
    """One isolated item outcome, including its stable payload."""

    path: Path
    output: Path
    payload: dict


@dataclass(frozen=True, slots=True)
class RunResult:
    """The complete ordered result of one planned request."""

    results: tuple[ItemResult, ...]
    batch: bool

    @property
    def payload(self) -> dict:
        if self.batch:
            return {"total": len(self.results), "results": [item.payload for item in self.results]}
        return self.results[0].payload


def _error_payload(path: Path, output: Path, error: Exception) -> dict:
    return {
        "kind": "unknown",
        "input": str(path),
        "output": str(output),
        **act.report_actions([act.failed(error)]),
        "error": str(error),
        "exit_code": status_to_exit_code(OperationStatus.FAILED),
    }


def _dry_run_payload(path: Path, output: Path | None, plan: CleanPlan, in_place: bool) -> dict:
    visible = plan.visible
    if visible is None:
        raise ValueError("dry-run requires a visible cleaning plan")
    destination = path if in_place else output or cleaned_path(path)
    if visible.mask_path is not None:
        localization = f"mask:{visible.mask_path}"
    elif visible.box is not None:
        localization = f"box:{','.join(map(str, visible.box))}"
    else:
        localization = "external-detector"
    actions = [
        act.plan_localize(localization),
        act.plan_refine_mask(visible.dilation_radius),
        act.plan_inpaint(visible.backend),
        act.plan_strip_metadata(),
    ]
    if plan.degrade is not None:
        actions.append(act.plan_degrade(plan.degrade.strategy))
    actions.append(act.plan_publish(str(visible.mask_output), str(destination)))
    return {
        "kind": "image",
        "status": "dry-run",
        "input": str(path),
        "output": str(destination),
        "mask": str(visible.mask_output),
        "backend": visible.backend,
        "timeout": visible.timeout,
        **act.report_actions(actions),
        "exit_code": status_to_exit_code(OperationStatus.VERIFIED),
    }


def _execute_item(
    path: Path,
    output_path: Path | None,
    request: CleanRequest,
    plan: CleanPlan,
    *,
    on_token: TokenSink | None,
) -> ItemResult:
    destination = path if request.in_place else output_path or cleaned_path(path)
    try:
        if request.dry_run:
            payload = _dry_run_payload(path, output_path, plan, request.in_place)
        else:
            rewrite_plan = plan.text.rewrite_plan
            if rewrite_plan is not None and (warning := remote_warning(rewrite_plan.base_url)):
                # Warnings belong to the presentation adapter; this callback is
                # intentionally omitted from the execution result.
                payload = {"_warning": warning}
            else:
                payload = {}
            result = clean_asset(path, destination, plan, on_token=on_token)
            payload.update(result.to_dict())
            status = (
                OperationStatus.VERIFIED if not result.residual else OperationStatus.RESIDUAL_RISK
            )
            payload["exit_code"] = status_to_exit_code(status)
            dropped = dropped_text_transforms(request, result.kind)
            if dropped:
                payload["skipped_text_transforms"] = list(dropped)
    except Exception as error:
        payload = _error_payload(path, destination, error)
    payload.pop("_warning", None)
    return ItemResult(path, destination, payload)


def execute_item(
    path: Path,
    output_path: Path | None,
    request: CleanRequest,
    plan: CleanPlan,
    *,
    on_token: TokenSink | None = None,
) -> dict:
    """Execute a preflighted item; retained for bridge seam tests and callbacks."""
    return _execute_item(path, output_path, request, plan, on_token=on_token).payload


def execute(
    request: CleanRequest,
    *,
    on_token: TokenSink | None = None,
    on_item: Callable[[ItemResult], None] | None = None,
) -> RunResult:
    """Select, preflight, and execute a request without printing or auditing."""
    selection = select_request_inputs(request)
    work = plan_work(selection.items, request, selection.batch)
    if request.output and selection.batch and not request.in_place and not request.dry_run:
        request.output.mkdir(parents=True, exist_ok=True)
    outcomes: list[ItemResult] = []
    for item, output, plan in work:
        outcome = _execute_item(item.path, output, request, plan, on_token=on_token)
        outcomes.append(outcome)
        if on_item is not None:
            on_item(outcome)
    return RunResult(tuple(outcomes), selection.batch)
