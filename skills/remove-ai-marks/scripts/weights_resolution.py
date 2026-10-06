"""Shared pinned-weights resolution prologue for the single-file hub classifiers.

``source_classify`` and ``source_classify_v2`` resolve one pinned model file
the same way: a weights-env override (a file, or a directory holding the
model file) wins over a pinned-revision Hugging Face Hub download, and the
override gate must keep identical error wording in both classifiers. This
module holds exactly that shared prologue.

The Hub download itself and its failure handling stay in each classifier on
purpose: ``source_classify`` wraps download failures in a RuntimeError
carrying the pinned repo/revision context (a deliberate modification over
upstream), while ``source_classify_v2`` propagates the bare transport error.
Only the behavior-identical parts live here.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["import_hub_or_raise", "resolve_env_override"]


def resolve_env_override(env_var: str, filename: str) -> Path | None:
    """Resolve a weights-env override to one existing model file, or None.

    The override may name the model file itself or a directory holding it.
    A set-but-missing override raises a labeled RuntimeError naming the
    variable — it never silently falls back to the Hub download.
    """
    override = os.environ.get(env_var, "").strip()
    if not override:
        return None
    path = Path(override).expanduser()
    candidate = path / filename if path.is_dir() else path
    if not candidate.is_file():
        raise RuntimeError(f"{env_var} does not contain {filename}: {path}")
    return candidate


def import_hub_or_raise(install_hint: str) -> Any:
    """Import ``huggingface_hub``, or raise the caller's install hint."""
    try:
        hub: Any = import_module("huggingface_hub")
    except ModuleNotFoundError as error:
        raise RuntimeError(install_hint) from error
    return hub
