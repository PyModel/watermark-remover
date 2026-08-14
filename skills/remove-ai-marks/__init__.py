"""Watermark-remover agent skill package.

The on-disk directory uses a hyphen (``skills/remove-ai-marks/``) while the
importable package is ``skills.remove_ai_marks``. pyproject.toml maps the two
through ``package-dir``; this file makes the mapped directory a real package.
"""

from __future__ import annotations
