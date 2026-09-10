"""Watermark-remover agent skill package.

The on-disk directory keeps its Claude-skill layout
(``skills/remove-ai-marks/``) while the installed package is a single top-level
``watermark_remover`` root. pyproject.toml maps the two through
``package-dir``; this file makes the mapped directory a real package.
"""

from __future__ import annotations
