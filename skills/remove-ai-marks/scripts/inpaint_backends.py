"""Inpaint backend protocol and adapters for existing backends.

Defines ``InpaintBackend`` as a Protocol so new backends can plug in
without modifying existing code.  The three wrapper classes adapt the
incumbent implementations (texture_patch, simple_fill, external_command)
to this protocol.

Pipeline
--------
    existing mask acquisition
        ↓
    [component filtering, hole fill, dilation]   ← morphomod.py
        ↓
    InpaintRequest
        ↓
    InpaintBackend.inpaint()  →  InpaintResult
        ↓
    composite()  ← morphomod.py (single outside-mask seam)
        ↓
    verification  ← verification.py
"""

from __future__ import annotations

import math
import tempfile
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import external_command
from domain_types import InpaintRequest, InpaintResult
from morphomod import (
    decode_png,
    encode_png,
    simple_inpaint,
    texture_patch_inpaint,
)

# ---------------------------------------------------------------------------
# Protocol — backend contract
# ---------------------------------------------------------------------------


class InpaintBackend(Protocol):
    """Single-method protocol every inpaint backend must implement."""

    name: str

    @property  # type: ignore[override]
    def available(self) -> bool: ...

    def supports(self, request: InpaintRequest) -> bool:
        """Return True if this backend can handle the request."""

    @abstractmethod
    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        """Run inpainting.  May raise on hard errors."""


# ---------------------------------------------------------------------------
# Texture-patch backend adapter (wraps existing texture_patch_inpaint)
# ---------------------------------------------------------------------------


class TexturePatchBackend:
    """Adapter: existing texture_patch_inpaint → InpaintBackend."""

    name = "texture-patch"

    @property
    def available(self) -> bool:
        return True  # stdlib-only, always available

    def supports(self, request: InpaintRequest) -> bool:
        # Existing texture patch only works on PNG (RGB/RGBA)
        if request.source.channels not in (3, 4):
            return False
        # Texture patch struggles with very large masks
        mask_ratio = request.mask.marked / max(1, request.source.width * request.source.height)
        return mask_ratio < 0.5

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        try:
            restored, match = texture_patch_inpaint(
                request.source,
                request.mask,
                feather=request.params.get("feather", 0),
            )
            return InpaintResult(
                inpainted=restored,
                backend_name=self.name,
                quality_estimate=0.85,
                diagnostics={
                    "texture_match": f"({match.x},{match.y},{match.width},{match.height})",
                    "edge_mse": round(match.score, 4),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"{self.name} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Simple-fill backend adapter (wraps existing simple_inpaint)
# ---------------------------------------------------------------------------


class SimpleFillBackend:
    """Adapter: existing simple_inpaint → InpaintBackend."""

    name = "simple"

    @property
    def available(self) -> bool:
        return True  # stdlib-only, always available

    def supports(self, request: InpaintRequest) -> bool:
        # Simple fill works on any channel count
        if request.source.channels not in (1, 3, 4):
            return False
        mask_ratio = request.mask.marked / max(1, request.source.width * request.source.height)
        return mask_ratio < 0.7

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        try:
            filled = simple_inpaint(request.source, request.mask)
            return InpaintResult(
                inpainted=filled,
                backend_name=self.name,
                quality_estimate=0.4,
            )
        except Exception as exc:
            raise RuntimeError(f"{self.name} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# External command backend adapter (wraps existing external_command.run_command)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExternalCommandBackend:
    """Adapter: existing external_command.run_command → InpaintBackend."""

    command: str = ""
    timeout: float = 1800.0
    output_limit: int = 64 * 1024

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("command must not be empty")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if isinstance(self.output_limit, bool) or not isinstance(self.output_limit, int):
            raise TypeError("output_limit must be an integer")
        if self.output_limit <= 0:
            raise ValueError("output_limit must be positive")

    @property
    def name(self) -> str:
        return "external"

    @property
    def available(self) -> bool:
        try:
            argv = external_command.command_from_template(
                self.command,
                input="input.png",
                mask="mask.pgm",
                output="output.png",
                prompt="prompt",
            )
        except ValueError:
            return False
        # Check first token is an executable.
        import shutil

        return bool(argv) and shutil.which(argv[0]) is not None

    def supports(self, request: InpaintRequest) -> bool:
        return True  # external can handle anything it receives

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        with tempfile.TemporaryDirectory(prefix="wm-inpaint-") as temp_dir:
            tmp_path = Path(temp_dir)
            img_path = tmp_path / "input.png"
            mask_path = tmp_path / "mask.pgm"
            out_path = tmp_path / "output.png"

            # Write inputs
            img_path.write_bytes(encode_png(request.source))
            mask_data = bytes(request.mask.data)
            mask_path.write_bytes(
                f"P5\n{request.mask.width} {request.mask.height}\n255\n".encode() + mask_data
            )

            # Run command
            try:
                argv = external_command.command_from_template(
                    self.command,
                    input=str(img_path),
                    mask=str(mask_path),
                    output=str(out_path),
                    prompt=request.params.get("prompt", "Remove watermark, fill with background"),
                )
                result = external_command.run_command(
                    argv,
                    timeout=self.timeout,
                    output_limit=self.output_limit,
                )
            except external_command.ExternalCommandTimeout as exc:
                raise RuntimeError(f"external command timed out: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"external command failed: {exc}") from exc

            if result.returncode != 0:
                diag = (result.stderr_text or result.stdout_text).strip()[:500]
                raise RuntimeError(f"external command returned {result.returncode}: {diag}")

            if not out_path.is_file():
                raise RuntimeError("external command did not create output file")

            inpainted = decode_png(out_path.read_bytes())
            return InpaintResult(
                inpainted=inpainted,
                backend_name=self.name,
                quality_estimate=0.75,
                diagnostics={
                    "returncode": result.returncode,
                    "stdout_truncated": result.stdout_truncated,
                    "stderr_truncated": result.stderr_truncated,
                },
            )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

#: Registry of available backends, instantiated on import.
_REGISTRY: list[InpaintBackend] = []


def _bootstrap_registry() -> None:
    """Populate the backend registry."""
    _REGISTRY.append(TexturePatchBackend())
    _REGISTRY.append(SimpleFillBackend())


_bootstrapped = False


def get_backends() -> list[InpaintBackend]:
    """Return the current backend registry (lazy-initialised)."""
    global _bootstrapped
    if not _bootstrapped:
        _bootstrap_registry()
        _bootstrapped = True
    return _REGISTRY


def register_backend(backend: InpaintBackend) -> None:
    """Add a backend to the registry (call before first use)."""
    get_backends().append(backend)
