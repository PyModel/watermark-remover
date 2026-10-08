"""Contracts for the watermark-profile data module.

The profiles are pure data plus pure helpers: profile ids, the per-vendor
strength tables, and the resolvers. The module configures which removal profile
a stage uses, so it must be importable without any pixel or GPU dependency — a
caller that only wants to read the profile list must not pay for torch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/remove-ai-marks/scripts"
sys.path.insert(0, str(SCRIPTS))

from watermark_profiles import (
    AUTO_PROFILE,
    CHROMA_ZIMAGE_PROFILE,
    DEFAULT_PROFILE,
    PROFILE_ADAPTIVE_POLISH,
    PROFILE_CHOICES,
    QWEN_ZIMAGE_OPENAI_STRENGTH,
    QWEN_ZIMAGE_PROFILE,
    REMOVAL_MODULES,
    SDXL_ZIMAGE_PROFILE,
    global_offload_supported,
    normalize_profile,
    resolve_adaptive_polish,
    resolve_auto_profile,
    resolve_effective_profile,
    resolve_seed,
    resolve_strength,
    strength_default_help,
)


def test_importing_the_profiles_module_loads_no_pixel_or_gpu_stack() -> None:
    """The module must stay data-only: no torch, numpy, or cv2 at import."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import watermark_profiles; print(sorted(sys.modules))",
        ],
        cwd=str(SCRIPTS),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "watermark_profiles" in proc.stdout
    for heavy in ("torch", "numpy", "cv2"):
        assert f"'{heavy}'" not in proc.stdout, f"{heavy} was imported at module load"


def test_profile_ids_are_closed_and_stable() -> None:
    assert DEFAULT_PROFILE == QWEN_ZIMAGE_PROFILE
    assert PROFILE_CHOICES == (
        QWEN_ZIMAGE_PROFILE,
        SDXL_ZIMAGE_PROFILE,
        CHROMA_ZIMAGE_PROFILE,
        AUTO_PROFILE,
    )
    assert REMOVAL_MODULES == ("torch", "diffusers", "diffsynth")
    assert set(PROFILE_ADAPTIVE_POLISH) == set(PROFILE_CHOICES)


def test_normalize_profile_accepts_the_underscore_spellings() -> None:
    assert normalize_profile("Qwen_ZImage") == QWEN_ZIMAGE_PROFILE
    assert normalize_profile(" sdxl_zimage ") == SDXL_ZIMAGE_PROFILE
    assert normalize_profile("chroma_zimage") == CHROMA_ZIMAGE_PROFILE
    # An unknown spelling is returned normalized, never silently defaulted.
    assert normalize_profile("mystery") == "mystery"


def test_auto_profile_resolves_from_the_vendor() -> None:
    assert resolve_auto_profile("google") == SDXL_ZIMAGE_PROFILE
    assert resolve_auto_profile("Microsoft") == CHROMA_ZIMAGE_PROFILE
    assert resolve_auto_profile("openai") == QWEN_ZIMAGE_PROFILE
    assert resolve_auto_profile(None) == QWEN_ZIMAGE_PROFILE
    assert resolve_auto_profile("") == QWEN_ZIMAGE_PROFILE


def test_profile_selection_changes_the_effective_profile() -> None:
    """The core selector contract: the same vendor, two configured profiles."""
    assert resolve_effective_profile(AUTO_PROFILE, "google") == SDXL_ZIMAGE_PROFILE
    assert resolve_effective_profile(SDXL_ZIMAGE_PROFILE, "google") == SDXL_ZIMAGE_PROFILE
    assert resolve_effective_profile(QWEN_ZIMAGE_PROFILE, "google") == QWEN_ZIMAGE_PROFILE
    # An explicit profile is never overridden by the vendor.
    assert resolve_effective_profile(CHROMA_ZIMAGE_PROFILE, "openai") == CHROMA_ZIMAGE_PROFILE


def test_resolve_strength_follows_the_selected_profile_and_vendor() -> None:
    sdxl_openai = resolve_strength(None, "openai", SDXL_ZIMAGE_PROFILE)
    qwen_openai = resolve_strength(None, "openai", QWEN_ZIMAGE_PROFILE, size=(1024, 1024))
    assert sdxl_openai != qwen_openai
    # An explicit strength always wins.
    assert resolve_strength(0.42, "openai", SDXL_ZIMAGE_PROFILE) == 0.42


def test_resolve_strength_is_total_for_every_profile() -> None:
    for profile in (QWEN_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE, CHROMA_ZIMAGE_PROFILE, AUTO_PROFILE):
        value = resolve_strength(None, "google", profile, size=(2048, 2048))
        assert 0.0 < value <= 1.0


def test_qwen_requires_a_size_for_the_unknown_cohort() -> None:
    # The qwen unknown-content cohort keys on image area, so it needs a size.
    with pytest.raises(ValueError, match="size"):
        resolve_strength(None, "unlisted-vendor", QWEN_ZIMAGE_PROFILE)


def test_seed_and_polish_and_offload_defaults_are_profile_data() -> None:
    assert resolve_seed(None) == 0
    assert resolve_seed(7) == 7
    assert resolve_adaptive_polish(None, SDXL_ZIMAGE_PROFILE) is True
    assert resolve_adaptive_polish(None, QWEN_ZIMAGE_PROFILE) is False
    assert resolve_adaptive_polish(True, QWEN_ZIMAGE_PROFILE) is True
    assert global_offload_supported(QWEN_ZIMAGE_PROFILE) is True
    assert global_offload_supported(CHROMA_ZIMAGE_PROFILE) is False
    assert str(QWEN_ZIMAGE_OPENAI_STRENGTH) in strength_default_help()
