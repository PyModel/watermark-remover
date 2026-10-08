# Ported from wiltodelta/remove-ai-watermarks src/remove_ai_watermarks/_internal/watermark_profiles.py,
# Apache-2.0; modified: the resolution-adaptive denoise curve is ported in from
# the reference tree's ``_internal/two_stage_pipeline.resolution_adaptive_denoise``
# so the module has no in-tree import at all, the extras vocabulary is this
# repo's (``detect``/``pixel``/``visible``), and the C2PA/IPTI vendor probe is
# function-local and fail-soft. Profile ids, aliases, per-vendor strength tables,
# the adaptive-polish map, and the offload set are verbatim.
"""Project-owned configuration for invisible-watermark regeneration profiles.

The profiles are pure data plus pure helpers: profile ids, the per-vendor
strength tables, and the resolvers that turn a configured profile into the
concrete engine for one image. Nothing here imports torch, numpy, or cv2 — the
CLI's availability gate and the remover's own precondition both read this
module, and a caller that only wants the profile list must not pay for a GPU
stack. Heavy imports, where a helper needs one, are function-local.
"""

from __future__ import annotations

# SDXL base is no longer a profile of its own, but it is still the global stage of
# sdxl-zimage, so the checkpoint id stays. Named for what it is rather than
# ``DEFAULT_MODEL_ID``: there is no user-selectable model any more, so "default"
# implied an override that every profile rejects.
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_CANNY_MODEL = "xinsir/controlnet-canny-sdxl-1.0"

QWEN_ZIMAGE_PROFILE = "qwen-zimage"
SDXL_ZIMAGE_PROFILE = "sdxl-zimage"
CHROMA_ZIMAGE_PROFILE = "chroma-zimage"
AUTO_PROFILE = "auto"
DEFAULT_PROFILE = QWEN_ZIMAGE_PROFILE
PROFILE_CHOICES = (QWEN_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE, CHROMA_ZIMAGE_PROFILE, AUTO_PROFILE)

# The modules a real removal run needs, and the extra that installs them. Both live
# here, in the only profile module that imports nothing heavy, because the CLI's
# availability gate and the remover's own precondition must agree: when they drifted,
# the CLI passed on a torch+diffusers environment and the run then died at the
# DiffSynth face stage, telling the user to install an extra that does not contain it.
REMOVAL_MODULES = ("torch", "diffusers", "diffsynth")
# Shell-quoted, because this string is printed as a command the user copy-pastes.
# Bare brackets are a glob in zsh (the macOS default shell): an unquoted
# ``pip install watermark-remover[qwen-zimage]`` dies with "no matches found"
# before pip ever runs -- another install hint that does not install.
INVISIBLE_EXTRA = "'watermark-remover[pixel]'"
# The pixel stack every visible-mark command needs. The default package ships without
# it -- Homebrew installs exactly that build -- so ``visible``, ``erase``, ``all`` and
# ``batch`` used to die on a bare ``ModuleNotFoundError: No module named 'cv2'``
# traceback on a first run that followed the project's own install instructions.
PIXELS_MODULES = ("cv2", "numpy")
VISIBLE_EXTRA = "'watermark-remover[visible]'"

# qwen-zimage's output already matches the input's detail level, so polishing it is a
# no-op at best. sdxl-zimage's global pass leaves the softer output the polish exists
# for. This is per-profile data, not a CLI concern: the flag defaults to None so that
# "the user did not choose" stays a value rather than an inference from Click state.
PROFILE_ADAPTIVE_POLISH = {
    QWEN_ZIMAGE_PROFILE: False,
    SDXL_ZIMAGE_PROFILE: True,
    CHROMA_ZIMAGE_PROFILE: False,
    # This fallback applies only before a vendor is known. The engine resolves
    # auto first so Google-to-SDXL receives SDXL's enabled default.
    AUTO_PROFILE: False,
}

# Which profiles let ``--cpu-offload`` reach their GLOBAL stack. The face stage
# honours the flag everywhere, because residency lives on the shared base, but the
# global stack is each profile's own: only qwen-zimage streams it. chroma-zimage and
# sdxl-zimage load theirs with a plain ``.to(device)``, so on a card below the
# face-stage residency floor the flag has nothing left to change and the run behaves
# as if it were absent. Data rather than a subclass attribute, because the caller
# that must warn cannot import a profile module without pulling in Diffusers.
GLOBAL_OFFLOAD_PROFILES = frozenset({QWEN_ZIMAGE_PROFILE})

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_PATTERN = "sdxl_lightning_4step_lora.safetensors"

# All profiles are certified at a fixed seed because SynthID removal near the
# strength floor is seed-dependent. The step count and CFG are not settable at all --
# each stage owns them.
PROFILE_SEED = 0

# sdxl-zimage runs the qwen-zimage recipe on an SDXL global stage, and strength is
# architecture-bound. The source-fresh 2026-09-15 calibration found Google CJK
# detectable through 0.45 and clean at 0.50. OpenAI is the easier oracle: the profile
# cleared openai.com/verify at 0.1102, so 0.15 sits above what was verified rather
# than on it. Unknown follows Google, the stricter measured cohort. Unlike
# qwen-zimage this is a flat vendor policy rather than a resolution curve, because
# flat values are what was measured.
SDXL_ZIMAGE_OPENAI_STRENGTH = 0.15
SDXL_ZIMAGE_GEMINI_STRENGTH = 0.50
SDXL_ZIMAGE_UNKNOWN_STRENGTH = SDXL_ZIMAGE_GEMINI_STRENGTH

# qwen-zimage keeps its resolution curve for unknown content, but measured vendor
# cohorts bypass it. A source-fresh 2026-09-15 sweep found Latin scenes and faces
# clean at 0.15 but CJK and Cyrillic text requiring 0.35, independently replicated
# clean on busy, flat-text, and face-plus-text CJK carriers.
QWEN_ZIMAGE_GOOGLE_STRENGTH = 0.35

# OpenAI source-fresh expansion (official Content Provenance API, 2026-09-21): n=8.
# Add one full observed cross-source spread to the worst clean boundary.
QWEN_ZIMAGE_OPENAI_STRENGTH = 0.15625

# Microsoft's public detector returned Inconclusive rather than a watermark-negative
# verdict; three valid Paint sources first cleared at 0.04125 / 0.055 / 0.095. Add
# one full observed cross-source spread to the worst clean boundary, rounded up.
QWEN_ZIMAGE_MICROSOFT_STRENGTH = 0.15

# Meta Muse Image's retained standalone IPTC AI tag is not a vendor-unique signature,
# so this floor requires an explicit vendor override. Worst clean boundary (0.06)
# plus one full observed spread (0.0525 - 0.015) = 0.0975, rounded up to 0.1.
QWEN_ZIMAGE_META_STRENGTH = 0.1

_QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR: dict[str, float] = {
    "google": QWEN_ZIMAGE_GOOGLE_STRENGTH,
    "openai": QWEN_ZIMAGE_OPENAI_STRENGTH,
    "microsoft": QWEN_ZIMAGE_MICROSOFT_STRENGTH,
    "meta": QWEN_ZIMAGE_META_STRENGTH,
}

# sdxl-zimage always picks its strength from the vendor. An unlisted or unknown SDXL
# vendor falls back to the Gemini value.
_SDXL_ZIMAGE_STRENGTH_BY_VENDOR: dict[str, float] = {
    "openai": SDXL_ZIMAGE_OPENAI_STRENGTH,
    "google": SDXL_ZIMAGE_GEMINI_STRENGTH,
}

# chroma-zimage floors, from the 2026-08-29/30 four-cohort oracle calibration, derived
# by the same worst-clean-plus-one-spread rule as the other profiles. OpenAI keeps a
# FLAT floor because the face and zero-face classes differ by 0.025 while the spread
# within each is wider; Microsoft clears below qwen; Google follows the source-fresh
# 2026-09-15 sweep; Meta's Content Seal scatters wider than qwen's.
CHROMA_ZIMAGE_OPENAI_STRENGTH = 0.20
CHROMA_ZIMAGE_MICROSOFT_STRENGTH = 0.125
CHROMA_ZIMAGE_GOOGLE_STRENGTH = 0.50
CHROMA_ZIMAGE_META_STRENGTH = 0.17
CHROMA_ZIMAGE_UNKNOWN_STRENGTH = CHROMA_ZIMAGE_GOOGLE_STRENGTH

_CHROMA_ZIMAGE_STRENGTH_BY_VENDOR: dict[str, float] = {
    "openai": CHROMA_ZIMAGE_OPENAI_STRENGTH,
    "google": CHROMA_ZIMAGE_GOOGLE_STRENGTH,
    "microsoft": CHROMA_ZIMAGE_MICROSOFT_STRENGTH,
    "meta": CHROMA_ZIMAGE_META_STRENGTH,
}
_ALIASES = {
    "qwen_zimage": QWEN_ZIMAGE_PROFILE,
    "sdxl_zimage": SDXL_ZIMAGE_PROFILE,
    "chroma_zimage": CHROMA_ZIMAGE_PROFILE,
    "auto": AUTO_PROFILE,
}

# The deterministic per-cohort selection policy: which profile wins on which vendor,
# from the 2026-08-29/30 four-cohort calibration. chroma-zimage has the lower measured
# Microsoft floor. SDXL wins Google on paired quality at the independently clean
# operating points. Qwen wins OpenAI and Meta; unknown stays on Qwen, the shipped
# conservative fallback.
_ENGINE_BY_VENDOR: dict[str, str] = {
    "google": SDXL_ZIMAGE_PROFILE,
    "microsoft": CHROMA_ZIMAGE_PROFILE,
}

# Both edges of the resolution-adaptive denoise curve, and the operator level that
# shifts it. These are the reference tree's own defaults for the qwen unknown-content
# cohort (``two_stage_pipeline.resolution_adaptive_denoise``).
_RDN_DENOISE_MIN = 0.08
_RDN_DENOISE_MAX = 0.15
_RDN_ADAPTIVE_LEVEL = 6
_RDN_AREA_LOW_MP = 0.30
_RDN_AREA_HIGH_MP = 3.40
_RDN_POSITIVE_DELTA_FRACTION = 0.285714 / 5.0
_RDN_NEGATIVE_DELTA_FRACTION = 0.257143 / 4.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolution_adaptive_denoise(
    width: int,
    height: int,
    *,
    adaptive_level: int = _RDN_ADAPTIVE_LEVEL,
    denoise_min: float = _RDN_DENOISE_MIN,
    denoise_max: float = _RDN_DENOISE_MAX,
) -> float:
    """Choose the calibrated global strength from image area and operator level.

    Ported verbatim (pure arithmetic) from the reference tree's two-stage pipeline,
    so the profile module stays import-free.
    """
    low, high = sorted((float(denoise_min), float(denoise_max)))
    megapixels = max(1.0, float(width) * float(height)) * 1e-6
    area_fraction = _clamp(
        (megapixels - _RDN_AREA_LOW_MP) / (_RDN_AREA_HIGH_MP - _RDN_AREA_LOW_MP), 0.0, 1.0
    )
    strength_range = high - low
    strength = low + strength_range * area_fraction
    level_delta = float(int(adaptive_level)) - 5.0
    if level_delta >= 0.0:
        strength += level_delta * strength_range * _RDN_POSITIVE_DELTA_FRACTION
    else:
        strength += level_delta * strength_range * _RDN_NEGATIVE_DELTA_FRACTION
    return _clamp(strength, 0.0001, 1.0)


def resolve_auto_profile(vendor: str | None) -> str:
    """Pick the engine for --pipeline auto from the provenance vendor."""
    return _ENGINE_BY_VENDOR.get((vendor or "").casefold(), QWEN_ZIMAGE_PROFILE)


def normalize_profile(profile: str) -> str:
    """Normalize spelling and resolve the underscore spellings."""
    value = profile.strip().casefold()
    return _ALIASES.get(value, value)


def resolve_effective_profile(profile: str, vendor: str | None) -> str:
    """Resolve a configured profile to the concrete engine for one image."""
    normalized = normalize_profile(profile)
    if normalized == AUTO_PROFILE:
        return resolve_auto_profile(vendor)
    return normalized


def resolve_seed(seed: int | None) -> int:
    """Keep every profile reproducible by default."""
    return PROFILE_SEED if seed is None else seed


def resolve_adaptive_polish(adaptive_polish: bool | None, pipeline: str) -> bool:
    """Return an explicit polish choice, or the profile's calibrated default."""
    if adaptive_polish is not None:
        return adaptive_polish
    return PROFILE_ADAPTIVE_POLISH.get(normalize_profile(pipeline), True)


def global_offload_supported(pipeline: str) -> bool:
    """Whether ``--cpu-offload`` reaches this profile's global stack.

    Per-image ``auto`` runs resolve before loading a stack. An explicit preload is
    the only unresolved ``auto`` load, and that path deliberately uses qwen-zimage
    as its fallback, so it has the same offload support as qwen-zimage.
    """
    profile = resolve_effective_profile(pipeline, None)
    return profile in GLOBAL_OFFLOAD_PROFILES


def strength_default_help() -> str:
    """Describe the live default policy without duplicating its values."""
    return (
        "profile-adaptive (qwen-zimage uses resolution-adaptive denoise, with a "
        f"flat OpenAI {QWEN_ZIMAGE_OPENAI_STRENGTH} / Google {QWEN_ZIMAGE_GOOGLE_STRENGTH} / "
        f"Microsoft InvisMark {QWEN_ZIMAGE_MICROSOFT_STRENGTH} / Meta Content Seal "
        f"{QWEN_ZIMAGE_META_STRENGTH} floors; sdxl-zimage "
        f"uses OpenAI {SDXL_ZIMAGE_OPENAI_STRENGTH} / Google {SDXL_ZIMAGE_GEMINI_STRENGTH} / "
        f"unknown {SDXL_ZIMAGE_UNKNOWN_STRENGTH}; chroma-zimage uses OpenAI "
        f"{CHROMA_ZIMAGE_OPENAI_STRENGTH} / Microsoft {CHROMA_ZIMAGE_MICROSOFT_STRENGTH} / "
        f"Google {CHROMA_ZIMAGE_GOOGLE_STRENGTH} / Meta {CHROMA_ZIMAGE_META_STRENGTH}, "
        "from the C2PA issuer)"
    )


def resolve_strength(
    strength: float | None,
    vendor: str | None = None,
    pipeline: str | None = None,
    *,
    size: tuple[int, int] | None = None,
) -> float:
    """Resolve a user override or the calibrated policy for a profile and vendor.

    Total by design. qwen-zimage picks its strength from image area rather than
    from the vendor, so it needs ``size``; returning ``None`` for it instead would
    push that branch onto every caller and move one of the two strength policies
    outside this module. Measured vendor exceptions bypass the area curve: OpenAI
    and Microsoft use one additional observed cross-source spread over their worst
    clean boundaries, while Google uses a separately repeated cross-source
    candidate. See the constants' comments for the exact derivations.
    """
    if strength is not None:
        return strength
    normalized = resolve_effective_profile(pipeline or "", vendor)
    if normalized == SDXL_ZIMAGE_PROFILE:
        return _SDXL_ZIMAGE_STRENGTH_BY_VENDOR.get(
            (vendor or "").casefold(), SDXL_ZIMAGE_UNKNOWN_STRENGTH
        )
    if normalized == CHROMA_ZIMAGE_PROFILE:
        vendor_key = (vendor or "").casefold()
        return _CHROMA_ZIMAGE_STRENGTH_BY_VENDOR.get(vendor_key, CHROMA_ZIMAGE_UNKNOWN_STRENGTH)
    if size is None:
        raise ValueError("qwen-zimage resolves strength from image area, so size is required")
    vendor_strength = _QWEN_ZIMAGE_FLAT_STRENGTH_BY_VENDOR.get((vendor or "").casefold())
    if vendor_strength is not None:
        return vendor_strength
    return _resolution_adaptive_denoise(*size)
