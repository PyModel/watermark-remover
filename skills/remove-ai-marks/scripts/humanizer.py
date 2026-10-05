"""Post-processing filters for the cleaned output.

Ported from wiltodelta/remove-ai-watermarks ``humanizer.py``, Apache-2.0;
modified: grain is seeded (upstream drew from the unseeded global RNG, which
made a ``--humanize`` run unreproducible), and :func:`humanize_stage`
composes the three phases in the order upstream's engine applies them so a
caller does not re-derive the order (and its reasoning) at every call site.

``apply_analog_humanizer`` injects film grain and chromatic aberration to
reduce overly uniform digital surfaces; ``unsharp_mask`` counters the soft,
over-smoothed look that a fill pass leaves behind (itself a common
"this is AI" tell).

numpy/cv2 are imported lazily inside the functions so importing this module
loads no third-party code; a call without the pixel extra installed raises
the labeled :class:`~image_io.PixelDependencyUnavailable` error naming
``watermark-remover[pixel]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from image_io import PixelDependencyUnavailable, _import_pixel

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# ── Adaptive polish (target the input's detail level; spare text) ──────────────
# A capped unsharp scaled to the sharpness deficit, then edge-masked grain to close
# the rest -- tunable constants (upstream-calibrated).
_ADAPTIVE_MAX_UNSHARP = 1.0
_ADAPTIVE_UNSHARP_GAIN = 0.4  # unsharp amount per unit of (deficit - 1), before the cap
_ADAPTIVE_MAX_GRAIN = 8.0
_MASK_EDGE_PERCENTILE = 85.0  # local-energy percentile above which a pixel is an "edge/text"
_MASK_EDGE_DILATE = 5  # grow the edge mask so grain is suppressed in a margin around text
_MASK_GAMMA = 2.0  # push the smooth weight toward 0 except in genuinely flat areas


def apply_analog_humanizer(
    image: NDArray,
    grain_intensity: float = 4.0,
    chromatic_shift: int = 1,
    seed: int | None = None,
) -> NDArray:
    """Apply shared-luminance grain and a small lateral color offset.

    Args:
        image: BGR image as numpy array (uint8).
        grain_intensity: Standard deviation of the Gaussian noise (film grain).
        chromatic_shift: Number of pixels to shift the red/blue color channels.
        seed: Optional RNG seed; the same seed reproduces the same grain
            (upstream relied on the unseeded global RNG).

    Returns:
        Humanized BGR image.
    """
    np = _import_pixel("numpy")
    cv2 = _import_pixel("cv2")

    # Ensure image is BGR
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image.copy()

    # Translate the outer color channels without circular edge wrapping.
    b, g, r = cv2.split(image)
    shift = min(max(0, chromatic_shift), max(0, image.shape[1] - 1))
    if shift > 0:
        shifted_b = np.empty_like(b)
        shifted_b[:, :shift] = b[:, :1]
        shifted_b[:, shift:] = b[:, :-shift]
        b = shifted_b

        shifted_r = np.empty_like(r)
        shifted_r[:, :-shift] = r[:, shift:]
        shifted_r[:, -shift:] = r[:, -1:]
        r = shifted_r

    merged = cv2.merge((b, g, r))

    if grain_intensity > 0:
        rng = np.random.default_rng(seed)
        img_f = merged.astype(np.float32)
        noise = rng.normal(0, grain_intensity, img_f.shape).astype(np.float32)
        humanized = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    else:
        humanized = merged

    return humanized


def unsharp_mask(image: NDArray, amount: float = 0.5, sigma: float = 1.0) -> NDArray:
    """Sharpen via unsharp masking: ``out = image + amount * (image - blur(image))``.

    Counters the soft, over-smoothed look of a fill pass, which reads as an AI
    tell. ``amount`` 0 = no-op (returns an unchanged copy); ~0.5-0.8 is a safe
    range -- higher risks bright edge halos that are their own artifact.
    ``sigma`` is the Gaussian radius of the unsharp kernel.

    Args:
        image: BGR image as numpy array (uint8).
        amount: Sharpening strength (0 = off).
        sigma: Gaussian blur sigma for the unsharp kernel.

    Returns:
        Sharpened BGR image (uint8).
    """
    np = _import_pixel("numpy")
    cv2 = _import_pixel("cv2")
    if amount <= 0.0:
        return image.copy()
    img_f = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img_f, 1.0 + amount, blurred, -amount, 0.0)
    return np.clip(np.rint(sharpened, out=sharpened), 0, 255).astype(np.uint8)


def _to_gray(image: NDArray) -> NDArray:
    """Single-channel grayscale; passes a 2D (already-gray) input through unchanged."""
    cv2 = _import_pixel("cv2")
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _laplacian_variance(image: NDArray) -> float:
    """Variance of the Laplacian -- a cheap proxy for high-frequency detail/sharpness."""
    cv2 = _import_pixel("cv2")
    return float(cv2.Laplacian(_to_gray(image), cv2.CV_64F).var())


def _smooth_grain_mask(image: NDArray) -> NDArray:
    """Per-pixel weight ~1 in flat/smooth regions, ~0 over text and hard edges.

    Grain in smooth ("AI-plastic") regions reads as natural sensor noise; grain
    over text/edges just speckles them, so this masks grain to the smooth
    regions only.
    """
    np = _import_pixel("numpy")
    cv2 = _import_pixel("cv2")
    energy = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(_to_gray(image).astype(np.float32), cv2.CV_32F)), (0, 0), sigmaX=2.0
    )
    thr = float(np.percentile(energy, _MASK_EDGE_PERCENTILE))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MASK_EDGE_DILATE, _MASK_EDGE_DILATE))
    edges = cv2.dilate((energy > thr).astype(np.uint8), kernel)
    mask = np.clip(1.0 - energy / (thr + 1e-6), 0.0, 1.0) ** _MASK_GAMMA
    mask[edges > 0] = 0.0
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)


def adaptive_polish(
    image: NDArray,
    reference: NDArray,
    seed: int | None = None,
    on_skip: Callable[[str], None] | None = None,
) -> NDArray:
    """Restore the detail level of ``reference`` in a softened ``image``, sparing text.

    A fill pass leaves an over-smoothed "AI-plastic" look, worst on photo/face
    regions. This targets the reference's Laplacian variance (the input's
    detail level): a capped unsharp scaled to the deficit, then edge-masked
    grain (smooth regions only) calibrated to close the remaining gap.
    **Self-limiting on text/graphics** -- they are already high-frequency, so
    the deficit is small and almost no polish is applied (text legibility is a
    generation-side concern, not a filter one). No-op when the image already
    meets the reference's detail level.

    That no-op used to be silent, which made ``--humanize`` look like it
    composed with the polish when it had in fact replaced it: grain raises the
    measured detail level past the reference's, so the polish sees no deficit
    and returns the grain untouched. ``on_skip`` reports it instead. (The
    stage runs the polish BEFORE the humanizer, so the two compose; a caller
    driving the helpers directly can still reach the no-op, and an output
    that is already sharper than its source reaches it whatever the order.)

    Args:
        image: the cleaned BGR output (uint8).
        reference: the original input BGR at the same resolution (the detail target).
        seed: optional RNG seed for reproducible grain.
        on_skip: optional callback invoked with a one-line reason when the polish
            self-limits to nothing, so the skip is not silent.

    Returns:
        Polished BGR image (uint8).
    """
    np = _import_pixel("numpy")
    target = _laplacian_variance(reference)
    current = _laplacian_variance(image)
    if target <= 0.0 or current >= target:
        if on_skip is not None:
            on_skip(
                "Adaptive polish: nothing to close -- the image already meets the source's "
                f"detail level (Laplacian variance {current:.1f} vs {target:.1f})."
            )
        return image.copy()

    deficit = target / max(current, 1.0)
    amount = min(_ADAPTIVE_MAX_UNSHARP, _ADAPTIVE_UNSHARP_GAIN * (deficit - 1.0))
    work = unsharp_mask(image, amount=amount, sigma=1.2) if amount > 0.0 else image.copy()
    if _laplacian_variance(work) >= target:
        return work

    # Calibrate the grain sigma by a short search: its lap-var contribution depends on
    # the per-pixel mask (no closed form), so step it up until the target is met. A few
    # full-image Laplacians here are negligible against the fill pass that precedes.
    mask = _smooth_grain_mask(work)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, work.shape[:2]).astype(np.float32) * mask
    best = work
    sigma = 2.0
    while sigma <= _ADAPTIVE_MAX_GRAIN:
        best = np.clip(
            work.astype(np.float32) + (noise * sigma)[:, :, np.newaxis], 0.0, 255.0
        ).astype(np.uint8)
        if _laplacian_variance(best) >= target:
            break
        sigma += 1.0
    return best


def humanize_stage(
    image: NDArray,
    reference: NDArray,
    seed: int | None = None,
    grain_intensity: float = 4.0,
    chromatic_shift: int = 1,
    unsharp_amount: float = 0.5,
    on_event: Callable[[str], None] | None = None,
) -> tuple[NDArray, dict[str, object]]:
    """Run the full ``--humanize`` post-stage on one decoded BGR image.

    The order is the upstream engine's: unsharp, adaptive polish, then grain.
    Grain comes LAST because the polish measures the image it is given against
    the reference's detail level, so grain applied first is read as detail
    that is already there (see :func:`adaptive_polish`).

    Returns:
        ``(humanized_image, report)`` where *report* names the phases applied
        (``stages``), the grain parameters, and a ``polish_skipped`` reason
        when the polish self-limited (never silent).
    """
    work = unsharp_mask(image, amount=unsharp_amount)
    stages = ["unsharp"]
    skip: list[str] = []
    work = adaptive_polish(work, reference, seed=seed, on_skip=skip.append)
    stages.append("polish")
    work = apply_analog_humanizer(
        work, grain_intensity=grain_intensity, chromatic_shift=chromatic_shift, seed=seed
    )
    stages.append("grain")
    if on_event is not None:
        on_event(f"Humanized: {', '.join(stages)}")
    return work, {
        "stages": stages,
        "grain_intensity": grain_intensity,
        "chromatic_shift": chromatic_shift,
        "polish_skipped": skip[0] if skip else None,
    }


__all__ = [
    "PixelDependencyUnavailable",
    "adaptive_polish",
    "apply_analog_humanizer",
    "humanize_stage",
    "unsharp_mask",
]
