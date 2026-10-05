"""Vectors for the ported humanizer (grain / aberration / unsharp / adaptive polish).

Ported from wiltodelta/remove-ai-watermarks ``humanizer.py``, Apache-2.0;
modified: grain is seeded (upstream used the unseeded global RNG), and the
composable stage order helper is covered here rather than upstream's engine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts")
)

import humanizer


@pytest.fixture()
def gradient():
    np = pytest.importorskip("numpy")
    # Deterministic 32x24 BGR ramp with a flat patch: sharp enough to measure,
    # flat enough to smooth.
    h, w = 24, 32
    b = (np.tile(np.arange(w, dtype=np.int16) * 7, (h, 1)) % 256).astype(np.uint8)
    g = (np.tile((np.arange(h, dtype=np.int16) * 11)[:, None], (1, w)) % 256).astype(np.uint8)
    r = ((b.astype(np.int16) + g) % 256).astype(np.uint8)
    image = np.dstack([b, g, r])
    image[4:10, 4:12] = 128
    return image


# --- apply_analog_humanizer: grain + chromatic aberration ------------------------


def test_humanizer_passes_non_color_input_through_unchanged(gradient):
    np = pytest.importorskip("numpy")
    gray = gradient[:, :, 0].copy()
    out = humanizer.apply_analog_humanizer(gray, grain_intensity=8.0, seed=7)
    assert np.array_equal(out, gray)


def test_chromatic_shift_moves_blue_right_and_red_left_without_wrapping(gradient):
    np = pytest.importorskip("numpy")
    out = humanizer.apply_analog_humanizer(gradient, grain_intensity=0.0, chromatic_shift=2)
    b_in, _, r_in = gradient[:, :, 0], gradient[:, :, 1], gradient[:, :, 2]
    # Blue shifted right: column x takes column x-2's value; edge replicates.
    assert np.array_equal(out[:, 2:, 0], b_in[:, :-2])
    assert np.array_equal(out[:, :2, 0], np.repeat(b_in[:, :1], 2, axis=1))
    # Red shifted left: column x takes column x+2's value; edge replicates.
    assert np.array_equal(out[:, :-2, 2], r_in[:, 2:])
    assert np.array_equal(out[:, -2:, 2], np.repeat(r_in[:, -1:], 2, axis=1))
    # Green untouched.
    assert np.array_equal(out[:, :, 1], gradient[:, :, 1])


def test_seeded_grain_is_reproducible_and_changes_pixels(gradient):
    np = pytest.importorskip("numpy")
    a = humanizer.apply_analog_humanizer(gradient, grain_intensity=6.0, seed=42)
    b = humanizer.apply_analog_humanizer(gradient, grain_intensity=6.0, seed=42)
    c = humanizer.apply_analog_humanizer(gradient, grain_intensity=6.0, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, gradient)
    assert not np.array_equal(a, c)


def test_zero_grain_and_zero_shift_is_the_identity(gradient):
    np = pytest.importorskip("numpy")
    out = humanizer.apply_analog_humanizer(gradient, grain_intensity=0.0, chromatic_shift=0)
    assert np.array_equal(out, gradient)


# --- unsharp_mask -----------------------------------------------------------------


def test_unsharp_zero_amount_is_an_unchanged_copy(gradient):
    np = pytest.importorskip("numpy")
    out = humanizer.unsharp_mask(gradient, amount=0.0)
    assert np.array_equal(out, gradient)
    assert out is not gradient


def test_unsharp_adds_high_frequency_energy(gradient):
    pytest.importorskip("numpy")
    import cv2

    def lap_var(img):
        return float(cv2.Laplacian(humanizer._to_gray(img), cv2.CV_64F).var())

    out = humanizer.unsharp_mask(gradient, amount=0.8, sigma=1.0)
    assert lap_var(out) > lap_var(gradient)


# --- adaptive_polish ---------------------------------------------------------------


def test_polish_noop_when_image_meets_reference_detail(gradient):
    np = pytest.importorskip("numpy")
    skips: list[str] = []
    out = humanizer.adaptive_polish(gradient, gradient, seed=1, on_skip=skips.append)
    assert np.array_equal(out, gradient)
    assert len(skips) == 1 and "already meets" in skips[0]


def test_polish_sharpens_a_softened_image_toward_the_reference(gradient):
    pytest.importorskip("numpy")
    import cv2

    def lap_var(img):
        return float(cv2.Laplacian(humanizer._to_gray(img), cv2.CV_64F).var())

    soft = cv2.GaussianBlur(gradient, (0, 0), sigmaX=2.0)
    reference = gradient
    assert lap_var(soft) < lap_var(reference)
    out = humanizer.adaptive_polish(soft, reference, seed=5)
    assert lap_var(out) > lap_var(soft)
    # The polish never overshoots into invented sharpness beyond the target
    # plus the grain search's final step.
    assert lap_var(out) < lap_var(reference) + 200.0


def test_polish_seeded_is_reproducible(gradient):
    np = pytest.importorskip("numpy")
    import cv2

    soft = cv2.GaussianBlur(gradient, (0, 0), sigmaX=2.0)
    a = humanizer.adaptive_polish(soft, gradient, seed=11)
    b = humanizer.adaptive_polish(soft, gradient, seed=11)
    assert np.array_equal(a, b)


# --- stage order (the composition the CLI flag drives) -----------------------------


def test_stage_applies_all_three_phases_and_reports(gradient):
    np = pytest.importorskip("numpy")
    out, report = humanizer.humanize_stage(gradient, gradient, seed=3)
    assert not np.array_equal(out, gradient)
    assert report["stages"] == ["unsharp", "polish", "grain"]
    assert report["grain_intensity"] == pytest.approx(4.0)
    assert report["chromatic_shift"] == 1


def test_stage_polish_skip_is_reported_not_silent(gradient):
    pytest.importorskip("numpy")
    _, report = humanizer.humanize_stage(gradient, gradient, seed=3)
    assert report["polish_skipped"] is not None


# --- pixel extra labeled-unavailable ------------------------------------------------


def _run_blocked(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    stubs = tmp_path / "blocked-stubs"
    stubs.mkdir()
    for name in ("numpy", "cv2", "PIL"):
        (stubs / f"{name}.py").write_text(
            f"raise ModuleNotFoundError('stub-blocked: {name}')\n", encoding="utf-8"
        )
    scripts = str(Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts")
    return subprocess.run(
        [sys.executable, "-c", source],
        input="",
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PYTHONPATH": f"{stubs}{__import__('os').pathsep}{scripts}", "PATH": "/usr/bin:/bin"},
        check=False,
    )


def test_humanizer_without_numpy_is_labeled_unavailable(tmp_path):
    probe = (
        "import humanizer\n"
        "try:\n"
        "    humanizer.unsharp_mask(object(), amount=0.5)\n"
        "except humanizer.PixelDependencyUnavailable as error:\n"
        "    print('LABELED:', error)\n"
        "else:\n"
        "    print('NO-RAISE')\n"
    )
    result = _run_blocked(probe, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "LABELED:" in result.stdout and "watermark-remover[pixel]" in result.stdout


def test_humanize_stage_without_cv2_is_labeled_unavailable(tmp_path):
    probe = (
        "import humanizer\n"
        "try:\n"
        "    humanizer.humanize_stage(object(), object(), seed=1)\n"
        "except humanizer.PixelDependencyUnavailable as error:\n"
        "    print('LABELED:', error)\n"
        "else:\n"
        "    print('NO-RAISE')\n"
    )
    result = _run_blocked(probe, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "LABELED:" in result.stdout and "watermark-remover[pixel]" in result.stdout
