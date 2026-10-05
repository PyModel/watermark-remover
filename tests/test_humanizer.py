"""Vectors for the ported humanizer (grain / aberration / unsharp / adaptive polish).

Ported from wiltodelta/remove-ai-watermarks ``humanizer.py``, Apache-2.0;
modified: grain is seeded (upstream used the unseeded global RNG), and the
composable stage order helper is covered here rather than upstream's engine.
"""

from __future__ import annotations

import base64
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


# --- wiring: --humanize on the image clean path -----------------------------------

# Fixture pinned at ticket base 241d5b8: a 24x16 gradient PNG and the output of
# the visible clean (box 4,2,3,3, texture backend, dilation 1) BEFORE this
# ticket existed. The default-off pipeline must reproduce those bytes exactly.
_PIN_INPUT_B64 = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAQCAIAAACDRijCAAAEDUlEQVR42gXBf4jQVBwA8O9VL4Kh9DjdkXDgdqgcLg9vXBMuyxsvxGBayNMd/bg/psOKG10Qa8pxf9yKHV4MoYM2K5AY5uBgCRbXg/4wnMItapgc0t5foeuHTBgeJDTW5wMASAAkApIAKYA0QDogA5AJyALkAPIA+YBCQDGgBFAGiAHKAZWAKkA1oBZQBwgAIwE/I2JBwlsVjDW8TcdDBt5h4mEL73TwiId3+3g0xEqMxxI8nuEJhg/keLLEL1V4qsakxYc7/CqAjARZEGUsyUOKPKzJI7o8ashjpjxhyZOOPOXJh33ZCOXjsTydyDOZfJrJ7+XyXCm7lTxfy4utvNTJIYCKBHWrqA5J6k5FHdXUcV2dNFRiqoalnnDUGU8946tzoXo2VhcTdTlTV5j6Va5eLtWsUtdq9XqrrnfqbwAECQSLZFgiowqZ0MiUTgyDTJvktEXmHDLvkSWfrITkUkxWE7KWkRuMlDnhJfmrIps1GWjJlo48B0CRQLeJdESi4wqd0uhrOp0xqGPSeYsuO/SiR1OfroX0Vkw3Eno/o5uMPpXTwZLKFd1f00MtPdbRtwFsJNhDoj0q2ZOKbWj2jG7PGfaiaa9Y9mXHXvPsdd/mof0wtgcSezCzdzFby+0jpf1GZc/W9kJrX+jsrwFcJLg7RHdMconiTmuuo7uLhvu56a5a7nXH3fDcB747ELpi7O5N3EOZS5n7bu4ulO5nlXuldn9s3dud+ydAgIRgWAwmpMBQgtNaMK8HK0awagY3rIA7waYXbPGDXWFwMA5oEsxmwccs+DIPrpXBz1Vwrw66NtjeBc8DREiIdorRpBSdUKI5LVrWo8tGdN2MuBX960SDXrTPj46E0ak4WkiiOIuusejXPPqnjJ6uIqmOXmyjk130AUCKhHRETKekdEZJ57X0op6uGemGmW5a6aCT7vfSY346G6bn4/RKkt7M0nssfTJPpTJ9uUrfqtNzbRp16fcADAlst8gOS+yMwpY0lups3WAPTLbFYvscdsxj7/vsQsiuxux2wh5lbDtjL+TsZMk+qlhUsx9a9nvH/gMokFCMioUhFXNKsaIVa3rBjWLALHZZxRGnmPWKC35xLSzuxkWXFFJWvMKKd/Li07L4tiru1MXjthjuiikAjgSuiPy4xM8q/JLGb+n8ocFFkx+0+CmHn/f4VZ/fDTnEfE/Cj2b8Q8a/yPlPJf+74rjmB1o+0/FPABokNGNiMy01i0qzqjUbejNgNHvNhlrNgtNc8ZrbftOFzZ64eT1pzmVNwppf8uZx2YxUjVE3bttc6pp1gB4J/bjYz0j9stKvaf19vR80+kNmP2v1sdPf9PpHfi+F/dG4P5f032T9HdY/kfdjZf9m1S/V/Xdt/0fXP/s/vdcUv3qe8pIAAAAASUVORK5CYII="
_PIN_EXPECTED_B64 = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAQCAIAAACDRijCAAAD3ElEQVR42l3LXWjeVBgA4LfqESFMDHUZCoUlpRtjcWUL8xvU6foRKRWyIiNbij+9yBqm0mAFOWal9KJRUlYJAwueTIUioe5AIQ6mfB7worJs0IiGOsow50q2+DMyCCs4MMT7PvcPACABkARIBqQC6gDqAjIAWYBsQC4gD5APKAQUAYoBJYAYoBRQDqgAVAKqATWAAEQkiE9JorDrWOKALe53xUFPPOCLh0JRjcThWDyWiMeZeCIVR3Lx5UIcLUW9Fsca8TUABQmKICnirmMpx21lxFVGPWXMV4xQORMpk7EylSjTTHkvVWZzBRfKfKks1spSo4QAGhK0pyVt365jaYatnXW1KU+74GuzoXYx0hZjbTnRVpj2Vaqt5VpSaL1S26i1zUb7DUBHgi5K+sCuY+nTtj7r6vOevuTrK6G+Gunrsd5L9BtMz1Od5/pfhb5T6n21vqfRnwMwkWA+K5mDu45lztvmsmte8Uzqm73QvBWZ27F5LzF3mPlEavbnplKYR0vzVG1ONObbAA4SnH2Sc0h2RlTH6DhTXWfWcBYtZ8V21lyn5zmbvsND50Hk9MVOf+IMMaeTOuO580bhzJTOQu1cbpyvATAS8PMSHpaxruLJDna7eNHAn1t43cYbLt728H0f94VYivDhGJ9KsMnwuyleyPFnBb5a4h9rvNXgPwECJAQDUnBcDgw1mO4E891gxQjWreCGHXA32PGCPX4wFAYno8CMg5kk+JgFX6bB9Tz4uQjulkFTB3ub4AUAggSyXyIjMjmrktkOWe6SNYNsWITb5F+X9HvkiE/GQ3I+IgsxiRJynZFfU/JPTp4siFySl2pyriEfAFAk0EGJjsp0SqXzHXqlS3sG3bbojk37XXrUoxM+nQnppYhejenNhN5l9PGUyjl9paBvlXSupqSh3wMwJLADEhuT2QWVLXUY7bJNg9232B6bHXHZhMfe99nlkF2L2FbMHiZsL2Mvpuxczj4qGCnZDzX7vWH/AWRIyA5JmSFns2q20sl63YwbWZ+VDdnZuJvNeNllP7seZneirIkzOcleZdk7afZpnn1bZLfL7FGdDTTZKABHAlclfkbmF1W+2uG3uvyBwSWLn7T5eZdf8vg1n98JOUT8YMxPJ/xDxr9I+U85/7vgYslP1Hyq4Z8AVEiohqVqUq4W1Wq9U213qz6jOmxVpl0tuNVVr9ryqyasDkbV63E1l1Qxq35Jq0d5NVhURlnhulptqk2AFgntMamdkttlte112nvdtt9oT1ntjN1GbnvTax/6rRy2p6N2Lm6/SdrbrH0sbYfz9s2iXSrb7+r2j6Z95n+JLx0+zuXS9wAAAABJRU5ErkJggg=="


def _pin_fixture(tmp_path):
    from clean_asset import CleanPlan
    from morphomod import VisiblePlan

    source = tmp_path / "input.png"
    source.write_bytes(base64.b64decode(_PIN_INPUT_B64))
    plan = CleanPlan(
        forced_kind="image",
        visible=VisiblePlan(
            box=(4, 2, 3, 3),
            backend="texture",
            mask_output=tmp_path / "mask.pgm",
            dilation_radius=1,
        ),
    )
    return source, plan


def test_humanize_default_off_is_byte_identical_to_the_base_pipeline(tmp_path):
    """The pinned guarantee: without --humanize, output bytes are the base pipeline's."""
    from clean_asset import clean_asset

    source, plan = _pin_fixture(tmp_path)
    dest = tmp_path / "out.png"
    result = clean_asset(source, dest, plan)
    assert dest.read_bytes() == base64.b64decode(_PIN_EXPECTED_B64)
    assert "humanize" not in result.to_dict()


def test_humanize_stage_applies_and_is_reported(tmp_path):
    np = pytest.importorskip("numpy")
    import cv2
    from clean_asset import CleanPlan, clean_asset
    from morphomod import VisiblePlan

    source, _ = _pin_fixture(tmp_path)
    dest_off = tmp_path / "off.png"
    off_plan = CleanPlan(
        forced_kind="image",
        visible=VisiblePlan(
            box=(4, 2, 3, 3), backend="texture", mask_output=tmp_path / "off.pgm", dilation_radius=1
        ),
    )
    clean_asset(source, dest_off, off_plan)

    dest_on = tmp_path / "on.png"
    on_plan = CleanPlan(
        forced_kind="image",
        visible=VisiblePlan(
            box=(4, 2, 3, 3), backend="texture", mask_output=tmp_path / "on.pgm", dilation_radius=1
        ),
        humanize=True,
        humanize_seed=9,
    )
    result = clean_asset(source, dest_on, on_plan).to_dict()
    assert result["humanize"]["status"] == "applied"
    assert result["humanize"]["stages"] == ["unsharp", "polish", "grain"]
    assert result["humanize"]["seed"] == 9
    # The stage visibly changed pixels against the default-off run.
    off = cv2.imread(str(dest_off))
    on = cv2.imread(str(dest_on))
    assert off is not None and on is not None
    assert not np.array_equal(off, on)


def test_humanize_is_refused_for_non_image_assets():
    from clean_request import CleanRequest, build_clean_plan

    with pytest.raises(ValueError, match="--humanize"):
        build_clean_plan(CleanRequest(humanize=True), Path("out.txt"), "text")


def test_humanize_requires_png_output(tmp_path):
    """A JPEG input (no visible plan) reaches the stage with non-PNG output and
    must refuse rather than silently re-encode lossy pixels."""
    cv2 = pytest.importorskip("cv2")

    from clean_asset import CleanPlan, clean_asset

    ok, encoded = cv2.imencode(".jpg", __import__("numpy").zeros((8, 8, 3), dtype="uint8"))
    assert ok
    source = tmp_path / "input.jpg"
    source.write_bytes(encoded.tobytes())
    plan = CleanPlan(forced_kind="image", humanize=True)
    with pytest.raises(ValueError, match="PNG"):
        clean_asset(source, tmp_path / "out.jpg", plan)


def test_clean_file_cli_accepts_humanize_flag(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "remove-ai-marks"
        / "scripts"
        / "clean_file.py"
    )
    source, _ = _pin_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            "-o",
            str(tmp_path / "out.png"),
            "--humanize",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PYTHONPATH": str(
                Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
            ),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert result.returncode == 0, result.stderr
    assert '"humanize"' in result.stdout


def test_clean_file_without_humanize_stays_default_off(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "remove-ai-marks"
        / "scripts"
        / "clean_file.py"
    )
    source, _ = _pin_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, str(script), str(source), "-o", str(tmp_path / "out.png"), "--json"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PYTHONPATH": str(
                Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
            ),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert result.returncode == 0, result.stderr
    assert '"humanize"' not in result.stdout


def test_humanize_without_pixel_deps_is_labeled_unavailable(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "remove-ai-marks"
        / "scripts"
        / "clean_file.py"
    )
    scripts = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
    source, _ = _pin_fixture(tmp_path)
    stubs = tmp_path / "blocked-stubs"
    stubs.mkdir()
    for name in ("numpy", "cv2", "PIL"):
        (stubs / f"{name}.py").write_text(
            f"raise ModuleNotFoundError('stub-blocked: {name}')\n", encoding="utf-8"
        )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            "-o",
            str(tmp_path / "out.png"),
            "--humanize",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": f"{stubs}{__import__('os').pathsep}{scripts}", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert '"unavailable"' in result.stdout
    assert "watermark-remover[pixel]" in result.stdout
