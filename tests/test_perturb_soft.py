"""Tests for perturb_text (P8) and inspect_soft_binding (P7)."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import inspect_soft_binding as soft_binding_module
from inspect_soft_binding import inspect_soft_binding
from perturb_text import perturb_text
from text_unicode import clean_text

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Watermark signals hide inside ordinary token choices."
) * 4


def test_perturb_zero_width_is_layer_a_reversible():
    out, stats = perturb_text(SAMPLE, mode="zero-width", strength=0.3, seed=42)
    assert stats["changed"] > 0
    assert stats["reversible_by_layer_a"] is True
    restored, _ = clean_text(out)
    assert restored == SAMPLE


def test_perturb_space_swap_is_layer_a_reversible():
    out, stats = perturb_text(SAMPLE, mode="space-swap", strength=0.5, seed=7)
    assert stats["changed"] > 0
    restored, _ = clean_text(out)
    assert restored == SAMPLE


def test_perturb_deterministic_with_seed():
    a, _ = perturb_text(SAMPLE, mode="zero-width", strength=0.2, seed=1)
    b, _ = perturb_text(SAMPLE, mode="zero-width", strength=0.2, seed=1)
    c, _ = perturb_text(SAMPLE, mode="zero-width", strength=0.2, seed=2)
    assert a == b
    assert a != c


def test_perturb_zero_strength_noop():
    out, stats = perturb_text(SAMPLE, mode="confusable", strength=0.0, seed=1)
    assert out == SAMPLE
    assert stats["changed"] == 0


def test_perturb_confusable_not_reversible():
    out, stats = perturb_text(SAMPLE, mode="confusable", strength=1.0, seed=3)
    assert stats["reversible_by_layer_a"] is False
    assert out != SAMPLE
    assert "а" in out  # Cyrillic a


def test_clean_file_character_perturbation(tmp_path: Path):
    src = tmp_path / "draft.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    dest = tmp_path / "draft.out.txt"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(src),
            "-o",
            str(dest),
            "--char-perturb",
            "--char-mode",
            "zero-width",
            "--char-strength",
            "0.3",
            "--seed",
            "42",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = __import__("json").loads(r.stdout)
    assert report["stats"]["char_perturb"]["changed"] > 0
    assert dest.read_text(encoding="utf-8") != SAMPLE


def test_soft_binding_detected(tmp_path: Path):
    f = tmp_path / "sealed.jpg"
    f.write_bytes(
        b"\xff\xd8jpeg-ish c2pa manifest store with assertion "
        b'{"label": "c2pa.soft-binding", "url": "https://verify.contentauth.example/manifest/abc"}'
        b"\xff\xd9"
    )
    report = inspect_soft_binding(f)
    assert report["has_c2pa"] is True
    assert report["soft_binding"]["found"] is True
    assert "c2pa.soft-binding" in report["soft_binding"]["labels"]
    assert any("manifest" in u for u in report["soft_binding"]["manifest_urls"])
    assert report["warning"]


def test_soft_binding_absent_in_clean_file(tmp_path: Path):
    f = tmp_path / "plain.txt"
    f.write_text("ordinary prose, no provenance", encoding="utf-8")
    report = inspect_soft_binding(f)
    assert report["soft_binding"]["found"] is False
    assert report["warning"] is None


def test_manifestish_url_without_c2pa_is_not_soft_binding(tmp_path: Path):
    f = tmp_path / "ordinary.txt"
    f.write_text("Documentation: https://example.test/manifest/format", encoding="utf-8")
    report = inspect_soft_binding(f)
    assert report["has_c2pa"] is False
    assert report["soft_binding"]["found"] is False


def test_soft_binding_cli_exit_codes(tmp_path: Path):
    sealed = tmp_path / "s.png"
    sealed.write_bytes(b"\x89PNG c2pa c2pa.remote-manifest http://x.example/manifest/1")
    plain = tmp_path / "p.txt"
    plain.write_text("clean", encoding="utf-8")
    r1 = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_soft_binding.py"), str(sealed)],
        capture_output=True,
    )
    r0 = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_soft_binding.py"), str(plain)],
        capture_output=True,
    )
    assert r1.returncode == 1
    assert r0.returncode == 0
