"""Tests for Layer A text Unicode scrub."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from text_unicode import clean_text, inspect_text


def test_strips_zero_width_and_soft_hyphen():
    raw = "Hello\u200bWorld\u00ad!"
    cleaned, stats = clean_text(raw)
    assert cleaned == "HelloWorld!"
    assert stats["removed_count"] >= 2


def test_normalizes_exotic_spaces():
    raw = "a\u2003b\u3000c"  # em space, ideographic space
    cleaned, stats = clean_text(raw)
    assert cleaned == "a b c"
    assert stats["replaced_count"] >= 2


def test_inspect_finds_zwsp():
    report = inspect_text("x\u200by")
    assert report.suspicious_total >= 1
    kinds = {h.kind for h in report.hits}
    assert "zwj_family" in kinds or "strip" in kinds


def test_inspect_tag_chars():
    # Language tag character U+E0041 (TAG LATIN CAPITAL LETTER A)
    raw = "hi" + chr(0xE0041) + "there"
    report = inspect_text(raw)
    assert report.suspicious_total >= 1
    assert any(h.kind == "tag_chars" for h in report.hits)
    cleaned, stats = clean_text(raw)
    assert chr(0xE0041) not in cleaned
    assert stats["removed_count"] >= 1


def test_inspect_bidi():
    raw = "ab\u202eef"  # RLO
    report = inspect_text(raw)
    assert any(h.kind == "bidi" for h in report.hits)
    cleaned, _ = clean_text(raw)
    assert "\u202e" not in cleaned


def test_preserves_contextual_zwj_and_variation_selector_by_default():
    raw = "👩‍💻 ❤️"
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["preserved_count"] == 2  # ZWJ + VS16

    aggressive, _ = clean_text(raw, preserve_semantic=False)
    assert "‍" not in aggressive
    assert "️" not in aggressive


def test_preserves_script_zwnj_and_balanced_bidi():
    persian = "می‌روم"  # meaningful ZWNJ
    bidi = "\u202babc\u202c"  # balanced RLE/PDF
    cleaned, stats = clean_text(f"{persian} {bidi}")
    assert cleaned == f"{persian} {bidi}"
    assert stats["preserved_count"] == 3


def test_removes_orphan_joiner_and_unbalanced_bidi():
    raw = "\u200dstart ab\u202eef"
    cleaned, _ = clean_text(raw)
    assert "\u200d" not in cleaned
    assert "\u202e" not in cleaned


def test_preserves_invisible_math_operator_in_context():
    raw = "f\u2061(x)"
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["preserved_count"] == 1


def test_clean_preserves_normal_text():
    raw = "Normal ASCII and café — fine."
    cleaned, stats = clean_text(raw)
    assert cleaned == raw
    assert stats["removed_count"] == 0


def test_clean_text_preserves_preexisting_backup_on_rejection(tmp_path: Path):
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")
    backup = src.with_suffix(".txt.bak")
    backup.write_text("old backup", encoding="utf-8")
    script = SCRIPTS / "clean_text.py"
    result = subprocess.run(
        [sys.executable, str(script), str(src), "--in-place"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert backup.read_text(encoding="utf-8") == "old backup"
    assert src.read_text(encoding="utf-8") == "a\u200bb"


def test_clean_text_in_place_rejects_backup_symlink(tmp_path: Path):
    src = tmp_path / "input.txt"
    src.write_text("a\u200bb", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve", encoding="utf-8")
    src.with_suffix(".txt.bak").symlink_to(outside)
    script = SCRIPTS / "clean_text.py"
    result = subprocess.run(
        [sys.executable, str(script), str(src), "--in-place"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert src.read_text(encoding="utf-8") == "a\u200bb"


def test_cli_roundtrips_invalid_utf8_and_backup_is_byte_exact(tmp_path: Path):
    src = tmp_path / "mixed.txt"
    original = b"abc\xffdef\xe2\x80\x8b"
    src.write_bytes(original)
    script = SCRIPTS / "clean_text.py"
    r = subprocess.run(
        [sys.executable, str(script), str(src), "--in-place"],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr
    assert src.with_suffix(".txt.bak").read_bytes() == original
    assert src.read_bytes() == b"abc\xffdef"


def test_aggressive_confusable():
    # Cyrillic 'а' (U+0430) looks like Latin 'a'
    raw = "p\u0430y"  # p + cyrillic a + y
    cleaned, _ = clean_text(raw, aggressive_homoglyphs=True)
    assert cleaned == "pay"
