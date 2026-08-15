"""F7: best-effort Claude/Anthropic detector-evasion risk scoring."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from claude_risk import BEST_EFFORT_NOTE, assess_claude_risk

RISK = SCRIPTS / "claude_risk.py"


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _png_with_c2pa() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa test contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _png_clean() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def test_clean_text_is_low_risk(tmp_path: Path) -> None:
    src = tmp_path / "clean.txt"
    src.write_text("plain text without watermark carriers", encoding="utf-8")
    report = assess_claude_risk(src)
    assert report["kind"] == "text"
    assert report["verdict"] == "low"
    assert report["score"] < 30
    assert report["signals"] == []
    assert BEST_EFFORT_NOTE in report["note"]


def test_residual_invisible_chars_raise_score(tmp_path: Path) -> None:
    src = tmp_path / "dirty.txt"
    src.write_text("a\u200bb\u200cc\u200d" * 5, encoding="utf-8")
    report = assess_claude_risk(src)
    assert report["kind"] == "text"
    assert any(s["signal"] == "invisible_unicode_carriers" for s in report["signals"])
    assert report["score"] > 5  # above the base-only floor


def test_image_with_residual_c2pa_is_high_risk(tmp_path: Path) -> None:
    src = tmp_path / "marked.png"
    src.write_bytes(_png_with_c2pa())
    report = assess_claude_risk(src)
    assert report["kind"] == "image"
    assert any(s["signal"] == "residual_c2pa" for s in report["signals"])
    assert report["verdict"] in ("medium", "high")


def test_clean_image_is_low_or_medium(tmp_path: Path) -> None:
    src = tmp_path / "clean.png"
    src.write_bytes(_png_clean())
    report = assess_claude_risk(src)
    assert report["kind"] == "image"
    assert report["verdict"] in ("low", "medium")


def test_fresh_process_import_path_has_no_heavy_deps() -> None:
    # Production hardening: the risk-scoring chain must stay stdlib-only.
    # Importing claude_risk in a fresh process must not pull cv2/skimage/torch.
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; import claude_risk; print(sorted(sys.modules))"],
        cwd=str(SCRIPTS),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "claude_risk" in proc.stdout
    assert "cv2" not in proc.stdout
    assert "skimage" not in proc.stdout
    assert "torch" not in proc.stdout


def test_cli_json_and_error(tmp_path: Path) -> None:
    src = tmp_path / "clean.txt"
    src.write_text("hello", encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, str(RISK), str(src), "--json"],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["kind"] == "text"

    missing = tmp_path / "nope.txt"
    bad = subprocess.run(
        [sys.executable, str(RISK), str(missing), "--json"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert "not a regular file" in bad.stderr
