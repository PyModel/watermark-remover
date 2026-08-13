"""Demo handler tests without importing optional Gradio."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from morphomod import Raster, encode_png

from demo import clean_upload


def test_demo_text_handler_and_layer_b_prompt(tmp_path: Path):
    src = tmp_path / "draft.txt"
    src.write_text("hello​world", encoding="utf-8")
    report, output, prompt = clean_upload(str(src), False, True, "tsapa")
    assert "**Kind:** `text`" in report
    assert output and Path(output).is_file()
    assert Path(output).read_text(encoding="utf-8") == "helloworld"
    assert "Pareto" in prompt


def test_demo_image_handler(tmp_path: Path):
    src = tmp_path / "image.png"
    src.write_bytes(encode_png(Raster(2, 2, 3, bytearray([1, 2, 3] * 4))))
    report, output, prompt = clean_upload(str(src), False, False, "paraphrase")
    assert "**Kind:** `image`" in report
    assert output and Path(output).is_file()
    assert prompt == ""
