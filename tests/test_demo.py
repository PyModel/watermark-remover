"""Demo handler tests without importing optional Gradio."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import layer_b_http
from morphomod import Raster, encode_png

from demo import clean_upload


def test_demo_text_handler_and_layer_b_prompt_is_always_offline(tmp_path: Path, monkeypatch):
    src = tmp_path / "draft.txt"
    src.write_text("hello\u200bworld", encoding="utf-8")
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "openai-compatible")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "must-not-be-used")
    monkeypatch.setattr(
        layer_b_http,
        "request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("demo prompt reached network")
        ),
    )
    report, output, prompt = clean_upload(str(src), False, True, "tsapa")
    assert "**Kind:** `text`" in report
    assert output and Path(output).is_file()
    assert Path(output).read_text(encoding="utf-8") == "helloworld"
    assert "Pareto" in prompt


def test_demo_binary_container_does_not_decode_layer_b_text(tmp_path: Path):
    src = tmp_path / "document.pdf"
    src.write_bytes(b"%PDF-1.4\n%%EOF\n")
    report, output, prompt = clean_upload(str(src), False, True, "paraphrase")
    assert "**Kind:** `container`" in report
    assert output and Path(output).is_file()
    assert prompt == "Layer B is available only for plain-text uploads."


def test_demo_routes_unknown_suffix_container_from_content(tmp_path: Path):
    src = tmp_path / "artifact.bin"
    src.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><metadata>AI provenance</metadata></svg>',
        encoding="utf-8",
    )

    report, output, prompt = clean_upload(str(src), False, True, "paraphrase")

    assert "**Kind:** `container`" in report
    assert output and Path(output).is_file()
    assert "<metadata" not in Path(output).read_text(encoding="utf-8")
    assert prompt == "Layer B is available only for plain-text uploads."


def test_demo_image_handler(tmp_path: Path):
    src = tmp_path / "image.png"
    src.write_bytes(encode_png(Raster(2, 2, 3, bytearray([1, 2, 3] * 4))))
    report, output, prompt = clean_upload(str(src), False, False, "paraphrase")
    assert "**Kind:** `image`" in report
    assert output and Path(output).is_file()
    assert prompt == ""
