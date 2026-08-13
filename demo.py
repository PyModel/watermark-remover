#!/usr/bin/env python3
"""Optional Gradio demo UI for watermark-remover.

    pip install gradio
    python3 demo.py            # serves on http://127.0.0.1:7860

Text runs Layer A (deterministic Unicode scrub). Images/containers run the
metadata strippers. For plain-text uploads, Layer B emits a rewrite prompt
(print-prompt backend; no model or network call). Binary document rewriting is
intentionally unavailable because it requires format-aware text extraction.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from asset_kind import classify_asset
from clean_asset import CleanPlan, clean_asset
from rewrite_text import RewritePlan, rewrite


def clean_upload(file_obj, keep_non_ai: bool, layer_b: bool, strength: str):
    """Gradio handler. Returns (report markdown, cleaned file path, rewrite prompt)."""
    if file_obj is None:
        return "Upload a file first.", None, ""
    src = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    workdir = Path(tempfile.mkdtemp(prefix="wmr-"))
    dest = workdir / f"{src.stem}.cleaned{src.suffix}"

    kind = classify_asset(src)
    try:
        result = clean_asset(
            src, dest, CleanPlan(forced_kind=kind, strip_all_metadata=not keep_non_ai)
        ).to_dict()
    except Exception as e:
        return f"**Error cleaning {src.name}:** `{e}`", None, ""

    prompt = ""
    if layer_b and kind == "text":
        body = dest.read_text(encoding="utf-8", errors="surrogateescape")
        if body.strip():
            prompt, _ = rewrite(body, RewritePlan.prompt(strength))
    elif layer_b:
        prompt = "Layer B is available only for plain-text uploads."

    report = (
        f"**Kind:** `{result.get('kind')}`  \n"
        f"**Output:** `{result.get('output')}`  \n"
        f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)[:4000]}\n```"
    )
    return report, str(dest), prompt


def main() -> None:
    try:
        import gradio as gr
    except ImportError:
        sys.exit("gradio not installed: pip install gradio")

    with gr.Blocks(title="watermark-remover") as demo:
        gr.Markdown(
            "# watermark-remover\n"
            "Strip AI provenance marks — Layer A Unicode, file metadata (C2PA/EXIF/XMP), "
            "optional Layer B rewrite prompt. For content **you own**."
        )
        with gr.Row():
            with gr.Column():
                file_in = gr.File(label="Upload file", type="filepath")
                keep_non_ai = gr.Checkbox(label="Images: keep non-AI metadata", value=False)
                layer_b = gr.Checkbox(label="Layer B: show rewrite prompt", value=False)
                strength = gr.Dropdown(
                    ["paraphrase", "backtranslate", "structural", "tsapa"],
                    value="paraphrase",
                    label="Layer B strength",
                )
                go = gr.Button("Clean", variant="primary")
            with gr.Column():
                report = gr.Markdown()
                file_out = gr.File(label="Download cleaned file")
                prompt_out = gr.Textbox(
                    label="Layer B prompt (paste into a non-origin model)", lines=8
                )
        go.click(
            clean_upload,
            [file_in, keep_non_ai, layer_b, strength],
            [report, file_out, prompt_out],
        )

    demo.launch()


if __name__ == "__main__":
    main()
