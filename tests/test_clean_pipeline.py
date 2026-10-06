from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import clean_pipeline
from clean_request import CleanRequest
from pipeline_actions import drop_png_chunk, report_actions


def test_reports_expose_structured_actions_without_prose_channel():
    payload = report_actions([drop_png_chunk("tEXt")])

    assert "actions" not in payload
    assert payload["action_details"][0]["code"] == "drop_png_chunk"


def test_action_producers_do_not_reintroduce_prose_field():
    producer_files = (
        "pipeline_actions.py",
        "clean_file.py",
        "clean_image.py",
        "clean_text.py",
        "morphomod.py",
        "heif_meta.py",
        "container_meta.py",
        "image_meta.py",
    )
    scripts = Path(__file__).parents[1] / "skills" / "remove-ai-marks" / "scripts"
    for filename in producer_files:
        source = (scripts / filename).read_text(encoding="utf-8")
        assert '"actions"' not in source


def test_edge_sentence_goldens_match_base_smoke_outputs(tmp_path):
    root = Path(__file__).parents[1]
    scripts = root / "skills" / "remove-ai-marks" / "scripts"
    fixtures = root / "tests" / "fixtures"

    def run(*args, input_text=None):
        result = subprocess.run(
            [sys.executable, *map(str, args)],
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout + result.stderr

    wm_a = tmp_path / "wm-a.txt"
    wm_b = tmp_path / "wm-b.txt"
    smoke_text = (fixtures / "sample_watermarked.txt").read_text(encoding="utf-8")
    wm_a.write_text(smoke_text, encoding="utf-8")
    wm_b.write_text(smoke_text, encoding="utf-8")
    wm_output = run(
        scripts / "clean_file.py",
        wm_a,
        wm_b,
        "-o",
        tmp_path / "wm-out",
    )
    wm_output = wm_output.replace(str(tmp_path), "<TMP>").replace("\\", "/")  # Windows
    assert wm_output == (fixtures / "t05_wm_cli.golden.txt").read_text()

    smoke_image = fixtures / "t05_smoke.png"
    dry_output = tmp_path / "dry-output.png"
    dry_run = run(
        scripts / "clean_file.py",
        smoke_image,
        "-o",
        dry_output,
        "--visible-box",
        "1,1,1,1",
        "--visible-backend",
        "simple",
        "--dilate",
        "0",
        "--dry-run",
    )
    dry_run = dry_run.replace(str(smoke_image), "<INPUT>")
    dry_run = dry_run.replace(str(dry_output.with_suffix(".mask.pgm")), "<MASK>")
    dry_run = dry_run.replace(str(dry_output), "<OUTPUT>")
    assert dry_run == (fixtures / "t05_wm_dry_run.golden.txt").read_text()

    text_output = run(
        scripts / "clean_text.py",
        "--stats",
        input_text="Hello\u200b world\n",
    )
    assert text_output == (fixtures / "t05_clean_text.golden.txt").read_text()

    image_output = tmp_path / "image-output.png"
    image_run = run(scripts / "clean_image.py", smoke_image, "-o", image_output)
    assert (
        image_run.replace(str(image_output), "<OUTPUT>")
        == (fixtures / "t05_clean_image.golden.txt").read_text()
    )

    morpho_input = fixtures / "t05_morpho_smoke.png"
    morpho_run = run(
        scripts / "morphomod.py",
        morpho_input,
        "--box",
        "1,1,1,1",
        "--backend",
        "print-plan",
    )
    morpho_mask = morpho_input.with_suffix(".visible.cleaned.mask.pgm")
    morpho_run = morpho_run.replace(str(morpho_mask), "<MASK>")
    assert morpho_run == (fixtures / "t05_morphomod.golden.txt").read_text()


def test_execute_preserves_batch_order_and_isolates_item_failure(tmp_path, monkeypatch):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\u200b", encoding="utf-8")
    second.write_text("two\u200b", encoding="utf-8")
    output = tmp_path / "out"
    original = clean_pipeline.clean_asset

    def fail_second(path, destination, plan, *, on_token=None):
        if path == second:
            raise RuntimeError("second failed")
        return original(path, destination, plan, on_token=on_token)

    monkeypatch.setattr(clean_pipeline, "clean_asset", fail_second)
    run = clean_pipeline.execute(CleanRequest(paths=(first, second), output=output))

    assert [item.path for item in run.results] == [first, second]
    assert run.results[0].payload["exit_code"] == 0
    assert run.results[1].payload["exit_code"] != 0
    assert (output / "first.txt").is_file()
    assert not (output / "second.txt").exists()


def test_execute_dry_run_plans_without_writing(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"not an image")
    mask = tmp_path / "mask.pgm"
    mask.write_bytes(b"P2\n1 1\n255\n0\n")
    output = tmp_path / "out"
    request = CleanRequest(
        paths=(source,), output=output, visible_mask=mask, dry_run=True, force_type="image"
    )

    run = clean_pipeline.execute(request)

    assert run.results[0].payload["status"] == "dry-run"
    assert not output.exists()
    assert not source.with_name("image.mask.pgm").exists()
