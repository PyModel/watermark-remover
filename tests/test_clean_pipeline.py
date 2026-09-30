from __future__ import annotations

from pathlib import Path

import clean_pipeline
from clean_request import CleanRequest


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
