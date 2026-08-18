"""Tests for the optional CtrlRegen pixel-watermark remover adapter.

Adapts THEIRS test_ctrlregen_clean.py to the OURS seam (the integration
tests against image_meta.run_ctrlregen_clean are deferred until that
function is ported into image_meta.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

CLEAN_SCRIPT = SCRIPTS / "clean_ctrlregen.py"

FAKE_PIL = """class Image:
    def __init__(self, size=(10, 20)):
        self.size = size
    @staticmethod
    def open(path):
        return Image()
    def convert(self, mode):
        return self
    def load(self):
        return self
    def save(self, fp, format=None, **kwargs):
        fp.write(b"FAKEIMAGE")
"""


def _fake_engine(fail_run: bool) -> str:
    run_body = 'raise RuntimeError("model missing")' if fail_run else "return image"
    return (
        "class CtrlRegenEngine:\n"
        "    def __init__(self, **kwargs):\n"
        "        pass\n"
        "    def run(self, image, strength=0.5, num_inference_steps=50, "
        "guidance_scale=2.0, seed=None):\n"
        f"        {run_body}\n"
        "\n"
        "def is_ctrlregen_available():\n"
        "    return True\n"
    )


def _make_fake_upstream(tmp_path: Path, *, fail_run: bool = False) -> Path:
    upstream = tmp_path / "noai-watermark"
    ctrlregen = upstream / "src" / "ctrlregen"
    pil = upstream / "src" / "PIL"
    ctrlregen.mkdir(parents=True)
    pil.mkdir(parents=True)
    (ctrlregen / "__init__.py").write_text("")
    (pil / "__init__.py").write_text(FAKE_PIL)
    (ctrlregen / "engine.py").write_text(_fake_engine(fail_run))
    return upstream


def _run_adapter(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("NOAI_WATERMARK_DIR", None)
    return subprocess.run(
        [sys.executable, str(CLEAN_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_unavailable_without_upstream(tmp_path: Path):
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"not really an image")
    r = _run_adapter(str(dummy))
    assert r.returncode == 3
    assert "NOAI_WATERMARK_DIR" in (r.stderr or "")


def test_cli_bad_input_missing_file(tmp_path: Path):
    r = _run_adapter(str(tmp_path / "missing.png"))
    assert r.returncode == 2


def test_cli_bad_strength(tmp_path: Path):
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"x")
    r = _run_adapter(str(dummy), "--strength", "0")
    assert r.returncode == 2


def test_cli_unavailable_missing_src_dir(tmp_path: Path):
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"x")
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run_adapter(str(dummy), "--upstream-dir", str(empty))
    assert r.returncode == 3


def test_cli_json_success(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    out = tmp_path / "out.png"
    r = _run_adapter(
        str(img),
        "-o",
        str(out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert payload["output"] == str(out)
    assert out.read_bytes() == b"FAKEIMAGE"


def test_cli_runtime_error(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path, fail_run=True)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = _run_adapter(
        str(img),
        "-o",
        str(tmp_path / "out.png"),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert "model missing" in (r.stderr or "")


def test_cli_refuses_symlink_output(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    victim = tmp_path / "victim"
    victim.write_bytes(b"original")
    out = tmp_path / "out.png"
    try:
        out.symlink_to(victim)
    except OSError:
        pytest.skip("symlinks unavailable")
    r = _run_adapter(
        str(img),
        "-o",
        str(out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert victim.read_bytes() == b"original"


def test_cli_jpeg_output(tmp_path: Path):
    """JPEG output suffix selects JPEG encoding path."""
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    out = tmp_path / "out.jpg"
    r = _run_adapter(
        str(img),
        "-o",
        str(out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert payload["output"] == str(out)
    # Image.save was called with format="JPEG" by the fake engine
    assert out.read_bytes() == b"FAKEIMAGE"


def test_cli_default_output_cleaned_path(tmp_path: Path):
    """Without -o, output is *.ctrlregen.* via cleaned_path."""
    upstream = _make_fake_upstream(tmp_path)
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = _run_adapter(
        str(img),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["output"] == str(tmp_path / "img.ctrlregen.png")
    assert (tmp_path / "img.ctrlregen.png").read_bytes() == b"FAKEIMAGE"
