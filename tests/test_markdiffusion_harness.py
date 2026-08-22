"""Tests for the optional MarkDiffusion image-watermark harness adapter.

Adapts THEIRS test_markdiffusion_harness.py to the OURS seam (the integration
tests against image_meta.run_markdiffusion_purify are deferred until that
function is ported into image_meta.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

HARNESS_SCRIPT = SCRIPTS / "markdiffusion_harness.py"


def _run_adapter(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("MARKDIFFUSION_DIR", None)
    return subprocess.run(
        [sys.executable, str(HARNESS_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_unavailable_missing_package(tmp_path: Path):
    """No upstream + no installed package -> exit 3."""
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = _run_adapter("detect", str(img), "--scheme", "tr")
    # The harness reports unavailability (exit 3) when markdiffusion is not
    # importable; we already removed the package from the env, so we expect a
    # 3 (or 2 if the upstream-dir check trips first).
    assert r.returncode in (2, 3)
    assert any(msg in (r.stderr or "") for msg in ("markdiffusion", "scheme", "MARKDIFFUSION_DIR"))


def test_cli_bad_scheme(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = _run_adapter("detect", str(img), "--scheme", "nope")
    assert r.returncode in (2, 3)


def test_cli_missing_file(tmp_path: Path):
    r = _run_adapter("detect", str(tmp_path / "missing.png"), "--scheme", "tr")
    assert r.returncode in (2, 3)


def test_cli_normalize_scheme_known_aliases():
    """normalize_scheme maps user-facing aliases to canonical names."""
    from markdiffusion_harness import normalize_scheme

    assert normalize_scheme("tr") == "TR"
    assert normalize_scheme("ringid") == "RI"
    assert normalize_scheme("gaussianshading") == "GS"
    assert normalize_scheme("gaussmarker") == "GM"
    assert normalize_scheme("TR") == "TR"
    assert normalize_scheme("GS") == "GS"


def test_cli_normalize_scheme_unknown_raises():
    """normalize_scheme raises ValueError for unknown schemes."""
    import pytest
    from markdiffusion_harness import normalize_scheme

    with pytest.raises(ValueError):
        normalize_scheme("nope")


def test_cli_resolve_device_explicit():
    """Explicit device passes through unchanged."""
    from markdiffusion_harness import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_cli_resolve_device_auto_falls_back_to_cpu():
    """With no torch import, 'auto' falls back to cpu."""
    from markdiffusion_harness import resolve_device

    assert resolve_device("auto") == "cpu"


def test_cli_resolve_config_too_large(tmp_path: Path):
    big = tmp_path / "huge.json"
    big.write_bytes(b"x" * (1024 * 1024 + 1))
    from markdiffusion_harness import _resolve_config, _Unavailable

    with __import__("pytest").raises(_Unavailable):
        _resolve_config(None, str(big))


def test_cli_resolve_config_missing(tmp_path: Path):
    from markdiffusion_harness import _resolve_config, _Unavailable

    with __import__("pytest").raises(_Unavailable):
        _resolve_config(None, str(tmp_path / "missing.json"))


_VALID_SHA = "f71d7867a2745c420aa93441638b119c85995963"


def test_split_model_revision_accepts_full_commit_sha():
    """A full 40-char lowercase hex commit ID passes through verbatim."""
    from markdiffusion_harness import _split_model_revision

    assert _split_model_revision(f"org/repo@{_VALID_SHA}") == ("org/repo", _VALID_SHA)


def test_split_model_revision_unrevisioned_returns_none():
    """No '@' -> repo id unchanged and revision None."""
    from markdiffusion_harness import _split_model_revision

    assert _split_model_revision("org/repo") == ("org/repo", None)


def test_split_model_revision_rejects_mutable_refs_and_malformed():
    """Branches, tags, short/long/non-hex SHAs and malformed specs raise."""
    import pytest
    from markdiffusion_harness import _split_model_revision

    for bad in (
        "org/repo@main",
        "org/repo@v1.0.0",
        "org/repo@" + "a" * 39,
        "org/repo@" + "a" * 41,
        "org/repo@" + "g" * 40,
        "@" + "a" * 40,
        "org/repo@",
    ):
        with pytest.raises(ValueError):
            _split_model_revision(bad)


def test_default_model_is_pinned_to_full_sha():
    """The built-in default carries an immutable full-commit-SHA revision."""
    from markdiffusion_harness import (
        DEFAULT_MODEL,
        DEFAULT_MODEL_REVISION,
        _split_model_revision,
    )

    repo, revision = _split_model_revision(f"{DEFAULT_MODEL}@{DEFAULT_MODEL_REVISION}")
    assert repo == DEFAULT_MODEL
    assert revision == DEFAULT_MODEL_REVISION


def test_cli_unpinned_online_model_rejected(tmp_path: Path):
    """Online + unrevisioned --model -> exit 2 before any upstream access."""
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    r = _run_adapter(
        "detect",
        str(img),
        "--scheme",
        "tr",
        "--model",
        "huanzi05/stable-diffusion-2-1-base",
    )
    assert r.returncode == 2
    assert "unpinned" in (r.stderr or "")
