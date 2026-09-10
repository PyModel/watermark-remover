"""Tests for the optional MarkLLM text-watermark harness adapter."""

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

DETECT_SCRIPT = SCRIPTS / "detect_text_watermark.py"

FAKE_TRANSFORMERS = (
    "import sys\n"
    "class _LM:\n"
    "    def to(self, device):\n"
    "        return self\n"
    "\n"
    "class AutoModelForCausalLM:\n"
    "    @staticmethod\n"
    "    def from_pretrained(name, **kwargs):\n"
    "        print('MARKLLM_PRETRAINED_KWARGS=' + repr(kwargs), file=sys.stderr)\n"
    "        return _LM()\n"
    "\n"
    "class _Tok:\n"
    "    def encode(self, s):\n"
    "        return [0] * 512\n"
    "\n"
    "class AutoTokenizer:\n"
    "    @staticmethod\n"
    "    def from_pretrained(name, **kwargs):\n"
    "        return _Tok()\n"
)

FAKE_TRANSFORMERS_CONFIG = (
    "class TransformersConfig:\n"
    "    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', **kwargs):\n"
    "        self.device = device\n"
    "        self.model = model\n"
    "        self.tokenizer = tokenizer\n"
    "        self.vocab_size = vocab_size\n"
    "        self.gen_kwargs = {}\n"
    "        self.gen_kwargs.update(kwargs)\n"
)

KGW_CONFIG = '{"algorithm_name": "KGW", "z_threshold": 4.0}'
SYNTHID_CONFIG = '{"algorithm_name": "SynthID", "threshold": 0.52, "detector_type": "mean"}'


def _fake_auto_watermark(*, fail_detect: bool = False, fail_generate: bool = False) -> str:
    detect_body = (
        'raise RuntimeError("boom")'
        if fail_detect
        else 'return {"is_watermarked": True, "score": 3.5}'
    )
    gen_body = 'raise RuntimeError("boom")' if fail_generate else "return 'WATERMARKED SAMPLE'"
    return (
        "from types import SimpleNamespace\n"
        "class _WM:\n"
        "    def __init__(self):\n"
        "        self.config = SimpleNamespace(gen_kwargs={})\n"
        "    def detect_watermark(self, text, return_dict=True):\n"
        f"        {detect_body}\n"
        "    def generate_watermarked_text(self, prompt):\n"
        f"        {gen_body}\n"
        "    def generate_unwatermarked_text(self, prompt):\n"
        "        return 'PLAIN SAMPLE'\n"
        "\n"
        "class AutoWatermark:\n"
        "    @staticmethod\n"
        "    def load(algorithm_name, algorithm_config=None, transformers_config=None):\n"
        "        return _WM()\n"
    )


def _fake_auto_watermark_detect(detect_literal: str) -> str:
    """Fake AutoWatermark returning a caller-supplied detect dict literal."""
    return (
        "from types import SimpleNamespace\n"
        "class _WM:\n"
        "    def __init__(self):\n"
        "        self.config = SimpleNamespace(gen_kwargs={})\n"
        "    def detect_watermark(self, text, return_dict=True):\n"
        f"        return {detect_literal}\n"
        "\n"
        "class AutoWatermark:\n"
        "    @staticmethod\n"
        "    def load(algorithm_name, algorithm_config=None, transformers_config=None):\n"
        "        return _WM()\n"
    )


def _make_fake_upstream(
    tmp_path: Path,
    *,
    with_config: bool = True,
    fail_detect: bool = False,
    fail_generate: bool = False,
    missing_watermark_dir: bool = False,
    detect_literal: str | None = None,
) -> Path:
    upstream = tmp_path / "MarkLLM"
    config_dir = upstream / "config"
    config_dir.mkdir(parents=True)
    if with_config:
        (config_dir / "KGW.json").write_text(KGW_CONFIG)
        (config_dir / "SynthID.json").write_text(SYNTHID_CONFIG)
    if not missing_watermark_dir:
        watermark = upstream / "watermark"
        watermark.mkdir(parents=True)
        (watermark / "__init__.py").write_text("")
        fake = (
            _fake_auto_watermark_detect(detect_literal)
            if detect_literal is not None
            else _fake_auto_watermark(fail_detect=fail_detect, fail_generate=fail_generate)
        )
        (watermark / "auto_watermark.py").write_text(fake)
    utils_dir = upstream / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "__init__.py").write_text("")
    (utils_dir / "transformers_config.py").write_text(FAKE_TRANSFORMERS_CONFIG)
    transformers_dir = upstream / "transformers"
    transformers_dir.mkdir(parents=True)
    (transformers_dir / "__init__.py").write_text(FAKE_TRANSFORMERS)
    return upstream


def _run_adapter(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("MARKLLM_DIR", None)
    return subprocess.run(
        [sys.executable, str(DETECT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_unavailable_without_upstream(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "kgw")
    assert r.returncode == 3
    assert "MARKLLM_DIR" in (r.stderr or "")


def test_cli_unavailable_incomplete_checkout(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(empty))
    assert r.returncode == 3

    upstream = _make_fake_upstream(tmp_path, missing_watermark_dir=True)
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 3


def test_cli_unavailable_missing_config(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    upstream = _make_fake_upstream(tmp_path, with_config=False)
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 3
    assert "config" in (r.stderr or "").lower()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX perms")
def test_cli_unreadable_config_detect(tmp_path: Path):
    """A config that becomes unreadable after resolution exits 3, not a traceback."""
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    upstream = _make_fake_upstream(tmp_path)
    config = upstream / "config" / "KGW.json"
    config.chmod(0)
    try:
        r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
        assert r.returncode == 3
        assert "cannot read watermarking config" in (r.stderr or "")
    finally:
        config.chmod(0o644)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX perms")
def test_cli_unreadable_config_watermark(tmp_path: Path):
    """The watermark path reports the same unreadable-config failure, not a traceback."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    upstream = _make_fake_upstream(tmp_path)
    config = upstream / "config" / "KGW.json"
    config.chmod(0)
    try:
        r = _run_adapter(
            "watermark",
            str(prompt),
            "--scheme",
            "kgw",
            "-o",
            str(tmp_path / "wm.txt"),
            "--upstream-dir",
            str(upstream),
        )
        assert r.returncode == 3
        assert "cannot read watermarking config" in (r.stderr or "")
    finally:
        config.chmod(0o644)


def test_cli_unavailable_missing_deps(tmp_path: Path):
    upstream = tmp_path / "MarkLLM"
    (upstream / "config").mkdir(parents=True)
    (upstream / "config" / "KGW.json").write_text(KGW_CONFIG)
    watermark = upstream / "watermark"
    watermark.mkdir()
    (watermark / "__init__.py").write_text("")
    (watermark / "auto_watermark.py").write_text("import does_not_exist_123\n")
    (upstream / "utils").mkdir()
    (upstream / "utils" / "__init__.py").write_text("")
    (upstream / "utils" / "transformers_config.py").write_text(FAKE_TRANSFORMERS_CONFIG)
    (upstream / "transformers").mkdir()
    (upstream / "transformers" / "__init__.py").write_text(FAKE_TRANSFORMERS)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 3
    assert "dependencies missing" in (r.stderr or "")


def test_cli_bad_input_missing_file(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    r = _run_adapter(
        "detect", str(tmp_path / "missing.txt"), "--scheme", "kgw", "--upstream-dir", str(upstream)
    )
    assert r.returncode == 2


def test_cli_bad_input_binary(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    png = tmp_path / "img.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nnot really")
    r = _run_adapter("detect", str(png), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r.returncode == 2
    assert "refusing" in (r.stderr or "")


def test_cli_bad_scheme(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "nope")
    assert r.returncode == 2


def test_cli_detect_json_success(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert payload["scheme"] == "KGW"
    assert payload["is_watermarked"] is True
    assert payload["score"] == 3.5
    assert payload["threshold"] == 4.0
    assert payload["device"] == "cpu"


def test_cli_detect_synthid_alias(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "synthid-text",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["scheme"] == "SynthID"
    assert payload["threshold"] == 0.52


def test_cli_detect_runtime_error(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path, fail_detect=True)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert "boom" in (r.stderr or "")


def test_cli_detect_offline_flag(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
        "--offline",
    )
    assert r.returncode == 0, r.stderr
    assert "'local_files_only': True" in (r.stderr or "")


def test_cli_config_too_large(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    big = tmp_path / "huge.json"
    big.write_bytes(b"x" * (1024 * 1024 + 1))
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--config",
        str(big),
        "--upstream-dir",
        str(upstream),
    )
    assert r.returncode == 3
    assert "too large" in (r.stderr or "")


def test_cli_watermark_json_success(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    wm_out = tmp_path / "wm.txt"
    uwm_out = tmp_path / "uwm.txt"
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "-o",
        str(wm_out),
        "-o2",
        str(uwm_out),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert wm_out.read_text() == "WATERMARKED SAMPLE"
    assert uwm_out.read_text() == "PLAIN SAMPLE"


def test_cli_watermark_default_stdout(tmp_path: Path):
    """When -o is omitted, watermarked text goes to stdout, not a file named '-'."""
    upstream = _make_fake_upstream(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
    )
    assert r.returncode == 0, r.stderr
    assert "WATERMARKED SAMPLE" in (r.stdout or "")
    # Must NOT have created a file literally named '-'
    assert not (Path.cwd() / "-").exists()


def test_cli_watermark_runtime_error(tmp_path: Path):
    upstream = _make_fake_upstream(tmp_path, fail_generate=True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 1
    assert "boom" in (r.stderr or "")


def test_cli_watermark_json_stdout_stays_pure_without_output(tmp_path: Path):
    """--json must never mix generated text with the JSON payload on stdout.

    With no -o, the watermarked sample previously went to stdout and made
    json.loads() fail; it must be routed to stderr instead.
    """
    upstream = _make_fake_upstream(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["available"] is True
    assert payload["watermarked_output"] == "-"
    assert "WATERMARKED SAMPLE" in (r.stderr or "")
    assert "WATERMARKED SAMPLE" not in (r.stdout or "")


def test_cli_detect_positive_verdict_detected(tmp_path: Path):
    """A positive observation becomes DETECTED even with unknown provenance."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": True, "score": 9.2}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "DETECTED"
    assert payload["detector_verdict"] == "DETECTED"
    assert payload["is_watermarked"] is True
    assert payload["provenance_match"] is None
    assert payload["input_tokens"] == 512


def test_cli_detect_positive_inside_abstention_band_stays_detected(tmp_path: Path):
    """A positive observation is not downgraded by the abstention band."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": True, "score": 0.54}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "synthid-text",
        "--upstream-dir",
        str(upstream),
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "DETECTED"
    assert payload["detector_verdict"] == "DETECTED"
    assert payload["threshold"] == 0.52


def test_cli_detect_positive_without_threshold_stays_detected(tmp_path: Path):
    """A positive observation remains DETECTED without a config threshold."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": True, "score": 2.0}'
    )
    config = tmp_path / "no-threshold.json"
    config.write_text('{"algorithm_name": "KGW"}')
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--config",
        str(config),
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "DETECTED"
    assert payload["detector_verdict"] == "DETECTED"
    assert payload["threshold"] is None
    assert payload["verdict_reason"] == "detector reported watermarked"


def test_cli_detect_negative_no_key_inconclusive(tmp_path: Path):
    """A negative with unknown provenance must NOT become 'clean'."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["detector_verdict"] == "NOT_DETECTED"
    assert payload["is_watermarked"] is False
    assert payload["provenance_match"] is None
    assert "provenance/key match not confirmed" in payload["verdict_reason"]
    # Human output carries the warning line (non-JSON mode).
    r2 = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream))
    assert r2.returncode == 0, r2.stderr
    assert "does NOT establish that the document is watermark-free" in r2.stdout


def test_cli_detect_negative_with_key_not_detected(tmp_path: Path):
    """A negative with an asserted key and enough tokens is a strong negative."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "kgw-test-v1",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "NOT_DETECTED"
    assert payload["provenance_match"] is True
    assert payload["key_id"] == "kgw-test-v1"


def test_cli_detect_abstention_band_inconclusive(tmp_path: Path):
    """A near-threshold score abstains even with a matching key."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.503}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "synthid-text",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "synthid-test-v1",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["threshold"] == 0.52
    assert "abstention band" in payload["verdict_reason"]


def test_cli_detect_short_text_inconclusive(tmp_path: Path):
    """Too few scored tokens abstains even with a matching key."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "kgw-test-v1",
        "--min-tokens",
        "1000",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert "below minimum calibrated length" in payload["verdict_reason"]


def test_cli_detect_unsupported_sidecar_scheme(tmp_path: Path):
    """A sidecar declaring an unknown scheme yields UNSUPPORTED, no model load."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": True, "score": 9.2}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    (tmp_path / "t.txt.wm.json").write_text(
        json.dumps({"scheme": "anthropic-synthid", "key_id": "claude-key"})
    )
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "UNSUPPORTED"
    assert payload["provenance_match"] is False
    assert "anthropic-synthid" in payload["verdict_reason"]


def test_cli_detect_sidecar_config_hash_match(tmp_path: Path):
    """Sidecar config_hash matching the detected config confirms provenance."""
    import hashlib

    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    (tmp_path / "t.txt.wm.json").write_text(
        json.dumps(
            {
                "scheme": "KGW",
                "key_id": "kgw-test-v1",
                "config_hash": hashlib.sha256(KGW_CONFIG.encode()).hexdigest(),
            }
        )
    )
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "NOT_DETECTED"
    assert payload["provenance_match"] is True
    assert payload["key_id"] == "kgw-test-v1"


def test_cli_detect_sidecar_config_hash_mismatch(tmp_path: Path):
    """Sidecar config_hash differing from the config is a provenance mismatch."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    (tmp_path / "t.txt.wm.json").write_text(
        json.dumps({"scheme": "KGW", "key_id": "kgw-test-v1", "config_hash": "deadbeef"})
    )
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["provenance_match"] is False


def test_cli_detect_sidecar_different_scheme_mismatch(tmp_path: Path):
    """A sidecar declaring a different supported scheme is a mismatch."""
    import hashlib

    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    (tmp_path / "t.txt.wm.json").write_text(
        json.dumps(
            {
                "scheme": "SynthID",
                "key_id": "synthid-key",
                "config_hash": hashlib.sha256(SYNTHID_CONFIG.encode()).hexdigest(),
            }
        )
    )
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["provenance_match"] is False


def test_cli_detect_sidecar_key_id_alone_stays_inconclusive(tmp_path: Path):
    """A document-supplied key_id (no config_hash) must not grant a clean result."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    (tmp_path / "t.txt.wm.json").write_text(json.dumps({"scheme": "KGW", "key_id": "attacker"}))
    r = _run_adapter("detect", str(f), "--scheme", "kgw", "--upstream-dir", str(upstream), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["provenance_match"] is None
    assert "provenance/key match not confirmed" in payload["verdict_reason"]


def test_cli_detect_few_scored_tokens_inconclusive(tmp_path: Path):
    """The length gate reads the detector's scored-token count, not the raw input."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25, "num_tokens": 5}'
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "kgw-test-v1",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    # The fake tokenizer reports 512 input tokens, so only the scored count
    # (5) can trip the default 200-token minimum.
    assert payload["input_tokens"] == 512
    assert payload["effective_scored_tokens"] == 5
    assert payload["verdict"] == "INCONCLUSIVE"
    assert "below minimum calibrated length (5 < 200 tokens)" in payload["verdict_reason"]


def test_cli_detect_unknown_scored_tokens_inconclusive(tmp_path: Path):
    """An unknown scored-token count must not fall through to a strong negative."""
    upstream = _make_fake_upstream(
        tmp_path, detect_literal='{"is_watermarked": False, "score": 0.25}'
    )
    # A tokenizer without .encode() makes both counts unknown.
    (upstream / "transformers" / "__init__.py").write_text(
        FAKE_TRANSFORMERS.replace(
            "    def encode(self, s):\n        return [0] * 512\n", "    pass\n"
        )
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "kgw-test-v1",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["input_tokens"] is None
    assert payload["effective_scored_tokens"] is None
    assert payload["verdict"] == "INCONCLUSIVE"
    assert "scored token count unknown" in payload["verdict_reason"]


def test_cli_detect_missing_score_is_error(tmp_path: Path):
    """A negative observation with no usable score is an ERROR, never clean."""
    upstream = _make_fake_upstream(tmp_path, detect_literal='{"is_watermarked": False}')
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "kgw-test-v1",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "ERROR"
    assert payload["detector_verdict"] == "NOT_DETECTED"
    assert payload["score"] is None
    assert "no score or threshold" in payload["verdict_reason"]
    r2 = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--key-id",
        "kgw-test-v1",
    )
    assert r2.returncode == 0, r2.stderr
    assert "does NOT establish that the document is watermark-free" in r2.stdout


def test_cli_watermark_writes_sidecar(tmp_path: Path):
    """watermark -o writes a provenance sidecar next to the sample."""
    upstream = _make_fake_upstream(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    wm_out = tmp_path / "wm.txt"
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "-o",
        str(wm_out),
        "--key-id",
        "kgw-test-v1",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    sidecar = json.loads((tmp_path / "wm.txt.wm.json").read_text())
    assert sidecar["scheme"] == "KGW"
    assert sidecar["key_id"] == "kgw-test-v1"
    assert sidecar["config_hash"]
    assert "implementation_commit" in sidecar  # fake upstream has no git
    assert "timestamp" in sidecar
    assert sidecar["watermark_parameters"] == {"algorithm_name": "KGW", "z_threshold": 4.0}


def test_cli_watermark_sidecar_redacts_secret_config_fields(tmp_path: Path):
    """The sidecar travels with the sample, so it must carry no key material."""
    upstream = _make_fake_upstream(tmp_path)
    secret_config = tmp_path / "secret_kgw.json"
    secret_config.write_text(
        json.dumps(
            {
                "algorithm_name": "KGW",
                "z_threshold": 4.0,
                "hash_key": 15485863,
                "nested": {"prf_keys": [1, 2, 3], "gamma": 0.5},
            }
        )
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("write about capybaras")
    wm_out = tmp_path / "wm.txt"
    r = _run_adapter(
        "watermark",
        str(prompt),
        "--scheme",
        "kgw",
        "-o",
        str(wm_out),
        "--config",
        str(secret_config),
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
    )
    assert r.returncode == 0, r.stderr
    raw = (tmp_path / "wm.txt.wm.json").read_text()
    assert "15485863" not in raw
    sidecar = json.loads(raw)
    params = sidecar["watermark_parameters"]
    assert params["hash_key"] == "[redacted]"
    assert params["nested"]["prf_keys"] == "[redacted]"
    assert params["nested"]["gamma"] == 0.5
    assert params["z_threshold"] == 4.0
    assert sidecar["config_hash"]


def test_cli_detect_applies_rlimit_as_in_child(tmp_path: Path):
    """--rlimit-as must cap the child's address space before heavy imports."""
    if os.name == "nt":
        pytest.skip("RLIMIT_AS is POSIX-only")
    upstream = _make_fake_upstream(tmp_path)
    (upstream / "transformers" / "__init__.py").write_text(
        FAKE_TRANSFORMERS + "import resource, sys\n"
        "print('RLIMIT_AS=' + str(resource.getrlimit(resource.RLIMIT_AS)[0]), file=sys.stderr)\n"
    )
    f = tmp_path / "t.txt"
    f.write_text("hello world")
    # 512 GiB: high enough to clear the macOS arm64 dyld VM reservation, low
    # enough to prove the cap is applied (the default soft limit is unlimited).
    limit = 512 * 2**30
    r = _run_adapter(
        "detect",
        str(f),
        "--scheme",
        "kgw",
        "--upstream-dir",
        str(upstream),
        "--device",
        "cpu",
        "--json",
        "--rlimit-as",
        str(limit),
    )
    assert r.returncode == 0, r.stderr
    assert f"RLIMIT_AS={limit}" in (r.stderr or "")
