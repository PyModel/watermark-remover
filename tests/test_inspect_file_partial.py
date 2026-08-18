"""inspect_file.py must mark unscanned inputs and exit EXIT_PARTIAL (3).

Unrecognized and refused inputs were previously reported as clean (exit 0),
which let an incomplete audit pass CI as a clean signal. Unscanned results
carry ``"unscanned": true`` and the CLI exits 3, taking precedence over both
clean and suspicious findings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"
INSPECT_FILE = SCRIPTS / "inspect_file.py"

EXIT_PARTIAL = 3


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, str(INSPECT_FILE), *args],
        capture_output=True,
        text=True,
        env=merged,
        check=False,
    )


def test_unknown_input_is_unscanned_and_partial(tmp_path: Path) -> None:
    blob = tmp_path / "no_extension"
    blob.write_text("no magic, no extension\n", encoding="utf-8")

    r = _run(str(blob), "--json")
    assert r.returncode == EXIT_PARTIAL
    payload = json.loads(r.stdout)
    assert payload["kind"] == "unknown"
    assert payload["unscanned"] is True
    assert "note" in payload  # refusal message preserved


def test_oversized_input_is_unscanned_and_partial(tmp_path: Path) -> None:
    blob = tmp_path / "big.txt"
    blob.write_text("x" * 64, encoding="utf-8")

    r = _run(str(blob), "--json", env={"WATERMARKS_MAX_INPUT_BYTES": "16"})
    assert r.returncode == EXIT_PARTIAL
    payload = json.loads(r.stdout)
    assert payload["kind"] == "refused"
    assert payload["unscanned"] is True
    assert "larger than" in payload["note"]  # refusal message preserved


def test_clean_text_input_stays_clean(tmp_path: Path) -> None:
    src = tmp_path / "clean.txt"
    src.write_text("ordinary prose, no hidden marks\n", encoding="utf-8")

    r = _run(str(src), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["kind"] == "text"
    assert "unscanned" not in payload


def test_suspicious_text_input_keeps_exit_1(tmp_path: Path) -> None:
    src = tmp_path / "suspicious.txt"
    src.write_text("hidden\u200bmark\n", encoding="utf-8")

    r = _run(str(src), "--json")
    assert r.returncode == 1
    payload = json.loads(r.stdout)
    assert payload["kind"] == "text"
    assert payload["suspicious"] is True
    assert "unscanned" not in payload


def test_mixed_batch_partial_takes_precedence(tmp_path: Path) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("ordinary prose\n", encoding="utf-8")
    suspicious = tmp_path / "suspicious.txt"
    suspicious.write_text("hidden\u200bmark\n", encoding="utf-8")
    unknown = tmp_path / "no_extension"
    unknown.write_text("no magic, no extension\n", encoding="utf-8")

    r = _run(str(clean), str(suspicious), str(unknown), "--json")
    assert r.returncode == EXIT_PARTIAL
    payload = json.loads(r.stdout)
    assert payload["total"] == 3
    by_path = {Path(item["path"]).name: item for item in payload["results"]}
    assert "unscanned" not in by_path["clean.txt"]
    assert by_path["suspicious.txt"]["suspicious"] is True
    assert by_path["no_extension"]["unscanned"] is True
