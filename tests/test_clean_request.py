"""CleanRequest owns the option surface both the CLI and the TUI build plans from."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "remove-ai-marks" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_file
from clean_request import (
    CleanRequest,
    build_clean_plan,
    build_rewrite_plan,
    describe_dropped_text_transforms,
    dropped_text_transforms,
)

ZWSP = "a​b"


def _parse(argv: list[str]) -> CleanRequest:
    args = clean_file._build_parser().parse_args(argv)
    return CleanRequest.from_args(args)


# --- the argparse adapter must stay mechanical -------------------------------


def test_from_args_covers_every_parser_destination():
    """Any flag added to the parser must be carried on CleanRequest.

    Without this, a new flag silently reaches the CLI and never the TUI — the
    exact drift the seam exists to prevent.
    """
    parser = clean_file._build_parser()
    destinations = {
        action.dest for action in parser._actions if action.dest not in ("help", "path")
    }
    request = _parse(["file.txt"])
    fields = set(CleanRequest.__dataclass_fields__)
    missing = destinations - fields
    assert not missing, f"parser flags absent from CleanRequest: {sorted(missing)}"
    assert request.paths == (Path("file.txt"),)


def test_defaults_match_bare_cli_invocation():
    assert _parse(["file.txt"]) == CleanRequest(paths=(Path("file.txt"),))


def test_rejects_unknown_enumerations():
    with pytest.raises(ValueError, match="unknown rewrite strength"):
        CleanRequest(rewrite="not-a-strength")
    with pytest.raises(ValueError, match="unsupported forced asset kind"):
        CleanRequest(force_type="binary")
    with pytest.raises(ValueError, match="unknown quality profile"):
        CleanRequest(quality="ultra")
    with pytest.raises(ValueError, match="timeout must be"):
        CleanRequest(timeout=0)


# --- Layer B reaches every strength, not just tsapa --------------------------


@pytest.mark.parametrize(
    "strength",
    ["paraphrase", "backtranslate", "structural", "humanize", "code", "tsapa"],
)
def test_every_rewrite_strength_builds_a_live_plan(monkeypatch, strength):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "test-model")
    plan = build_rewrite_plan(CleanRequest(rewrite=strength))
    assert plan is not None
    assert plan.strength == strength
    assert plan.backend == "ollama"


def test_tsapa_flag_is_an_alias_for_rewrite_tsapa(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "test-model")
    assert _parse(["f.txt", "--tsapa"]).rewrite_strength == "tsapa"
    assert _parse(["f.txt", "--rewrite", "tsapa"]).rewrite_strength == "tsapa"
    assert _parse(["f.txt"]).rewrite_strength is None


def test_explicit_settings_win_over_environment(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_BACKEND", "ollama")
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "from-env")
    monkeypatch.setenv("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434")
    plan = build_rewrite_plan(
        CleanRequest(
            rewrite="humanize",
            rewrite_model="explicit-model",
            rewrite_base_url="http://localhost:9999",
            rewrite_candidates=4,
            rewrite_temperature=0.5,
        )
    )
    assert plan.model == "explicit-model"
    assert plan.base_url == "http://localhost:9999"
    assert plan.candidates == 4
    assert plan.temperature == 0.5


def test_tsapa_label_survives_in_the_error_message(monkeypatch):
    monkeypatch.delenv("WATERMARKS_REWRITE_BACKEND", raising=False)
    with pytest.raises(ValueError, match=r"--tsapa requires a live backend"):
        build_rewrite_plan(CleanRequest(tsapa=True))
    with pytest.raises(ValueError, match=r"--rewrite humanize requires a live backend"):
        build_rewrite_plan(CleanRequest(rewrite="humanize"))


def test_api_key_never_appears_in_a_repr():
    request = CleanRequest(rewrite_api_key="unit-test-secret-never-real")
    assert "unit-test-secret-never-real" not in repr(request)


# --- text-only transforms off the text path ----------------------------------


def test_non_text_kinds_name_the_transforms_they_drop():
    request = CleanRequest(rewrite="humanize", char_perturb=True, nfkc=True)
    assert dropped_text_transforms(request, "text") == ()
    dropped = dropped_text_transforms(request, "container")
    assert "Layer B rewrite (humanize)" in dropped
    assert "character perturbation" in dropped
    assert "NFKC normalization" in dropped
    message = describe_dropped_text_transforms(request, "container", Path("draft.md"))
    assert "draft.md" in message
    assert "--as text" in message


def test_dropping_is_a_note_not_a_failure(monkeypatch):
    """A mixed batch must still plan: text-only flags are no-ops on images."""
    monkeypatch.delenv("WATERMARKS_REWRITE_BACKEND", raising=False)
    plan = build_clean_plan(CleanRequest(rewrite="humanize"), Path("out.png"), "image")
    assert plan.text.rewrite_plan is None


def test_markdown_run_reports_the_skipped_rewrite(tmp_path: Path):
    source = tmp_path / "draft.md"
    source.write_text(ZWSP, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "clean_file.py"),
            str(source),
            "-o",
            str(tmp_path / "draft.cleaned.md"),
            "--rewrite",
            "humanize",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "skipped Layer B rewrite (humanize)" in result.stderr


# --- the generated command line ----------------------------------------------


def test_command_line_round_trips_through_the_parser():
    request = _parse(
        [
            "a.txt",
            "b.txt",
            "-o",
            "out",
            "--nfkc",
            "--rewrite",
            "humanize",
            "--rewrite-model",
            "qwen3",
            "--rewrite-candidates",
            "3",
            "--char-perturb",
            "--char-mode",
            "confusable",
            "--seed",
            "7",
        ]
    )
    argv = request.command_line()
    assert argv[0] == "wm"
    assert _parse(argv[1:]) == request


def test_command_line_never_carries_an_api_key():
    request = CleanRequest(
        paths=(Path("a.txt"),),
        rewrite="humanize",
        rewrite_api_key="unit-test-secret-never-real",
    )
    assert "unit-test-secret-never-real" not in " ".join(request.command_line())


def test_command_line_of_a_default_request_is_just_the_paths():
    assert CleanRequest(paths=(Path("a.txt"),)).command_line() == ["wm", "a.txt"]
