"""Import boundaries for the single cleaning execution seam."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "remove-ai-marks" / "scripts"


def _imports(module: str) -> set[str]:
    tree = ast.parse((SCRIPTS / module).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _defines_or_imports_dispatcher(module: str) -> bool:
    tree = ast.parse((SCRIPTS / module).read_text(encoding="utf-8"))
    return any(
        (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_clean_item"
        )
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "run_clean_item" for alias in node.names)
        )
        for node in ast.walk(tree)
    )


def test_cli_adapter_does_not_import_concrete_cleaners_or_dispatcher():
    imports = _imports("clean_file.py")
    assert not {"clean_asset", "image_meta", "container_meta", "text_unicode"} & imports
    assert not _defines_or_imports_dispatcher("clean_file.py")


def test_execution_dispatcher_is_owned_only_by_clean_pipeline():
    pipeline_source = (SCRIPTS / "clean_pipeline.py").read_text(encoding="utf-8")
    assert "clean_asset" in _imports("clean_pipeline.py")
    assert "def execute(" in pipeline_source
    for path in SCRIPTS.glob("*.py"):
        if path.name in {"clean_pipeline.py", "clean_asset.py"}:
            continue
        assert not _defines_or_imports_dispatcher(path.name), path.name


def test_tui_surfaces_do_not_import_cli_or_concrete_cleaners():
    for module in ("tui_core.py", "tui_bridge.py"):
        imports = _imports(module)
        assert "clean_file" not in imports, module
        assert not {"image_meta", "container_meta", "text_unicode"} & imports, module
