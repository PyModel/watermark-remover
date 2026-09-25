#!/usr/bin/env python3
"""wm-tui: launch the interactive terminal UI over the watermark-remover pipeline.

The UI itself is a TypeScript program on Bun (``tui/``); every decision about a
clean is made by the Python bridge it spawns (``tui_bridge.py``, contract in
``tui/PROTOCOL.md``).  This module only does what must happen before a
full-screen program takes the terminal:

* refuse a missing path the way ``wm`` does, instead of opening a UI whose only
  content is an error;
* find ``bun`` and say how to get it when it is absent, rather than dying with
  ``FileNotFoundError``;
* install the frontend's dependencies once, into a writable copy when the
  package lives somewhere read-only (a wheel in site-packages);
* hand the frontend the interpreter it must spawn the bridge with, so the
  bridge never runs on some other ``python`` from PATH.

It imports nothing from the pipeline on purpose: this runs on every start, and
the bridge pays for the pipeline import once it is actually needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

BUN_INSTALL_HINT = (
    "wm-tui needs Bun (https://bun.sh) to run its terminal frontend.\n"
    "Install it with:  curl -fsSL https://bun.sh/install | bash\n"
    '(Windows: powershell -c "irm bun.sh/install.ps1 | iex"), then run wm-tui again.\n'
    "The plain CLI (`wm FILE`) works without it."
)

#: Files copied when the frontend has to be staged into a writable cache.
#: ``node_modules`` is never copied: it is platform-specific and rebuilt there.
_COPY_IGNORE = shutil.ignore_patterns("node_modules", ".git", "*.log")


def build_parser() -> argparse.ArgumentParser:
    """The ``wm-tui`` argument surface. The bridge re-parses ``WM_TUI_ARGV`` with it."""
    parser = argparse.ArgumentParser(
        prog="wm-tui",
        description="Interactive terminal UI for watermark-remover.",
    )
    parser.add_argument(
        "path",
        nargs="*",
        help="File(s) or director(ies) to open. Defaults to the current directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into directories when listing files",
    )
    parser.add_argument("--glob", default="*", help="Directory glob for the file list")
    parser.add_argument("--extensions", default=None, help="Comma-separated extension allow-list")
    setup = parser.add_mutually_exclusive_group()
    setup.add_argument(
        "--setup",
        action="store_true",
        help="Open the first-run setup screen even though a saved setup exists",
    )
    setup.add_argument(
        "--no-setup",
        action="store_true",
        help="Never open the setup screen, even on a first run",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Debug: print the WM_TUI_* environment the frontend would get, as JSON, and exit",
    )
    return parser


def package_version() -> str:
    """The installed distribution's version, or ``dev`` from a bare checkout."""
    try:
        return version("watermark-remover")
    except PackageNotFoundError:
        return "dev"


def cache_root() -> Path:
    """The per-user cache root: ``XDG_CACHE_HOME``, else the platform default."""
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"])
    return Path.home() / ".cache"


def frontend_dir() -> Path:
    """Where the frontend lives: ``WM_TUI_DIR``, else ``tui/`` beside ``scripts/``.

    The fallback resolves in a checkout (``skills/remove-ai-marks/tui``) and in
    a wheel (``watermark_remover/tui``) alike, because package-data ships the
    directory at the same place relative to this file.
    """
    override = os.environ.get("WM_TUI_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent.parent / "tui"


def bridge_path() -> Path:
    return Path(__file__).resolve().parent / "tui_bridge.py"


def frontend_env(argv: list[str]) -> dict[str, str]:
    """The ``WM_TUI_*`` variables the frontend receives (PROTOCOL "Environment").

    ``WM_TUI_CWD`` is the one addition: the frontend must run with its own
    directory as cwd (Bun reads ``bunfig.toml`` from there), so the bridge needs
    to be told where the operator's relative paths are relative to.
    """
    log = os.environ.get("WM_TUI_LOG") or str(cache_root() / "watermark-remover" / "tui.log")
    return {
        "WM_TUI_PYTHON": sys.executable,
        "WM_TUI_BRIDGE": str(bridge_path()),
        "WM_TUI_ARGV": json.dumps(argv),
        "WM_TUI_LOG": log,
        "WM_TUI_CWD": os.getcwd(),
    }


def _stage_frontend(source: Path) -> Path:
    """Copy a read-only frontend into the cache so ``bun install`` has somewhere to write.

    Keyed by package version: an upgrade gets a fresh copy instead of running
    new sources against the previous version's dependency tree.
    """
    target = cache_root() / "watermark-remover" / f"tui-{package_version()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    # copyfile, not copy2: a read-only source (a Nix-style store, 0o444 files)
    # must not produce a read-only copy that the next launch cannot refresh.
    shutil.copytree(
        source,
        target,
        ignore=_COPY_IGNORE,
        dirs_exist_ok=True,
        copy_function=shutil.copyfile,
    )
    # copytree still copies each directory's mode; bun install writes into
    # the root, and the next refresh writes into every directory.
    for directory, _subdirs, _files in os.walk(target):
        mode = os.stat(directory).st_mode
        os.chmod(directory, mode | stat.S_IWUSR | stat.S_IXUSR)
    return target


def _install(directory: Path, bun: str) -> int:
    """``bun install --frozen-lockfile``: the lockfile, not the network, decides versions."""
    print(
        f"wm-tui: installing frontend dependencies in {directory} (first run only)...",
        file=sys.stderr,
    )
    result = subprocess.run(
        [bun, "install", "--frozen-lockfile"],
        cwd=directory,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"wm-tui: `bun install --frozen-lockfile` failed in {directory} "
            f"(exit {result.returncode})",
            file=sys.stderr,
        )
    return result.returncode


def prepare_frontend(source: Path, bun: str) -> tuple[Path | None, int]:
    """Return a directory with installed dependencies to run from, or an exit code."""
    if not (source / "package.json").is_file() or not (source / "src" / "index.tsx").is_file():
        print(
            f"wm-tui: the terminal frontend is missing from {source} "
            "(expected package.json and src/index.tsx). Set WM_TUI_DIR to its directory.",
            file=sys.stderr,
        )
        return None, 1
    run_dir = source
    if not (source / "node_modules").is_dir():
        if not os.access(source, os.W_OK):
            try:
                run_dir = _stage_frontend(source)
            except OSError as error:
                print(f"wm-tui: cannot stage the frontend into the cache: {error}", file=sys.stderr)
                return None, 1
        if not (run_dir / "node_modules").is_dir():
            code = _install(run_dir, bun)
            if code != 0:
                return None, 1
    return run_dir, 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    # Refuse a typo here, the way ``wm`` does, rather than open a full-screen
    # UI whose only content is an error in the status bar.
    missing = [path for path in (args.path or ["."]) if not Path(path).exists()]
    if missing:
        for path in missing:
            print(f"wm-tui: no such file or directory: {path}", file=sys.stderr)
        return 2

    # The frontend gets the arguments minus the launcher's own debug switch.
    passed = [item for item in raw if item != "--print-env"]
    wm_env = frontend_env(passed)
    if args.print_env:
        # Only the WM_TUI_* values: the inherited environment can carry the
        # rewrite API key, and a debug flag must not be how it leaks.
        print(json.dumps(wm_env, indent=2, sort_keys=True))
        return 0

    bun = shutil.which("bun")
    if bun is None:
        print(BUN_INSTALL_HINT, file=sys.stderr)
        return 1

    run_dir, code = prepare_frontend(frontend_dir(), bun)
    if run_dir is None:
        return code

    log_dir = Path(wm_env["WM_TUI_LOG"]).parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"wm-tui: cannot create the log directory {log_dir}: {error}", file=sys.stderr)
        return 1

    env = {**os.environ, **wm_env}
    command = [bun, "run", "src/index.tsx"]
    if os.name == "nt":
        # No exec on Windows that keeps the console; wait and pass the code on.
        return subprocess.run(command, cwd=run_dir, env=env, check=False).returncode
    os.chdir(run_dir)
    # Replace this process: the frontend owns the terminal and its signals,
    # and a Python parent waiting on it would only be one more thing to kill.
    os.execvpe(bun, command, env)  # noqa: S606 - argv list, no shell
    return 0  # pragma: no cover - execvpe does not return


if __name__ == "__main__":
    raise SystemExit(main())
