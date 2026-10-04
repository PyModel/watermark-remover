"""Wheel install verification for v0.6.0.

Skip-unless-env is the designed pattern from ticket 06 (goal contract,
Execution order step 4): "MARKED skip-unless-env WM_PACKAGING_E2E=1 — CI-fast
default, explicit e2e on demand". The default suite collects this file and
skips it. ``WM_PACKAGING_E2E=1`` runs the release gate.

Build and verify outputs use pytest temporary storage (WM_PACKAGING_TMP can
choose its parent). Set WM_PACKAGING_DIST to export only successfully verified
artifacts into a new directory for CI publication. Servers are killed by recorded PID only.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Ticket 06: skip-unless-env WM_PACKAGING_E2E=1 (CI-fast default, explicit e2e
# on demand). Not a disabled test — the gate is the release verification command.
pytestmark = pytest.mark.skipif(
    os.environ.get("WM_PACKAGING_E2E") != "1",
    reason="ticket 06: skip unless WM_PACKAGING_E2E=1 (CI-fast default, explicit e2e on demand)",
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def packaging_output(tmp_path):
    parent = Path(os.environ.get("WM_PACKAGING_TMP", str(tmp_path)))
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wm-packaging-", dir=parent) as directory:
        output = Path(directory)
        try:
            yield output
        finally:
            _make_writable(output)


ENTRYPOINTS = ("wm", "wm-tui", "wm-serve", "wm-audit-dir", "wm-audit-site")
PRINT_ENV_KEYS = {
    "WM_TUI_PYTHON",
    "WM_TUI_BRIDGE",
    "WM_TUI_ARGV",
    "WM_TUI_LOG",
    "WM_TUI_CWD",
}
DEV_PACKAGES = ("typescript", "@types/bun", "bun-types")
NOTES_MARKERS = (
    "pi-tui",
    "CSI 2026",
    "500",
    "3 packages",
    "history",
    "undo",
    "paste",
    "autocomplete",
    "emoji",
    "code point",
    "pageup",
    "pagedown",
    "ctrl+k",
    "ctrl+j",
    "scrollback",
)


def _emit(capfd: pytest.CaptureFixture[str], text: str) -> None:
    """Print into the pytest process stdout so ``goal.mjs run`` logs the line."""
    with capfd.disabled():
        print(text, flush=True)


def _run(
    capfd: pytest.CaptureFixture[str],
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    display = " ".join(argv)
    _emit(capfd, f"$ {display}")
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    _emit(capfd, f"exit {result.returncode}")
    if result.stdout.strip():
        _emit(capfd, result.stdout.rstrip())
    if result.stderr.strip():
        _emit(capfd, result.stderr.rstrip())
    return result


def _chmod_tree(root: Path, directory_mode: int, file_mode: int) -> None:
    """chmod a tree without following symlinks.

    ``os.chmod`` follows links. A venv's ``bin/python`` is a symlink to the
    real interpreter; chmod'ing it would strip execute from that binary.
    """
    if not root.exists():
        return
    for directory, _subdirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        if not current.is_symlink():
            os.chmod(current, directory_mode)
        for name in files:
            path = current / name
            if path.is_symlink():
                continue
            os.chmod(path, file_mode)


def _chmod_readonly(root: Path) -> None:
    """Drop write bits. Owner read/exec stays so the launcher can copy the tree."""
    _chmod_tree(root, stat.S_IRUSR | stat.S_IXUSR, stat.S_IRUSR)


def _make_writable(root: Path) -> None:
    """Undo a read-only install simulation so the next run can delete the tree."""
    _chmod_tree(root, stat.S_IRWXU, stat.S_IRUSR | stat.S_IWUSR)


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return int(text) if text.isdigit() else None


def _kill(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return


def _assert_release_sources(capfd: pytest.CaptureFixture[str]) -> None:
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    version_line = next(line for line in pyproject.splitlines() if line.startswith("version = "))
    _emit(capfd, f"pyproject {version_line}")
    assert version_line == 'version = "0.6.0"'

    notes = REPO / "docs" / "release-notes" / "v0.6.0.md"
    assert notes.is_file(), "docs/release-notes/v0.6.0.md is absent"
    body = notes.read_text(encoding="utf-8")
    first = body.splitlines()[0]
    _emit(capfd, f"notes {first}")
    assert first.startswith("# v0.6.0 —")
    assert "## Changed" in body and "## Fixed" in body
    lowered = body.lower()
    missing = [marker for marker in NOTES_MARKERS if marker.lower() not in lowered]
    assert not missing, f"release notes omit {missing}"
    assert "Honesty contract:" in body
    assert "compare/v0.5.0...v0.6.0" in body

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    folded = readme.lower()
    assert "opentui" not in folded
    assert "solidjs" not in folded and "solid.js" not in folded
    assert "@earendil-works/pi-tui" in readme
    assert "PI_TUI_WRITE_LOG" in readme
    assert "--frozen-lockfile --production" in readme
    assert "three runtime packages" in readme
    assert all(f"`{name}`" in readme for name in ENTRYPOINTS)
    _emit(capfd, "readme: no OpenTUI; pi-tui + three-package production install + PI_TUI_WRITE_LOG")


def _write_helpers(out: Path, real_bun: str) -> tuple[Path, Path]:
    bun_wrap = out / "bun-wrap.sh"
    bun_wrap.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$WM_PACKAGING_BUN_LOG"\n'
        'exec "$WM_PACKAGING_REAL_BUN" "$@"\n',
        encoding="utf-8",
    )
    bun_wrap.chmod(0o755)
    tee = out / "tee-python.py"
    tee.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, subprocess\n"
        "from pathlib import Path\n"
        "Path(os.environ['WM_PACKAGING_WRAPPER_PID']).write_text(str(os.getpid()))\n"
        "proc = subprocess.Popen(\n"
        "    [os.environ['WM_PACKAGING_REAL_PYTHON'], *sys.argv[1:]],\n"
        "    stdin=sys.stdin.buffer,\n"
        "    stdout=subprocess.PIPE,\n"
        "    stderr=sys.stderr,\n"
        ")\n"
        "Path(os.environ['WM_PACKAGING_CHILD_PID']).write_text(str(proc.pid))\n"
        "assert proc.stdout is not None\n"
        "with open(os.environ['WM_PACKAGING_FRAME_LOG'], 'wb') as log:\n"
        "    while True:\n"
        "        chunk = proc.stdout.read(4096)\n"
        "        if not chunk:\n"
        "            break\n"
        "        log.write(chunk)\n"
        "        log.flush()\n"
        "        sys.stdout.buffer.write(chunk)\n"
        "        sys.stdout.buffer.flush()\n"
        "raise SystemExit(proc.wait())\n",
        encoding="utf-8",
    )
    tee.chmod(0o755)
    assert real_bun
    return bun_wrap, tee


def test_wheel_install_exercises_every_entrypoint(
    capfd: pytest.CaptureFixture[str], packaging_output: Path
) -> None:
    """Build, install the wheel, and exercise every console script plus first-run staging."""
    assert os.name != "nt", "the read-only staging check uses POSIX permissions"
    assert not (hasattr(os, "geteuid") and os.geteuid() == 0), "chmod a-w does not bind root"
    real_bun = shutil.which("bun")
    assert real_bun, "bun is required for the frontend boot smoke"

    _assert_release_sources(capfd)

    out = packaging_output
    dist = out / "dist"
    work = out / "work"
    work.mkdir()
    env = os.environ.copy()
    env["TMPDIR"] = str(work)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    installed = _run(
        capfd,
        [sys.executable, "-m", "pip", "install", "--upgrade", "build"],
        env=env,
        timeout=180,
    )
    assert installed.returncode == 0, installed.stderr
    built = _run(
        capfd,
        [sys.executable, "-m", "build", "--outdir", str(dist)],
        cwd=REPO,
        env=env,
        timeout=300,
    )
    assert built.returncode == 0, built.stderr
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1
    assert "0.6.0" in wheels[0].name
    _emit(capfd, f"sdist {sdists[0].name}")
    _emit(capfd, f"wheel {wheels[0].name}")

    venv = out / "venv"
    created = _run(capfd, [sys.executable, "-m", "venv", str(venv)], env=env)
    assert created.returncode == 0, created.stderr
    py = venv / "bin" / "python"
    pip = venv / "bin" / "pip"
    fresh_env = env.copy()
    fresh_env["VIRTUAL_ENV"] = str(venv)
    installed_wheel = _run(capfd, [str(pip), "install", str(wheels[0])], env=fresh_env, timeout=180)
    assert installed_wheel.returncode == 0, installed_wheel.stderr
    shown = _run(capfd, [str(pip), "show", "watermark-remover"], env=fresh_env)
    assert shown.returncode == 0
    assert "Version: 0.6.0" in shown.stdout

    for name in ENTRYPOINTS:
        script = venv / "bin" / name
        assert script.is_file(), f"missing console script {name}"
        helped = _run(capfd, [str(script), "--help"], env=fresh_env)
        assert helped.returncode == 0, helped.stderr

    cleaned = out / "out.cleaned.txt"
    fixture = REPO / "tests" / "fixtures" / "sample_watermarked.txt"
    clean = _run(
        capfd,
        [str(venv / "bin" / "wm"), str(fixture), "-o", str(cleaned)],
        cwd=REPO,
        env=fresh_env,
    )
    assert clean.returncode == 0, clean.stderr
    assert cleaned.is_file() and cleaned.stat().st_size > 0
    assert "removed=" in clean.stderr
    removed = int(clean.stderr.split("removed=", 1)[1].split()[0])
    assert removed > 0
    assert "\u200b" not in cleaned.read_text(encoding="utf-8")

    spec_path = out / "openapi.json"
    dumped = _run(
        capfd,
        [
            str(py),
            "-c",
            "import json,sys; from watermark_remover.scripts.server import openapi_spec; "
            "json.dump(openapi_spec(), open(sys.argv[1], 'w'))",
            str(spec_path),
        ],
        env=fresh_env,
    )
    assert dumped.returncode == 0, dumped.stderr
    validated = _run(
        capfd,
        [
            sys.executable,
            "-c",
            "import json,sys; from openapi_spec_validator import validate; "
            "spec = json.load(open(sys.argv[1])); validate(spec); "
            "print('openapi', spec['openapi'], 'valid')",
            str(spec_path),
        ],
        env=env,
    )
    assert validated.returncode == 0, validated.stderr
    assert "openapi 3.0.3 valid" in validated.stdout

    printed = _run(capfd, [str(venv / "bin" / "wm-tui"), "--print-env"], cwd=out, env=fresh_env)
    assert printed.returncode == 0, printed.stderr
    wm_env = json.loads(printed.stdout)
    assert set(wm_env) == PRINT_ENV_KEYS
    # sys.executable is the venv's python3.x symlink; Path.resolve() follows it
    # out to the base interpreter, so compare the un-resolved bin directory.
    reported_python = Path(wm_env["WM_TUI_PYTHON"])
    assert reported_python.parent == venv / "bin"
    assert reported_python.name.startswith("python")
    bridge = Path(wm_env["WM_TUI_BRIDGE"])
    assert bridge.is_file()
    assert "watermark_remover" in bridge.parts

    frontend = bridge.parent.parent / "tui"
    assert (frontend / "src" / "index.ts").is_file()
    _chmod_readonly(frontend)
    assert not os.access(frontend, os.W_OK)
    _emit(capfd, f"read-only install frontend {frontend}")

    cache = out / "cache"
    cache.mkdir()
    bun_log = out / "bun-argv.log"
    bun_wrap, tee = _write_helpers(out, real_bun)
    stage_env = fresh_env.copy()
    stage_env.update(
        {
            "XDG_CACHE_HOME": str(cache),
            "WM_PACKAGING_BUN": str(bun_wrap),
            "WM_PACKAGING_BUN_LOG": str(bun_log),
            "WM_PACKAGING_REAL_BUN": real_bun,
        }
    )
    staged = _run(
        capfd,
        [
            str(py),
            "-c",
            "import os, sys\n"
            "from watermark_remover.scripts import tui as launcher\n"
            "source = launcher.frontend_dir()\n"
            "print(f'source={source}')\n"
            "print(f'writable={os.access(source, os.W_OK)}')\n"
            "run_dir, status = launcher.prepare_frontend(source, os.environ['WM_PACKAGING_BUN'])\n"
            "print(f'run_dir={run_dir}')\n"
            "print(f'status={status}')\n"
            "sys.exit(0 if run_dir else 1)\n",
        ],
        env=stage_env,
        timeout=180,
    )
    assert staged.returncode == 0, staged.stderr
    run_dir = Path(
        next(
            line.split("=", 1)[1]
            for line in staged.stdout.splitlines()
            if line.startswith("run_dir=")
        )
    )
    expected = cache / "watermark-remover" / "tui-0.6.0"
    assert run_dir.resolve() == expected.resolve()
    assert os.access(run_dir, os.W_OK)
    assert not (frontend / "node_modules").exists()
    argv_log = bun_log.read_text(encoding="utf-8")
    _emit(capfd, f"bun argv: {argv_log.strip()}")
    assert "install --frozen-lockfile --production" in argv_log
    installed_names = sorted(
        path.name for path in (run_dir / "node_modules").iterdir() if not path.name.startswith(".")
    )
    _emit(capfd, "runtime node_modules: " + " ".join(installed_names))
    modules = run_dir / "node_modules"
    assert (modules / "@earendil-works" / "pi-tui").is_dir()
    assert (modules / "marked").is_dir()
    assert (modules / "get-east-asian-width").is_dir()
    for package in DEV_PACKAGES:
        assert not (modules / package).exists(), package
    assert not (modules / "typescript").exists()
    assert not (modules / "@types").exists()

    frame_log = out / "bridge-frames.log"
    wrapper_pid_path = out / "wrapper.pid"
    child_pid_path = out / "bridge.pid"
    frontend_out = out / "frontend.out"
    frontend_err = out / "frontend.err"
    boot_env = os.environ.copy()
    boot_env.update(
        {
            "WM_TUI_PYTHON": str(tee),
            "WM_TUI_BRIDGE": str(bridge),
            "WM_TUI_ARGV": json.dumps([str(fixture)]),
            "WM_TUI_LOG": str(out / "wm-tui-bridge.log"),
            "WM_TUI_CWD": str(REPO),
            "WM_PACKAGING_REAL_PYTHON": str(py),
            "WM_PACKAGING_FRAME_LOG": str(frame_log),
            "WM_PACKAGING_WRAPPER_PID": str(wrapper_pid_path),
            "WM_PACKAGING_CHILD_PID": str(child_pid_path),
            "TMPDIR": str(work),
        }
    )
    _emit(capfd, f"$ {real_bun} run src/index.ts  # pty-less, cwd={run_dir}")
    recorded: list[int] = []
    with frontend_out.open("wb") as stdout, frontend_err.open("wb") as stderr:
        proc = subprocess.Popen(
            [real_bun, "run", "src/index.ts"],
            cwd=run_dir,
            env=boot_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
        recorded.append(proc.pid)
        _emit(capfd, f"frontend pid {proc.pid}")
        # stdin is /dev/null, so the frontend quits after the handshake. Wait
        # for that exit, then read the tee — the ready frame flushes on close.
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.2)
        finally:
            # Kill the process we spawned before reading child pid files, so a
            # bad pid file cannot skip that kill.
            _kill(proc.pid)
            for pid in (_read_pid(wrapper_pid_path), _read_pid(child_pid_path)):
                if pid is not None:
                    recorded.append(pid)
                    _kill(pid)
            proc.wait(timeout=5)
    _emit(capfd, f"exit {proc.returncode} (stopped after handshake; pid {proc.pid})")
    # Re-read after the process exits: the tee flushes the ready frame as the
    # pipe closes, which can land after the last poll.
    frames = frame_log.read_text(encoding="utf-8", errors="replace") if frame_log.exists() else ""
    _emit(capfd, f"bridge protocol stdout: {frames.strip() or '<empty>'}")
    _emit(
        capfd,
        "mechanism: pty-less bun run src/index.ts; ready is a protocol stdout frame "
        "(not WM_TUI_LOG/stderr). A tee on WM_TUI_PYTHON records it; the frontend still consumes it.",
    )
    assert '"event": "ready"' in frames and '"version": "0.6.0"' in frames, frames
    # The frontend's hello request is answered with the preset list. That is the
    # handshake; stdin EOF then quits, so the alt-screen bytes may not flush.
    assert "Hidden marks" in frames and "Verifiable" in frames, frames
    for pid in recorded:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        else:
            raise AssertionError(f"process {pid} still alive")

    audited = _run(
        capfd,
        [str(venv / "bin" / "wm-audit-dir"), str(REPO / "tests" / "fixtures")],
        cwd=REPO,
        env=fresh_env,
    )
    assert audited.returncode in {0, 1}, audited.stderr
    assert "Files scanned:" in audited.stdout
    site_help = _run(capfd, [str(venv / "bin" / "wm-audit-site"), "--help"], env=fresh_env)
    assert site_help.returncode == 0, site_help.stderr
    _emit(capfd, "audit dry invocation: wm-audit-dir tests/fixtures; wm-audit-site --help")

    # Promote the very wheel installed above, not a second untested rebuild.
    # Refuse an existing directory so stale distributions cannot be published.
    if export := os.environ.get("WM_PACKAGING_DIST"):
        destination = Path(export).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        for artifact in (*wheels, *sdists):
            shutil.copy2(artifact, destination / artifact.name)
        _emit(capfd, f"verified distributions exported to {destination}")
