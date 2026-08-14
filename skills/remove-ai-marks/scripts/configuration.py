"""Configuration system for the watermark-remover tool.

Reads settings from pyproject.toml → .env → environment variables → defaults.
Each setting tracks its source for observability.

Supported environment variables (all optional):
    WATERMARKS_MAX_FILE_SIZE        — Max input file size in bytes (default: 100 MB)
    WATERMARKS_MAX_IMAGE_PIXELS     — Max pixels for image ops (default: 40 000 000)
    WATERMARKS_MAX_CONCURRENCY      — Batch concurrency limit (default: 4)
    WATERMARKS_LOG_LEVEL            — DEBUG/INFO/WARNING/ERROR (default: INFO)
    WATERMARKS_REWRITE_TIMEOUT      — Seconds for Layer B rewrite (default: 120)
    WATERMARKS_MAX_REWRITE_GENERATIONS — TSAPA max generations (default: 5)
    WATERMARKS_MAX_REWRITE_POPULATION  — TSAPA max population (default: 12)
    WATERMARKS_REWRITE_BACKEND      — ollama/openai-compatible/print-prompt
    WATERMARKS_REWRITE_BASE_URL     — Backend base URL
    WATERMARKS_REWRITE_MODEL        — Backend model name
    WATERMARKS_REWRITE_API_KEY      — Backend API key
    WATERMARKS_REWRITE_DISABLE_THINKING — Disable thinking on reasoning models
    WATERMARKS_PLL_MODEL            — PLL scorer model
    WATERMARKS_EMBED_MODEL          — Embedding scorer model
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any


class ConfigSource(Enum):
    """Where a setting value came from."""

    PYPROJECT = auto()
    ENV_FILE = auto()
    ENV_VAR = auto()
    DEFAULT = auto()


@dataclass(frozen=True)
class ConfigSetting:
    """A single configuration value with provenance."""

    value: Any
    source: ConfigSource
    description: str


@dataclass(frozen=True)
class ConfigSummary:
    """Full configuration summary with provenance."""

    settings: dict[str, ConfigSetting]
    pyproject_path: Path | None
    env_file_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: {"value": setting.value, "source": setting.source.name}
            for key, setting in self.settings.items()
        }


# ---------------------------------------------------------------------------
# TOML reader (stdlib only, Python 3.11+)
# ---------------------------------------------------------------------------

def _read_toml(path: Path) -> dict[str, Any] | None:
    """Minimal TOML reader for pyproject.toml [tool.watermark] section.

    Falls back to None for Python < 3.11 or if toml is unavailable.
    """
    try:
        import tomllib  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        # Python < 3.11 without tomlkit
        return None

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    return data.get("tool", {}).get("watermark", None)


def _read_env_file(path: Path) -> dict[str, str]:
    """Read key=value lines from a .env-style file, ignoring comments."""
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                result[key] = value
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------
# Settings definitions
# ---------------------------------------------------------------------------

_SETTINGS_DEF: list[tuple[str, Any, str, ConfigSource]] = [
    (
        "max_file_size",
        100 * 1024 * 1024,
        "Max input file size in bytes",
        ConfigSource.DEFAULT,
    ),
    (
        "max_image_pixels",
        40_000_000,
        "Max pixel count for image operations",
        ConfigSource.DEFAULT,
    ),
    (
        "max_concurrency",
        4,
        "Concurrent batch file limit",
        ConfigSource.DEFAULT,
    ),
    (
        "log_level",
        "INFO",
        "Logging level: DEBUG, INFO, WARNING, ERROR",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_timeout",
        120.0,
        "Seconds timeout for Layer B rewrite operations",
        ConfigSource.DEFAULT,
    ),
    (
        "max_rewrite_generations",
        5,
        "Default TSAPA maximum generations",
        ConfigSource.DEFAULT,
    ),
    (
        "max_rewrite_population",
        12,
        "Default TSAPA maximum population size",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_backend",
        "print-prompt",
        "Layer B rewrite backend: print-prompt, ollama, openai-compatible",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_base_url",
        "http://127.0.0.1:11434",
        "Layer B backend base URL",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_model",
        None,
        "Layer B backend model name",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_api_key",
        None,
        "Layer B backend API key",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_disable_thinking",
        False,
        "Disable thinking mode on reasoning-capable models",
        ConfigSource.DEFAULT,
    ),
    (
        "pll_model",
        None,
        "Pseudo-log-likelihood scorer model",
        ConfigSource.DEFAULT,
    ),
    (
        "embed_model",
        None,
        "Embedding scorer model",
        ConfigSource.DEFAULT,
    ),
]


def _bool_from_str(value: str) -> bool:
    """Convert a string to bool, raising ValueError on invalid input."""
    low = value.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a bool: {value!r}")


def _int_from_str(value: str, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _float_from_str(value: str, default: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_configuration(project_root: Path | None = None) -> ConfigSummary:
    """Load configuration from all sources with provenance tracking.

    Priority (highest wins):
        1. Environment variables
        2. .env file in project root
        3. pyproject.toml [tool.watermark]
        4. Built-in defaults
    """
    if project_root is None:
        project_root = Path.cwd()

    pyproject_path = project_root / "pyproject.toml"
    env_file_path = project_root / ".env"

    # 1. Read pyproject.toml
    toml_section = _read_toml(pyproject_path) if pyproject_path.is_file() else None

    # 2. Read .env file
    env_values = _read_env_file(env_file_path) if env_file_path.is_file() else {}

    settings: dict[str, ConfigSetting] = {}
    env_var_map: dict[str, tuple[str, str]] = {
        "max_file_size": ("WATERMARKS_MAX_FILE_SIZE", "int"),
        "max_image_pixels": ("WATERMARKS_MAX_IMAGE_PIXELS", "int"),
        "max_concurrency": ("WATERMARKS_MAX_CONCURRENCY", "int"),
        "log_level": ("WATERMARKS_LOG_LEVEL", "str"),
        "rewrite_timeout": ("WATERMARKS_REWRITE_TIMEOUT", "float"),
        "max_rewrite_generations": ("WATERMARKS_MAX_REWRITE_GENERATIONS", "int"),
        "max_rewrite_population": ("WATERMARKS_MAX_REWRITE_POPULATION", "int"),
        "rewrite_backend": ("WATERMARKS_REWRITE_BACKEND", "str"),
        "rewrite_base_url": ("WATERMARKS_REWRITE_BASE_URL", "str"),
        "rewrite_model": ("WATERMARKS_REWRITE_MODEL", "str"),
        "rewrite_api_key": ("WATERMARKS_REWRITE_API_KEY", "str"),
        "rewrite_disable_thinking": ("WATERMARKS_REWRITE_DISABLE_THINKING", "bool"),
        "pll_model": ("WATERMARKS_PLL_MODEL", "str"),
        "embed_model": ("WATERMARKS_EMBED_MODEL", "str"),
    }

    for key, default_val, description, _ in _SETTINGS_DEF:
        env_var_name = env_var_map.get(key)
        value = default_val
        source = ConfigSource.DEFAULT

        # Check pyproject.toml
        if toml_section is not None and key in toml_section:
            value = toml_section[key]
            source = ConfigSource.PYPROJECT

        # Check .env file
        if env_var_name and env_var_name[0] in env_values:
            raw = env_values[env_var_name[0]]
            vtype = env_var_name[1]
            if vtype == "int":
                value = _int_from_str(raw, default_val)
            elif vtype == "float":
                value = _float_from_str(raw, default_val)
            elif vtype == "bool":
                value = _bool_from_str(raw)
            else:
                value = raw
            source = ConfigSource.ENV_FILE

        # Check environment variable
        if env_var_name and env_var_name[0] in os.environ:
            raw = os.environ[env_var_name[0]]
            vtype = env_var_name[1]
            if vtype == "int":
                value = _int_from_str(raw, default_val)
            elif vtype == "float":
                value = _float_from_str(raw, default_val)
            elif vtype == "bool":
                value = _bool_from_str(raw)
            else:
                value = raw
            source = ConfigSource.ENV_VAR

        settings[key] = ConfigSetting(
            value=value,
            source=source,
            description=description,
        )

    return ConfigSummary(
        settings=settings,
        pyproject_path=pyproject_path if pyproject_path.is_file() else None,
        env_file_path=env_file_path if env_file_path.is_file() else None,
    )


def get_config(summary: ConfigSummary) -> dict[str, Any]:
    """Extract flat settings dict from a ConfigSummary for easy access."""
    return {key: setting.value for key, setting in summary.settings.items()}


def print_config_summary(summary: ConfigSummary) -> None:
    """Print a human-readable configuration summary to stderr."""
    sys.stderr.write("Configuration:\n")
    for key, setting in summary.settings.items():
        marker = {"pyproject": "toml", "env_file": ".env", "env_var": "env"}.get(
            setting.source.name.lower(), "?"
        )
        sys.stderr.write(f"  {key} = {setting.value!r}  [{marker}]\n")
    sys.stderr.flush()
