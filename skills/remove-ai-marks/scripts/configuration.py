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
    WATERMARKS_REWRITE_ALLOW_REMOTE — Allow non-loopback rewrite endpoints
    WATERMARKS_REWRITE_REASONING_EFFORT — OpenAI-compatible reasoning effort
    WATERMARKS_PLL_MODEL            — PLL scorer model
    WATERMARKS_EMBED_MODEL          — Embedding scorer model
    WATERMARKS_MAX_INPUT_BYTES      — Max input bytes (default: 256 MiB)
    WATERMARKS_MAX_STDIN_BYTES      — Max stdin bytes (default: 64 MiB)
    WATERMARKS_MAX_FILE_SIZE        — Deprecated alias of WATERMARKS_MAX_INPUT_BYTES
    WATERMARKS_SERVER_HOST          — HTTP service bind host
    WATERMARKS_SERVER_PORT          — HTTP service bind port
    WATERMARKS_SERVER_API_KEY       — HTTP service bearer API key
    WATERMARKS_GEMINI_API_KEY       — Gemini text-detector API key
    WATERMARKS_GEMINI_MODEL         — Gemini text-detector model
    WATERMARKS_GEMINI_TIMEOUT       — Gemini text-detector timeout
    WATERMARKS_GEMINI_MAX_CHARS     — Gemini text-detector max chars
    WATERMARKS_MARKLLM_SCHEME       — MarkLLM detection scheme
    WATERMARKS_MARKLLM_TIMEOUT      — MarkLLM timeout
    WATERMARKS_MARKLLM_RLIMIT_AS    — MarkLLM child RLIMIT_AS
    WATERMARKS_SYNTHID_SCORER_URL   — SynthID scorer HTTP URL
    WATERMARKS_SYNTHID_SCORER_API_KEY — SynthID scorer bearer API key
    WATERMARKS_SYNTHID_SCORER_TIMEOUT — SynthID scorer timeout
    WATERMARKS_SYNTHID_SERVER_HOST  — SynthID sidecar bind host
    WATERMARKS_SYNTHID_SERVER_PORT  — SynthID sidecar bind port
    WATERMARKS_SYNTHID_SERVER_VERSION — SynthID sidecar version
    WATERMARKS_SERVICE_URL          — Optional thin-client service URL
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
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
    """Full configuration summary with provenance and parse failures."""

    settings: dict[str, ConfigSetting]
    pyproject_path: Path | None
    env_file_path: Path | None
    parse_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            key: {"value": setting.value, "source": setting.source.name}
            for key, setting in self.settings.items()
        }
        data["parse_errors"] = dict(self.parse_errors)
        return data


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
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
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
        "rewrite_allow_remote",
        False,
        "Allow non-loopback Layer B rewrite endpoints",
        ConfigSource.DEFAULT,
    ),
    (
        "rewrite_reasoning_effort",
        "none",
        "OpenAI-compatible reasoning effort: none, low, medium, high, off",
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
    (
        "max_input_bytes",
        256 * 1024 * 1024,
        "Max input bytes (256 MiB default; WATERMARKS_MAX_FILE_SIZE aliases here)",
        ConfigSource.DEFAULT,
    ),
    (
        "max_stdin_bytes",
        64 * 1024 * 1024,
        "Max stdin bytes (64 MiB default)",
        ConfigSource.DEFAULT,
    ),
    (
        "server_host",
        "127.0.0.1",
        "HTTP service bind host",
        ConfigSource.DEFAULT,
    ),
    (
        "server_port",
        8765,
        "HTTP service bind port",
        ConfigSource.DEFAULT,
    ),
    (
        "server_api_key",
        None,
        "HTTP service bearer API key",
        ConfigSource.DEFAULT,
    ),
    (
        "gemini_api_key",
        None,
        "Gemini text-detector API key",
        ConfigSource.DEFAULT,
    ),
    (
        "gemini_model",
        "gemini-2.5-flash",
        "Gemini text-detector model",
        ConfigSource.DEFAULT,
    ),
    (
        "gemini_timeout",
        30.0,
        "Gemini text-detector timeout seconds",
        ConfigSource.DEFAULT,
    ),
    (
        "gemini_max_chars",
        1_000_000,
        "Gemini text-detector max chars",
        ConfigSource.DEFAULT,
    ),
    (
        "markllm_scheme",
        "kgw",
        "MarkLLM detection scheme",
        ConfigSource.DEFAULT,
    ),
    (
        "markllm_timeout",
        600.0,
        "MarkLLM timeout seconds",
        ConfigSource.DEFAULT,
    ),
    (
        "markllm_rlimit_as",
        8 * 1024 * 1024 * 1024,
        "MarkLLM child RLIMIT_AS",
        ConfigSource.DEFAULT,
    ),
    (
        "synthid_scorer_url",
        None,
        "SynthID scorer HTTP URL",
        ConfigSource.DEFAULT,
    ),
    (
        "synthid_scorer_api_key",
        None,
        "SynthID scorer bearer API key",
        ConfigSource.DEFAULT,
    ),
    (
        "synthid_scorer_timeout",
        60.0,
        "SynthID scorer timeout seconds",
        ConfigSource.DEFAULT,
    ),
    (
        "synthid_server_host",
        "127.0.0.1",
        "SynthID sidecar bind host",
        ConfigSource.DEFAULT,
    ),
    (
        "synthid_server_port",
        8766,
        "SynthID sidecar bind port",
        ConfigSource.DEFAULT,
    ),
    (
        "synthid_server_version",
        None,
        "SynthID sidecar version",
        ConfigSource.DEFAULT,
    ),
    (
        "service_url",
        None,
        "Optional thin-client service URL",
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


def _convert_raw(raw: str, vtype: str) -> Any:
    """Convert a raw config string to its declared type, raising on failure."""
    if vtype == "int":
        return int(raw)
    if vtype == "float":
        return float(raw)
    if vtype == "bool":
        return _bool_from_str(raw)
    return raw


def _convert_or_record(parse_errors: dict[str, str], key: str, raw: str, vtype: str) -> Any | None:
    """Convert a raw value; record failures and return None to keep the default.

    Loaded configuration never crashes on a bad value: the failure is reported
    through ``ConfigSummary.parse_errors`` instead of being silently absorbed.
    """
    try:
        return _convert_raw(raw, vtype)
    except ValueError as error:
        parse_errors.setdefault(key, str(error))
        return None


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
    parse_errors: dict[str, str] = {}
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
        "rewrite_allow_remote": ("WATERMARKS_REWRITE_ALLOW_REMOTE", "bool"),
        "rewrite_reasoning_effort": ("WATERMARKS_REWRITE_REASONING_EFFORT", "str"),
        "pll_model": ("WATERMARKS_PLL_MODEL", "str"),
        "embed_model": ("WATERMARKS_EMBED_MODEL", "str"),
        "max_input_bytes": ("WATERMARKS_MAX_INPUT_BYTES", "int"),
        "max_stdin_bytes": ("WATERMARKS_MAX_STDIN_BYTES", "int"),
        "server_host": ("WATERMARKS_SERVER_HOST", "str"),
        "server_port": ("WATERMARKS_SERVER_PORT", "int"),
        "server_api_key": ("WATERMARKS_SERVER_API_KEY", "str"),
        "gemini_api_key": ("WATERMARKS_GEMINI_API_KEY", "str"),
        "gemini_model": ("WATERMARKS_GEMINI_MODEL", "str"),
        "gemini_timeout": ("WATERMARKS_GEMINI_TIMEOUT", "float"),
        "gemini_max_chars": ("WATERMARKS_GEMINI_MAX_CHARS", "int"),
        "markllm_scheme": ("WATERMARKS_MARKLLM_SCHEME", "str"),
        "markllm_timeout": ("WATERMARKS_MARKLLM_TIMEOUT", "float"),
        "markllm_rlimit_as": ("WATERMARKS_MARKLLM_RLIMIT_AS", "int"),
        "synthid_scorer_url": ("WATERMARKS_SYNTHID_SCORER_URL", "str"),
        "synthid_scorer_api_key": ("WATERMARKS_SYNTHID_SCORER_API_KEY", "str"),
        "synthid_scorer_timeout": ("WATERMARKS_SYNTHID_SCORER_TIMEOUT", "float"),
        "synthid_server_host": ("WATERMARKS_SYNTHID_SERVER_HOST", "str"),
        "synthid_server_port": ("WATERMARKS_SYNTHID_SERVER_PORT", "int"),
        "synthid_server_version": ("WATERMARKS_SYNTHID_SERVER_VERSION", "str"),
        "service_url": ("WATERMARKS_SERVICE_URL", "str"),
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
            converted = _convert_or_record(
                parse_errors, key, env_values[env_var_name[0]], env_var_name[1]
            )
            if converted is not None:
                value = converted
                source = ConfigSource.ENV_FILE

        # Check environment variable
        if env_var_name and env_var_name[0] in os.environ:
            converted = _convert_or_record(
                parse_errors, key, os.environ[env_var_name[0]], env_var_name[1]
            )
            if converted is not None:
                value = converted
                source = ConfigSource.ENV_VAR

        # Deprecated alias: WATERMARKS_MAX_FILE_SIZE maps onto max_input_bytes.
        # The canonical key WATERMARKS_MAX_INPUT_BYTES wins when both are set.
        if (
            key == "max_input_bytes"
            and "WATERMARKS_MAX_FILE_SIZE" in os.environ
            and "WATERMARKS_MAX_INPUT_BYTES" not in os.environ
        ):
            converted = _convert_or_record(
                parse_errors, key, os.environ["WATERMARKS_MAX_FILE_SIZE"], "int"
            )
            if converted is not None:
                value = converted
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
        parse_errors=parse_errors,
    )


def get_config(summary: ConfigSummary) -> dict[str, Any]:
    """Extract flat settings dict from a ConfigSummary for easy access."""
    return {key: setting.value for key, setting in summary.settings.items()}


def print_config_summary(summary: ConfigSummary) -> None:
    """Print a human-readable configuration summary to stderr."""
    sys.stderr.write("Configuration:\n")
    for key, setting in summary.settings.items():
        marker = {
            "pyproject": "toml",
            "env_file": ".env",
            "env_var": "env",
            "default": "default",
        }.get(setting.source.name.lower(), "?")
        sys.stderr.write(f"  {key} = {setting.value!r}  [{marker}]\n")
    for key, error in summary.parse_errors.items():
        sys.stderr.write(f"  ! {key}: {error}\n")
    sys.stderr.flush()
