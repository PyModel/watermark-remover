"""Structured logging for the watermark-remover tool.

Provides a module-level logger singleton with structured JSON output.
All log output goes to stderr so stdout stays reserved for CLI payloads
(including machine-readable --json output).

Usage:
    from structured_log import log_warning, log_info, log_error, init_logger

    log_warning("C2PA still present", module="container_meta")
    log_info("wrote cleaned file", module="clean_asset", path="/tmp/out.png")
    log_error("invalid input", module="clean_file", exit_code=2)

    # Or full logger for structured output:
    logger = init_logger()
    logger.info("cleaning asset", module="clean_asset", path=str(path))

The default log level comes from the shared configuration system
(configuration.load_configuration): WATERMARKS_LOG_LEVEL env var, .env, or
pyproject.toml [tool.watermark], falling back to INFO.
"""

from __future__ import annotations

import datetime
import json
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class LogLevel(Enum):
    """Logging levels matching Python's logging module."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass(frozen=True, slots=True)
class _LogEntry:
    """A single structured log entry."""

    ts: str
    level: str
    module: str
    msg: str
    extra: dict[str, Any] = field(default_factory=dict)


def _make_entry(level: str, msg: str, module: str = "", **kwargs: Any) -> _LogEntry:
    return _LogEntry(
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        level=level,
        module=module,
        msg=msg,
        extra=kwargs,
    )


def _emit(entry: _LogEntry) -> None:
    """Write one log entry to stderr as JSON (with extras) or plain text."""
    if entry.extra:
        data = {
            "ts": entry.ts,
            "level": entry.level,
            "module": entry.module,
            "msg": entry.msg,
            "extra": entry.extra,
        }
        print(json.dumps(data, ensure_ascii=False, default=str), file=sys.stderr)
    else:
        sys.stderr.write(f"[{entry.level}] {entry.module}: {entry.msg}\n")
    sys.stderr.flush()


# Quick helpers — always on stderr; JSON when extra fields are provided.
def log_debug(msg: str, module: str = "", **kwargs: Any) -> None:
    _emit(_make_entry("DEBUG", msg, module, **kwargs))


def log_info(msg: str, module: str = "", **kwargs: Any) -> None:
    _emit(_make_entry("INFO", msg, module, **kwargs))


def log_warning(msg: str, module: str = "", **kwargs: Any) -> None:
    _emit(_make_entry("WARNING", msg, module, **kwargs))


def log_error(msg: str, module: str = "", **kwargs: Any) -> None:
    _emit(_make_entry("ERROR", msg, module, **kwargs))


def log_critical(msg: str, module: str = "", **kwargs: Any) -> None:
    _emit(_make_entry("CRITICAL", msg, module, **kwargs))


# ---------------------------------------------------------------------------
# Logger class — full structured logging
# ---------------------------------------------------------------------------


class Logger:
    """Structured logger that writes JSON lines to stderr."""

    def __init__(self, level: LogLevel = LogLevel.INFO) -> None:
        self._level = level
        self._level_map = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
        }
        self._min_level = self._level_map.get(level, 1)

    def _should_log(self, level: LogLevel) -> bool:
        return self._level_map.get(level, 0) >= self._min_level

    def _log(self, level: LogLevel, msg: str, module: str = "", **kwargs: Any) -> None:
        if not self._should_log(level):
            return
        entry = _make_entry(level.name, msg, module, **kwargs)
        _emit(entry)

    def debug(self, msg: str, module: str = "", **kwargs: Any) -> None:
        self._log(LogLevel.DEBUG, msg, module=module, **kwargs)

    def info(self, msg: str, module: str = "", **kwargs: Any) -> None:
        self._log(LogLevel.INFO, msg, module=module, **kwargs)

    def warning(self, msg: str, module: str = "", **kwargs: Any) -> None:
        self._log(LogLevel.WARNING, msg, module=module, **kwargs)

    def error(self, msg: str, module: str = "", **kwargs: Any) -> None:
        self._log(LogLevel.ERROR, msg, module=module, **kwargs)

    def critical(self, msg: str, module: str = "", **kwargs: Any) -> None:
        self._log(LogLevel.CRITICAL, msg, module=module, **kwargs)

    def structured(self, level: str, msg: str, module: str = "", **kwargs: Any) -> None:
        """Log a structured entry with a custom level string."""
        entry = _make_entry(level, msg, module, **kwargs)
        _emit(entry)


def init_logger(log_level: str | LogLevel | None = None) -> Logger:
    """Create and return a Logger configured from shared configuration.

    Precedence when ``log_level`` is omitted: WATERMARKS_LOG_LEVEL env var,
    .env file, pyproject.toml [tool.watermark], then the INFO default.
    Invalid level strings fall back to INFO; unknown levels never crash.
    """
    if log_level is None:
        from configuration import load_configuration

        summary = load_configuration()
        raw = summary.settings.get("log_level")
        log_level = raw.value if raw is not None else "INFO"
    if isinstance(log_level, str):
        log_level = _parse_log_level(log_level)
    return Logger(level=log_level)


def _parse_log_level(raw: str) -> LogLevel:
    mapping = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "WARN": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
    }
    return mapping.get(raw.upper().strip(), LogLevel.INFO)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_logger: Logger | None = None


def get_logger() -> Logger:
    """Return the global logger singleton, initializing it lazily."""
    global _default_logger  # noqa: PLW0603
    if _default_logger is None:
        _default_logger = init_logger()
    return _default_logger
