"""Structured logging for the watermark-remover tool.

Provides a module-level logger singleton with structured JSON output.
Falls back to plain stderr printing when JSON is not requested.

Usage:
    from structured_log import log_warning, log_info, log_error, init_logger

    log_warning("C2PA still present", module="container_meta")
    log_info("wrote cleaned file", module="clean_asset", path="/tmp/out.png")
    log_error("invalid input", module="clean_file", exit_code=2)

    # Or full logger for structured output:
    logger = init_logger()
    logger.info("cleaning asset", module="clean_asset", path=str(path))
"""

from __future__ import annotations

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

    @classmethod
    def _iso_now(cls) -> str:
        """Return current time as ISO 8601 string."""
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _make_entry(level: str, msg: str, module: str = "", **kwargs: Any) -> _LogEntry:
    return _LogEntry(
        ts=_LogEntry._iso_now(),
        level=level,
        module=module,
        msg=msg,
        extra=kwargs,
    )


def _emit(entry: _LogEntry, stderr: bool = False) -> None:
    """Output a log entry as a structured JSON line or plain text."""
    if entry.extra:
        data = {
            "ts": entry.ts,
            "level": entry.level,
            "module": entry.module,
            "msg": entry.msg,
            "extra": entry.extra,
        }
        print(json.dumps(data, ensure_ascii=False, default=str), file=sys.stderr if stderr else sys.stdout)
    else:
        target = sys.stderr if stderr else sys.stdout
        target.write(f"[{entry.level}] {entry.module}: {entry.msg}\n")
    sys.stderr.flush()


# Quick helpers — always formatted, never JSON unless extra provided
def log_debug(msg: str, module: str = "", **kwargs: Any) -> None:
    entry = _make_entry("DEBUG", msg, module, **kwargs)
    _emit(entry, stderr=(bool(kwargs) or len(kwargs) > 0))


def log_info(msg: str, module: str = "", **kwargs: Any) -> None:
    entry = _make_entry("INFO", msg, module, **kwargs)
    _emit(entry, stderr=False)


def log_warning(msg: str, module: str = "", **kwargs: Any) -> None:
    entry = _make_entry("WARNING", msg, module, **kwargs)
    _emit(entry, stderr=True)


def log_error(msg: str, module: str = "", **kwargs: Any) -> None:
    entry = _make_entry("ERROR", msg, module, **kwargs)
    _emit(entry, stderr=True)


def log_critical(msg: str, module: str = "", **kwargs: Any) -> None:
    entry = _make_entry("CRITICAL", msg, module, **kwargs)
    _emit(entry, stderr=True)


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
        _emit(entry, stderr=True)

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
        _emit(entry, stderr=True)


def init_logger(log_level: LogLevel | None = None) -> Logger:
    """Create and return a Logger configured from the environment.

    Reads WATERMARKS_LOG_LEVEL from the environment; defaults to INFO.
    """
    if log_level is None:
        raw = _get_env_log_level()
        log_level = _parse_log_level(raw)
    return Logger(level=log_level)


def _get_env_log_level() -> str:
    return (
        sys.modules.get("__main__", {}).get("WATERMARKS_LOG_LEVEL", "INFO")
        if "__main__" in sys.modules
        else ""
    )


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
    global _default_logger
    if _default_logger is None:
        _default_logger = init_logger()
    return _default_logger
