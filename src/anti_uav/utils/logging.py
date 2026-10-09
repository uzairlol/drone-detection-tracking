"""Logging setup: human-readable on a TTY, single-line JSON otherwise.

The edge deployment is headless and log lines get shipped over MQTT, so machine
parsing matters more than colour. ``--log-json`` forces JSON even on a console;
a TTY without that flag gets a compact colourised format.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Final

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)-38s %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

_RESERVED: Final = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"asctime", "message", "taskName"}
)

_configured = False


class _JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per record.

    Anything passed via ``extra=`` is hoisted to the top level so log queries
    like ``jq '.dataset == "dvb"'`` work without nesting gymnastics.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str, separators=(",", ":"))


class _ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        grey = "\x1b[38;20m"
        bold = "\x1b[1m"
        reset = "\x1b[0m"
        self._plain = not sys.stderr.isatty() or os.environ.get("NO_COLOR") is not None
        if not self._plain:
            self._plain = True
            return
        self._fmt = _LOG_FORMAT
        self._bold = bold
        self._grey = grey
        self._reset = reset

    def format(self, record: logging.LogRecord) -> str:
        if self._plain:
            return super().format(record)
        colour = {
            logging.DEBUG: self._grey,
            logging.INFO: "",
            logging.WARNING: "\x1b[33m",
            logging.ERROR: "\x1b[31m",
            logging.CRITICAL: "\x1b[1;31m",
        }.get(record.levelno, "")
        stamp = self.formatTime(record, _DATE_FORMAT)
        return (
            f"{self._grey}{stamp}{self._reset} {colour}{record.levelname:<7}{self._reset} "
            f"{self._grey}{record.name:<38}{self._reset} {record.getMessage()}"
        )


def setup_logging(
    level: str | int | None = None,
    *,
    json_output: bool | None = None,
    log_file: str | None = None,
) -> None:
    """Configure root logging once, idempotently."""
    global _configured

    if level is None:
        level = os.environ.get("ANTI_UAV_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if json_output is None:
        json_output = os.environ.get("ANTI_UAV_LOG_JSON", "").lower() in {"1", "true", "yes"}

    if _configured:
        logging.getLogger().setLevel(level)
        return

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = _JsonFormatter() if json_output else _ConsoleFormatter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)

    # Third-party chatter we do not control.
    for noisy in ("urllib3", "filelock", "matplotlib", "PIL", "httpx", "httpcore", "engineio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Module logger. Always route through this instead of ``logging.getLogger``."""
    if not name.startswith("anti_uav"):
        name = f"anti_uav.{name}"
    return logging.getLogger(name)
