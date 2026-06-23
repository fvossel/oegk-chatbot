"""Logging configuration for the OEKG chatbot.

Two log streams are provided:

- a **request log** (``logs/requests.log``) capturing every user request and
  its lifecycle (cache hits, generated query, result size, duration);
- an **error log** (``logs/errors.log``) capturing warnings and exceptions.

Use :func:`configure_logging` once at start-up and :func:`get_request_logger`
wherever request-level events should be recorded.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from oekg.config import AppConfig, get_config

REQUEST_LOGGER_NAME = "oekg.requests"

_configured = False


def _build_handler(path, level: int, fmt: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def configure_logging(config: AppConfig | None = None) -> None:
    """Configure application logging. Idempotent across reruns/imports."""
    global _configured
    if _configured:
        return

    config = config or get_config()
    config.log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, config.log_level.upper(), logging.INFO)

    # Request logger: structured, info-level lifecycle events.
    request_logger = logging.getLogger(REQUEST_LOGGER_NAME)
    request_logger.setLevel(level)
    request_logger.propagate = False
    request_logger.handlers.clear()
    request_logger.addHandler(
        _build_handler(
            config.request_log_path,
            level,
            "%(asctime)s | %(levelname)s | %(message)s",
        )
    )

    # Root logger: everything WARNING and above also lands in the error log.
    error_handler = _build_handler(
        config.error_log_path,
        logging.WARNING,
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    root = logging.getLogger()
    if not any(getattr(h, "_oekg_error_handler", False) for h in root.handlers):
        error_handler._oekg_error_handler = True  # type: ignore[attr-defined]
        root.addHandler(error_handler)
    if root.level > logging.WARNING or root.level == logging.NOTSET:
        root.setLevel(logging.WARNING)

    _configured = True


def get_request_logger() -> logging.Logger:
    """Return the request logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(REQUEST_LOGGER_NAME)
