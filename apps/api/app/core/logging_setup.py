"""Shared structlog/stdlib logging bootstrap for API and ARQ worker processes."""

import logging
import logging.config
import sys

import structlog
from structlog.dev import ConsoleRenderer

_CONFIGURED = False


def _is_local_environment() -> bool:
    """Check if running in local development environment."""
    import os

    env = os.environ.get("ENVIRONMENT", "").lower()
    return env in ("", "local", "development", "dev")


def configure_logging(log_level: str) -> None:
    """Configure structured logging with environment-appropriate format.

    - Local/development: Human-readable console output with colors
    - Production: JSON output for log aggregation (Datadog, Splunk, etc.)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Avoid Windows console encoding crashes when payloads contain non-ASCII text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    is_local = _is_local_environment()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if is_local:
        # Local development: Human-readable console output
        # TimeStamper already adds timestamp in shared_processors
        formatter_config = {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                ConsoleRenderer(
                    colors=True,
                    pad_event=40,
                ),
            ],
            "foreign_pre_chain": shared_processors,
        }
    else:
        # Production: JSON output for log aggregation
        formatter_config = {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            "foreign_pre_chain": shared_processors,
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": formatter_config,
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                },
                **(
                    {
                        "file": {
                            "class": "logging.handlers.RotatingFileHandler",
                            "filename": "logs/arq.log",
                            "maxBytes": 10 * 1024 * 1024,
                            "backupCount": 5,
                            "formatter": "json",
                            "encoding": "utf-8",
                            "errors": "backslashreplace",
                        }
                    }
                    if is_local
                    else {}
                ),
            },
            "root": {
                "handlers": ["default", *(["file"] if is_local else [])],
                "level": log_level,
            },
            "loggers": {
                "uvicorn.access": {"level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "asyncpg": {"level": "WARNING"},
            },
        }
    )

    _CONFIGURED = True
