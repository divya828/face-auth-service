"""Structured CloudWatch-ready JSON logging."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as a flat JSON object CloudWatch can index."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote structured metrics attached via `extra={"metrics": {...}}`.
        metrics = getattr(record, "metrics", None)
        if isinstance(metrics, dict):
            payload.update(metrics)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("face-auth")


log = configure_logging()


def emit(action: str, **metrics: Any) -> None:
    """Emit a structured metric line (inference_ms, db_ms, total_ms, user_id...)."""
    log.info(action, extra={"metrics": {"action": action, **metrics}})
