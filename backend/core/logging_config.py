"""Logging setup.

Two things worth knowing:

1. `job_id` / `call_id` are contextvars. Anything logged inside a pipeline
   task automatically carries them, so a single call can be traced across
   the ffmpeg -> whisper -> translate -> classify stages without every log
   line having to pass identifiers around by hand.
2. Output goes to stdout *and* a rotating file under `<storage>/logs`.
   Set LOG_JSON=true for one-JSON-object-per-line, which is what you want
   if these logs are shipped anywhere.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from backend.core.config import Settings

job_id_var: ContextVar[str] = ContextVar("job_id", default="-")
call_id_var: ContextVar[str] = ContextVar("call_id", default="-")
stage_var: ContextVar[str] = ContextVar("stage", default="-")

_CONFIGURED = False


class ContextFilter(logging.Filter):
    """Stamp every record with the current job/call/stage."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = job_id_var.get()
        record.call_id = call_id_var.get()
        record.stage = stage_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "job_id": getattr(record, "job_id", "-"),
            "call_id": getattr(record, "call_id", "-"),
            "stage": getattr(record, "stage", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


TEXT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-38s | "
    "job=%(job_id)s call=%(call_id)s stage=%(stage)s | %(message)s"
)


def configure_logging(settings: Settings) -> None:
    """Idempotent - safe to call from both the CLI and the app factory."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter: logging.Formatter = (
        JsonFormatter() if settings.LOG_JSON else logging.Formatter(TEXT_FORMAT)
    )
    context_filter = ContextFilter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(context_filter)

    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_dir / "theme_analytics.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # uvicorn installs its own handlers; drop them so we do not double-print.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # These are chatty at DEBUG and add nothing here.
    for name in ("httpx", "httpcore", "openai", "multipart", "numba", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def log_context(
    job_id: str | None = None,
    call_id: str | None = None,
    stage: str | None = None,
) -> Iterator[None]:
    """Bind identifiers for the duration of a block, then restore."""
    tokens = []
    if job_id is not None:
        tokens.append((job_id_var, job_id_var.set(job_id)))
    if call_id is not None:
        tokens.append((call_id_var, call_id_var.set(call_id)))
    if stage is not None:
        tokens.append((stage_var, stage_var.set(stage)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
