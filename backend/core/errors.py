"""Typed application errors + the handlers that turn them into JSON."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.core.logging_config import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for anything we raise on purpose."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class StageError(AppError):
    """A pipeline stage (ffmpeg / stt / translation / classification) failed.

    Carries the stage name so the failure can be reported per call instead of
    sinking the whole batch.
    """

    status_code = 502
    code = "stage_failed"

    def __init__(self, stage: str, message: str, *, details: dict | None = None):
        super().__init__(message, details=details)
        self.stage = stage


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError [%s]: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {},
                }
            },
        )
