"""FastAPI application factory and entrypoint.

Run with:
    python -m backend.main
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend import __version__
from backend.api.v1.router import api_router
from backend.container import ServiceContainer
from backend.core.config import Settings, get_settings
from backend.core.errors import register_exception_handlers
from backend.core.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    settings.ensure_directories()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = ServiceContainer(settings)
        app.state.container = container
        logger.info("Starting %s v%s (%s)", settings.APP_NAME, __version__, settings.ENV)
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    app = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        description=(
            "Call centre theme detection: Whisper large-v3 transcription, "
            "LLM translation, and prompt-driven theme/issue classification, "
            "reported as Excel."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # The UI is a separate Streamlit process (streamlit_app/), so this service
    # serves the API only. "/" goes to the docs rather than 404ing.
    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.ENV.lower() == "dev",
        log_config=None,  # logging_config.py owns the handlers
    )
