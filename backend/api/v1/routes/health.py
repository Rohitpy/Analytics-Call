"""Liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend import __version__
from backend.api.deps import ContainerDep, SettingsDep
from backend.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter()


class ClientConfigResponse(BaseModel):
    """Server-owned limits, published so the UI validates the same way the
    backend does instead of hardcoding a second copy."""

    app_name: str
    max_upload_mb: int
    max_files_per_job: int
    allowed_extensions: list[str]
    pipeline_workers: int
    stt_backend: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok", app=settings.APP_NAME, version=__version__, env=settings.ENV
    )


@router.get(
    "/config",
    response_model=ClientConfigResponse,
    summary="Upload limits and backend configuration for clients",
)
async def client_config(settings: SettingsDep) -> ClientConfigResponse:
    return ClientConfigResponse(
        app_name=settings.APP_NAME,
        max_upload_mb=settings.MAX_UPLOAD_MB,
        max_files_per_job=settings.MAX_FILES_PER_JOB,
        allowed_extensions=sorted(settings.allowed_extensions),
        pipeline_workers=settings.PIPELINE_WORKERS,
        stt_backend=settings.STT_BACKEND,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe - checks ffmpeg, the STT model and the LLM server",
)
async def readiness(container: ContainerDep, settings: SettingsDep) -> ReadinessResponse:
    details: dict[str, str] = {}

    ffmpeg_path = container.audio.resolve_ffmpeg()
    ffmpeg_available = ffmpeg_path is not None
    if ffmpeg_available:
        details["ffmpeg"] = ffmpeg_path or ""
    else:
        details["ffmpeg"] = (
            f"not found at '{settings.FFMPEG_PATH}' or on PATH - conversion will fail"
        )

    llm_reachable, llm_model, llm_error = await container.llm.health()
    if llm_error:
        details["llm"] = llm_error

    taxonomy = container.taxonomy
    if not taxonomy.is_loaded:
        details["taxonomy"] = taxonomy.load_error or "using the built-in fallback"

    transcription = container.transcription
    stt_loaded = transcription.model_loaded
    if not stt_loaded and settings.STT_BACKEND.lower() != "mock":
        if transcription.loaded_count == 0:
            details["stt"] = "no instance loaded - it loads on the first call"
        else:
            # Partial load means one card failed; the rest still work.
            details["stt"] = (
                f"{transcription.loaded_count}/{transcription.instance_count} "
                f"instances loaded across {', '.join(transcription.devices)}"
            )

    ready = ffmpeg_available and llm_reachable and taxonomy.is_loaded
    return ReadinessResponse(
        status="ready" if ready else "degraded",
        stt_backend=settings.STT_BACKEND,
        stt_model_loaded=stt_loaded,
        stt_mode="multi" if settings.is_multi_gpu else "single",
        stt_devices=transcription.devices,
        stt_instances=transcription.instance_count,
        stt_instances_loaded=transcription.loaded_count,
        llm_base_url=settings.LLM_BASE_URL,
        llm_reachable=llm_reachable,
        llm_model=llm_model or settings.LLM_MODEL or None,
        ffmpeg_available=ffmpeg_available,
        taxonomy_themes=len(taxonomy.taxonomy.themes),
        workers=container.jobs.worker_count,
        queue_depth=container.jobs.queue_depth,
        details=details,
    )
