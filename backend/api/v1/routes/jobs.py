"""Job lifecycle: upload, inspect, stream progress, cancel, delete."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api.deps import ContainerDep, JobManagerDep, StoreDep, UploadDep
from backend.core.errors import AppError, ValidationError
from backend.core.logging_config import get_logger, log_context
from backend.schemas.common import JobStatus, MessageResponse
from backend.schemas.job import JobCreateResponse, JobDetail, JobRecord, JobSummary

logger = get_logger(__name__)
router = APIRouter()

# How often the SSE stream polls the store, and how often it emits a comment
# line to stop idle proxies from closing the connection.
SSE_POLL_SECONDS = 1.0
SSE_HEARTBEAT_SECONDS = 15.0


class JobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    jobs: list[JobSummary]


@router.post(
    "",
    response_model=JobCreateResponse,
    status_code=202,
    summary="Upload one or more calls and start processing",
)
async def create_job(
    container: ContainerDep,
    store: StoreDep,
    manager: JobManagerDep,
    uploads: UploadDep,
    files: Annotated[list[UploadFile], File(description="Audio files")],
    name: Annotated[str, Form()] = "",
) -> JobCreateResponse:
    """Returns as soon as the files are on disk; processing runs in the
    background. Poll GET /jobs/{id} or subscribe to /jobs/{id}/events."""
    if not files:
        raise ValidationError("No files were uploaded.")

    settings = container.settings
    if len(files) > settings.MAX_FILES_PER_JOB:
        raise ValidationError(
            f"{len(files)} files exceeds the per-job limit of "
            f"{settings.MAX_FILES_PER_JOB}."
        )

    job = JobRecord(
        name=name.strip() or f"Batch {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}"
    )
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []

    with log_context(job_id=job.id):
        sr_no = 1
        for upload in files:
            try:
                call = await uploads.save(upload, job.id, sr_no)
            except AppError as exc:
                # One bad file must not sink the batch.
                rejected.append(
                    {"filename": upload.filename or "unknown", "reason": exc.message}
                )
                continue
            except Exception as exc:
                rejected.append(
                    {
                        "filename": upload.filename or "unknown",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            job.calls.append(call)
            accepted.append(call.filename)
            sr_no += 1

        if not job.calls:
            raise ValidationError(
                "None of the uploaded files could be accepted.",
                details={"rejected": json.dumps(rejected)},
            )

        await store.create(job)
        await manager.submit(job)
        logger.info(
            "Job %s created: %d accepted, %d rejected",
            job.id,
            len(accepted),
            len(rejected),
        )

    return JobCreateResponse(
        job_id=job.id,
        status=job.status,
        total=job.total,
        accepted=accepted,
        rejected=rejected,
    )


@router.get("", response_model=JobListResponse, summary="List jobs, newest first")
async def list_jobs(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: JobStatus | None = None,
) -> JobListResponse:
    jobs, total = await store.list_jobs(limit=limit, offset=offset, status=status)
    return JobListResponse(
        total=total,
        limit=limit,
        offset=offset,
        jobs=[JobSummary.from_record(j) for j in jobs],
    )


@router.get("/{job_id}", response_model=JobDetail, summary="Job status and per-call progress")
async def get_job(job_id: str, store: StoreDep) -> JobDetail:
    return JobDetail.from_record(await store.get(job_id))


@router.get(
    "/{job_id}/events",
    summary="Server-sent progress stream",
    response_class=StreamingResponse,
)
async def job_events(job_id: str, request: Request, store: StoreDep) -> StreamingResponse:
    """Emits a JobDetail payload whenever the job changes, then closes once it
    reaches a terminal state. Falls back gracefully - the UI polls if the
    EventSource errors."""
    await store.get(job_id)  # 404 before opening the stream

    async def event_stream() -> AsyncIterator[str]:
        last_revision = -1
        since_heartbeat = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    return

                job = store.get_optional(job_id)
                if job is None:
                    yield _sse("deleted", {"job_id": job_id})
                    return

                if job.revision != last_revision:
                    last_revision = job.revision
                    since_heartbeat = 0.0
                    yield _sse(
                        "update", json.loads(JobDetail.from_record(job).model_dump_json())
                    )

                if job.status.is_terminal:
                    yield _sse("done", {"job_id": job_id, "status": job.status.value})
                    return

                await asyncio.sleep(SSE_POLL_SECONDS)
                since_heartbeat += SSE_POLL_SECONDS
                if since_heartbeat >= SSE_HEARTBEAT_SECONDS:
                    since_heartbeat = 0.0
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
        },
    )


@router.post(
    "/{job_id}/cancel",
    response_model=JobDetail,
    summary="Cancel queued calls (running calls finish)",
)
async def cancel_job(job_id: str, manager: JobManagerDep) -> JobDetail:
    return JobDetail.from_record(await manager.cancel(job_id))


@router.delete(
    "/{job_id}", response_model=MessageResponse, summary="Delete a job and its files"
)
async def delete_job(
    job_id: str, store: StoreDep, container: ContainerDep
) -> MessageResponse:
    job = await store.get(job_id)
    if not job.status.is_terminal:
        raise ValidationError(
            "This job is still running. Cancel it first, then delete."
        )

    settings = container.settings
    await store.delete(job_id)

    def _cleanup() -> None:
        shutil.rmtree(settings.upload_dir / job_id, ignore_errors=True)
        shutil.rmtree(settings.work_dir / job_id, ignore_errors=True)
        (settings.results_dir / f"{job_id}.xlsx").unlink(missing_ok=True)

    await asyncio.to_thread(_cleanup)
    return MessageResponse(message=f"Job {job_id} and its artefacts were deleted.")


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
