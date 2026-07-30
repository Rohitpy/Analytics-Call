"""Results: the report rows the UI table renders, per-call detail, and the
Excel download."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.api.deps import ContainerDep, StoreDep
from backend.core.errors import NotFoundError
from backend.core.logging_config import get_logger
from backend.schemas.common import CallStatus
from backend.schemas.job import ResultsResponse, StageTimings
from backend.schemas.theme import (
    ClassificationResult,
    TranscriptionResult,
    TranslationResult,
)
from backend.services import excel

logger = get_logger(__name__)
router = APIRouter()


class CallDetail(BaseModel):
    """Everything known about one call - powers the UI detail drawer."""

    id: str
    sr_no: int
    filename: str
    status: CallStatus
    error: str | None = None
    failed_stage: str | None = None
    size_bytes: int = 0
    transcription: TranscriptionResult | None = None
    translation: TranslationResult | None = None
    classification: ClassificationResult | None = None
    timings: StageTimings


@router.get(
    "/{job_id}/results",
    response_model=ResultsResponse,
    summary="Report rows - identical in shape to the Excel sheet",
)
async def get_results(
    job_id: str,
    store: StoreDep,
    include_pending: Annotated[bool, Query()] = True,
    transcript_chars: Annotated[
        int,
        Query(
            ge=0,
            description=(
                "Clip transcripts to this many characters in the list payload. "
                "0 returns them in full. A 200-call batch of full transcripts "
                "is tens of megabytes, so the table asks for previews and the "
                "detail endpoint serves the full text."
            ),
        ),
    ] = 1200,
) -> ResultsResponse:
    job = await store.get(job_id)
    rows = excel.build_result_rows(job)
    if not include_pending:
        rows = [r for r in rows if r.status.is_terminal]

    truncated = False
    if transcript_chars:
        for row in rows:
            if len(row.transcription) > transcript_chars:
                row.transcription = row.transcription[:transcript_chars] + " ..."
                truncated = True
            row.original_transcription = ""  # full text lives on the detail route

    return ResultsResponse(
        job_id=job.id,
        status=job.status,
        columns=excel.REPORT_COLUMNS,
        fields=excel.REPORT_FIELDS,
        total=len(rows),
        rows=rows,
        excel_available=bool(job.excel_path),
        transcripts_truncated=truncated,
    )


@router.get(
    "/{job_id}/results/{call_id}",
    response_model=CallDetail,
    summary="Full detail for a single call",
)
async def get_call_detail(job_id: str, call_id: str, store: StoreDep) -> CallDetail:
    job = await store.get(job_id)
    call = job.find_call(call_id)
    if call is None:
        raise NotFoundError(f"Call '{call_id}' was not found in job '{job_id}'.")
    return CallDetail(
        id=call.id,
        sr_no=call.sr_no,
        filename=call.filename,
        status=call.status,
        error=call.error,
        failed_stage=call.failed_stage,
        size_bytes=call.size_bytes,
        transcription=call.transcription,
        translation=call.translation,
        classification=call.classification,
        timings=call.timings,
    )


@router.get(
    "/{job_id}/export",
    response_class=FileResponse,
    summary="Download the Excel report",
)
async def export_excel(job_id: str, store: StoreDep, container: ContainerDep):
    """Available before the job finishes too - rows still in flight simply show
    as PENDING, which is handy for a long batch."""
    job = await store.get(job_id)

    path = Path(job.excel_path) if job.excel_path else None
    if path is None or not path.exists():
        path = container.settings.results_dir / f"{job_id}.xlsx"
        logger.info("Generating report on demand for job %s", job_id)
        await asyncio.to_thread(excel.write_report, job, path)

        def _attach(record) -> None:
            record.excel_path = str(path)

        await store.update(job_id, _attach)

    safe_name = (job.name or job_id).replace(" ", "_").replace("/", "-")
    return FileResponse(
        path=path,
        filename=f"theme_analysis_{safe_name}.xlsx",
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )
