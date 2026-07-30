"""Job / call records (the persisted state machine) and the API response
models built from them."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from backend.schemas.common import CallStatus, JobStatus, Stage
from backend.schemas.theme import (
    ClassificationResult,
    TranscriptionResult,
    TranslationResult,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class StageTimings(BaseModel):
    """Wall-clock seconds per stage - useful for spotting the bottleneck."""

    convert: float | None = None
    transcribe: float | None = None
    translate: float | None = None
    classify: float | None = None
    total: float | None = None


# --------------------------------------------------------------------------
# Persisted records
# --------------------------------------------------------------------------
class CallRecord(BaseModel):
    """One uploaded audio file and everything we learn about it."""

    id: str = Field(default_factory=lambda: _new_id("call"))
    sr_no: int
    filename: str
    stored_path: str
    size_bytes: int = 0

    status: CallStatus = CallStatus.QUEUED
    stage: Stage = Stage.QUEUED
    error: str | None = None
    failed_stage: str | None = None
    attempts: int = 0

    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    transcription: TranscriptionResult | None = None
    translation: TranslationResult | None = None
    classification: ClassificationResult | None = None
    timings: StageTimings = Field(default_factory=StageTimings)

    # ---- convenience accessors used by the report + API ----
    @property
    def final_transcript(self) -> str:
        """The English transcript the classifier saw - that is what belongs in
        the report."""
        if self.translation and self.translation.text:
            return self.translation.text
        if self.transcription and self.transcription.text:
            return self.transcription.text
        return ""

    @property
    def original_transcript(self) -> str:
        return self.transcription.text if self.transcription else ""

    @property
    def progress(self) -> int:
        if self.status is CallStatus.COMPLETED:
            return 100
        if self.status in (CallStatus.FAILED, CallStatus.CANCELLED):
            return 100
        return self.stage.progress


class JobRecord(BaseModel):
    """A batch upload. One job -> many calls -> one Excel file."""

    id: str = Field(default_factory=lambda: _new_id("job"))
    name: str = ""
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    calls: list[CallRecord] = Field(default_factory=list)
    excel_path: str | None = None
    error: str | None = None
    cancel_requested: bool = False

    # Bumped on every mutation so the SSE stream knows when to emit.
    revision: int = 0

    # ---- counters ----
    @property
    def total(self) -> int:
        return len(self.calls)

    @property
    def completed(self) -> int:
        return sum(1 for c in self.calls if c.status is CallStatus.COMPLETED)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.calls if c.status is CallStatus.FAILED)

    @property
    def cancelled(self) -> int:
        return sum(1 for c in self.calls if c.status is CallStatus.CANCELLED)

    @property
    def pending(self) -> int:
        return sum(1 for c in self.calls if not c.status.is_terminal)

    @property
    def progress(self) -> int:
        if not self.calls:
            return 0
        return int(sum(c.progress for c in self.calls) / len(self.calls))

    def find_call(self, call_id: str) -> CallRecord | None:
        for call in self.calls:
            if call.id == call_id:
                return call
        return None


# --------------------------------------------------------------------------
# API responses
# --------------------------------------------------------------------------
class CallSummary(BaseModel):
    id: str
    sr_no: int
    filename: str
    status: CallStatus
    stage: Stage
    progress: int
    theme: str | None = None
    issue: str | None = None
    confidence: float | None = None
    language: str | None = None
    duration_seconds: float | None = None
    error: str | None = None

    @classmethod
    def from_record(cls, call: CallRecord) -> "CallSummary":
        cls_result = call.classification
        return cls(
            id=call.id,
            sr_no=call.sr_no,
            filename=call.filename,
            status=call.status,
            stage=call.stage,
            progress=call.progress,
            theme=cls_result.theme if cls_result else None,
            issue=cls_result.issue if cls_result else None,
            confidence=cls_result.confidence if cls_result else None,
            language=call.transcription.language if call.transcription else None,
            duration_seconds=(
                call.transcription.duration_seconds if call.transcription else None
            ),
            error=call.error,
        )


class JobSummary(BaseModel):
    id: str
    name: str
    status: JobStatus
    total: int
    completed: int
    failed: int
    pending: int
    progress: int
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    excel_available: bool = False

    @classmethod
    def from_record(cls, job: JobRecord) -> "JobSummary":
        return cls(
            id=job.id,
            name=job.name,
            status=job.status,
            total=job.total,
            completed=job.completed,
            failed=job.failed,
            pending=job.pending,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=job.updated_at,
            finished_at=job.finished_at,
            excel_available=bool(job.excel_path),
        )


class JobDetail(JobSummary):
    calls: list[CallSummary] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_record(cls, job: JobRecord) -> "JobDetail":
        summary = JobSummary.from_record(job)
        return cls(
            **summary.model_dump(),
            calls=[CallSummary.from_record(c) for c in job.calls],
            error=job.error,
        )


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    total: int
    accepted: list[str]
    rejected: list[dict[str, str]] = Field(default_factory=list)


class ResultRow(BaseModel):
    """Exactly the Excel report shape, so the table on the frontend and the
    workbook can never drift apart.

    Field names stay snake_case over the wire; the human column headings live
    in excel.REPORT_COLUMNS and are handed to the client in `columns`.
    """

    sr_no: int
    file_name: str
    theme: str
    specific_issue: str
    transcription: str
    ai_reasoning: str

    # Extras - shown in the UI drawer and on the workbook's "Details" sheet,
    # never in the six required columns.
    call_id: str
    status: CallStatus
    reason_for_issue: str = ""
    confidence: float = 0.0
    language: str = "unknown"
    duration_seconds: float = 0.0
    original_transcription: str = ""
    evidence: list[str] = Field(default_factory=list)
    error: str | None = None


class ResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    columns: list[str] = Field(description="Display headings, in report order.")
    fields: list[str] = Field(description="ResultRow keys matching `columns`.")
    total: int
    rows: list[ResultRow]
    excel_available: bool = False
    transcripts_truncated: bool = False
