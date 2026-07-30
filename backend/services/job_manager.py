"""Work queue + worker pool.

Concurrency model
-----------------
One asyncio.Queue of (job_id, call_id) and PIPELINE_WORKERS coroutines pulling
from it. Uploading returns immediately; the queue absorbs the batch.

Each stage is throttled where it actually costs something:
  * ffmpeg      - a subprocess, naturally parallel
  * Whisper     - bounded by the STT thread pool (STT_CONCURRENCY)
  * translate   - bounded by the LLM semaphore (LLM_MAX_CONCURRENCY)
  * classify    - same semaphore

So PIPELINE_WORKERS can safely exceed STT_CONCURRENCY: while one call sits on
the GPU, the others are waiting on the LLM, and neither starves the other.

Job completion is decided inside a store mutation, which runs under the store
lock. That is what stops two workers finishing the last two calls at the same
instant and both trying to write the report.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import Settings
from backend.core.errors import NotFoundError
from backend.core.logging_config import get_logger, log_context
from backend.schemas.common import CallStatus, JobStatus, Stage
from backend.schemas.job import JobRecord
from backend.services import excel
from backend.services.job_store import JobStore
from backend.services.pipeline import CallPipeline

logger = get_logger(__name__)

FLUSH_INTERVAL_SECONDS = 5.0


class JobManager:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        pipeline: CallPipeline,
    ):
        self._settings = settings
        self._store = store
        self._pipeline = pipeline
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._flusher: asyncio.Task | None = None
        self._running = False

    # ---- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        count = max(1, self._settings.PIPELINE_WORKERS)
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"pipeline-worker-{i}")
            for i in range(count)
        ]
        self._flusher = asyncio.create_task(self._flush_loop(), name="job-flusher")
        logger.info("Job manager started with %d worker(s)", count)

    async def stop(self) -> None:
        self._running = False
        for task in [*self._workers, self._flusher]:
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *[t for t in [*self._workers, self._flusher] if t is not None],
            return_exceptions=True,
        )
        self._workers.clear()
        self._flusher = None
        await self._store.flush_dirty()
        logger.info("Job manager stopped")

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    # ---- submission --------------------------------------------------------
    async def submit(self, job: JobRecord) -> None:
        for call in job.calls:
            self._queue.put_nowait((job.id, call.id))
        logger.info("Queued job %s with %d call(s)", job.id, job.total)

    async def cancel(self, job_id: str) -> JobRecord:
        """Cooperative: calls already running finish, queued ones are dropped."""

        def _mutate(job: JobRecord) -> None:
            job.cancel_requested = True
            for call in job.calls:
                if call.status is CallStatus.QUEUED:
                    call.status = CallStatus.CANCELLED
                    call.error = "Cancelled before processing started."
            if job.pending == 0 and not job.status.is_terminal:
                job.status = JobStatus.CANCELLED
                job.finished_at = _now()

        job, _ = await self._store.update(job_id, _mutate)
        logger.info("Cancellation requested for job %s", job_id)
        return job

    # ---- workers -----------------------------------------------------------
    async def _worker(self, index: int) -> None:
        logger.info("Worker %d ready", index)
        while True:
            try:
                job_id, call_id = await self._queue.get()
            except asyncio.CancelledError:
                logger.info("Worker %d stopping", index)
                raise
            try:
                await self._process_call(job_id, call_id)
            except asyncio.CancelledError:
                logger.info("Worker %d cancelled mid-call", index)
                raise
            except Exception:
                # A worker must survive anything a single call can throw.
                logger.exception("Worker %d crashed on call %s", index, call_id)
            finally:
                self._queue.task_done()

    async def _process_call(self, job_id: str, call_id: str) -> None:
        try:
            job = await self._store.get(job_id)
        except NotFoundError:
            logger.warning("Job %s vanished before call %s ran", job_id, call_id)
            return

        call = job.find_call(call_id)
        if call is None:
            logger.warning("Call %s not found in job %s", call_id, job_id)
            return
        if call.status.is_terminal:
            return  # already cancelled

        if job.cancel_requested:
            await self._complete_call(
                job_id, call_id, CallStatus.CANCELLED, error="Job was cancelled."
            )
            return

        # ---- mark running ----
        def _start(record: JobRecord) -> None:
            target = record.find_call(call_id)
            if target is None:
                return
            target.status = CallStatus.RUNNING
            target.stage = Stage.CONVERTING
            target.started_at = _now()
            target.attempts += 1
            if record.status is JobStatus.QUEUED:
                record.status = JobStatus.RUNNING
                record.started_at = record.started_at or _now()

        await self._store.update(job_id, _start)

        # ---- run ----
        async def on_stage(stage: Stage) -> None:
            def _set(record: JobRecord) -> None:
                target = record.find_call(call_id)
                if target is not None:
                    target.stage = stage

            # persist=False: progress ticks are recoverable, IO is not free.
            await self._store.update(job_id, _set, persist=False)

        work_dir = self._settings.work_dir / job_id / call_id
        outcome = await self._pipeline.run(call, job_id, work_dir, on_stage)

        # ---- record ----
        def _finish(record: JobRecord) -> None:
            target = record.find_call(call_id)
            if target is None:
                return
            target.transcription = outcome.transcription
            target.translation = outcome.translation
            target.classification = outcome.classification
            target.timings = outcome.timings
            target.finished_at = _now()
            target.stage = Stage.DONE
            if outcome.ok:
                target.status = CallStatus.COMPLETED
            else:
                target.status = CallStatus.FAILED
                target.error = outcome.error
                target.failed_stage = outcome.failed_stage

        await self._store.update(job_id, _finish)
        await self._maybe_finalize(job_id)

    async def _complete_call(
        self, job_id: str, call_id: str, status: CallStatus, error: str | None = None
    ) -> None:
        def _mutate(record: JobRecord) -> None:
            target = record.find_call(call_id)
            if target is None:
                return
            target.status = status
            target.error = error
            target.finished_at = _now()

        await self._store.update(job_id, _mutate)
        await self._maybe_finalize(job_id)

    # ---- finalisation ------------------------------------------------------
    async def _maybe_finalize(self, job_id: str) -> None:
        """Claim the job for finalisation, atomically, exactly once.

        The status is set inside the mutation, so a second worker arriving a
        microsecond later sees a terminal job and claims nothing.
        """

        def _claim(record: JobRecord) -> bool:
            if record.pending > 0 or record.status.is_terminal:
                return False
            if record.cancel_requested:
                record.status = JobStatus.CANCELLED
            elif record.failed == record.total and record.total > 0:
                record.status = JobStatus.FAILED
            elif record.failed > 0:
                record.status = JobStatus.COMPLETED_WITH_ERRORS
            else:
                record.status = JobStatus.COMPLETED
            record.finished_at = _now()
            return True

        job, claimed = await self._store.update(job_id, _claim)
        if not claimed:
            return

        logger.info(
            "Job %s finished: %s (%d ok, %d failed of %d)",
            job_id,
            job.status.value,
            job.completed,
            job.failed,
            job.total,
        )

        with log_context(job_id=job_id, stage="report"):
            try:
                output = self._settings.results_dir / f"{job_id}.xlsx"
                await asyncio.to_thread(excel.write_report, job, output)

                def _attach(record: JobRecord) -> None:
                    record.excel_path = str(output)

                await self._store.update(job_id, _attach)
            except Exception as exc:
                logger.exception("Failed to write the report for job %s", job_id)

                def _note(record: JobRecord) -> None:
                    record.error = f"Report generation failed: {exc}"

                await self._store.update(job_id, _note)

    # ---- background flush --------------------------------------------------
    async def _flush_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
                await self._store.flush_dirty()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Job state flush failed")


def _now() -> datetime:
    return datetime.now(timezone.utc)
