"""Job/call state, kept in memory and mirrored to disk.

In-memory is the source of truth while the process runs; the JSON mirror in
`<storage>/jobs/` survives restarts so finished reports stay downloadable.

Every mutation goes through `update()`, which holds a single lock and bumps
`revision` - that is what makes the SSE stream cheap and what keeps the
"is this job finished?" check race-free when several workers finish at once.

Persistence rules, learned the hard way:
  * the snapshot is serialised INSIDE the state lock, so a mirror can never
    catch a job halfway through a mutation;
  * each write goes to a uniquely named temp file, because several workers do
    finish the same job's calls at the same moment and a shared temp name
    means one rename wins and the other raises;
  * writes are ordered by revision, so a slow older snapshot cannot overwrite
    a newer one.

Swapping this for Redis or Postgres later means reimplementing this one class;
nothing else in the codebase touches job state directly.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from backend.core.config import Settings
from backend.core.errors import NotFoundError
from backend.core.logging_config import get_logger
from backend.schemas.common import CallStatus, JobStatus
from backend.schemas.job import JobRecord

logger = get_logger(__name__)

T = TypeVar("T")


class JobStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._dir: Path = settings.jobs_dir
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        # Jobs changed by a persist=False update, awaiting the periodic flush.
        self._dirty: set[str] = set()
        # Highest revision already on disk, per job.
        self._written: dict[str, int] = {}

    # ---- persistence -------------------------------------------------------
    def _path_for(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.json"

    def _write_sync(self, job_id: str, payload: str) -> None:
        """Atomic write. The temp name is unique per call so two concurrent
        writes for the same job cannot steal each other's file."""
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._path_for(job_id)
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)

    async def _persist(self, job_id: str, payload: str, revision: int) -> None:
        async with self._io_lock:
            if self._written.get(job_id, -1) > revision:
                return  # a newer snapshot already reached the disk
            try:
                await asyncio.to_thread(self._write_sync, job_id, payload)
                self._written[job_id] = revision
            except Exception:
                # Losing the mirror must never fail the request that caused it.
                logger.exception("Failed to persist job %s", job_id)

    async def load_existing(self) -> int:
        """Rehydrate on startup.

        The work queue is in-process, so anything still marked running when we
        died is not coming back - mark it interrupted rather than leaving a
        job that lies about being in progress forever.
        """
        if not self._dir.exists():
            return 0

        loaded = 0
        for path in sorted(self._dir.glob("*.json")):
            try:
                job = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping unreadable job file %s: %s", path.name, exc)
                continue

            self._jobs[job.id] = job
            self._written[job.id] = job.revision
            loaded += 1

            if not job.status.is_terminal:
                for call in job.calls:
                    if not call.status.is_terminal:
                        call.status = CallStatus.FAILED
                        call.error = "Interrupted by a service restart."
                        call.failed_stage = call.stage.value
                job.status = (
                    JobStatus.COMPLETED_WITH_ERRORS if job.completed else JobStatus.FAILED
                )
                job.error = "Interrupted by a service restart."
                job.finished_at = job.finished_at or _now()
                job.revision += 1
                await self._persist(job.id, job.model_dump_json(indent=2), job.revision)

        if loaded:
            logger.info("Rehydrated %d job(s) from %s", loaded, self._dir)
        return loaded

    # ---- CRUD --------------------------------------------------------------
    async def create(self, job: JobRecord) -> JobRecord:
        async with self._lock:
            self._jobs[job.id] = job
            payload = job.model_dump_json(indent=2)
            revision = job.revision
        await self._persist(job.id, payload, revision)
        return job

    async def get(self, job_id: str) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"Job '{job_id}' was not found.")
        return job

    def get_optional(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def list_jobs(
        self, limit: int = 50, offset: int = 0, status: JobStatus | None = None
    ) -> tuple[list[JobRecord], int]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        if status is not None:
            jobs = [j for j in jobs if j.status is status]
        return jobs[offset : offset + limit], len(jobs)

    async def update(
        self, job_id: str, mutator: Callable[[JobRecord], T], persist: bool = True
    ) -> tuple[JobRecord, T]:
        """Apply `mutator` under the lock, bump the revision, persist.

        The mutator runs inside the critical section, so read-modify-write
        decisions made in it (such as "am I the worker that finished this
        job?") are atomic.

        `persist=False` marks the job dirty instead of writing immediately.
        Use it for high-frequency progress updates: a job record embeds every
        transcript, so rewriting it on each stage change of each call would
        multiply into hundreds of megabytes of pointless IO on a large batch.
        `flush_dirty()` writes them out on a timer.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(f"Job '{job_id}' was not found.")
            outcome = mutator(job)
            job.revision += 1
            job.updated_at = _now()

            if persist:
                self._dirty.discard(job_id)
                # Serialise here, not in the IO thread: the record is mutated
                # from the event loop and a snapshot taken outside this lock
                # could catch it mid-change.
                payload = job.model_dump_json(indent=2)
                revision = job.revision
            else:
                self._dirty.add(job_id)
                payload = None

        if payload is not None:
            await self._persist(job_id, payload, revision)
        return job, outcome

    async def flush_dirty(self) -> int:
        """Persist everything touched by a persist=False update."""
        async with self._lock:
            snapshots = [
                (job_id, job.model_dump_json(indent=2), job.revision)
                for job_id in self._dirty
                if (job := self._jobs.get(job_id)) is not None
            ]
            self._dirty.clear()

        for job_id, payload, revision in snapshots:
            await self._persist(job_id, payload, revision)
        return len(snapshots)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.pop(job_id, None)
            self._dirty.discard(job_id)
        if job is None:
            raise NotFoundError(f"Job '{job_id}' was not found.")

        async with self._io_lock:
            self._written.pop(job_id, None)
            await asyncio.to_thread(self._path_for(job_id).unlink, True)
        logger.info("Deleted job %s", job_id)

    # ---- housekeeping ------------------------------------------------------
    async def purge_expired(self) -> int:
        """Drop jobs older than JOB_RETENTION_DAYS."""
        days = self._settings.JOB_RETENTION_DAYS
        if days <= 0:
            return 0
        cutoff = _now().timestamp() - days * 86400
        stale = [j.id for j in self._jobs.values() if j.created_at.timestamp() < cutoff]
        for job_id in stale:
            try:
                await self.delete(job_id)
            except NotFoundError:
                pass
        if stale:
            logger.info("Purged %d job(s) older than %d days", len(stale), days)
        return len(stale)

    @property
    def count(self) -> int:
        return len(self._jobs)


def _now() -> datetime:
    return datetime.now(timezone.utc)
