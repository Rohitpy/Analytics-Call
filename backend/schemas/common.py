"""Enums and small shared response models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_ERRORS,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }


class CallStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.CANCELLED}


class Stage(str, Enum):
    """Pipeline stages, in execution order."""

    QUEUED = "queued"
    CONVERTING = "converting"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    CLASSIFYING = "classifying"
    DONE = "done"

    @property
    def progress(self) -> int:
        """Rough percentage for the progress bar in the UI."""
        return {
            Stage.QUEUED: 0,
            Stage.CONVERTING: 10,
            Stage.TRANSCRIBING: 30,
            Stage.TRANSLATING: 70,
            Stage.CLASSIFYING: 85,
            Stage.DONE: 100,
        }[self]


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    env: str


class ReadinessResponse(BaseModel):
    """Dependency probe - used by the UI banner and by orchestrators."""

    status: str = Field(description='"ready" | "degraded"')
    stt_backend: str
    stt_model_loaded: bool
    stt_mode: str = Field(default="single", description='"single" | "multi"')
    stt_devices: list[str] = Field(default_factory=list)
    stt_instances: int = 0
    stt_instances_loaded: int = 0
    llm_base_url: str
    llm_reachable: bool
    llm_model: str | None = None
    ffmpeg_available: bool
    taxonomy_themes: int
    workers: int
    queue_depth: int
    details: dict[str, str] = Field(default_factory=dict)
