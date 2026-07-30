"""The per-call pipeline: convert -> transcribe -> translate -> classify.

Deliberately stateless. It takes a CallRecord, produces a CallOutcome, and
never touches the store; the job manager owns all state transitions. That
split is what makes the pipeline straightforward to unit test and the
concurrency easy to reason about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from backend.core.config import Settings
from backend.core.errors import StageError
from backend.core.logging_config import get_logger, log_context
from backend.schemas.common import Stage
from backend.schemas.job import CallRecord, StageTimings
from backend.schemas.theme import (
    ClassificationResult,
    TranscriptionResult,
    TranslationResult,
)
from backend.services.audio import AudioService
from backend.services.classification import ClassificationService, no_content_result
from backend.services.taxonomy import TaxonomyService
from backend.services.transcription import TranscriptionService
from backend.services.translation import TranslationService

logger = get_logger(__name__)

StageCallback = Callable[[Stage], Awaitable[None]]


@dataclass
class CallOutcome:
    transcription: TranscriptionResult | None = None
    translation: TranslationResult | None = None
    classification: ClassificationResult | None = None
    timings: StageTimings = field(default_factory=StageTimings)
    error: str | None = None
    failed_stage: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class CallPipeline:
    def __init__(
        self,
        settings: Settings,
        audio: AudioService,
        transcription: TranscriptionService,
        translation: TranslationService,
        classification: ClassificationService,
        taxonomy: TaxonomyService,
    ):
        self._settings = settings
        self._audio = audio
        self._transcription = transcription
        self._translation = translation
        self._classification = classification
        self._taxonomy = taxonomy

    async def run(
        self,
        call: CallRecord,
        job_id: str,
        work_dir: Path,
        on_stage: StageCallback,
    ) -> CallOutcome:
        outcome = CallOutcome()
        started = time.perf_counter()
        source = Path(call.stored_path)
        work_dir.mkdir(parents=True, exist_ok=True)
        pcm_path: Path | None = None

        with log_context(job_id=job_id, call_id=call.id):
            logger.info("Pipeline start: %s", call.filename)
            try:
                if not source.exists():
                    raise StageError(
                        "converting", f"Uploaded file is missing: {source}"
                    )

                # ---- 1. convert ------------------------------------------
                await on_stage(Stage.CONVERTING)
                with log_context(stage="converting"):
                    stage_started = time.perf_counter()
                    pcm_path = await self._audio.to_pcm_wav(source, work_dir)
                    outcome.timings.convert = _elapsed(stage_started)

                # ---- 2. transcribe ---------------------------------------
                await on_stage(Stage.TRANSCRIBING)
                with log_context(stage="transcribing"):
                    stage_started = time.perf_counter()
                    outcome.transcription = await self._transcription.transcribe(
                        pcm_path, work_dir / f"{source.stem}_transcription.json"
                    )
                    outcome.timings.transcribe = _elapsed(stage_started)
                    logger.info(
                        "Transcribed %.1fs of audio into %d chars (language=%s)",
                        outcome.transcription.duration_seconds,
                        len(outcome.transcription.text),
                        outcome.transcription.language,
                    )

                taxonomy = self._taxonomy.taxonomy

                # ---- 3. translate ----------------------------------------
                if outcome.transcription.no_speech:
                    # Nothing was said; stages 3 and 4 have nothing to work on.
                    logger.warning("No speech detected - skipping translation and LLM")
                    outcome.translation = TranslationResult(
                        text="", translated=False,
                        source_language=outcome.transcription.language,
                    )
                    outcome.classification = no_content_result(taxonomy)
                else:
                    await on_stage(Stage.TRANSLATING)
                    with log_context(stage="translating"):
                        stage_started = time.perf_counter()
                        outcome.translation = await self._translation.translate(
                            outcome.transcription,
                            taxonomy,
                            work_dir / f"{source.stem}_translation.txt",
                        )
                        outcome.timings.translate = _elapsed(stage_started)

                    # ---- 4. classify -------------------------------------
                    await on_stage(Stage.CLASSIFYING)
                    with log_context(stage="classifying"):
                        stage_started = time.perf_counter()
                        outcome.classification = await self._classification.classify(
                            transcript=outcome.translation.text
                            or outcome.transcription.text,
                            filename=call.filename,
                            taxonomy=taxonomy,
                        )
                        outcome.timings.classify = _elapsed(stage_started)
                        logger.info(
                            "Classified as %s / %s (confidence %.2f)",
                            outcome.classification.theme,
                            outcome.classification.issue,
                            outcome.classification.confidence,
                        )

                await on_stage(Stage.DONE)

            except StageError as exc:
                outcome.error = exc.message
                outcome.failed_stage = exc.stage
                logger.error("Stage '%s' failed: %s", exc.stage, exc.message)
            except Exception as exc:
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.failed_stage = "unknown"
                logger.exception("Unexpected pipeline failure")
            finally:
                outcome.timings.total = _elapsed(started)
                if pcm_path is not None and not self._settings.KEEP_INTERMEDIATE_FILES:
                    pcm_path.unlink(missing_ok=True)
                logger.info(
                    "Pipeline end: %s (%.1fs, %s)",
                    call.filename,
                    outcome.timings.total or 0.0,
                    "ok" if outcome.ok else f"failed at {outcome.failed_stage}",
                )

        return outcome


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 2)
