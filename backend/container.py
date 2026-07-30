"""Composition root.

Every service is constructed once, here, and hung off `app.state.container`.
Routes receive them through FastAPI dependencies (see api/deps.py) and never
import or instantiate a service themselves - which is what keeps the modules
independently testable.
"""

from __future__ import annotations

from backend.core.config import Settings
from backend.core.logging_config import get_logger
from backend.services.audio import AudioService
from backend.services.classification import ClassificationService
from backend.services.job_manager import JobManager
from backend.services.job_store import JobStore
from backend.services.llm_client import LLMClient
from backend.services.pipeline import CallPipeline
from backend.services.prompt_builder import PromptLibrary
from backend.services.taxonomy import TaxonomyService
from backend.services.transcription import TranscriptionService
from backend.services.translation import TranslationService
from backend.services.upload import UploadService

logger = get_logger(__name__)


class ServiceContainer:
    def __init__(self, settings: Settings):
        self.settings = settings

        # ---- stateless helpers ----
        self.taxonomy = TaxonomyService(settings)
        self.prompts = PromptLibrary(settings)
        self.upload = UploadService(settings)
        self.audio = AudioService(settings)

        # ---- clients / pools ----
        self.llm = LLMClient(settings)
        self.transcription = TranscriptionService(settings)

        # ---- pipeline stages ----
        self.translation = TranslationService(settings, self.llm, self.prompts)
        self.classification = ClassificationService(settings, self.llm, self.prompts)
        self.pipeline = CallPipeline(
            settings=settings,
            audio=self.audio,
            transcription=self.transcription,
            translation=self.translation,
            classification=self.classification,
            taxonomy=self.taxonomy,
        )

        # ---- orchestration ----
        self.store = JobStore(settings)
        self.jobs = JobManager(settings, self.store, self.pipeline)

    async def startup(self) -> None:
        self.settings.ensure_directories()
        self.taxonomy.load()
        await self.store.load_existing()
        await self.store.purge_expired()
        await self.jobs.start()
        # Blocking model load, done up front so the first upload is not slow.
        await self.transcription.warmup()
        logger.info(
            "Startup complete | stt=%s workers=%d llm=%s",
            self.settings.STT_BACKEND,
            self.settings.PIPELINE_WORKERS,
            self.settings.LLM_BASE_URL,
        )

    async def shutdown(self) -> None:
        await self.jobs.stop()
        self.transcription.shutdown()
        await self.llm.aclose()
        logger.info("Shutdown complete")
