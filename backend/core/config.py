"""Central application settings.

Everything tunable lives here and is overridable from `.env` or the real
environment. Nothing else in the codebase reads `os.getenv` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root: <root>/backend/core/config.py -> parents[2]
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- app ---------------------------------------------------------------
    APP_NAME: str = "Theme Analytics"
    ENV: str = "dev"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    # Comma separated, or "*" for all. Parsed by `cors_origins`.
    CORS_ORIGINS: str = "*"

    # ---- storage -----------------------------------------------------------
    STORAGE_DIR: Path = BASE_DIR / "storage"

    # ---- upload limits -----------------------------------------------------
    MAX_UPLOAD_MB: int = 200
    MAX_FILES_PER_JOB: int = 200
    ALLOWED_AUDIO_EXTENSIONS: str = (
        ".wav,.mp3,.m4a,.ogg,.flac,.aac,.wma,.amr,.opus,.webm,.mp4"
    )

    # ---- ffmpeg ------------------------------------------------------------
    FFMPEG_PATH: str = "/opt/genaiadm/call_quality/ffmpeg/ffmpeg"

    # ---- speech to text ----------------------------------------------------
    STT_BACKEND: str = "whisper"  # "whisper" | "mock"
    WHISPER_MODEL_PATH: str = "/data0/genaiadm_bkp/GenAi-LLM/whisper/large-v3.pt"
    WHISPER_DEVICE: str = "cuda:0"
    WHISPER_LANGUAGE: str = "ar"
    WHISPER_INITIAL_PROMPT: str = ""
    STT_CONCURRENCY: int = 1
    STT_WARMUP: bool = True
    CHUNK_DURATION: int = 30
    SILENCE_PAD: float = 0.3
    TARGET_SAMPLE_RATE: int = 16000

    # ---- LLM ---------------------------------------------------------------
    LLM_BASE_URL: str = "http://localhost:8003/v1"
    LLM_API_KEY: str = "EMPTY"
    LLM_MODEL: str = "gemma-4"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 7000
    LLM_TIMEOUT: float = 600.0
    LLM_MAX_RETRIES: int = 3
    LLM_MAX_CONCURRENCY: int = 8
    LLM_USE_GUIDED_JSON: bool = True

    # ---- pipeline ----------------------------------------------------------
    PIPELINE_WORKERS: int = 4
    KEEP_INTERMEDIATE_FILES: bool = True
    AUTO_DETECT_LANGUAGE: bool = True
    JOB_RETENTION_DAYS: int = 30

    # ---- derived paths -----------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.STORAGE_DIR / "uploads"

    @property
    def work_dir(self) -> Path:
        return self.STORAGE_DIR / "work"

    @property
    def results_dir(self) -> Path:
        return self.STORAGE_DIR / "results"

    @property
    def jobs_dir(self) -> Path:
        return self.STORAGE_DIR / "jobs"

    @property
    def log_dir(self) -> Path:
        return self.STORAGE_DIR / "logs"

    @property
    def prompts_dir(self) -> Path:
        return BASE_DIR / "backend" / "prompts"

    @property
    def themes_file(self) -> Path:
        return BASE_DIR / "backend" / "data" / "themes.yaml"

    @property
    def frontend_dir(self) -> Path:
        return BASE_DIR / "frontend"

    # ---- parsed helpers ----------------------------------------------------
    @property
    def allowed_extensions(self) -> set[str]:
        return {
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in self.ALLOWED_AUDIO_EXTENSIONS.split(",")
            if ext.strip()
        }

    @property
    def cors_origins(self) -> list[str]:
        raw = self.CORS_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create every runtime directory. Called once on startup."""
        for path in (
            self.STORAGE_DIR,
            self.upload_dir,
            self.work_dir,
            self.results_dir,
            self.jobs_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Safe to use as a FastAPI dependency."""
    return Settings()
