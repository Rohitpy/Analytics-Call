"""Audio normalisation: anything ffmpeg can read -> 16 kHz mono PCM WAV.

Same conversion as the original script, but driven through
asyncio.create_subprocess_exec so a slow file never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from backend.core.config import Settings
from backend.core.errors import StageError
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

CONVERSION_TIMEOUT_SECONDS = 600


class AudioService:
    def __init__(self, settings: Settings):
        self._settings = settings

    # ---- capability probe --------------------------------------------------
    def resolve_ffmpeg(self) -> str | None:
        """Configured path first, then whatever is on PATH."""
        configured = self._settings.FFMPEG_PATH
        if configured and Path(configured).is_file():
            return configured
        found = shutil.which("ffmpeg")
        if found:
            return found
        return None

    def is_available(self) -> bool:
        return self.resolve_ffmpeg() is not None

    # ---- conversion --------------------------------------------------------
    async def to_pcm_wav(self, input_path: Path, output_dir: Path) -> Path:
        ffmpeg = self.resolve_ffmpeg()
        if ffmpeg is None:
            raise StageError(
                "converting",
                f"ffmpeg not found. Checked FFMPEG_PATH="
                f"'{self._settings.FFMPEG_PATH}' and the system PATH.",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_pcm.wav"

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-acodec", "pcm_s16le",
            "-ar", str(self._settings.TARGET_SAMPLE_RATE),
            "-ac", "1",
            str(output_path),
        ]

        logger.info("Converting %s -> %s", input_path.name, output_path.name)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=CONVERSION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise StageError(
                "converting",
                f"ffmpeg timed out after {CONVERSION_TIMEOUT_SECONDS}s on "
                f"{input_path.name}",
            ) from exc

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()[-1500:]
            raise StageError(
                "converting",
                f"ffmpeg failed (exit {process.returncode}) on {input_path.name}",
                details={"stderr": detail},
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise StageError(
                "converting",
                f"ffmpeg produced no output for {input_path.name}",
            )

        logger.info("PCM ready: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
        return output_path
