"""Validating and storing uploaded audio.

Files are streamed to disk in chunks and the size limit is enforced *while*
streaming, so an oversized upload is aborted partway instead of being buffered
whole and rejected afterwards.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from backend.core.config import Settings
from backend.core.errors import UnsupportedMediaError, ValidationError
from backend.core.logging_config import get_logger
from backend.schemas.job import CallRecord

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
_UNSAFE = re.compile(r"[^A-Za-z0-9._\- ]")


def safe_filename(name: str) -> str:
    """Strip directory components and anything that could escape the folder."""
    base = Path(name or "audio").name
    base = unicodedata.normalize("NFKD", base)
    base = _UNSAFE.sub("_", base).strip(" .") or "audio"
    return base[:180]


class UploadService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def validate(self, filename: str) -> None:
        suffix = Path(filename).suffix.lower()
        allowed = self._settings.allowed_extensions
        if suffix not in allowed:
            shown = suffix or "(none)"
            raise UnsupportedMediaError(
                f"'{filename}' has an unsupported extension {shown}.",
                details={"allowed": ", ".join(sorted(allowed))},
            )

    async def save(
        self, upload: UploadFile, job_id: str, sr_no: int
    ) -> CallRecord:
        original_name = upload.filename or f"upload_{sr_no}"
        self.validate(original_name)

        record = CallRecord(sr_no=sr_no, filename=original_name, stored_path="")
        target_dir = self._settings.upload_dir / job_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Prefix with the call id so two files of the same name cannot collide.
        target = target_dir / f"{record.id}_{safe_filename(original_name)}"

        written = 0
        limit = self._settings.max_upload_bytes
        try:
            async with aiofiles.open(target, "wb") as out:
                while chunk := await upload.read(CHUNK_SIZE):
                    written += len(chunk)
                    if written > limit:
                        raise ValidationError(
                            f"'{original_name}' exceeds the "
                            f"{self._settings.MAX_UPLOAD_MB} MB limit.",
                        )
                    await out.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if written == 0:
            target.unlink(missing_ok=True)
            raise ValidationError(f"'{original_name}' is empty.")

        record.stored_path = str(target)
        record.size_bytes = written
        logger.info("Stored %s (%.1f KB) as %s", original_name, written / 1024, target.name)
        return record
