"""UI settings.

Separate from the backend's `Settings` on purpose: the UI is a client and may
well run on a different host, so it must not import backend config or assume
it can read the backend's .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class UiSettings:
    # Where this UI reaches the API. Default assumes both run on one box.
    api_base_url: str = "http://127.0.0.1:8000"
    api_prefix: str = "/api/v1"

    # No poll interval on purpose: the page never reruns on a timer. Progress
    # is refreshed only when the user asks for it.

    request_timeout: float = 30.0
    # Uploads can be hundreds of megabytes over a slow link.
    upload_timeout: float = 900.0
    # Excel generation for a large batch is not instant.
    export_timeout: float = 300.0

    # Transcript characters requested for the table; the detail tab fetches
    # the full text separately.
    transcript_preview_chars: int = 600

    page_title: str = "Theme Analytics"
    page_icon: str = "\N{TELEPHONE RECEIVER}"

    @property
    def api_url(self) -> str:
        return f"{self.api_base_url.rstrip('/')}{self.api_prefix}"


@lru_cache(maxsize=1)
def get_ui_settings() -> UiSettings:
    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, "").strip() or default)
        except ValueError:
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "").strip() or default)
        except ValueError:
            return default

    return UiSettings(
        api_base_url=os.getenv("THEME_ANALYTICS_API_URL", "").strip()
        or "http://127.0.0.1:8000",
        request_timeout=_float("THEME_ANALYTICS_REQUEST_TIMEOUT", 30.0),
        upload_timeout=_float("THEME_ANALYTICS_UPLOAD_TIMEOUT", 900.0),
        transcript_preview_chars=_int("THEME_ANALYTICS_PREVIEW_CHARS", 600),
    )
