"""Thin wrapper over the Theme Analytics REST API.

Every backend error comes back in the same envelope
(`{"error": {"code", "message", "details"}}`), so unwrapping it once here means
the views can just catch `ApiError` and show `exc.message`.
"""

from __future__ import annotations

from typing import Any, Iterable

import requests


class ApiError(Exception):
    def __init__(self, message: str, *, status: int | None = None, code: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code

    @property
    def is_not_found(self) -> bool:
        return self.status == 404 or self.code == "not_found"


class ApiClient:
    def __init__(self, api_url: str, timeout: float = 30.0):
        self._url = api_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    # ---- plumbing ----------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        try:
            response = self._session.request(method, f"{self._url}{path}", **kwargs)
        except requests.exceptions.ConnectionError as exc:
            raise ApiError(
                f"Cannot reach the API at {self._url}. Is the backend running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ApiError(f"The API did not respond in time ({method} {path}).") from exc

        if response.status_code >= 400:
            message = f"{response.status_code} {response.reason}"
            code = ""
            try:
                payload = response.json().get("error", {})
                message = payload.get("message") or message
                code = payload.get("code", "")
            except ValueError:
                pass
            raise ApiError(message, status=response.status_code, code=code)
        return response

    def _get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs).json()

    # ---- health ------------------------------------------------------------
    def readiness(self) -> dict:
        return self._get("/health/ready")

    def config(self) -> dict:
        return self._get("/config")

    # ---- jobs --------------------------------------------------------------
    def create_job(
        self, files: Iterable[tuple[str, bytes, str]], name: str, timeout: float
    ) -> dict:
        payload = [
            ("files", (filename, content, content_type or "application/octet-stream"))
            for filename, content, content_type in files
        ]
        if not payload:
            raise ApiError("No files were selected.")
        return self._request(
            "POST", "/jobs", files=payload, data={"name": name}, timeout=timeout
        ).json()

    def list_jobs(self, limit: int = 60) -> dict:
        return self._get("/jobs", params={"limit": limit})

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict:
        return self._request("POST", f"/jobs/{job_id}/cancel").json()

    def delete_job(self, job_id: str) -> dict:
        return self._request("DELETE", f"/jobs/{job_id}").json()

    # ---- results -----------------------------------------------------------
    def results(self, job_id: str, transcript_chars: int = 600) -> dict:
        return self._get(
            f"/jobs/{job_id}/results", params={"transcript_chars": transcript_chars}
        )

    def call_detail(self, job_id: str, call_id: str) -> dict:
        return self._get(f"/jobs/{job_id}/results/{call_id}")

    def export(self, job_id: str, timeout: float = 300.0) -> bytes:
        return self._request("GET", f"/jobs/{job_id}/export", timeout=timeout).content

    # ---- taxonomy ----------------------------------------------------------
    def themes(self) -> dict:
        return self._get("/themes")

    def reload_themes(self) -> dict:
        return self._request("POST", "/themes/reload").json()
