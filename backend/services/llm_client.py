"""Async client for any OpenAI-compatible endpoint (vLLM serving gemma-4 here).

Responsibilities kept in one place so the translation and classification
services stay declarative:
  * one shared AsyncOpenAI instance and connection pool
  * a global semaphore, so N pipeline workers can never open more than
    LLM_MAX_CONCURRENCY sockets against the inference server
  * retry with exponential backoff and jitter on transient failures
  * model-id auto-discovery when LLM_MODEL is left empty
  * optional vLLM guided_json structured decoding
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from backend.core.config import Settings
from backend.core.errors import StageError
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

# Anything else is treated as a permanent failure and is not retried.
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY or "EMPTY",
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
            max_retries=0,  # we do our own, with logging
        )
        self._semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)
        self._model: str | None = settings.LLM_MODEL or None
        self._model_lock = asyncio.Lock()

    # ---- model resolution --------------------------------------------------
    async def resolve_model(self) -> str:
        """Return the model id, discovering it from /v1/models if unset."""
        if self._model:
            return self._model
        async with self._model_lock:
            if self._model:
                return self._model
            models = await self._client.models.list()
            if not models.data:
                raise StageError("llm", "Inference server advertises no models.")
            self._model = models.data[0].id
            logger.info("Auto-discovered LLM model: %s", self._model)
            return self._model

    @property
    def model_name(self) -> str | None:
        return self._model

    # ---- health ------------------------------------------------------------
    async def health(self) -> tuple[bool, str | None, str | None]:
        """(reachable, model_id, error). Never raises - it feeds a probe."""
        try:
            models = await asyncio.wait_for(self._client.models.list(), timeout=10)
            model_id = self._model or (models.data[0].id if models.data else None)
            return True, model_id, None
        except Exception as exc:
            return False, self._model, f"{type(exc).__name__}: {exc}"

    # ---- completion --------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        guided_json: dict[str, Any] | None = None,
        stream: bool = True,
        stage: str = "llm",
    ) -> str:
        """Run a chat completion and return the assistant text.

        Streaming is on by default: these are long generations and streaming
        stops idle proxies from cutting the connection mid-answer.
        """
        settings = self._settings
        model = await self.resolve_model()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": (
                settings.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        }
        if guided_json and settings.LLM_USE_GUIDED_JSON:
            kwargs["extra_body"] = {"guided_json": guided_json}

        _log_prompt(stage, model, kwargs)

        last_error: Exception | None = None
        for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                async with self._semaphore:
                    text = (
                        await self._stream_call(kwargs)
                        if stream
                        else await self._single_call(kwargs)
                    )
                elapsed = time.perf_counter() - started
                logger.info(
                    "LLM ok (stage=%s attempt=%d, %.1fs, %d chars)",
                    stage,
                    attempt,
                    elapsed,
                    len(text),
                )
                _log_response(stage, text)
                if not text.strip():
                    raise StageError(stage, "LLM returned an empty response.")
                return text

            except _RETRYABLE as exc:
                last_error = exc
                if attempt >= settings.LLM_MAX_RETRIES:
                    break
                delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "LLM %s on attempt %d/%d (stage=%s); retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    settings.LLM_MAX_RETRIES,
                    stage,
                    delay,
                )
                await asyncio.sleep(delay)

            except APIStatusError as exc:
                # 5xx is worth another try; 4xx means the request is wrong.
                last_error = exc
                if exc.status_code < 500 or attempt >= settings.LLM_MAX_RETRIES:
                    raise StageError(
                        stage,
                        f"LLM returned HTTP {exc.status_code}: {exc.message}",
                        details={"status_code": str(exc.status_code)},
                    ) from exc
                delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "LLM HTTP %s on attempt %d; retrying in %.1fs",
                    exc.status_code,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)

        raise StageError(
            stage,
            f"LLM unreachable after {settings.LLM_MAX_RETRIES} attempts: "
            f"{type(last_error).__name__}: {last_error}",
        ) from last_error

    async def _single_call(self, kwargs: dict[str, Any]) -> str:
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def _stream_call(self, kwargs: dict[str, Any]) -> str:
        chunks: list[str] = []
        stream = await self._client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                chunks.append(content)
        return "".join(chunks)

    async def aclose(self) -> None:
        await self._client.close()


# --------------------------------------------------------------------------
# Prompt / response debug logging
#
# Off by default (DEBUG is below the default INFO level) so a normal run's
# logs stay readable and don't leak full transcripts. Turn it on with:
#
#   LOG_LEVEL=DEBUG python -m backend.main
#
# or, to avoid re-exporting every INFO line from every other module too, set
# just this logger from Python before starting the app:
#
#   import logging
#   logging.getLogger("backend.services.llm_client").setLevel(logging.DEBUG)
#
# Every line is prefixed "PROMPT>" / "RESPONSE>" so `grep "PROMPT>"` isolates
# them from the rest of the log even at DEBUG level.
# --------------------------------------------------------------------------
def _log_prompt(stage: str, model: str, kwargs: dict[str, Any]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    messages = kwargs.get("messages", [])
    lines = [
        "",
        "=" * 88,
        f"PROMPT> stage={stage} model={model} temperature={kwargs.get('temperature')} "
        f"max_tokens={kwargs.get('max_tokens')} "
        f"guided_json={'yes' if 'extra_body' in kwargs else 'no'}",
        "=" * 88,
    ]
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content", "")
        lines.append(f"PROMPT> --- {role} " + "-" * (78 - len(role)))
        lines.append(content if isinstance(content, str) else str(content))
    lines.append("=" * 88)
    logger.debug("\n".join(lines))


def _log_response(stage: str, text: str) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "\nRESPONSE> stage=%s\nRESPONSE> " + "-" * 78 + "\n%s\nRESPONSE> " + "=" * 78,
        stage,
        text,
    )
