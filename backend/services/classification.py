"""Stage 3: English transcript -> {theme, issue, reason} + reasoning.

Two defensive layers around the model:

  parsing    - accepts a bare JSON object, a ```json fence, or JSON with
               prose around it, because guided_json is not available on every
               server and gemma will occasionally editorialise.
  validation - the theme is forced back into the taxonomy and the issue is
               matched against that theme's issues. A hallucinated category
               silently corrupts every downstream count, so it is caught here
               and flagged on the row rather than trusted.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from backend.core.config import Settings
from backend.core.errors import StageError
from backend.core.logging_config import get_logger
from backend.schemas.theme import ClassificationResult, Taxonomy
from backend.services.llm_client import LLMClient
from backend.services.prompt_builder import (
    PromptLibrary,
    build_classification_messages,
    classification_json_schema,
)

logger = get_logger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Below this similarity a near-miss is not accepted as a match.
FUZZY_THRESHOLD = 0.82


class ClassificationService:
    def __init__(self, settings: Settings, llm: LLMClient, prompts: PromptLibrary):
        self._settings = settings
        self._llm = llm
        self._prompts = prompts

    async def classify(
        self, transcript: str, filename: str, taxonomy: Taxonomy
    ) -> ClassificationResult:
        if not transcript.strip():
            return no_content_result(taxonomy)

        messages = build_classification_messages(
            taxonomy=taxonomy,
            transcript=transcript,
            filename=filename,
            library=self._prompts,
        )

        raw = await self._llm.complete(
            messages,
            temperature=self._settings.LLM_TEMPERATURE,
            # The answer is a small JSON object; a big cap only invites rambling.
            max_tokens=1200,
            guided_json=classification_json_schema(taxonomy),
            stream=False,
            stage="classifying",
        )

        payload = extract_json(raw)
        if payload is None:
            raise StageError(
                "classifying",
                "Could not parse JSON from the classification response.",
                details={"response_preview": raw[:500]},
            )

        return self._validate(payload, taxonomy, raw)

    # ---- validation --------------------------------------------------------
    def _validate(
        self, payload: dict, taxonomy: Taxonomy, raw: str
    ) -> ClassificationResult:
        raw_theme = str(payload.get("theme", "")).strip()
        raw_issue = str(payload.get("issue", "")).strip()

        theme_def = taxonomy.find_theme(raw_theme)
        theme_matched = theme_def is not None

        if theme_def is None and raw_theme:
            match = _closest(raw_theme, taxonomy.theme_names())
            if match:
                logger.warning(
                    "LLM theme '%s' not exact - matched to '%s'", raw_theme, match
                )
                theme_def = taxonomy.find_theme(match)
                theme_matched = theme_def is not None

        if theme_def is None:
            logger.warning(
                "LLM returned unknown theme '%s' - falling back to '%s'",
                raw_theme,
                taxonomy.fallback_theme,
            )
            theme_def = taxonomy.find_theme(taxonomy.fallback_theme)
            theme_matched = False

        theme_name = theme_def.name if theme_def else taxonomy.fallback_theme

        issue_matched = False
        issue_name = raw_issue or taxonomy.fallback_issue
        if theme_def and raw_issue:
            issue_def = theme_def.find_issue(raw_issue)
            if issue_def is None:
                match = _closest(raw_issue, theme_def.issue_names())
                issue_def = theme_def.find_issue(match) if match else None
            if issue_def is not None:
                issue_name = issue_def.name
                issue_matched = True
            else:
                # Keep the model's wording: the prompt explicitly permits a
                # custom issue, and losing it would hide a taxonomy gap.
                logger.info(
                    "Custom issue '%s' kept for theme '%s'", raw_issue, theme_name
                )

        evidence = payload.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(e).strip() for e in evidence if str(e).strip()][:3]

        return ClassificationResult(
            theme=theme_name,
            issue=issue_name,
            reason=str(payload.get("reason", "")).strip(),
            reasoning=str(payload.get("reasoning", "")).strip(),
            evidence=evidence,
            confidence=_clamp_confidence(payload.get("confidence")),
            sentiment=_normalise_sentiment(payload.get("sentiment")),
            theme_matched=theme_matched,
            issue_matched=issue_matched,
            raw_response=raw[:4000],
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def no_content_result(taxonomy: Taxonomy) -> ClassificationResult:
    """Deterministic result for a silent call - no LLM round trip needed."""
    return ClassificationResult(
        theme=taxonomy.fallback_theme,
        issue="No usable speech in the call",
        reason="The transcript is empty - no speech was detected in the audio.",
        reasoning=(
            "Transcription produced no usable text, so there is nothing to "
            "classify. The call was not sent to the LLM."
        ),
        confidence=1.0,
        theme_matched=True,
        issue_matched=True,
    )



def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response."""
    if not text:
        return None
    candidate = text.strip()

    # 1. straight parse - the guided_json path lands here
    parsed = _try_load(candidate)
    if parsed is not None:
        return parsed

    # 2. fenced block
    fence = _FENCE.search(candidate)
    if fence:
        parsed = _try_load(fence.group(1).strip())
        if parsed is not None:
            return parsed

    # 3. first balanced {...}, ignoring braces inside strings
    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_load(candidate[start : index + 1])
                    if parsed is not None:
                        return parsed
                    break
        start = candidate.find("{", start + 1)

    return None


def _try_load(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _closest(value: str, options: list[str]) -> str | None:
    """Best fuzzy match above the threshold, else None."""
    if not options:
        return None
    target = _norm(value)
    best, best_score = None, 0.0
    for option in options:
        score = SequenceMatcher(None, target, _norm(option)).ratio()
        if score > best_score:
            best, best_score = option, score
    return best if best_score >= FUZZY_THRESHOLD else None


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


def _clamp_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Some models answer 85 instead of 0.85.
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _normalise_sentiment(value) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    for option in ("positive", "negative", "neutral"):
        if option in text:
            return option
    return None
