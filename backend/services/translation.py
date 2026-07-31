"""Stage 2: Arabic (or mixed) transcript -> diarized English transcript.

The LLM call is skipped entirely when the transcript is already English, which
is worth doing: it is the most expensive stage per call and a large share of
real traffic needs no translation.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.core.config import Settings
from backend.core.errors import StageError
from backend.core.logging_config import get_logger
from backend.schemas.theme import Taxonomy, TranscriptionResult, TranslationResult
from backend.services.llm_client import LLMClient
from backend.services.prompt_builder import PromptLibrary, build_translation_messages

logger = get_logger(__name__)

# Arabic, Arabic Supplement, Extended-A, and the two presentation-form blocks.
#
# Written as \u escapes on purpose: spelling the ranges with literal Arabic
# characters makes this line's meaning depend on the file's encoding surviving
# every copy, editor and terminal in between. It does not - one re-encode and
# the character class breaks with "unterminated character set". Keeping the
# source pure ASCII removes the failure mode.
#
# Note the last range stops at FEFC, the final Arabic ligature. U+FEFD..U+FEFF
# are not letters (U+FEFF is the byte-order mark), so counting them as Arabic
# would skew the ratio below.
_ARABIC = re.compile(
    "["
    "\u0600-\u06ff"  # Arabic
    "\u0750-\u077f"  # Arabic Supplement
    "\u08a0-\u08ff"  # Arabic Extended-A
    "\ufb50-\ufdff"  # Arabic Presentation Forms-A
    "\ufe70-\ufefc"  # Arabic Presentation Forms-B (letters only)
    "]"
)
_LATIN = re.compile(r"[A-Za-z]")

# Above this share of Arabic letters the text is treated as needing translation.
# Deliberately low: a mostly-English call with a few Arabic sentences still
# has to go through the LLM, or those sentences reach the classifier unread.
ARABIC_RATIO_THRESHOLD = 0.10


def script_ratio(text: str) -> tuple[float, float]:
    """(arabic_share, latin_share) over letters only."""
    arabic = len(_ARABIC.findall(text))
    latin = len(_LATIN.findall(text))
    total = arabic + latin
    if total == 0:
        return 0.0, 0.0
    return arabic / total, latin / total


def needs_translation(text: str, declared_language: str) -> bool:
    """Script beats the declared language: Whisper's language tag is often
    wrong on short or code-switched calls, the characters never are."""
    arabic_share, latin_share = script_ratio(text)
    if arabic_share >= ARABIC_RATIO_THRESHOLD:
        return True
    if latin_share > 0:
        return False
    # No letters at all - digits or punctuation only. Nothing to translate;
    # the classifier will route it to the fallback theme.
    return False


class TranslationService:
    def __init__(self, settings: Settings, llm: LLMClient, prompts: PromptLibrary):
        self._settings = settings
        self._llm = llm
        self._prompts = prompts

    async def translate(
        self,
        transcription: TranscriptionResult,
        taxonomy: Taxonomy,
        txt_out: Path | None = None,
    ) -> TranslationResult:
        source_text = transcription.text.strip()

        if not source_text:
            return TranslationResult(
                text="", translated=False, source_language=transcription.language
            )

        if self._settings.AUTO_DETECT_LANGUAGE and not needs_translation(
            source_text, transcription.language
        ):
            logger.info("Transcript is already English - skipping the LLM translation")
            result = TranslationResult(
                text=source_text, translated=False, source_language="en"
            )
            self._persist(result, txt_out)
            return result

        # Prefer the timestamped view; fall back to the paragraph if VAD gave
        # us no per-segment detail.
        srt_context = transcription.srt.strip() or source_text
        messages = build_translation_messages(
            srt_context=srt_context,
            whole_transcript=source_text,
            taxonomy=taxonomy,
            library=self._prompts,
        )

        logger.info(
            "Translating transcript (%d chars, language=%s)",
            len(source_text),
            transcription.language,
        )
        text = await self._llm.complete(
            messages,
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
            stream=True,
            stage="translating",
        )

        cleaned = text.strip()
        if not cleaned:
            raise StageError("translating", "Translation returned no text.")

        result = TranslationResult(
            text=cleaned, translated=True, source_language=transcription.language
        )
        self._persist(result, txt_out)
        return result

    @staticmethod
    def _persist(result: TranslationResult, txt_out: Path | None) -> None:
        if txt_out is None:
            return
        txt_out.parent.mkdir(parents=True, exist_ok=True)
        txt_out.write_text(result.text, encoding="utf-8")
        result.txt_path = str(txt_out)
