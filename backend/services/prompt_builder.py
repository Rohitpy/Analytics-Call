"""Turns the taxonomy + a transcript into the actual messages sent to the LLM.

Templating is intentionally tiny: `{{name}}` placeholders only. Prompts are
full of literal JSON braces, and str.format would choke on them, so nothing
here touches single braces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.core.config import Settings
from backend.core.logging_config import get_logger
from backend.core.yaml_compat import load_file
from backend.schemas.theme import Taxonomy

logger = get_logger(__name__)

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# Long transcripts are truncated before they reach the classifier. Roughly
# 4 chars/token, so 48k chars ~ 12k tokens, which leaves room for the taxonomy
# and the answer inside a typical 32k context.
MAX_TRANSCRIPT_CHARS = 48_000


def render(template: str, **values: Any) -> str:
    """Replace every {{key}} with str(values[key]); unknown keys stay put."""

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        logger.warning("Prompt placeholder {{%s}} has no value - left as-is", key)
        return match.group(0)

    return _PLACEHOLDER.sub(_sub, template)


def truncate_transcript(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Keep the head and the tail - the opening states the problem and the
    close carries the resolution; the middle is the most droppable part."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    tail = limit - head
    return (
        f"{text[:head]}\n\n"
        f"[... {len(text) - limit} characters of the middle of this call were "
        f"omitted because the transcript exceeded the model context ...]\n\n"
        f"{text[-tail:]}"
    )


class PromptLibrary:
    """Reads the YAML prompt files, caching on mtime so edits are picked up
    without a restart."""

    def __init__(self, settings: Settings):
        self._dir: Path = settings.prompts_dir
        self._cache: dict[str, tuple[float, dict[str, str]]] = {}

    def get(self, name: str) -> dict[str, str]:
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        mtime = path.stat().st_mtime
        cached = self._cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]

        data = load_file(path)
        if not isinstance(data, dict) or "system_message" not in data:
            raise ValueError(f"{path} must define at least 'system_message'")

        prompt = {
            "system_message": str(data.get("system_message", "")),
            "user_message": str(data.get("user_message", "")),
        }
        self._cache[name] = (mtime, prompt)
        logger.info("Loaded prompt '%s' from %s", name, path)
        return prompt


# --------------------------------------------------------------------------
# Taxonomy -> prompt text
# --------------------------------------------------------------------------
def build_taxonomy_block(taxonomy: Taxonomy) -> str:
    """Render every theme, issue, reason and example as readable text.

    Readable beats compact here: the examples are what actually teach the
    model where the boundaries between themes are.
    """
    lines: list[str] = []
    for t_index, theme in enumerate(taxonomy.themes, start=1):
        lines.append(f"### THEME {t_index}: {theme.name}")
        if theme.description:
            lines.append(f"When to use: {_one_line(theme.description)}")
        if theme.keywords:
            lines.append(f"Typical cues: {', '.join(theme.keywords)}")

        if not theme.issues:
            lines.append("  (no specific issues defined)")
        for i_index, issue in enumerate(theme.issues, start=1):
            lines.append(f"  {t_index}.{i_index} ISSUE: {issue.name}")
            if issue.description:
                lines.append(f"       meaning: {_one_line(issue.description)}")
            if issue.reasons:
                lines.append("       possible reasons:")
                lines.extend(f"         - {r}" for r in issue.reasons)
            if issue.examples:
                lines.append("       example calls:")
                lines.extend(f'         * "{_one_line(e)}"' for e in issue.examples)
        lines.append("")
    return "\n".join(lines).strip()


def build_classification_messages(
    taxonomy: Taxonomy,
    transcript: str,
    filename: str,
    library: PromptLibrary,
) -> list[dict[str, str]]:
    prompt = library.get("classification")
    common = {
        "domain": taxonomy.domain or "a retail bank contact centre",
        "taxonomy": build_taxonomy_block(taxonomy),
        "theme_list": "\n".join(f"  - {n}" for n in taxonomy.theme_names()),
        "transcript": truncate_transcript(transcript.strip()),
        "filename": filename,
    }
    return [
        {"role": "system", "content": render(prompt["system_message"], **common)},
        {"role": "user", "content": render(prompt["user_message"], **common)},
    ]


def build_translation_messages(
    srt_context: str,
    whole_transcript: str,
    taxonomy: Taxonomy,
    library: PromptLibrary,
) -> list[dict[str, str]]:
    prompt = library.get("translation")
    common = {
        "domain": taxonomy.domain or "a retail bank contact centre",
        "srt_context": srt_context,
        "whole_transcript": whole_transcript,
    }
    return [
        {"role": "system", "content": render(prompt["system_message"], **common)},
        {"role": "user", "content": render(prompt["user_message"], **common)},
    ]


def classification_json_schema(taxonomy: Taxonomy) -> dict[str, Any]:
    """JSON schema for vLLM's guided_json.

    `theme` is constrained to an enum, which removes the single most common
    failure mode - the model inventing a category that then breaks the
    reporting. Issue stays free text because the valid set depends on the
    theme, and it is validated in Python afterwards.
    """
    return {
        "type": "object",
        "properties": {
            "theme": {"type": "string", "enum": taxonomy.theme_names()},
            "issue": {"type": "string", "minLength": 2, "maxLength": 120},
            "reason": {"type": "string", "minLength": 2, "maxLength": 600},
            "reasoning": {"type": "string", "minLength": 2, "maxLength": 2000},
            "evidence": {
                "type": "array",
                "items": {"type": "string", "maxLength": 400},
                "maxItems": 3,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
            },
        },
        "required": ["theme", "issue", "reason", "reasoning", "confidence"],
        "additionalProperties": False,
    }


def _one_line(text: str) -> str:
    return " ".join(text.split())
