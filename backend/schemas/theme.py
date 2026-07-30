"""Taxonomy models (what the LLM is allowed to choose from) and the models
describing what the STT + LLM stages produce."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Taxonomy - loaded from backend/data/themes.yaml
# --------------------------------------------------------------------------
class IssueDefinition(BaseModel):
    """A specific issue inside a theme, plus the reasons it typically has."""

    name: str
    description: str = ""
    reasons: list[str] = Field(
        default_factory=list,
        description="Candidate root causes the LLM may pick from for this issue.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Short snippets of real calls that land on this issue.",
    )


class ThemeDefinition(BaseModel):
    name: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    issues: list[IssueDefinition] = Field(default_factory=list)

    def issue_names(self) -> list[str]:
        return [i.name for i in self.issues]

    def find_issue(self, name: str) -> IssueDefinition | None:
        target = _norm(name)
        for issue in self.issues:
            if _norm(issue.name) == target:
                return issue
        return None


class Taxonomy(BaseModel):
    version: str = "1"
    domain: str = ""
    fallback_theme: str = "Others"
    fallback_issue: str = "Unclassified"
    themes: list[ThemeDefinition] = Field(default_factory=list)

    def theme_names(self) -> list[str]:
        return [t.name for t in self.themes]

    def find_theme(self, name: str) -> ThemeDefinition | None:
        target = _norm(name)
        for theme in self.themes:
            if _norm(theme.name) == target:
                return theme
        return None

    def total_issues(self) -> int:
        return sum(len(t.issues) for t in self.themes)


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


# --------------------------------------------------------------------------
# Stage outputs
# --------------------------------------------------------------------------
class TranscriptSegment(BaseModel):
    start: str
    end: str
    text: str


class TranscriptionResult(BaseModel):
    text: str = ""
    srt: str = ""
    language: str = "unknown"
    duration_seconds: float = 0.0
    segment_count: int = 0
    silence_flags: dict[str, bool] = Field(default_factory=dict)
    json_path: str | None = None
    no_speech: bool = False


class TranslationResult(BaseModel):
    text: str = ""
    translated: bool = Field(
        default=False,
        description="False when the source was already English and we skipped the LLM.",
    )
    source_language: str = "unknown"
    txt_path: str | None = None


class ClassificationResult(BaseModel):
    """The three things the prompt asks the LLM for, plus provenance."""

    theme: str = "Others"
    issue: str = "Unclassified"
    reason: str = ""
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    sentiment: str | None = None

    # Set by the validator in services/classification.py.
    theme_matched: bool = Field(
        default=True,
        description="False when the LLM invented a theme outside the taxonomy.",
    )
    issue_matched: bool = Field(
        default=True,
        description="False when the issue is not one of the theme's known issues.",
    )
    raw_response: str | None = None

    def to_reasoning_cell(self) -> str:
        """Flatten reason + reasoning + evidence into the single 'AI Reasoning'
        column the report asks for."""
        parts: list[str] = []
        if self.reason:
            parts.append(f"Reason for Issue: {self.reason}")
        if self.reasoning:
            parts.append(f"Reasoning: {self.reasoning}")
        if self.evidence:
            quotes = "\n".join(f'  - "{e}"' for e in self.evidence)
            parts.append(f"Evidence:\n{quotes}")
        if self.confidence:
            parts.append(f"Confidence: {self.confidence:.2f}")
        if not self.theme_matched:
            parts.append("NOTE: theme was not in the taxonomy and was remapped.")
        elif not self.issue_matched:
            parts.append("NOTE: issue is not a predefined issue for this theme.")
        return "\n\n".join(parts)
