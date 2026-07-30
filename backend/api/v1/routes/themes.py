"""Taxonomy inspection, hot reload, and a prompt preview.

The preview endpoint matters more than it looks: it renders the exact messages
that would be sent for a given transcript, so prompt changes can be reviewed
without burning a GPU cycle or hunting through logs.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from pydantic import BaseModel

from backend.api.deps import ContainerDep, TaxonomyDep
from backend.schemas.theme import Taxonomy
from backend.services.prompt_builder import (
    build_classification_messages,
    build_taxonomy_block,
    classification_json_schema,
)

router = APIRouter()


class TaxonomyResponse(BaseModel):
    source_file: str
    loaded: bool
    load_error: str | None = None
    theme_count: int
    issue_count: int
    taxonomy: Taxonomy


class PromptPreviewResponse(BaseModel):
    system_message: str
    user_message: str
    json_schema_used: dict
    approx_prompt_chars: int


@router.get("", response_model=TaxonomyResponse, summary="Current theme taxonomy")
async def get_themes(taxonomy: TaxonomyDep) -> TaxonomyResponse:
    current = taxonomy.taxonomy
    return TaxonomyResponse(
        source_file=str(taxonomy.source_path),
        loaded=taxonomy.is_loaded,
        load_error=taxonomy.load_error,
        theme_count=len(current.themes),
        issue_count=current.total_issues(),
        taxonomy=current,
    )


@router.post(
    "/reload",
    response_model=TaxonomyResponse,
    summary="Re-read themes.yaml without restarting",
)
async def reload_themes(taxonomy: TaxonomyDep) -> TaxonomyResponse:
    taxonomy.reload()  # raises 422 with the parse error if the file is broken
    return await get_themes(taxonomy)


@router.get(
    "/rendered",
    summary="The taxonomy exactly as the model sees it",
    response_model=dict,
)
async def rendered_taxonomy(taxonomy: TaxonomyDep) -> dict:
    return {"rendered": build_taxonomy_block(taxonomy.taxonomy)}


@router.post(
    "/prompt-preview",
    response_model=PromptPreviewResponse,
    summary="Render the classification prompt for a sample transcript",
)
async def prompt_preview(
    container: ContainerDep,
    taxonomy: TaxonomyDep,
    transcript: Annotated[str, Body(embed=True)] = "Customer: my card was declined.",
    filename: Annotated[str, Body(embed=True)] = "sample.wav",
) -> PromptPreviewResponse:
    messages = build_classification_messages(
        taxonomy=taxonomy.taxonomy,
        transcript=transcript,
        filename=filename,
        library=container.prompts,
    )
    return PromptPreviewResponse(
        system_message=messages[0]["content"],
        user_message=messages[1]["content"],
        json_schema_used=classification_json_schema(taxonomy.taxonomy),
        approx_prompt_chars=sum(len(m["content"]) for m in messages),
    )
