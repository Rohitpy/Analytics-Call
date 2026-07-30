"""FastAPI dependencies - thin accessors over the container on app.state."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.container import ServiceContainer
from backend.core.config import Settings, get_settings
from backend.services.job_manager import JobManager
from backend.services.job_store import JobStore
from backend.services.taxonomy import TaxonomyService
from backend.services.upload import UploadService


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_store(container: ContainerDep) -> JobStore:
    return container.store


def get_job_manager(container: ContainerDep) -> JobManager:
    return container.jobs


def get_taxonomy_service(container: ContainerDep) -> TaxonomyService:
    return container.taxonomy


def get_upload_service(container: ContainerDep) -> UploadService:
    return container.upload


StoreDep = Annotated[JobStore, Depends(get_store)]
JobManagerDep = Annotated[JobManager, Depends(get_job_manager)]
TaxonomyDep = Annotated[TaxonomyService, Depends(get_taxonomy_service)]
UploadDep = Annotated[UploadService, Depends(get_upload_service)]
