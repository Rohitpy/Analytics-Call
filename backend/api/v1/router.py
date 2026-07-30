"""v1 API surface."""

from fastapi import APIRouter

from backend.api.v1.routes import health, jobs, results, themes

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(results.router, prefix="/jobs", tags=["results"])
api_router.include_router(themes.router, prefix="/themes", tags=["themes"])
