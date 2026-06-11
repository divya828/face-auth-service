"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1.compare import router as compare_router
from app.api.v1.health import router as health_router
from app.api.v1.register import router as register_router
from app.api.v1.verify import router as verify_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(register_router)
v1_router.include_router(verify_router)
v1_router.include_router(compare_router)
v1_router.include_router(health_router)

__all__ = ["v1_router"]
