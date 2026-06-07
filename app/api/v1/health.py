"""GET /api/v1/health -- liveness of PostgreSQL, Redis, and GPU visibility."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.database import db
from app.embeddings.tensorflow_setup import list_gpus
from app.services.rate_limiter import limiter

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    """Verify PostgreSQL, Redis, and physical GPU visibility via TensorFlow."""
    checks: dict[str, Any] = {}

    try:
        checks["postgres"] = await run_in_threadpool(db.ping)
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = False
        checks["postgres_error"] = str(exc)

    try:
        checks["redis"] = await limiter.ping()
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = False
        checks["redis_error"] = str(exc)

    gpus = list_gpus()
    checks["gpu_visible"] = len(gpus) > 0
    checks["gpu_count"] = len(gpus)

    healthy = bool(checks.get("postgres")) and bool(checks.get("redis")) and checks["gpu_visible"]
    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
