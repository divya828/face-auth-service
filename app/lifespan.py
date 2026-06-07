"""Application lifespan: GPU config, connection setup/teardown, VRAM warmup."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.database import db
from app.embeddings.tensorflow_setup import configure_tensorflow, warmup_model
from app.logging_config import emit
from app.services.rate_limiter import limiter
from app.storage import cold


@asynccontextmanager
async def lifespan(_: FastAPI):
    emit("startup_begin")
    configure_tensorflow()
    await run_in_threadpool(db.connect)
    await limiter.connect()
    cold.connect()
    # VRAM warmup with a dummy array so the first real request is not cold.
    await run_in_threadpool(warmup_model)
    emit("startup_complete")
    yield
    await run_in_threadpool(db.close)
    await limiter.close()
    emit("shutdown_complete")
