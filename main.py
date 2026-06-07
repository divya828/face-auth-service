"""
Strict face-verification pipeline.

Architecture
------------
* Facial core      : DeepFace + ArcFace (512-d embeddings)
* Detection        : retinaface backend, mandatory landmark alignment (align=True)
* Vector store     : PostgreSQL + pgvector, native cosine distance (<=>)
* Security cutoff  : cosine DISTANCE <= 0.40  ->  MATCH   (lower == more similar)
* Rate limit       : Redis, 3 verify requests / 60s / user_id
* Cold storage     : rejected/failed JPEGs streamed to S3 (fraud_reviews/ prefix)
* Runtime          : async FastAPI, heavy sync work offloaded via run_in_threadpool
* Target hardware  : AWS EC2 g6.xlarge, NVIDIA L4 (24GB), mixed_float16

NOTE on the threshold direction
-------------------------------
pgvector's `<=>` returns cosine DISTANCE in [0, 2]. Smaller means more similar.
The strict cutoff of 0.40 therefore ACCEPTS only very close matches and REJECTS
everything looser -- this is intentionally stricter than ArcFace's default (~0.68)
to suppress lookalikes and spoofs.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Configuration (env-driven; see .env.example)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model / pipeline
    model_name: str = "ArcFace"
    detector_backend: str = "retinaface"
    embedding_dim: int = 512
    cosine_threshold: float = 0.40  # DISTANCE cutoff: match iff distance <= this

    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "faces"
    pg_user: str = "postgres"
    pg_password: str = "postgres"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    rate_limit_max: int = 3
    rate_limit_window_s: int = 60

    # AWS S3 cold storage
    aws_region: str = "us-east-1"
    s3_bucket: str = "payment-fraud-review-snapshots"
    s3_prefix: str = "fraud_reviews/"


settings = Settings()


# ---------------------------------------------------------------------------
# Structured CloudWatch-ready JSON logging
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Render log records as a flat JSON object CloudWatch can index."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote structured metrics attached via `extra={"metrics": {...}}`.
        metrics = getattr(record, "metrics", None)
        if isinstance(metrics, dict):
            payload.update(metrics)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("face-auth")


log = _configure_logging()


def emit(action: str, **metrics: Any) -> None:
    """Emit a structured metric line (inference_ms, db_ms, total_ms, user_id...)."""
    log.info(action, extra={"metrics": {"action": action, **metrics}})


# ---------------------------------------------------------------------------
# Lazy/heavy imports done inside functions so the module imports fast and so
# TensorFlow GPU configuration happens exactly once, at process start.
# ---------------------------------------------------------------------------


def _configure_tensorflow() -> None:
    """Enable memory growth + mixed precision for the L4 GPU."""
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:  # already initialized
            log.warning("memory_growth_set_failed: %s", exc)
    try:
        mixed_precision.set_global_policy("mixed_float16")
    except Exception as exc:  # noqa: BLE001 - non-fatal on CPU-only boxes
        log.warning("mixed_precision_unavailable: %s", exc)


def _warmup_model() -> None:
    """Force ArcFace + retinaface weights to load and warm VRAM via a dummy array."""
    from deepface import DeepFace

    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    # enforce_detection=False so the warmup never raises on a blank image.
    DeepFace.represent(
        img_path=dummy,
        model_name=settings.model_name,
        detector_backend="skip",  # skip detection for warmup; just load the embedder
        align=False,
        enforce_detection=False,
    )


# ---------------------------------------------------------------------------
# Database (sync psycopg2 -> always called via run_in_threadpool)
# ---------------------------------------------------------------------------


class Database:
    """Thin psycopg2 wrapper. All methods are SYNC and must be threadpooled."""

    def __init__(self) -> None:
        self._conn = None

    def connect(self) -> None:
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self._conn = psycopg2.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname=settings.pg_db,
            user=settings.pg_user,
            password=settings.pg_password,
        )
        self._conn.autocommit = True
        register_vector(self._conn)

    def ping(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1;")
            return cur.fetchone()[0] == 1

    def upsert_face(self, user_id: str, embedding: np.ndarray) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO faces (user_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET embedding = EXCLUDED.embedding,
                              updated_at = now();
                """,
                (user_id, embedding),
            )

    def nearest(self, user_id: str, embedding: np.ndarray) -> Optional[float]:
        """Return cosine DISTANCE to the stored embedding for user_id, or None."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT embedding <=> %s AS distance FROM faces WHERE user_id = %s;",
                (embedding, user_id),
            )
            row = cur.fetchone()
            return float(row[0]) if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


db = Database()


# ---------------------------------------------------------------------------
# Redis (async) -- rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self) -> None:
        self._client = None

    async def connect(self) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def hit(self, user_id: str) -> bool:
        """Increment the per-user counter. Return True if WITHIN the limit."""
        key = f"ratelimit:verify:{user_id}"
        count = await self._client.incr(key)
        if count == 1:
            await self._client.expire(key, settings.rate_limit_window_s)
        return count <= settings.rate_limit_max

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


limiter = RateLimiter()


# ---------------------------------------------------------------------------
# S3 cold storage (boto3 is sync -> threadpooled)
# ---------------------------------------------------------------------------


class ColdStorage:
    def __init__(self) -> None:
        self._client = None

    def connect(self) -> None:
        import boto3

        self._client = boto3.client("s3", region_name=settings.aws_region)

    def put_jpeg(self, key: str, payload: bytes) -> None:
        self._client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="image/jpeg",
        )


cold = ColdStorage()


async def archive_rejected(user_id: str, payload: bytes, reason: str) -> None:
    """Stream a rejected/failed JPEG to the fraud-review S3 prefix (best-effort)."""
    # Deterministic-ish key without Date.now(): user + reason + size + content hash slice.
    digest = abs(hash((user_id, len(payload), reason))) % (10**12)
    key = f"{settings.s3_prefix}{user_id}/{reason}-{digest}.jpg"
    try:
        await run_in_threadpool(cold.put_jpeg, key, payload)
        emit("s3_archive", user_id=user_id, s3_key=key, reason=reason)
    except Exception as exc:  # noqa: BLE001 - archival must never break the response
        log.warning("s3_archive_failed: %s", exc, extra={"metrics": {"user_id": user_id}})


# ---------------------------------------------------------------------------
# Inference helpers (DeepFace is sync -> threadpooled)
# ---------------------------------------------------------------------------


def _decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode JPEG bytes into an RGB numpy array for DeepFace."""
    from PIL import Image

    img = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(img)


def _embed(payload: bytes) -> np.ndarray:
    """Detect (retinaface, align=True), embed (ArcFace). Returns a (dim,) float32 vector."""
    from deepface import DeepFace

    img = _decode_jpeg(payload)
    reps = DeepFace.represent(
        img_path=img,
        model_name=settings.model_name,
        detector_backend=settings.detector_backend,
        align=True,
        enforce_detection=True,
    )
    if not reps:
        raise ValueError("no_face_detected")
    return np.asarray(reps[0]["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    emit("startup_begin")
    _configure_tensorflow()
    await run_in_threadpool(db.connect)
    await limiter.connect()
    cold.connect()
    # VRAM warmup with a dummy array so the first real request is not cold.
    await run_in_threadpool(_warmup_model)
    emit("startup_complete")
    yield
    await run_in_threadpool(db.close)
    await limiter.close()
    emit("shutdown_complete")


app = FastAPI(title="Face Verification Pipeline", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/v1/register")
async def register(user_id: str = Form(...), image: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()
    payload = await image.read()

    t_inf = time.perf_counter()
    try:
        embedding = await run_in_threadpool(_embed, payload)
    except ValueError:
        raise HTTPException(status_code=422, detail="no_face_detected")
    inference_ms = (time.perf_counter() - t_inf) * 1000

    t_db = time.perf_counter()
    await run_in_threadpool(db.upsert_face, user_id, embedding)
    db_ms = (time.perf_counter() - t_db) * 1000

    total_ms = (time.perf_counter() - t0) * 1000
    emit(
        "register",
        user_id=user_id,
        result="stored",
        inference_ms=round(inference_ms, 2),
        db_ms=round(db_ms, 2),
        total_ms=round(total_ms, 2),
    )
    return {"user_id": user_id, "status": "registered"}


@app.post("/api/v1/verify")
async def verify(user_id: str = Form(...), image: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()

    # Rate limit BEFORE doing any expensive work.
    if not await limiter.hit(user_id):
        emit("verify", user_id=user_id, result="rate_limited")
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")

    payload = await image.read()

    t_inf = time.perf_counter()
    try:
        embedding = await run_in_threadpool(_embed, payload)
    except ValueError:
        await archive_rejected(user_id, payload, reason="no_face")
        emit("verify", user_id=user_id, result="no_face_detected")
        raise HTTPException(status_code=422, detail="no_face_detected")
    inference_ms = (time.perf_counter() - t_inf) * 1000

    t_db = time.perf_counter()
    distance = await run_in_threadpool(db.nearest, user_id, embedding)
    db_ms = (time.perf_counter() - t_db) * 1000

    if distance is None:
        emit("verify", user_id=user_id, result="unknown_user")
        raise HTTPException(status_code=404, detail="user_not_enrolled")

    matched = distance <= settings.cosine_threshold
    total_ms = (time.perf_counter() - t0) * 1000

    if not matched:
        await archive_rejected(user_id, payload, reason="rejected")

    emit(
        "verify",
        user_id=user_id,
        result="match" if matched else "reject",
        distance=round(distance, 4),
        threshold=settings.cosine_threshold,
        inference_ms=round(inference_ms, 2),
        db_ms=round(db_ms, 2),
        total_ms=round(total_ms, 2),
    )
    return {
        "user_id": user_id,
        "match": matched,
        "distance": round(distance, 4),
        "threshold": settings.cosine_threshold,
    }


@app.get("/api/v1/health")
async def health(request: Request) -> JSONResponse:
    """Verify PostgreSQL, Redis, and physical GPU visibility via TensorFlow."""
    import tensorflow as tf

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

    gpus = tf.config.list_physical_devices("GPU")
    checks["gpu_visible"] = len(gpus) > 0
    checks["gpu_count"] = len(gpus)

    healthy = bool(checks.get("postgres")) and bool(checks.get("redis")) and checks["gpu_visible"]
    status_code = 200 if healthy else 503
    return JSONResponse(status_code=status_code, content={"status": "ok" if healthy else "degraded", "checks": checks})
