"""
Strict face-verification pipeline -- application entrypoint.

This module only wires the application together: it constructs the FastAPI app,
attaches the lifespan handler, and mounts the API routers. All behavior lives in
the `app` package:

    app/config.py              configuration (Settings)
    app/logging_config.py      structured JSON logging
    app/lifespan.py            startup/shutdown (GPU config, warmup, connections)
    app/api/                   HTTP routes (v1: register, verify, health)
    app/services/              rate limiting + verification orchestration
    app/database/              PostgreSQL + pgvector access
    app/embeddings/            TensorFlow setup + ArcFace inference
    app/storage/               S3 cold storage

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
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api import v1_router
from app.lifespan import lifespan

app = FastAPI(title="Face Verification Pipeline", version="1.0.0", lifespan=lifespan)
app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
