"""Tier 2 fixtures: REAL PostgreSQL+pgvector and Redis; embedding still mocked.

These tests verify the things Tier 1 cannot: that the SQL is valid, the pgvector
adapter round-trips a vector(512), the `<=>` distance ordering is correct, and
the Redis INCR/EXPIRE rate limiter behaves. Inference (DeepFace/GPU) is out of
scope here and remains mocked.

Connection is driven by the standard env vars (PG_HOST/PG_PORT/REDIS_HOST/...).
With docker-compose.test.yml the offsets are PG_PORT=5433, REDIS_PORT=6380.
If the services are unreachable, the whole module is skipped (not failed).
"""

from __future__ import annotations

import numpy as np
import pytest


# Real ArcFace produces 512-d vectors; integration tests use the real column
# width so the pgvector(512) type and adapter are exercised exactly.
def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(512).astype(np.float32)


def _services_available() -> bool:
    """Return True only if both Postgres and Redis accept a connection."""
    from app.database import db
    from app.services.rate_limiter import limiter

    try:
        db.connect()
        db.ping()
    except Exception:
        return False
    # Redis check requires an event loop; do a minimal sync socket probe instead.
    import socket

    from app.config import settings

    try:
        with socket.create_connection((settings.redis_host, settings.redis_port), timeout=2):
            pass
    except OSError:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _require_services():
    if not _services_available():
        pytest.skip("PostgreSQL/Redis not reachable; start docker-compose.test.yml")


@pytest.fixture
def real_db():
    """The real Database singleton, with the faces table truncated per test."""
    from app.database import db

    if db._conn is None:
        db.connect()
    with db._conn.cursor() as cur:
        cur.execute("TRUNCATE faces;")
    return db


@pytest.fixture
def vec():
    return _vec
