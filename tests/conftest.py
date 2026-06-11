"""Tier 1 test fixtures.

These mock the heavy/external pieces (DeepFace embedding, PostgreSQL, Redis, S3)
so the FastAPI routing, error mapping, threshold logic, and rate-limit behavior
can be exercised in milliseconds with no GPU and no running services.

The app constructs `app` at import time with a lifespan that connects to real
services. We DON'T enter that lifespan here: TestClient is used WITHOUT its
context manager (i.e. `TestClient(app)` called directly), and every external
dependency is monkeypatched at its point of use before requests are made.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient


# --- Deterministic fake embeddings -----------------------------------------
# We key the returned vector off the uploaded bytes so tests can control the
# resulting cosine distance precisely (identical bytes -> identical vector ->
# distance 0.0; orthogonal canned vectors -> distance 1.0).

_VEC_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_VEC_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
# A vector ~0.30 cosine distance from _VEC_A (cos sim 0.70) for "close match".
_VEC_CLOSE = np.array([0.70, 0.714, 0.0], dtype=np.float32)


def _fake_embed_factory(no_face_for: set[bytes] | None = None):
    no_face_for = no_face_for or set()

    def _fake_embed(payload: bytes) -> np.ndarray:
        if payload in no_face_for:
            raise ValueError("no_face_detected")
        if payload == b"FACE_A":
            return _VEC_A.copy()
        if payload == b"FACE_B":
            return _VEC_B.copy()
        if payload == b"FACE_CLOSE":
            return _VEC_CLOSE.copy()
        # Default: derive a stable vector from the byte length so unknown
        # payloads still embed deterministically.
        v = np.array([len(payload) % 7, (len(payload) * 3) % 5, 1.0], dtype=np.float32)
        return v

    return _fake_embed


class FakeDB:
    """In-memory stand-in for the psycopg2 Database singleton."""

    def __init__(self) -> None:
        self.store: dict[str, np.ndarray] = {}

    def upsert_face(self, user_id: str, embedding: np.ndarray) -> None:
        self.store[user_id] = embedding

    def nearest(self, user_id: str, embedding: np.ndarray):
        from app.embeddings.arcface import cosine_distance

        stored = self.store.get(user_id)
        if stored is None:
            return None
        return cosine_distance(embedding, stored)

    def ping(self) -> bool:
        return True


class FakeLimiter:
    """In-memory rate limiter mirroring INCR/EXPIRE semantics (count per user)."""

    def __init__(self, max_hits: int = 3) -> None:
        self.counts: dict[str, int] = {}
        self.max_hits = max_hits

    async def hit(self, user_id: str) -> bool:
        self.counts[user_id] = self.counts.get(user_id, 0) + 1
        return self.counts[user_id] <= self.max_hits

    async def ping(self) -> bool:
        return True


@pytest.fixture
def fake_db():
    return FakeDB()


@pytest.fixture
def fake_limiter():
    return FakeLimiter(max_hits=3)


@pytest.fixture
def archived():
    """Collects (user_id, reason) tuples that would have been sent to S3."""
    return []


@pytest.fixture
def client(monkeypatch, fake_db, fake_limiter, archived):
    """A TestClient with all heavy/external dependencies mocked.

    Patches are applied at each module's point of use:
      * app.services.verification.embed / db / archive_rejected
      * app.services.rate_limiter.limiter  (the instance the verify route imports)
      * app.embeddings.tensorflow_setup.list_gpus  (for /health)
    """
    import app.api.v1.health as health_mod
    import app.api.v1.verify as verify_mod
    import app.services.verification as svc

    # Embedding: fake, no GPU/DeepFace.
    monkeypatch.setattr(svc, "embed", _fake_embed_factory({b"NO_FACE"}))
    # Database: in-memory.
    monkeypatch.setattr(svc, "db", fake_db)

    # S3 archival: record instead of uploading.
    async def _fake_archive(user_id, payload, reason):
        archived.append((user_id, reason))

    monkeypatch.setattr(svc, "archive_rejected", _fake_archive)

    # Rate limiter: the verify route holds its own reference `limiter`, and
    # health imports it too -- patch both names to the same fake.
    monkeypatch.setattr(verify_mod, "limiter", fake_limiter)
    monkeypatch.setattr(health_mod, "limiter", fake_limiter)
    # health also patches db + GPU listing.
    monkeypatch.setattr(health_mod, "db", fake_db)
    monkeypatch.setattr(health_mod, "list_gpus", lambda: ["GPU:0"])

    from main import app

    # NOTE: TestClient(app) without `with` does NOT run the lifespan, so the real
    # connect()/warmup() never fire. That's intentional for Tier 1.
    return TestClient(app)
