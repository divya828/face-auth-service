"""Tier 2: real pgvector round-trip, upsert semantics, and `<=>` distance order."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_upsert_and_nearest_roundtrip(real_db, vec):
    v = vec(1)
    real_db.upsert_face("alice", v)
    # nearest() against the same vector must be ~0 distance.
    dist = real_db.nearest("alice", v)
    assert dist is not None
    assert dist < 1e-5


def test_nearest_unknown_user_is_none(real_db, vec):
    assert real_db.nearest("nobody", vec(2)) is None


def test_upsert_overwrites_embedding(real_db, vec):
    real_db.upsert_face("bob", vec(10))
    real_db.upsert_face("bob", vec(20))  # second registration overwrites
    # Distance to vec(20) should now be ~0; to vec(10) should be larger.
    assert real_db.nearest("bob", vec(20)) < 1e-5
    assert real_db.nearest("bob", vec(10)) > 1e-5


def test_distance_ordering_matches_cosine(real_db, vec):
    """pgvector `<=>` must agree with our in-Python cosine_distance."""
    from app.embeddings.arcface import cosine_distance

    stored = vec(30)
    probe = vec(31)
    real_db.upsert_face("carol", stored)
    pg_distance = real_db.nearest("carol", probe)
    py_distance = cosine_distance(probe, stored)
    assert pg_distance == pytest.approx(py_distance, abs=1e-4)


def test_self_match_under_threshold(real_db, vec):
    from app.config import settings

    v = vec(42)
    real_db.upsert_face("dave", v)
    assert real_db.nearest("dave", v) <= settings.cosine_threshold
