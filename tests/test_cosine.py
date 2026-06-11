"""Tier 1: cosine_distance must match pgvector's `<=>` semantics."""

import numpy as np

from app.embeddings.arcface import cosine_distance


def test_identical_is_zero():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_distance(a, a) == 0.0


def test_orthogonal_is_one():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert round(cosine_distance(a, b), 6) == 1.0


def test_opposite_is_two():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert round(cosine_distance(a, b), 6) == 2.0


def test_zero_vector_is_safe():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)
    assert cosine_distance(a, z) == 1.0


def test_magnitude_invariant():
    # Cosine distance ignores magnitude; scaling must not change the result.
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # 2x a
    assert round(cosine_distance(a, b), 6) == 0.0
