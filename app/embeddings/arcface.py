"""ArcFace embedding via DeepFace (sync -> must be called through run_in_threadpool)."""

from __future__ import annotations

import io

import numpy as np

from app.config import settings


def decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode JPEG bytes into an RGB numpy array for DeepFace."""
    from PIL import Image

    img = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(img)


def embed(payload: bytes) -> np.ndarray:
    """Detect (retinaface, align=True), embed (ArcFace). Returns a (dim,) float32 vector.

    Raises ValueError("no_face_detected") when no face is found.
    """
    from deepface import DeepFace

    img = decode_jpeg(payload)
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


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two embeddings, matching pgvector's `<=>` operator.

    Returns `1 - cosine_similarity` in [0, 2]; lower == more similar. Used for the
    stateless /compare path where no row exists in the database to run `<=>` against.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)
