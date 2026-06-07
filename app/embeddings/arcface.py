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
