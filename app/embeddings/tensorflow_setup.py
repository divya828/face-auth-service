"""TensorFlow / GPU runtime configuration and model warmup.

Heavy imports (tensorflow, deepface) are done inside functions so the module
imports fast and so TensorFlow GPU configuration happens exactly once, at
process start.
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.logging_config import log


def configure_tensorflow() -> None:
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


def warmup_model() -> None:
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


def list_gpus() -> list:
    """Return the physical GPU devices visible to TensorFlow."""
    import tensorflow as tf

    return tf.config.list_physical_devices("GPU")
