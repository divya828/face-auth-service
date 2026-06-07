# ---------------------------------------------------------------------------
# Production image for the face-verification pipeline.
# Target hardware: AWS EC2 g6.xlarge / NVIDIA L4 (Ada, 24GB), CUDA 11.8 + cuDNN 8.
#
# TensorFlow 2.14 is the matching wheel for CUDA 11.8 / cuDNN 8. Mixed precision
# (mixed_float16) is enabled at runtime in main.py via _configure_tensorflow().
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Make the L4 GPU visible to the NVIDIA container runtime.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Pre-create the DeepFace weight cache so warmup writes to a known path.
    DEEPFACE_HOME=/app/.deepface

# System deps: python, build toolchain for psycopg2, libGL for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        libpq-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install python deps first for layer caching.
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# Pre-bake model weights into the image so cold starts don't download at runtime.
RUN mkdir -p ${DEEPFACE_HOME}/weights

EXPOSE 8000

# Single worker: the model holds GPU memory; scale horizontally with more pods,
# not more in-process workers. Concurrency is handled by run_in_threadpool.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
