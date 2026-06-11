"""Verification orchestration: ties together embeddings, database, and S3 archival.

These functions contain the register/verify business logic and timing metrics,
so the API route handlers stay thin. DeepFace and psycopg2 calls are sync and
are offloaded via run_in_threadpool here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.database import db
from app.embeddings.arcface import cosine_distance, embed
from app.logging_config import emit
from app.storage import archive_rejected


class NoFaceDetected(Exception):
    """Raised when no face can be detected/embedded from the payload.

    `which` identifies the offending image for stateless multi-image flows
    ("selfie" / "document"); None for the single-image register/verify paths.
    """

    def __init__(self, which: Optional[str] = None) -> None:
        super().__init__(which or "no_face_detected")
        self.which = which


class UserNotEnrolled(Exception):
    """Raised when verifying a user_id that has no stored embedding."""


@dataclass
class VerifyResult:
    user_id: str
    match: bool
    distance: float
    threshold: float


@dataclass
class CompareResult:
    match: bool
    distance: float
    threshold: float


async def register_face(user_id: str, payload: bytes) -> None:
    """Embed and upsert a face. Raises NoFaceDetected on failure."""
    t0 = time.perf_counter()

    t_inf = time.perf_counter()
    try:
        embedding = await run_in_threadpool(embed, payload)
    except ValueError as exc:
        raise NoFaceDetected() from exc
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


async def verify_face(user_id: str, payload: bytes) -> VerifyResult:
    """Embed, look up nearest stored embedding, apply strict cutoff.

    Raises NoFaceDetected or UserNotEnrolled; archives rejected/failed JPEGs.
    """
    t0 = time.perf_counter()

    t_inf = time.perf_counter()
    try:
        embedding = await run_in_threadpool(embed, payload)
    except ValueError as exc:
        await archive_rejected(user_id, payload, reason="no_face")
        emit("verify", user_id=user_id, result="no_face_detected")
        raise NoFaceDetected() from exc
    inference_ms = (time.perf_counter() - t_inf) * 1000

    t_db = time.perf_counter()
    distance: Optional[float] = await run_in_threadpool(db.nearest, user_id, embedding)
    db_ms = (time.perf_counter() - t_db) * 1000

    if distance is None:
        emit("verify", user_id=user_id, result="unknown_user")
        raise UserNotEnrolled()

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
    return VerifyResult(
        user_id=user_id,
        match=matched,
        distance=round(distance, 4),
        threshold=settings.cosine_threshold,
    )


async def compare_faces(selfie: bytes, document: bytes) -> CompareResult:
    """Stateless 1:1 comparison of a live selfie against a document (OVD) photo.

    Embeds both images, computes cosine distance in-process (no database, no
    storage), and applies the same strict cutoff as verify. Raises NoFaceDetected
    identifying which image failed. Nothing is persisted or archived.
    """
    t0 = time.perf_counter()

    t_inf = time.perf_counter()
    try:
        selfie_vec = await run_in_threadpool(embed, selfie)
    except ValueError as exc:
        emit("compare", result="no_face_detected", which="selfie")
        raise NoFaceDetected("selfie") from exc
    try:
        document_vec = await run_in_threadpool(embed, document)
    except ValueError as exc:
        emit("compare", result="no_face_detected", which="document")
        raise NoFaceDetected("document") from exc
    inference_ms = (time.perf_counter() - t_inf) * 1000

    distance = cosine_distance(selfie_vec, document_vec)
    matched = distance <= settings.cosine_threshold
    total_ms = (time.perf_counter() - t0) * 1000

    emit(
        "compare",
        result="match" if matched else "reject",
        distance=round(distance, 4),
        threshold=settings.cosine_threshold,
        inference_ms=round(inference_ms, 2),
        total_ms=round(total_ms, 2),
    )
    return CompareResult(
        match=matched,
        distance=round(distance, 4),
        threshold=settings.cosine_threshold,
    )
