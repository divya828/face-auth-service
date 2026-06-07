"""POST /api/v1/verify -- verify a face against the enrolled embedding."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.logging_config import emit
from app.services.rate_limiter import limiter
from app.services.verification import (
    NoFaceDetected,
    UserNotEnrolled,
    verify_face,
)

router = APIRouter()


@router.post("/verify")
async def verify(user_id: str = Form(...), image: UploadFile = File(...)) -> dict:
    # Rate limit BEFORE doing any expensive work.
    if not await limiter.hit(user_id):
        emit("verify", user_id=user_id, result="rate_limited")
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")

    payload = await image.read()
    try:
        result = await verify_face(user_id, payload)
    except NoFaceDetected:
        raise HTTPException(status_code=422, detail="no_face_detected")
    except UserNotEnrolled:
        raise HTTPException(status_code=404, detail="user_not_enrolled")

    return {
        "user_id": result.user_id,
        "match": result.match,
        "distance": result.distance,
        "threshold": result.threshold,
    }
