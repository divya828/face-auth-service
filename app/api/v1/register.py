"""POST /api/v1/register -- enroll a face."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.verification import NoFaceDetected, register_face

router = APIRouter()


@router.post("/register")
async def register(user_id: str = Form(...), image: UploadFile = File(...)) -> dict:
    payload = await image.read()
    try:
        await register_face(user_id, payload)
    except NoFaceDetected:
        raise HTTPException(status_code=422, detail="no_face_detected")
    return {"user_id": user_id, "status": "registered"}
