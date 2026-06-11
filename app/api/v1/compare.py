"""POST /api/v1/compare -- stateless 1:1 selfie vs document (OVD) photo comparison."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.verification import NoFaceDetected, compare_faces

router = APIRouter()


@router.post("/compare")
async def compare(
    selfie: UploadFile = File(...),
    document: UploadFile = File(...),
) -> dict:
    """Compare a live selfie against a document photo. Stores nothing.

    Both faces are embedded and compared by cosine distance against the strict
    cutoff. Returns 422 identifying which image had no detectable face.
    """
    selfie_bytes = await selfie.read()
    document_bytes = await document.read()
    try:
        result = await compare_faces(selfie_bytes, document_bytes)
    except NoFaceDetected as exc:
        raise HTTPException(
            status_code=422,
            detail=f"no_face_detected:{exc.which}" if exc.which else "no_face_detected",
        )

    return {
        "match": result.match,
        "distance": result.distance,
        "threshold": result.threshold,
    }
