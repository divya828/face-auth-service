"""Tier 1: /api/v1/compare routing, threshold, and per-image error reporting."""


def _files(selfie: bytes, document: bytes):
    return {
        "selfie": ("selfie.jpg", selfie, "image/jpeg"),
        "document": ("doc.jpg", document, "image/jpeg"),
    }


def test_identical_faces_match(client):
    r = client.post("/api/v1/compare", files=_files(b"FACE_A", b"FACE_A"))
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["distance"] == 0.0
    assert body["threshold"] == 0.4


def test_different_faces_reject(client):
    # FACE_A vs FACE_B are orthogonal -> distance 1.0 > 0.40 -> reject.
    r = client.post("/api/v1/compare", files=_files(b"FACE_A", b"FACE_B"))
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is False
    assert body["distance"] == 1.0


def test_close_faces_match_under_threshold(client):
    # FACE_CLOSE is ~0.30 distance from FACE_A -> under 0.40 -> match.
    r = client.post("/api/v1/compare", files=_files(b"FACE_A", b"FACE_CLOSE"))
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["distance"] <= 0.4


def test_no_face_in_selfie_reports_which(client):
    r = client.post("/api/v1/compare", files=_files(b"NO_FACE", b"FACE_A"))
    assert r.status_code == 422
    assert r.json()["detail"] == "no_face_detected:selfie"


def test_no_face_in_document_reports_which(client):
    r = client.post("/api/v1/compare", files=_files(b"FACE_A", b"NO_FACE"))
    assert r.status_code == 422
    assert r.json()["detail"] == "no_face_detected:document"


def test_compare_persists_nothing(client, fake_db, archived):
    client.post("/api/v1/compare", files=_files(b"FACE_A", b"FACE_B"))
    # Stateless: no DB rows, no S3 archival regardless of match/reject.
    assert fake_db.store == {}
    assert archived == []
