"""Tier 1: /api/v1/register + /api/v1/verify flow, rate limiting, archival."""


def _img(payload: bytes):
    return {"image": ("face.jpg", payload, "image/jpeg")}


def _form(user_id: str):
    return {"user_id": user_id}


def test_register_then_verify_match(client, fake_db):
    r = client.post("/api/v1/register", data=_form("alice"), files=_img(b"FACE_A"))
    assert r.status_code == 200
    assert r.json()["status"] == "registered"
    assert "alice" in fake_db.store

    r = client.post("/api/v1/verify", data=_form("alice"), files=_img(b"FACE_A"))
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["distance"] == 0.0


def test_verify_reject_archives(client, archived):
    client.post("/api/v1/register", data=_form("bob"), files=_img(b"FACE_A"))
    # Verify with an orthogonal face -> distance 1.0 -> reject -> archived.
    r = client.post("/api/v1/verify", data=_form("bob"), files=_img(b"FACE_B"))
    assert r.status_code == 200
    assert r.json()["match"] is False
    assert ("bob", "rejected") in archived


def test_verify_unknown_user_404(client):
    r = client.post("/api/v1/verify", data=_form("ghost"), files=_img(b"FACE_A"))
    assert r.status_code == 404
    assert r.json()["detail"] == "user_not_enrolled"


def test_verify_no_face_422_and_archives(client, archived):
    client.post("/api/v1/register", data=_form("carol"), files=_img(b"FACE_A"))
    r = client.post("/api/v1/verify", data=_form("carol"), files=_img(b"NO_FACE"))
    assert r.status_code == 422
    assert r.json()["detail"] == "no_face_detected"
    assert ("carol", "no_face") in archived


def test_register_no_face_422_no_store(client, fake_db):
    r = client.post("/api/v1/register", data=_form("dave"), files=_img(b"NO_FACE"))
    assert r.status_code == 422
    assert "dave" not in fake_db.store


def test_rate_limit_after_three(client):
    client.post("/api/v1/register", data=_form("eve"), files=_img(b"FACE_A"))
    # 3 allowed, 4th -> 429.
    for _ in range(3):
        r = client.post("/api/v1/verify", data=_form("eve"), files=_img(b"FACE_A"))
        assert r.status_code == 200
    r = client.post("/api/v1/verify", data=_form("eve"), files=_img(b"FACE_A"))
    assert r.status_code == 429
    assert r.json()["detail"] == "rate_limit_exceeded"


def test_rate_limit_is_per_user(client):
    client.post("/api/v1/register", data=_form("u1"), files=_img(b"FACE_A"))
    client.post("/api/v1/register", data=_form("u2"), files=_img(b"FACE_A"))
    for _ in range(3):
        client.post("/api/v1/verify", data=_form("u1"), files=_img(b"FACE_A"))
    # u1 exhausted, u2 should still be allowed.
    r = client.post("/api/v1/verify", data=_form("u2"), files=_img(b"FACE_A"))
    assert r.status_code == 200


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["postgres"] is True
    assert body["checks"]["redis"] is True
    assert body["checks"]["gpu_visible"] is True
