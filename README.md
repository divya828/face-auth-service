# Face Verification Pipeline

A strict, production-grade face-verification service built for payment-fraud
defense. It enrolls and verifies faces with **DeepFace + ArcFace**, stores
512-dimensional embeddings in **PostgreSQL/pgvector**, and enforces a deliberately
tight similarity cutoff to suppress lookalikes and spoofs.

---

## Architecture

| Concern            | Choice |
|--------------------|--------|
| Embedding model    | DeepFace `ArcFace` (512-d) |
| Face detection     | `retinaface` backend, mandatory landmark alignment (`align=True`) |
| Vector store       | PostgreSQL + `pgvector`, native cosine distance (`<=>`) |
| Security cutoff    | cosine **distance ≤ 0.40 → match** (lower = more similar) |
| API framework      | Async **FastAPI** with `contextlib` lifespan VRAM warmup |
| Concurrency        | Sync DL/DB work offloaded via `run_in_threadpool` |
| Rate limiting      | Redis, **3 verify requests / 60s / user_id** |
| Cold storage       | Rejected/failed JPEGs streamed to **S3** (`fraud_reviews/`) |
| Logging            | CloudWatch-ready structured JSON |
| Target hardware    | AWS EC2 **g6.xlarge** / NVIDIA **L4** (24GB), CUDA 11.8, mixed precision |

> **Threshold direction matters.** pgvector's `<=>` returns cosine *distance* in
> `[0, 2]`; smaller is more similar. The cutoff of `0.40` accepts only very close
> matches — intentionally stricter than ArcFace's ~0.68 default.

---

## Endpoints

| Method | Path                 | Description |
|--------|----------------------|-------------|
| `POST` | `/api/v1/register`   | Enroll a face. Form fields: `user_id`, `image` (JPEG). |
| `POST` | `/api/v1/verify`     | Verify a face against the enrolled embedding. Rate-limited per `user_id`. |
| `GET`  | `/api/v1/health`     | Liveness of PostgreSQL, Redis, and GPU visibility. |

### Example

```bash
# Enroll
curl -X POST http://localhost:8000/api/v1/register \
  -F "user_id=alice" -F "image=@alice.jpg"

# Verify
curl -X POST http://localhost:8000/api/v1/verify \
  -F "user_id=alice" -F "image=@probe.jpg"
# -> {"user_id":"alice","match":true,"distance":0.2317,"threshold":0.4}
```

Verify responses: `429` rate-limited, `404` user not enrolled, `422` no face
detected, `200` with `match: true|false`. Failed/rejected attempts have their
JPEG streamed to `s3://payment-fraud-review-snapshots/fraud_reviews/<user_id>/`.

---

## Setup

### 1. PostgreSQL + pgvector

```bash
psql "$DATABASE_URL" -f schema.sql
```

This creates the `vector` extension, a `faces(user_id, embedding vector(512), ...)`
table, and an IVFFlat cosine index.

### 2. Configuration

```bash
cp .env.example .env
# edit .env with your PG / Redis / AWS values
```

All settings are environment-driven (see `.env.example`). On the EC2 target,
prefer an **instance IAM role** for S3 access over static keys.

### 3. Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Run with Docker (GPU)

```bash
docker build -t face-auth-service .
docker run --gpus all --env-file .env -p 8000:8000 face-auth-service
```

The image is based on `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` and requires
the NVIDIA Container Toolkit on the host.

### 5. Apply the S3 lifecycle rule

Transitions objects under `fraud_reviews/` to Glacier Deep Archive after 90 days:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket payment-fraud-review-snapshots \
  --lifecycle-configuration file://s3-lifecycle-policy.json
```

---

## Log format

Every log line is a single JSON object suitable for CloudWatch Logs Insights.
Metric lines include precise timing and request metadata:

```json
{
  "timestamp": "2026-06-07T17:56:03+0000",
  "level": "INFO",
  "logger": "face-auth",
  "message": "verify",
  "action": "verify",
  "user_id": "alice",
  "result": "match",
  "distance": 0.2317,
  "threshold": 0.4,
  "inference_ms": 41.88,
  "db_ms": 3.12,
  "total_ms": 47.55
}
```

Fields emitted per action:

| action      | key fields |
|-------------|-----------|
| `register`  | `user_id`, `result`, `inference_ms`, `db_ms`, `total_ms` |
| `verify`    | `user_id`, `result` (`match`/`reject`/`rate_limited`/`no_face_detected`/`unknown_user`), `distance`, `threshold`, `inference_ms`, `db_ms`, `total_ms` |
| `s3_archive`| `user_id`, `s3_key`, `reason` |
| startup/shutdown | lifecycle markers |

Example CloudWatch Logs Insights query — p95 verify latency for matches:

```
fields total_ms
| filter action = "verify" and result = "match"
| stats pct(total_ms, 95) as p95_ms
```

---

## Operational notes

- **Single uvicorn worker per GPU.** The model holds VRAM; scale out with more
  pods/instances, not in-process workers. Throughput within a process comes from
  the threadpool offload of DeepFace and DB calls.
- **VRAM warmup** runs at startup via a dummy array so the first real request is
  not cold.
- **Mixed precision** (`mixed_float16`) is enabled at startup for the L4.
- **Rate limiting** uses `INCR` + `EXPIRE`; the first request in a window sets the
  60-second TTL.
