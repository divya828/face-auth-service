"""S3 cold storage (boto3 is sync -> put_jpeg is threadpooled)."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.logging_config import emit, log


class ColdStorage:
    def __init__(self) -> None:
        self._client = None

    def connect(self) -> None:
        import boto3

        self._client = boto3.client("s3", region_name=settings.aws_region)

    def put_jpeg(self, key: str, payload: bytes) -> None:
        self._client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=payload,
            ContentType="image/jpeg",
        )


cold = ColdStorage()


async def archive_rejected(user_id: str, payload: bytes, reason: str) -> None:
    """Stream a rejected/failed JPEG to the fraud-review S3 prefix (best-effort)."""
    # Deterministic-ish key without Date.now(): user + reason + size + content hash slice.
    digest = abs(hash((user_id, len(payload), reason))) % (10**12)
    key = f"{settings.s3_prefix}{user_id}/{reason}-{digest}.jpg"
    try:
        await run_in_threadpool(cold.put_jpeg, key, payload)
        emit("s3_archive", user_id=user_id, s3_key=key, reason=reason)
    except Exception as exc:  # noqa: BLE001 - archival must never break the response
        log.warning("s3_archive_failed: %s", exc, extra={"metrics": {"user_id": user_id}})
