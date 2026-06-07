"""Redis-backed per-user rate limiting (3 verify requests / 60s / user_id)."""

from __future__ import annotations

from app.config import settings


class RateLimiter:
    def __init__(self) -> None:
        self._client = None

    async def connect(self) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def hit(self, user_id: str) -> bool:
        """Increment the per-user counter. Return True if WITHIN the limit."""
        key = f"ratelimit:verify:{user_id}"
        count = await self._client.incr(key)
        if count == 1:
            await self._client.expire(key, settings.rate_limit_window_s)
        return count <= settings.rate_limit_max

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


limiter = RateLimiter()
