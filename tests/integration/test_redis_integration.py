"""Tier 2: real Redis INCR/EXPIRE rate-limiter behavior."""

import pytest

pytestmark = pytest.mark.integration


async def _fresh_limiter():
    from app.services.rate_limiter import RateLimiter

    lim = RateLimiter()
    await lim.connect()
    return lim


async def test_allows_up_to_max_then_blocks():
    from app.config import settings

    lim = await _fresh_limiter()
    user = "rl-user-1"
    # Clear any leftover key from a previous run.
    await lim._client.delete(f"ratelimit:verify:{user}")
    try:
        for _ in range(settings.rate_limit_max):
            assert await lim.hit(user) is True
        # Next hit exceeds the limit.
        assert await lim.hit(user) is False
    finally:
        await lim._client.delete(f"ratelimit:verify:{user}")
        await lim.close()


async def test_counter_is_per_user():
    lim = await _fresh_limiter()
    try:
        for u in ("rl-a", "rl-b"):
            await lim._client.delete(f"ratelimit:verify:{u}")
        # Exhaust rl-a.
        for _ in range(3):
            await lim.hit("rl-a")
        assert await lim.hit("rl-a") is False
        # rl-b is independent.
        assert await lim.hit("rl-b") is True
    finally:
        for u in ("rl-a", "rl-b"):
            await lim._client.delete(f"ratelimit:verify:{u}")
        await lim.close()


async def test_expiry_is_set():
    lim = await _fresh_limiter()
    user = "rl-ttl"
    await lim._client.delete(f"ratelimit:verify:{user}")
    try:
        await lim.hit(user)
        ttl = await lim._client.ttl(f"ratelimit:verify:{user}")
        # TTL set on first hit, within the configured window.
        from app.config import settings

        assert 0 < ttl <= settings.rate_limit_window_s
    finally:
        await lim._client.delete(f"ratelimit:verify:{user}")
        await lim.close()
