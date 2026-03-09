"""Per-user rate limiting for API endpoints using Redis sliding window.

Simple implementation: each user gets N requests per time window.
Uses Redis INCR + EXPIRE for atomic rate limiting without race conditions.
"""

import time

import structlog
from fastapi import HTTPException

from app.core.config import settings
from app.core.redis import get_redis

logger = structlog.get_logger(__name__)


async def check_user_rate_limit(
    user_id: str,
    endpoint: str,
    *,
    limit: int | None = None,
    window_seconds: int | None = None,
) -> None:
    """
    Check if user has exceeded rate limit for the endpoint.

    Raises HTTPException 429 if limit exceeded.
    Uses Redis sliding window with atomic INCR + EXPIRE.

    Args:
        user_id: User ID to check
        endpoint: Endpoint name (e.g., "chat", "followup")
        limit: Optional per-endpoint limit override (defaults to CHAT_RATE_LIMIT)
        window_seconds: Optional per-endpoint window override (defaults to CHAT_RATE_WINDOW_SECONDS)

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    if limit is None:
        limit = settings.FOLLOWUP_RATE_LIMIT if endpoint == "followup" else settings.CHAT_RATE_LIMIT
    if window_seconds is None:
        window_seconds = (
            settings.FOLLOWUP_RATE_WINDOW_SECONDS
            if endpoint == "followup"
            else settings.CHAT_RATE_WINDOW_SECONDS
        )

    try:
        redis = await get_redis()
        window = int(time.time()) // window_seconds
        key = f"ratelimit:user:{user_id}:{endpoint}:{window}"

        # Atomic increment + check
        count = await redis.incr(key)

        if count == 1:
            # First request in this window - set expiry
            await redis.expire(key, window_seconds * 2)

        if count > limit:
            logger.warning(
                "rate_limit.exceeded",
                user_id=user_id,
                endpoint=endpoint,
                count=count,
                limit=limit,
                window_seconds=window_seconds,
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. Maximum "
                    f"{limit} requests per "
                    f"{window_seconds} seconds."
                ),
                headers={"Retry-After": str(window_seconds)},
            )

        logger.debug(
            "rate_limit.check",
            user_id=user_id,
            endpoint=endpoint,
            count=count,
            limit=limit,
        )

    except HTTPException:
        raise  # Re-raise rate limit exception
    except Exception:
        # If Redis fails, allow request through (fail-open for availability)
        # This follows AGENTS.md Rule 8: graceful degradation
        logger.exception(
            "rate_limit.redis_error",
            user_id=user_id,
            endpoint=endpoint,
        )


def check_chat_rate_limit(current_user: dict):  # type: ignore[no-untyped-def]
    """
    FastAPI dependency factory for per-user chat rate limiting.

    Returns an async function that checks rate limits.
    This allows the dependency to access current_user from the outer scope.

    Usage in endpoint:
        async def chat(
            request: ChatRequest,
            current_user: dict = Depends(get_current_user),
        ):
            await check_chat_rate_limit(current_user)
            ...
    """

    async def _check():
        user_id = current_user.get("user_id")
        if not user_id:
            logger.warning("rate_limit.no_user_id")
            return

        await check_user_rate_limit(user_id, "chat")

    return _check()
