"""Project-scoped cache version helpers."""

from redis.asyncio import Redis

from app.core.constants import RedisKeys


async def get_project_cache_version(redis: Redis, project_id: str) -> str:
    """Return project cache version string, defaulting to '0'."""
    value = await redis.get(RedisKeys.PROJECT_CACHE_VERSION.format(project_id=project_id))
    return str(value) if value is not None else "0"


async def bump_project_cache_version(redis: Redis, project_id: str) -> int:
    """Atomically bump and return the project cache version."""
    return int(await redis.incr(RedisKeys.PROJECT_CACHE_VERSION.format(project_id=project_id)))

