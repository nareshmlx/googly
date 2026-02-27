from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.arq import close_arq_pool
from app.core.auth import verify_internal_token
from app.core.config import settings
from app.core.db import close_db_pools, get_db_pool
from app.core.logging_setup import configure_logging
from app.core.middleware import RequestIDMiddleware
from app.core.redis import close_redis, get_redis

configure_logging(settings.LOG_LEVEL)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app.startup", environment=settings.ENVIRONMENT)

    if not settings.OPENAI_API_KEY:
        logger.warning(
            "app.startup.openai_key_missing",
            hint="Set OPENAI_API_KEY — agent and embedding features will fail without it",
        )

    try:
        pool = await get_db_pool()
        logger.info("db.connected", pool_size=pool.get_size())
    except Exception as e:
        logger.warning("db.connection_failed", error=str(e))

    try:
        redis = await get_redis()
        await redis.ping()
        logger.info("redis.connected")
    except Exception as e:
        logger.warning("redis.connection_failed", error=str(e))

    yield

    logger.info("app.shutdown")
    await close_arq_pool()
    await close_db_pools()
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Googly API",
        description="Googly Agentic Research Platform - Backend API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-Internal-Token",
            "X-User-ID",
        ],
        max_age=3600,
    )

    app.add_middleware(RequestIDMiddleware)

    from app.api.v1 import chat, kb, projects, users

    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
    app.include_router(kb.router, prefix="/api/v1/kb", tags=["kb"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

    @app.get("/health", dependencies=[Depends(verify_internal_token)])
    async def health_check():
        """
        Liveness + readiness check.

        Returns 200 only when both DB and Redis are reachable.
        Returns 503 if either dependency is down.
        """
        from fastapi import status
        from fastapi.responses import JSONResponse

        db_ok = False
        redis_ok = False

        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception as exc:
            logger.warning("health.db_unreachable", error=str(exc))

        try:
            r = await get_redis()
            await r.ping()
            redis_ok = True
        except Exception as exc:
            logger.warning("health.redis_unreachable", error=str(exc))

        body = {
            "status": "healthy" if (db_ok and redis_ok) else "degraded",
            "db": "ok" if db_ok else "unavailable",
            "redis": "ok" if redis_ok else "unavailable",
        }
        http_status = (
            status.HTTP_200_OK if (db_ok and redis_ok) else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=body, status_code=http_status)

    return app


app = create_app()
