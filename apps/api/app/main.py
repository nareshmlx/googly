from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.arq import close_arq_pool
from app.core.config import settings
from app.core.db import close_db_pools
from app.core.logging_setup import configure_logging
from app.core.middleware import RequestIDMiddleware
from app.core.redis import close_redis, get_redis
from app.repositories.health import check_db_ready
from app.repositories.insights import reset_stuck_generating_reports_for_service

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
        db_ready = await check_db_ready()
        if db_ready:
            logger.info("db.connected")
        else:
            logger.warning("db.connection_failed", error="db_not_ready")
    except Exception as e:
        logger.warning("db.connection_failed", error=str(e))

    try:
        redis = await get_redis()
        await redis.ping()
        logger.info("redis.connected")
    except Exception as e:
        logger.warning("redis.connection_failed", error=str(e))

    try:
        reset_count = await reset_stuck_generating_reports_for_service()
        logger.info("insights.startup_report_status_reset", reset_count=reset_count)
    except Exception as exc:
        logger.warning("insights.startup_report_status_reset_failed", error=str(exc))

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
        max_age=settings.CORS_MAX_AGE_SECONDS,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=settings.GZIP_MINIMUM_SIZE_BYTES)

    from app.api.v1 import chat, cluster_followup, insights, kb, projects, users

    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
    app.include_router(kb.router, prefix="/api/v1/kb", tags=["kb"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
    app.include_router(insights.router, prefix="/api/v1", tags=["insights"])
    app.include_router(cluster_followup.router, prefix="/api/v1", tags=["insights-followup"])

    @app.get("/health")
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
            db_ok = await check_db_ready()
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

    @app.get("/healthz")
    async def healthz():
        """
        Lightweight liveness check without authentication.

        Use this for Kubernetes liveness/readiness probes when
        APIM_INTERNAL_TOKEN is configured.
        """
        return {"status": "ok"}

    return app


app = create_app()
