import logging
import logging.config
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.arq import close_arq_pool
from app.core.auth import verify_internal_token
from app.core.config import settings
from app.core.db import close_db_pools, get_db_pool
from app.core.middleware import RequestIDMiddleware
from app.core.redis import close_redis, get_redis

# Shared processors used by both structlog and the stdlib logging bridge.
# Defined once so both pipelines produce identical JSON output.
_SHARED_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
]

structlog.configure(
    processors=[
        *_SHARED_PROCESSORS,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Route ALL stdlib loggers (uvicorn, arq, httpx, asyncpg, …) through structlog
# so every log line is JSON with the same shape as structlog output.
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                "foreign_pre_chain": _SHARED_PROCESSORS,
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "root": {
            "handlers": ["default"],
            "level": settings.LOG_LEVEL,
        },
        # Silence chatty libs that spam debug logs at INFO level
        "loggers": {
            "uvicorn.access": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "asyncpg": {"level": "WARNING"},
        },
    }
)

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
