import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# structlog is configured once in main.py at application startup.
# Do NOT call structlog.configure() here — double-configure corrupts the
# processor chain and causes duplicate log entries in production.

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        logger.info("request.started", method=request.method, path=request.url.path)

        response = await call_next(request)

        logger.info(
            "request.completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        response.headers["X-Request-ID"] = request_id
        return response


async def log_error(request: Request, exc: Exception) -> None:
    logger.exception(
        "request.error",
        method=request.method,
        path=request.url.path,
        error=str(exc),
    )
