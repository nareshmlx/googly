"""Shared HTTP client singleton for efficient connection pooling.

This module provides a single httpx.AsyncClient instance that is reused
across all API calls, avoiding the overhead of creating new TCP connection
pools per request.
"""

from functools import cache

import httpx

# functools.cache creates a module-level singleton on first call — one
# connection pool per worker process, reused across all tools.
@cache
def get_http_client() -> httpx.AsyncClient:
    """Return (or lazily create) the module-level httpx.AsyncClient."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=True,  # HTTP/2 multiplexing — fewer connections, better throughput
    )
