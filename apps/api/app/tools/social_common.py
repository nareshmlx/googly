"""Common utilities and base classes for social media API tools."""

import asyncio
import os
from typing import Annotated, Any

import certifi
import structlog
from pydantic import BeforeValidator

try:
    from ensembledata.api import EDClient
except ImportError:
    EDClient = None  # type: ignore[assignment]

from app.core.circuit import ensemble_breaker
from app.core.config import settings
from app.core.retry import retry_with_backoff

logger = structlog.get_logger(__name__)
_SDK_WARNED = False


def strip_commas_for_int(v: Any) -> Any:
    """Helper for Pydantic to handle string integers with commas from social APIs."""
    if isinstance(v, str):
        return v.replace(",", "")
    return v


RobustInt = Annotated[int, BeforeValidator(strip_commas_for_int)]


def ensure_ssl_ca_bundle() -> None:
    """Ensure a valid CA bundle path exists for SDK/httpx TLS verification."""
    certifi_path = certifi.where()
    if certifi_path and os.path.exists(certifi_path):
        os.environ.setdefault("SSL_CERT_FILE", certifi_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi_path)


def sdk_available(platform: str) -> bool:
    """Return whether EnsembleData SDK is importable in this runtime."""
    global _SDK_WARNED
    if EDClient is None:
        if not _SDK_WARNED:
            logger.warning(f"social_{platform}.sdk_missing")
            _SDK_WARNED = True
        return False
    return True


def extract_items_from_payload(payload: object) -> list[dict]:
    """Extract list payloads from common EnsembleData/HTTP JSON response variants."""
    if not isinstance(payload, dict):
        return []

    # Case 1: Nested under 'data' key
    data = payload.get("data")
    if isinstance(data, dict):
        # Nested data.data or data.items
        raw_items = data.get("data") or data.get("items") or []
    elif isinstance(data, list):
        # Direct list under 'data'
        raw_items = data
    else:
        # Case 2: Direct top-level 'items' or 'results' or 'data' (already checked)
        raw_items = payload.get("items") or payload.get("results") or []

    if not isinstance(raw_items, list):
        return []

    return [item for item in raw_items if isinstance(item, dict)]


def extract_items_from_sdk_result(result: Any) -> list[dict]:
    """Extract list payloads from EnsembleData SDK result objects."""
    if not result or not hasattr(result, "data"):
        return []

    data = result.data
    if isinstance(data, dict):
        raw_items = data.get("data") or data.get("items") or []
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    if not isinstance(raw_items, list):
        return []

    return [item for item in raw_items if isinstance(item, dict)]


async def ensemble_sdk_call(platform: str, method: str, *args: Any, **kwargs: Any) -> Any:
    """Async wrapper for synchronous EnsembleData SDK calls with circuit breaker.

    Offloads the blocking call to a thread pool and protects it with a circuit breaker.
    Returns the result object on success, or None on failure/blocked.
    Never raises to the caller.
    """
    if not (settings.ENSEMBLE_API_TOKEN and sdk_available(platform)):
        return None

    ensure_ssl_ca_bundle()

    def _sync_call() -> Any:
        client = EDClient(token=settings.ENSEMBLE_API_TOKEN or "")
        # Resolve path e.g. client.tiktok.hashtag_search
        attr = client
        for part in method.split("."):
            attr = getattr(attr, part)
        return attr(*args, **kwargs)

    async def _async_call() -> Any:
        return await asyncio.to_thread(_sync_call)

    # Wrap the whole thing in retry (inner) + circuit breaker (outer)
    # Actually, tenacity usually wraps the actual call.
    # aiobreaker should be outer to fail immediately if circuit is open.
    try:
        protected_call = ensemble_breaker(_async_call)
        result = await retry_with_backoff(
            protected_call,
            max_attempts=int(settings.INGEST_RETRY_ATTEMPTS or 2),
            base_delay=1.0,
        )
        return result
    except Exception as exc:
        logger.exception(f"social_{platform}.sdk_call.failed", method=method, error=str(exc))
        return None
