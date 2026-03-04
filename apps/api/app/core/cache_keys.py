"""Shared cache key helpers for deterministic project-scoped keys."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


def stable_hash(parts: Sequence[object]) -> str:
    """Return short deterministic hash for cache-key input parts."""
    # Canonical JSON avoids ambiguous delimiter collisions (e.g. ["a:b", "c"] vs ["a", "b:c"]).
    raw = json.dumps(list(parts), separators=(",", ":"), sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_search_cache_key(
    *,
    project_id: str,
    provider: str,
    query_type: str,
    parts: list[object],
) -> str:
    """Build project-scoped search cache key."""
    return (
        f"search:cache:{project_id}:{provider}:{query_type}:{stable_hash(parts)}"
    )


def build_stale_cache_key(cache_key: str) -> str:
    """Build stale-fallback key paired to a primary cache key."""
    return f"{cache_key}:stale"
