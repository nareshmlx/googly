"""Shared utility helpers for lightweight payload normalization."""

from __future__ import annotations

import hashlib
import orjson
from collections.abc import Mapping, Sequence


def parse_metadata(raw: object) -> dict:
    """Normalize metadata payloads from asyncpg/jsonb into a plain dict using orjson."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = orjson.loads(text)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    if isinstance(raw, bytes):
        try:
            parsed = orjson.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}



def build_stable_signature(values: list[str], delimiter: str = "|") -> str:
    """Build a deterministic SHA256 signature for ordered string values."""
    if not values:
        return ""
    payload = delimiter.join(values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_non_empty_value(*values: object) -> object | None:
    """Return the first non-empty candidate value from ordered fallbacks."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
            continue
        return value
    return None


def metadata_pick(
    metadata: Mapping[str, object],
    keys: tuple[str, ...],
    *fallback_values: object,
) -> object | None:
    """Pick the first non-empty metadata value by ordered key preference."""
    candidates = [metadata.get(key) for key in keys]
    return first_non_empty_value(*candidates, *fallback_values)


def dedup_terms(*lists: Sequence[object] | None, limit: int | None = None) -> list[str]:
    """Merge and deduplicate term lists, preserving insertion order.

    Each input list is iterated; values are stringified and stripped.
    Deduplication is case-insensitive. Returns at most ``limit`` items
    if specified.
    """
    seen: set[str] = set()
    result: list[str] = []
    for terms in lists:
        for raw in terms or []:
            value = str(raw).strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                result.append(value)
    return result[:limit] if limit is not None else result
