"""Shared utility helpers for lightweight payload normalization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def parse_metadata(raw: object) -> dict:
    """Normalize metadata payloads from asyncpg/jsonb into a plain dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError):
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
