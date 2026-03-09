"""Shared serialization helpers used across services and tasks."""

from __future__ import annotations

import orjson
from typing import Any

import numpy as np


def to_list(value: Any) -> list[Any]:
    """Convert asyncpg/json payloads to list using orjson."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = orjson.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    if isinstance(value, bytes):
        try:
            parsed = orjson.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def to_dict(value: Any) -> dict[str, Any]:
    """Convert asyncpg/json payloads to dict using orjson."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = orjson.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if isinstance(value, bytes):
        try:
            parsed = orjson.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}



def sse_frame(payload: dict[str, Any]) -> str:
    """Format one SSE JSON frame using orjson."""
    return f"data: {orjson.dumps(payload).decode()}\n\n"



def parse_embedding(
    raw: Any,
    *,
    expected_dim: int | None = None,
) -> np.ndarray | None:
    """Parse embedding text/list to float32 vector.

    Handles pgvector text representations, raw lists, and None.
    Returns None when parsing fails or dimension check fails.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            vector = np.asarray(raw, dtype=np.float32)
            if expected_dim is not None and vector.size != expected_dim:
                return None
            return vector
        except Exception:
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        try:
            parts = text.split(",")
            vector = np.array(parts, dtype=np.float32)
            if vector.size == 0:
                return None
            if expected_dim is not None and vector.size != expected_dim:
                return None
            return vector
        except Exception:
            return None
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity with zero-vector safety."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / denom)
