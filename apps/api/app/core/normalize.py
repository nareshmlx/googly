"""Shared normalization helpers for primitive coercion."""

from __future__ import annotations


def coerce_int(value: object, default: int = 0) -> int:
    """Convert int/float/string numeric values to int safely."""
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        trimmed = value.strip().replace(",", "")
        if not trimmed:
            return default
        try:
            return int(float(trimmed))
        except ValueError:
            return default
    return default
