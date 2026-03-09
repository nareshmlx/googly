"""Shared normalization helpers for primitive coercion."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, TypeAdapter, ValidationError


def _strip_commas_for_int(v: Any) -> Any:
    if isinstance(v, str):
        return v.replace(",", "")
    return v

RobustInt = Annotated[int, BeforeValidator(_strip_commas_for_int)]
_int_adapter = TypeAdapter(RobustInt)

def coerce_int(value: object, default: int = 0) -> int:
    """Convert int/float/string numeric values to int safely via Pydantic."""
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        return default
    try:
        return _int_adapter.validate_python(value)
    except ValidationError:
        return default
