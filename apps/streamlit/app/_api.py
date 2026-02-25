"""Thin HTTP client for the Googly FastAPI backend.

All functions are synchronous (httpx.Client) so they can be called directly
from Streamlit code without an async runtime.  SSE streaming is handled via
httpx streaming context so tokens are yielded as they arrive.

Auth strategy (dev): unsigned JWT with sub claim generated locally.
Auth strategy (prod): Bearer token from Clerk session passed as FASTAPI_BEARER_TOKEN env var.
"""

import json
import os
import re
from collections.abc import Generator

import httpx
import jwt  # PyJWT
import structlog

import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_DEV_TOKEN: str | None = None


def _get_bearer_token() -> str:
    """
    Return a Bearer token for FastAPI calls.

    In production, FASTAPI_BEARER_TOKEN contains a valid Clerk-signed JWT.
    In local dev, we generate an unsigned HS256 token with sub = a fixed UUID
    so that get_current_user() can extract a user_id.  This works because
    the dev auth path (ENVIRONMENT=local) skips JWKS verification.
    """
    env_token = os.getenv("FASTAPI_BEARER_TOKEN")
    if env_token:
        return env_token

    global _DEV_TOKEN
    if _DEV_TOKEN is None:
        _DEV_TOKEN = jwt.encode(
            {"sub": "00000000-0000-0000-0000-000000000001"},
            key="",
            algorithm="HS256",
        )
    return _DEV_TOKEN


# NOTE: APIM_INTERNAL_TOKEN must be set to the SAME value in both the Streamlit
# environment and the FastAPI environment. In local dev the default "dev-internal"
# works only because FastAPI's verify_internal_token() skips the check when
# APIM_INTERNAL_TOKEN is not configured server-side.
def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_bearer_token()}",
        "X-User-ID": "00000000-0000-0000-0000-000000000001",
        "X-Internal-Token": os.getenv("APIM_INTERNAL_TOKEN", "dev-internal"),
    }


def _sanitize_chat_query(query: str) -> str:
    """Remove accidentally pasted assistant sections from user query text."""
    text = str(query or "")
    for marker in ("\n\nAnswer", "\nAnswer", "\n\nEvidence", "\nEvidence"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_api_error_detail(response: httpx.Response) -> str:
    """
    Return a concise, user-facing error detail string from a FastAPI response.

    FastAPI validation errors can include the full invalid input payload. This
    helper extracts only the useful message fields to avoid noisy UI and logs.
    """
    try:
        payload = response.json()
    except Exception:
        return (response.text or "Unknown error")[:200]

    detail = payload.get("detail")
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail[:3]:
            if not isinstance(item, dict):
                continue
            loc = ".".join(str(x) for x in (item.get("loc") or []) if x != "body")
            msg = str(item.get("msg") or "").strip()
            if loc and msg:
                parts.append(f"{loc}: {msg}")
            elif msg:
                parts.append(msg)
        if not parts:
            return "Request validation failed."
        suffix = " ..." if len(detail) > 3 else ""
        return "; ".join(parts) + suffix

    if detail is None:
        return "Unknown error"
    return str(detail)[:200]


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def list_projects() -> list[dict]:
    """Return all projects for the current user, newest first."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/projects/", headers=_headers()
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("_api.list_projects.error", error=str(exc))
        return []


def create_project(
    title: str,
    description: str,
    refresh_strategy: str = "once",
    tiktok_enabled: bool = True,
    instagram_enabled: bool = True,
    openalex_enabled: bool = True,
) -> tuple[dict | None, str | None]:
    """
    Create a project.

    Returns:
      - (project_dict, None) on success
      - (None, user_facing_error_message) on failure
    """
    try:
        with httpx.Client(timeout=settings.FASTAPI_CREATE_PROJECT_TIMEOUT) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/projects/",
                headers={**_headers(), "Content-Type": "application/json"},
                json={
                    "title": title,
                    "description": description,
                    "refresh_strategy": refresh_strategy,
                    "tiktok_enabled": tiktok_enabled,
                    "instagram_enabled": instagram_enabled,
                    "openalex_enabled": openalex_enabled,
                },
            )
            resp.raise_for_status()
            return resp.json(), None
    except httpx.TimeoutException as exc:
        logger.warning(
            "_api.create_project.timeout",
            timeout_seconds=settings.FASTAPI_CREATE_PROJECT_TIMEOUT,
            error=str(exc),
        )
        return None, "Project creation timed out. Please try again."
    except httpx.HTTPStatusError as exc:
        detail = _format_api_error_detail(exc.response)
        logger.warning(
            "_api.create_project.http_error",
            status_code=exc.response.status_code,
            detail=detail,
        )
        return None, f"Project creation failed ({exc.response.status_code}): {detail or 'Unknown error'}"
    except Exception as exc:
        logger.warning("_api.create_project.error", error=str(exc))
        return None, "Failed to reach backend while creating project."


def delete_project(project_id: str) -> bool:
    """Delete a project. Returns True on success."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}",
                headers=_headers(),
            )
            return resp.status_code == 204
    except Exception as exc:
        logger.warning("_api.delete_project.error", error=str(exc))
        return False


def get_discover_feed(project_id: str) -> list[dict]:
    """Return social KB items (TikTok + Instagram) for the Discover feed.

    Returns an empty list on error or if the project has no social content yet.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/discover",
                headers=_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("_api.get_discover_feed.error", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


def get_kb_status(project_id: str) -> dict | None:
    """Return KB status for a project, or None on error."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/kb/{project_id}/status",
                headers=_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("_api.get_kb_status.error", error=str(exc))
        return None


def upload_document(project_id: str, filename: str, file_bytes: bytes) -> dict | None:
    """
    Upload a document to a project's KB.

    Returns the UploadResponse dict (with upload_id and status) or None on error.
    Processing is async — poll get_kb_status() to track chunk count growth.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/kb/{project_id}/upload",
                headers=_headers(),
                files={"file": (filename, file_bytes)},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("_api.upload_document.error", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# SSE Chat streaming
# ---------------------------------------------------------------------------


def stream_chat(
    query: str, project_id: str, session_id: str
) -> Generator[str, None, None]:
    """
    Stream SSE tokens from POST /api/v1/chat/.

    Yields plain text tokens (not the raw SSE framing).
    On connection or parse error, yields an error string and stops.
    """
    payload = {
        "query": _sanitize_chat_query(query),
        "project_id": project_id,
        "session_id": session_id,
    }
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                f"{settings.FASTAPI_URL}/api/v1/chat/",
                headers={
                    **_headers(),
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if raw_line.startswith("data: "):
                        data_str = raw_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                            # Support both {token: "..."} and {choices: [{delta: {content: "..."}}]}
                            token = obj.get("token") or (
                                (obj.get("choices") or [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            # Raw text token (non-JSON SSE)
                            if data_str:
                                yield data_str
    except httpx.HTTPStatusError as exc:
        logger.warning("_api.stream_chat.http_error", error=str(exc))
        yield f"\n\n[API error {exc.response.status_code}]"
    except Exception as exc:
        logger.warning("_api.stream_chat.connection_error", error=str(exc))
        yield f"\n\n[Connection error: {exc}]"


def get_project_chat_history(project_id: str, session_id: str) -> list[dict]:
    """
    Fetch persisted chat history for a project+session from FastAPI.

    Returns a list of {role, content} dicts, oldest first.
    Returns an empty list on any error or if no history exists yet.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/chat/history/{project_id}",
                headers=_headers(),
                params={"session_id": session_id},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("_api.get_project_chat_history.error", error=str(exc))
        return []
