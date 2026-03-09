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
import settings
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_DEV_TOKEN: str | None = None
_DEV_USER_ID = os.getenv("FASTAPI_USER_ID", "00000000-0000-0000-0000-000000000001")


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
            {"sub": _DEV_USER_ID},
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
        "X-User-ID": _DEV_USER_ID,
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
    youtube_enabled: bool = True,
    reddit_enabled: bool = True,
    x_enabled: bool = True,
    papers_enabled: bool = True,
    patents_enabled: bool = True,
    perigon_enabled: bool = True,
    tavily_enabled: bool = True,
    exa_enabled: bool = True,
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
                    "youtube_enabled": youtube_enabled,
                    "reddit_enabled": reddit_enabled,
                    "x_enabled": x_enabled,
                    "papers_enabled": papers_enabled,
                    "patents_enabled": patents_enabled,
                    "perigon_enabled": perigon_enabled,
                    "tavily_enabled": tavily_enabled,
                    "exa_enabled": exa_enabled,
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
        return (
            None,
            f"Project creation failed ({exc.response.status_code}): {detail or 'Unknown error'}",
        )
    except Exception as exc:
        logger.warning("_api.create_project.error", error=str(exc))
        return None, "Failed to reach backend while creating project."


def wizard_evaluate(
    title: str,
    description: str,
    qa_pairs: list[dict],
    max_questions: int = 5,
) -> tuple[dict | None, str | None]:
    """Evaluate wizard sufficiency and fetch next dynamic question."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_CREATE_PROJECT_TIMEOUT) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/projects/wizard/evaluate",
                headers={**_headers(), "Content-Type": "application/json"},
                json={
                    "title": title,
                    "description": description,
                    "qa_pairs": qa_pairs,
                    "max_questions": max_questions,
                },
            )
            resp.raise_for_status()
            return resp.json(), None
    except httpx.TimeoutException as exc:
        logger.warning("_api.wizard_evaluate.timeout", error=str(exc))
        return None, "Wizard evaluation timed out. Please try again."
    except httpx.HTTPStatusError as exc:
        detail = _format_api_error_detail(exc.response)
        logger.warning(
            "_api.wizard_evaluate.http_error",
            status_code=exc.response.status_code,
            detail=detail,
        )
        return (
            None,
            f"Wizard evaluation failed ({exc.response.status_code}): {detail or 'Unknown error'}",
        )
    except Exception as exc:
        logger.warning("_api.wizard_evaluate.error", error=str(exc))
        return None, "Failed to evaluate wizard state."


def wizard_synthesize(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict],
    structured_intent: dict,
    source_toggles: dict[str, bool],
) -> tuple[dict | None, str | None]:
    """Generate phase-2 review payload from wizard context."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_CREATE_PROJECT_TIMEOUT) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/projects/wizard/synthesize",
                headers={**_headers(), "Content-Type": "application/json"},
                json={
                    "title": title,
                    "description": description,
                    "qa_pairs": qa_pairs,
                    "structured_intent": structured_intent,
                    "source_toggles": source_toggles,
                },
            )
            resp.raise_for_status()
            return resp.json(), None
    except httpx.TimeoutException as exc:
        logger.warning("_api.wizard_synthesize.timeout", error=str(exc))
        return None, "Wizard synthesis timed out. Please try again."
    except httpx.HTTPStatusError as exc:
        detail = _format_api_error_detail(exc.response)
        logger.warning(
            "_api.wizard_synthesize.http_error",
            status_code=exc.response.status_code,
            detail=detail,
        )
        return (
            None,
            f"Wizard synthesis failed ({exc.response.status_code}): {detail or 'Unknown error'}",
        )
    except Exception as exc:
        logger.warning("_api.wizard_synthesize.error", error=str(exc))
        return None, "Failed to synthesize wizard review."


def wizard_create(
    *,
    title: str,
    description: str,
    qa_pairs: list[dict],
    refresh_strategy: str,
    enriched_description: str,
    domain_focus: str,
    key_entities: list[str],
    must_match_terms: list[str],
    time_horizon: str,
    target_sources: dict[str, bool],
) -> tuple[dict | None, str | None]:
    """Create a project via two-phase wizard context and manual overrides."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_CREATE_PROJECT_TIMEOUT) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/projects/wizard/create",
                headers={**_headers(), "Content-Type": "application/json"},
                json={
                    "title": title,
                    "description": description,
                    "qa_pairs": qa_pairs,
                    "refresh_strategy": refresh_strategy,
                    "enriched_description": enriched_description,
                    "domain_focus": domain_focus,
                    "key_entities": key_entities,
                    "must_match_terms": must_match_terms,
                    "time_horizon": time_horizon,
                    "target_sources": target_sources,
                },
            )
            resp.raise_for_status()
            return resp.json(), None
    except httpx.TimeoutException as exc:
        logger.warning("_api.wizard_create.timeout", error=str(exc))
        return None, "Wizard project creation timed out. Please try again."
    except httpx.HTTPStatusError as exc:
        detail = _format_api_error_detail(exc.response)
        logger.warning(
            "_api.wizard_create.http_error",
            status_code=exc.response.status_code,
            detail=detail,
        )
        return (
            None,
            f"Wizard project creation failed ({exc.response.status_code}): {detail or 'Unknown error'}",
        )
    except Exception as exc:
        logger.warning("_api.wizard_create.error", error=str(exc))
        return None, "Failed to create project with wizard flow."


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


def get_insights(project_id: str) -> list[dict]:
    """Return insight cards for a project, or empty list on error."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_DISCOVER_TIMEOUT) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/insights",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("_api.get_insights.error", project_id=project_id, error=str(exc))
        return []


def get_insight_detail(project_id: str, insight_id: str) -> dict | None:
    """Return one insight detail object."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_DISCOVER_TIMEOUT) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/insights/{insight_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning(
            "_api.get_insight_detail.error",
            project_id=project_id,
            insight_id=insight_id,
            error=str(exc),
        )
        return None


def refresh_insights(project_id: str) -> tuple[bool, str]:
    """Enqueue insight refresh and return (accepted, message)."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_DISCOVER_TIMEOUT) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/insights/refresh",
                headers=_headers(),
            )
            if resp.status_code != 202:
                detail = _format_api_error_detail(resp)
                return False, f"Refresh failed ({resp.status_code}): {detail or 'Unknown error'}"
            payload = resp.json() if resp.content else {}
            if isinstance(payload, dict):
                status = str(payload.get("status") or "").strip().lower()
                message = str(payload.get("message") or "").strip()
                if status == "accepted":
                    return True, (message or "Insights refresh enqueued")
                return False, (message or "Insights refresh skipped")
            return True, "Insights refresh enqueued"
    except Exception as exc:
        logger.warning("_api.refresh_insights.error", project_id=project_id, error=str(exc))
        return False, "Could not enqueue insights refresh"


def stream_full_report(project_id: str, insight_id: str) -> Generator[str, None, None]:
    """Stream insight full report tokens via SSE endpoint."""
    try:
        with (
            httpx.Client(timeout=None) as client,
            client.stream(
                "GET",
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/insights/{insight_id}/report/stream",
                headers={**_headers(), "Accept": "text/event-stream"},
            ) as response,
        ):
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    payload = json.loads(data_str)
                    if payload.get("token"):
                        yield str(payload["token"])
                    elif payload.get("type") == "status":
                        logger.info(
                            "_api.stream_full_report.status",
                            insight_id=insight_id,
                            status=str(payload.get("status") or "unknown"),
                        )
                    elif payload.get("error"):
                        logger.warning(
                            "_api.stream_full_report.frame_error",
                            insight_id=insight_id,
                            error=str(payload.get("error") or ""),
                        )
                except json.JSONDecodeError:
                    if data_str.strip():
                        yield data_str
    except httpx.HTTPStatusError as exc:
        logger.warning("_api.stream_full_report.http_error", insight_id=insight_id, error=str(exc))
        yield f"\n\n[API error {exc.response.status_code}]"
    except Exception as exc:
        logger.warning("_api.stream_full_report.error", insight_id=insight_id, error=str(exc))
        yield f"\n\n[Connection error: {exc}]"


def stream_followup(insight_id: str, message: str) -> Generator[str, None, None]:
    """Stream follow-up tokens and convert control frames to readable UI text."""
    payload = {"message": message}
    try:
        with (
            httpx.Client(timeout=None) as client,
            client.stream(
                "POST",
                f"{settings.FASTAPI_URL}/api/v1/insights/{insight_id}/followup",
                headers={
                    **_headers(),
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
            ) as response,
        ):
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    frame = json.loads(data_str)
                except json.JSONDecodeError:
                    if data_str.strip():
                        yield data_str
                    continue

                if frame.get("token"):
                    yield str(frame["token"])
                    continue
                if frame.get("type") == "context_source":
                    source = str(frame.get("source") or "")
                    if source == "cluster_docs_expanded":
                        yield "\n\n_Using related chunks from this insight's source documents._\n\n"
                    else:
                        yield "\n\n_Using direct cluster evidence._\n\n"
                    continue
                if frame.get("type") == "no_context":
                    yield "\n\n[No relevant context found inside this insight's sources.]"
                    continue
                if frame.get("error"):
                    yield f"\n\n[{frame['error']}]"
    except httpx.HTTPStatusError as exc:
        logger.warning("_api.stream_followup.http_error", insight_id=insight_id, error=str(exc))
        yield f"\n\n[API error {exc.response.status_code}]"
    except Exception as exc:
        logger.warning("_api.stream_followup.error", insight_id=insight_id, error=str(exc))
        yield f"\n\n[Connection error: {exc}]"


def get_followup_history(insight_id: str) -> list[dict]:
    """Return follow-up history rows for one insight."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_DISCOVER_TIMEOUT) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/insights/{insight_id}/followup/history",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("_api.get_followup_history.error", insight_id=insight_id, error=str(exc))
        return []


def get_ingest_status(project_id: str) -> dict | None:
    """Return current ingest lifecycle status for a project."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_DISCOVER_TIMEOUT) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/ingest-status",
                headers=_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(
            "_api.get_ingest_status.error",
            project_id=project_id,
            timeout_seconds=settings.FASTAPI_DISCOVER_TIMEOUT,
            error=str(exc),
        )
        return None


def bootstrap_project(project_id: str, upload_ids: list[str]) -> dict | None:
    """Trigger project bootstrap with optional uploaded document IDs."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_DISCOVER_TIMEOUT) as client:
            resp = client.post(
                f"{settings.FASTAPI_URL}/api/v1/projects/{project_id}/bootstrap",
                headers={**_headers(), "Content-Type": "application/json"},
                json={"upload_ids": upload_ids},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(
            "_api.bootstrap_project.error",
            project_id=project_id,
            upload_count=len(upload_ids),
            timeout_seconds=settings.FASTAPI_DISCOVER_TIMEOUT,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


def get_kb_status(project_id: str) -> dict | None:
    """Return KB status for a project, or None on error."""
    try:
        with httpx.Client(timeout=settings.FASTAPI_KB_STATUS_TIMEOUT) as client:
            resp = client.get(
                f"{settings.FASTAPI_URL}/api/v1/kb/{project_id}/status",
                headers=_headers(),
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(
            "_api.get_kb_status.error",
            project_id=project_id,
            timeout_seconds=settings.FASTAPI_KB_STATUS_TIMEOUT,
            error=str(exc),
        )
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
        with (
            httpx.Client(timeout=None) as client,
            client.stream(
                "POST",
                f"{settings.FASTAPI_URL}/api/v1/chat/",
                headers={
                    **_headers(),
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
            ) as response,
        ):
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
