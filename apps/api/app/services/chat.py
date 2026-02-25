import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncGenerator, Callable

import structlog

from app.agents.orchestrator import run_query
from app.core.cache_version import get_project_cache_version
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.core.metrics import (
    cache_hits_total,
    cache_misses_total,
    chat_messages_per_request,
    chat_requests_total,
    chat_response_duration_seconds,
    chat_stream_tokens_total,
)
from app.core.redis import get_redis
from app.repositories import chat_history as chat_history_repo
from app.repositories.project import fetch_project_papers_enabled, list_projects_summary

logger = structlog.get_logger(__name__)

_LATEST_RESEARCH_PATTERN = re.compile(
    r"\b(latest|recent|newest|current)\b.*\b(paper|papers|study|studies|research|journal|literature)\b"
)
_TRENDING_QUERY_PATTERN = re.compile(
    r"\b(latest|recent|newest|current|today|trending|trend|breaking|this week|this month)\b"
)

# In-flight streaming request registry for deduplication
# Maps query_key -> (chunks_list, subscribers_list, completion_event)
_streaming_requests: dict[str, tuple[list[str], list[asyncio.Queue], asyncio.Event]] = {}
_streaming_lock = asyncio.Lock()


async def _deduplicated_stream(
    query_key: str,
    generator_fn: Callable[[], AsyncGenerator[str, None]],
) -> AsyncGenerator[str, None]:
    """
    Deduplicate concurrent streaming requests.

    When 1000 users ask the same question simultaneously during the 8-second
    streaming window:
    - First request executes generator_fn() and broadcasts chunks to all subscribers
    - Subsequent 999 requests subscribe and receive chunks as they arrive
    - All 1000 requests receive identical SSE streams

    This prevents 1000 concurrent LLM calls, reducing cost from $500/burst to $0.50/burst.

    Args:
        query_key: Deduplication key (e.g., "chat_stream:project_id:query_hash")
        generator_fn: Async generator function to call for the first request

    Yields:
        SSE chunks from the first request
    """
    is_first_request = False
    subscriber_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async with _streaming_lock:
        if query_key in _streaming_requests:
            # Another request is already streaming - subscribe to it
            existing_chunks, existing_subscribers, completion = _streaming_requests[query_key]
            existing_subscribers.append(subscriber_queue)
            logger.info("chat.stream_dedup.subscriber", query_key=query_key[:60])
        else:
            # This is the first request - create registry entry
            chunks_list: list[str] = []
            subscribers_list: list[asyncio.Queue[str | None]] = [subscriber_queue]
            completion = asyncio.Event()
            _streaming_requests[query_key] = (chunks_list, subscribers_list, completion)
            is_first_request = True
            logger.info("chat.stream_dedup.first_request", query_key=query_key[:60])

    if is_first_request:
        # Execute the generator and broadcast to all subscribers
        try:
            async for chunk in generator_fn():
                # Broadcast to all subscribers (including ourselves)
                async with _streaming_lock:
                    if query_key in _streaming_requests:
                        current_chunks, current_subscribers, _ = _streaming_requests[query_key]
                        current_chunks.append(chunk)
                        for queue in current_subscribers:
                            await queue.put(chunk)
                yield chunk

            # Signal completion to all subscribers
            async with _streaming_lock:
                if query_key in _streaming_requests:
                    _, current_subscribers, current_completion = _streaming_requests[query_key]
                    for queue in current_subscribers:
                        await queue.put(None)  # Sentinel value
                    current_completion.set()
        except Exception:
            # Broadcast exception to all subscribers
            logger.exception("chat.stream_dedup.error", query_key=query_key[:60])
            async with _streaming_lock:
                if query_key in _streaming_requests:
                    _, current_subscribers, _ = _streaming_requests[query_key]
                    for queue in current_subscribers:
                        await queue.put(None)  # Signal termination
            raise
        finally:
            # Clean up registry entry
            async with _streaming_lock:
                _streaming_requests.pop(query_key, None)
            logger.info("chat.stream_dedup.cleanup", query_key=query_key[:60])
    else:
        # We're a subscriber - replay chunks from the queue
        while True:
            queued_chunk = await subscriber_queue.get()
            if queued_chunk is None:  # Sentinel value - stream complete
                break
            yield queued_chunk
        logger.info("chat.stream_dedup.subscriber_complete", query_key=query_key[:60])


def _extract_token_from_sse_chunk(chunk: str) -> str | None:
    """
    Extract token text from one SSE frame string.

    Returns None for non-token frames (including [DONE]). For non-JSON data
    frames, returns the raw payload text so persistence still captures what was
    streamed to the client.
    """
    if not chunk.startswith("data:"):
        return None

    raw = chunk[len("data:") :].strip()
    if not raw or raw == "[DONE]":
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    token = payload.get("token")
    if isinstance(token, str):
        return token

    # Compatibility: support OpenAI-like streaming payloads.
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first, dict) else {}
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str):
            return content
    return None


def _should_bypass_semantic_cache(query: str) -> bool:
    """
    Return True for freshness-sensitive research-paper queries.

    These queries should not reuse semantic cache because users explicitly ask
    for latest/recent material and retrieval/ranking logic may evolve rapidly.
    """
    q = (query or "").lower()
    return bool(_LATEST_RESEARCH_PATTERN.search(q))


def _is_trending_query(query: str) -> bool:
    """Return True when query likely targets fast-changing information."""
    q = (query or "").lower()
    return bool(_TRENDING_QUERY_PATTERN.search(q))


async def get_cached_response(
    query: str,
    project_id: str,
) -> str | None:
    """
    Check semantic cache for a cached response.

    Returns cached response if found, None otherwise.
    """
    try:
        redis = await get_redis()
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        version = await get_project_cache_version(redis, project_id)
        cache_key = RedisKeys.SEMANTIC_CACHE.format(
            project_id=project_id, hash=f"{version}:{query_hash}"
        )
        cached = await redis.get(cache_key)
        if cached:
            cache_hits_total.labels(cache_tier="L2", cache_type="semantic").inc()
            logger.info("chat.cache_hit", project_id=project_id, query_hash=query_hash)
            return cached
        cache_misses_total.labels(cache_tier="L2", cache_type="semantic").inc()
        logger.info("chat.cache_miss", project_id=project_id, query_hash=query_hash)
        return None
    except Exception:
        logger.warning("chat.cache_read_error", project_id=project_id)
        return None


async def cache_response(
    query: str,
    project_id: str,
    response: str,
    is_trending: bool,
) -> None:
    """
    Store response in semantic cache.

    Trending queries get shorter TTL (1hr), stable queries get 24hr.
    """
    try:
        redis = await get_redis()
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        version = await get_project_cache_version(redis, project_id)
        cache_key = RedisKeys.SEMANTIC_CACHE.format(
            project_id=project_id, hash=f"{version}:{query_hash}"
        )
        ttl = (
            RedisTTL.SEMANTIC_CACHE_TRENDING.value
            if is_trending
            else RedisTTL.SEMANTIC_CACHE_STABLE.value
        )
        await redis.setex(cache_key, ttl, response)
        logger.info("chat.cached", project_id=project_id, query_hash=query_hash, ttl=ttl)
    except Exception:
        logger.warning("chat.cache_write_error", project_id=project_id)


async def _list_projects_summary_cached(pool, user_id: str) -> list[dict]:
    """Read project summaries from Redis cache, falling back to DB."""
    cache_key = RedisKeys.PROJECTS_SUMMARY.format(user_id=user_id)
    try:
        redis = await get_redis()
        cached = await redis.get(cache_key)
        if cached:
            payload = json.loads(cached)
            if isinstance(payload, list):
                return payload
    except Exception:
        logger.warning("chat.projects_summary_cache_read_error", user_id=user_id)

    projects = await list_projects_summary(pool, user_id)
    try:
        redis = await get_redis()
        await redis.setex(cache_key, RedisTTL.PROJECTS_SUMMARY.value, json.dumps(projects))
    except Exception:
        logger.warning("chat.projects_summary_cache_write_error", user_id=user_id)
    return projects


async def persist_chat_turn(
    *,
    user_id: str,
    project_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Persist a single user/assistant turn durably.

    Source of truth is Postgres (no TTL). Redis is best-effort mirror cache for
    short-term reads and backward compatibility.
    """
    pool = await get_db_pool()
    try:
        await chat_history_repo.insert_chat_turn(
            pool=pool,
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
    except Exception:
        logger.warning(
            "chat.history_write_db_error",
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            user_length=len(user_message),
            assistant_length=len(assistant_message),
        )
        return

    try:
        history_key = RedisKeys.CHAT_HISTORY.format(
            user_id=user_id, project_id=project_id, session_id=session_id
        )
        user_payload = json.dumps({"role": "user", "content": user_message}, ensure_ascii=False)
        assistant_payload = json.dumps(
            {"role": "assistant", "content": assistant_message},
            ensure_ascii=False,
        )
        redis = await get_redis()
        pipe = redis.pipeline(transaction=True)
        pipe.rpush(history_key, user_payload)
        pipe.rpush(history_key, assistant_payload)
        pipe.expire(history_key, RedisTTL.CHAT_HISTORY.value)
        await pipe.execute()
    except Exception:
        logger.warning(
            "chat.history_write_redis_mirror_error",
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
        )


async def get_chat_history_messages(
    *,
    project_id: str,
    user_id: str,
    session_id: str,
) -> list[dict]:
    """
    Return ordered chat history for project+session.

    Reads durable history from Postgres first. Falls back to Redis only when
    Postgres returns no rows, so pre-migration sessions still render.
    """
    db_messages: list[dict] = []
    pool = await get_db_pool()
    try:
        db_messages = await chat_history_repo.fetch_chat_history(
            pool=pool,
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        logger.warning(
            "chat.history_read_db_error",
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
        )

    redis_messages: list[dict] = []
    try:
        history_key = RedisKeys.CHAT_HISTORY.format(
            user_id=user_id, project_id=project_id, session_id=session_id
        )
        redis = await get_redis()
        raw_messages: list[str] = await redis.lrange(history_key, 0, -1)
        out: list[dict] = []
        for msg in raw_messages:
            try:
                parsed = json.loads(msg)
            except Exception:
                continue
            role = parsed.get("role")
            content = parsed.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                out.append({"role": role, "content": content})
        redis_messages = out
    except Exception:
        logger.warning(
            "chat.history_read_redis_fallback_error",
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
        )
        redis_messages = []

    # One-time backfill path: if Redis has more messages than DB (legacy sessions),
    # replace DB history with Redis copy so future reads remain durable.
    if redis_messages and len(redis_messages) > len(db_messages):
        try:
            await chat_history_repo.replace_chat_history(
                pool=pool,
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
                messages=redis_messages,
            )
            logger.info(
                "chat.history_backfilled_from_redis",
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
                message_count=len(redis_messages),
            )
            return redis_messages
        except Exception:
            logger.warning(
                "chat.history_backfill_failed",
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
            )

    if db_messages:
        return db_messages
    return redis_messages


async def stream_response(
    query: str,
    project_id: str,
    user_id: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream chat response via SSE.

    Checks cache first, falls back to agent orchestration.
    Yields SSE-formatted tokens.

    Tracks Prometheus metrics:
    - Request success/failure count
    - Response generation duration
    - Context message count
    - Token count
    """
    start_time = time.perf_counter()
    token_count = 0
    success = False

    try:
        # Get chat history for context and track message count
        history = await get_chat_history_messages(
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
        )
        chat_messages_per_request.observe(len(history))

        bypass_cache = _should_bypass_semantic_cache(query)
        if bypass_cache:
            logger.info(
                "chat.cache_bypassed", project_id=project_id, reason="latest_research_query"
            )
        else:
            cached = await get_cached_response(query, project_id)
            if cached:
                await persist_chat_turn(
                    user_id=user_id,
                    project_id=project_id,
                    session_id=session_id,
                    user_message=query,
                    assistant_message=cached,
                )
                # Track cached response tokens
                token_count = len(cached.split())
                chat_stream_tokens_total.inc(token_count)

                # Yield the full cached response as a single token — splitting on whitespace
                # destroys newlines, bullet points, and code blocks in markdown responses.
                yield f"data: {json.dumps({'token': cached})}\n\n"
                yield "data: [DONE]\n\n"
                success = True
                return

        pool = await get_db_pool()
        papers_enabled = (await fetch_project_papers_enabled(pool, project_id)) or False
        all_projects = await _list_projects_summary_cached(pool, user_id)
        if not all_projects:
            all_projects = [{"id": project_id, "title": "", "description": ""}]

        # Generate deduplication key for concurrent identical queries
        # Format: chat_stream:project_id:query_hash
        # TTL is implicit (only during active streaming, ~8-15 seconds)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        dedup_key = f"chat_stream:{project_id}:{query_hash}"

        tokens: list[str] = []
        streamed_frame_count = 0

        # Wrap run_query in deduplication layer to prevent thundering herd
        # When 1000 users ask identical question during streaming window:
        # - First request executes run_query and broadcasts chunks
        # - Remaining 999 requests receive chunks from first request
        # - Saves 999 LLM calls ($0.50-$5.00 per call = $495-$4995 per burst)
        async def _run_query_generator():
            async for chunk in run_query(
                query=query,
                primary_project_id=project_id,
                all_projects=all_projects,
                user_id=user_id,
                session_id=session_id,
                openalex_enabled=papers_enabled,
            ):
                yield chunk

        async for chunk in _deduplicated_stream(dedup_key, _run_query_generator):
            if chunk == "data: [DONE]\n\n":
                # Persist and cache BEFORE yielding the [DONE] frame.
                #
                # The Streamlit SSE client breaks its read loop the instant it
                # receives [DONE] and closes the HTTP connection.  Starlette
                # then cancels this async generator at the next await point,
                # making any code placed after "yield chunk" for [DONE]
                # unreachable in practice (confirmed: zero DB rows saved for
                # any query when persist lived after the loop).
                #
                # By doing the DB write here — while the connection is still
                # alive — we guarantee chat history is always persisted.
                full_text = "".join(tokens)
                token_count = len(tokens)
                chat_stream_tokens_total.inc(token_count)

                if not full_text and streamed_frame_count > 0:
                    logger.warning(
                        "chat.stream_completed_without_text",
                        project_id=project_id,
                        session_id=session_id,
                        frame_count=streamed_frame_count,
                        query_preview=query[:80],
                    )

                if full_text and not bypass_cache:
                    await cache_response(
                        query,
                        project_id,
                        full_text,
                        is_trending=_is_trending_query(query),
                    )

                if full_text:
                    await persist_chat_turn(
                        user_id=user_id,
                        project_id=project_id,
                        session_id=session_id,
                        user_message=query,
                        assistant_message=full_text,
                    )

                yield chunk  # Send [DONE] only after history is saved
                break
            else:
                streamed_frame_count += 1
                token = _extract_token_from_sse_chunk(chunk)
                if token:
                    tokens.append(token)
                yield chunk

        success = True
    except Exception:
        logger.exception(
            "chat.stream_error",
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
        )
        chat_requests_total.labels(status="error").inc()
        raise
    finally:
        duration = time.perf_counter() - start_time
        chat_response_duration_seconds.observe(duration)
        if success:
            chat_requests_total.labels(status="success").inc()
