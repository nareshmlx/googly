"""ARQ task: enrich a YouTube video via Gemini after KB ingest."""

from __future__ import annotations

import structlog
from google.genai.errors import ClientError

from app.core.constants import RedisKeys, RedisTTL
from app.kb.ingester import ingest_documents
from app.repositories import knowledge as knowledge_repo
from app.tools.video_enrichment import VideoEnrichmentTool

logger = structlog.get_logger(__name__)

# Sentinel substrings present in RuntimeErrors raised by _acquire_gemini_slot.
# Used to distinguish quota/rate-limit errors from transient failures without
# importing a custom exception class.
_DAILY_QUOTA_SENTINEL = "daily quota exhausted"
_RPM_EXHAUSTED_SENTINEL = "rate limit"

# Sentinel substring present in Gemini ClientError 400 INVALID_ARGUMENT when the
# video's token count exceeds Gemini's 1M context window. This is a permanent,
# unrecoverable failure for this video — no truncation fallback exists for FileData
# URI inputs. The video stays in the KB (title/description searchable) but will
# never be enriched. The enqueued key is kept to prevent infinite re-queuing.
_TOO_LONG_SENTINEL = "maximum number of tokens allowed 1048576"


async def _handle_transient_failure(redis, source_id: str, project_id: str) -> None:
    """Set enrichment status to failed and clear the enqueued marker.

    Called from both the non-quota/non-RPM RuntimeError branch and the catch-all
    Exception branch — they share identical cleanup logic. Each operation is wrapped
    in its own try/except so a Redis or DB outage during cleanup does not propagate.
    """
    try:
        await _set_enrichment_status(
            source_id=source_id,
            project_id=project_id,
            status="failed",
        )
    except Exception:
        logger.warning(
            "enrich_video.failed_to_set_failed_status",
            source_id=source_id,
        )
    enqueued_key = RedisKeys.VIDEO_ENRICH_ENQUEUED.format(source_id=source_id)
    try:
        await redis.delete(enqueued_key)
    except Exception:
        logger.warning(
            "enrich_video.failed_to_clear_enqueued_key",
            source_id=source_id,
        )


async def _set_enrichment_status(
    *,
    source_id: str,
    project_id: str,
    status: str,
) -> None:
    """Delegate enrichment status update to the knowledge repository.

    Calls the pool-acquiring wrapper which keeps get_db_pool() inside the
    repositories layer where it belongs.
    """
    await knowledge_repo.update_enrichment_status(
        source_id=source_id,
        project_id=project_id,
        status=status,
    )


async def enrich_video(
    ctx: dict,
    *,
    source_id: str,
    platform: str,
    video_id: str | None,
    project_id: str,
    original_description: str,
) -> None:
    """Enrich a video document with Gemini-extracted transcript and signals.

    Guard flow:
    1. Acquire Redis in-progress lock via SET NX — return early if already running
       (prevents duplicate concurrent runs from ARQ retries or race conditions).
    2. Set enrichment_status=pending in DB.
    3. Call VideoEnrichmentTool.enrich().
    4. Delegate RawDocument assembly to result.to_raw_document().
    5. Re-ingest via ingest_documents() to upsert by source_id and re-embed.
    6. Set enrichment_status=done in DB after successful ingest.
    7. On daily-quota RuntimeError: set enrichment_status=quota_exceeded, do NOT
       clear VIDEO_ENRICH_ENQUEUED — the video must not be retried until tomorrow.
    8. On any other exception: set enrichment_status=failed, clear
       VIDEO_ENRICH_ENQUEUED so the next ingest_project run can retry.
       The delete is wrapped in its own try/except so a Redis outage during cleanup
       does not propagate.
    9. Always release the in-progress lock (delete VIDEO_ENRICH_LOCK) in finally.

    Note: ingest_project sets a separate VIDEO_ENRICH_ENQUEUED key (7-day TTL) to
    prevent re-queuing already-processed videos. This task owns VIDEO_ENRICH_LOCK
    which is a short-lived (1-hour) in-progress mutex, distinct from the enqueued
    marker.

    Note: user_id is set to "" because this task runs in the ARQ worker context
    where the worker connects with the superuser (googly) role, bypassing RLS for
    writes. The project_id in metadata ensures correct project scoping on reads.
    """
    redis = ctx["redis"]
    lock_key = RedisKeys.VIDEO_ENRICH_LOCK.format(source_id=source_id)

    # SET NX — only acquire if not already held; returns True/1 if we got the lock,
    # None/falsy if another worker already holds it.
    acquired = await redis.set(lock_key, "1", ex=RedisTTL.VIDEO_ENRICH_LOCK, nx=True)
    if not acquired:
        logger.warning(
            "enrich_video.skipped.lock_held",
            source_id=source_id,
            platform=platform,
        )
        return

    logger.info("enrich_video.start", source_id=source_id, platform=platform)

    try:
        await _set_enrichment_status(
            source_id=source_id,
            project_id=project_id,
            status="pending",
        )

        result = await VideoEnrichmentTool(redis=redis).enrich(
            platform=platform,
            video_id=video_id or source_id,
        )

        raw_doc = result.to_raw_document(
            source_id=source_id,
            project_id=project_id,
            original_description=original_description,
        )

        await ingest_documents([raw_doc], overwrite=True)

        await _set_enrichment_status(
            source_id=source_id,
            project_id=project_id,
            status="done",
        )

        logger.info(
            "enrich_video.success",
            source_id=source_id,
            products_count=len(result.products_mentioned),
            claims_count=len(result.key_claims),
        )

    except RuntimeError as exc:
        if _DAILY_QUOTA_SENTINEL in str(exc):
            # Daily quota exhausted — do NOT clear the enqueued key. Keeping it
            # prevents ingest_project from re-queuing this video today. The key
            # has a 25-hour TTL and will naturally expire; the next ingest_project
            # run after midnight UTC will create a fresh key and re-enqueue.
            logger.warning(
                "enrich_video.daily_quota_exhausted",
                source_id=source_id,
                platform=platform,
                project_id=project_id,
            )
            try:
                await _set_enrichment_status(
                    source_id=source_id,
                    project_id=project_id,
                    status="quota_exceeded",
                )
            except Exception:
                logger.warning(
                    "enrich_video.failed_to_set_quota_exceeded_status",
                    source_id=source_id,
                )
        elif _RPM_EXHAUSTED_SENTINEL in str(exc):
            # RPM (requests-per-minute) limit exhausted — do NOT clear the enqueued
            # key. The rate window resets within a minute; ingest_project must not
            # re-queue this video during that window. Same retention policy as
            # daily quota: the enqueued key expires naturally.
            logger.warning(
                "enrich_video.rpm_exhausted",
                source_id=source_id,
                platform=platform,
                project_id=project_id,
            )
            try:
                await _set_enrichment_status(
                    source_id=source_id,
                    project_id=project_id,
                    status="rpm_exhausted",
                )
            except Exception:
                logger.warning(
                    "enrich_video.failed_to_set_rpm_exhausted_status",
                    source_id=source_id,
                )
        else:
            # Transient failure — clear the enqueued marker so the next
            # ingest_project run can retry this video.
            logger.exception(
                "enrich_video.failed",
                source_id=source_id,
                platform=platform,
                project_id=project_id,
            )
            await _handle_transient_failure(redis, source_id, project_id)

    except ClientError as exc:
        if exc.code == 429:
            # Gemini rejected the call directly with 429 RESOURCE_EXHAUSTED.
            # This happens when Google's server-side quota is hit before our
            # internal counter reaches the limit (e.g. after a pod restart that
            # reset the in-memory state, or concurrent workers from multiple pods).
            # Treat identically to the internal daily-quota path: do NOT clear
            # the enqueued key — the video must not be re-queued until tomorrow.
            logger.warning(
                "enrich_video.daily_quota_exhausted",
                source_id=source_id,
                platform=platform,
                project_id=project_id,
                gemini_status=exc.status,
            )
            try:
                await _set_enrichment_status(
                    source_id=source_id,
                    project_id=project_id,
                    status="quota_exceeded",
                )
            except Exception:
                logger.warning(
                    "enrich_video.failed_to_set_quota_exceeded_status",
                    source_id=source_id,
                )
        elif exc.code == 400 and _TOO_LONG_SENTINEL in str(exc):
            # Video token count exceeds Gemini's 1M context window. This is a
            # permanent failure — Gemini hard-rejects FileData URI inputs that
            # produce too many tokens (typically long-form videos > ~1 hour).
            # No truncation is possible at this layer. Keep the enqueued key so
            # the video is never re-queued and never burns another rate-limit slot.
            logger.warning(
                "enrich_video.skipped_too_long",
                source_id=source_id,
                platform=platform,
                project_id=project_id,
            )
            try:
                await _set_enrichment_status(
                    source_id=source_id,
                    project_id=project_id,
                    status="skipped_too_long",
                )
            except Exception:
                logger.warning(
                    "enrich_video.failed_to_set_skipped_too_long_status",
                    source_id=source_id,
                )
        else:
            # Non-quota, non-too-long ClientError (e.g. 400 bad URI, 403 Forbidden) —
            # treat as transient and allow retry on next ingest run.
            logger.exception(
                "enrich_video.failed",
                source_id=source_id,
                platform=platform,
                project_id=project_id,
                gemini_code=exc.code,
                gemini_status=exc.status,
            )
            await _handle_transient_failure(redis, source_id, project_id)

    except Exception:
        # Catch-all for non-RuntimeError, non-ClientError exceptions (e.g.
        # NotImplementedError for unsupported platforms, network errors, etc.)
        logger.exception(
            "enrich_video.failed",
            source_id=source_id,
            platform=platform,
            project_id=project_id,
        )
        await _handle_transient_failure(redis, source_id, project_id)

    finally:
        await redis.delete(lock_key)
