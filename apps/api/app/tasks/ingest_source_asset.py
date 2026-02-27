"""ARQ task that enriches paper/patent metadata rows with fulltext content."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import httpx
import structlog

from app.core.cache_version import bump_project_cache_version
from app.core.config import settings
from app.core.constants import RedisKeys, RedisTTL
from app.core.db import get_db_pool
from app.core.metrics import (
    fulltext_embed_total,
    fulltext_extract_total,
    fulltext_fetch_duration_seconds,
    fulltext_fetch_total,
    fulltext_upsert_total,
)
from app.core.url_safety import is_safe_public_url
from app.kb.ingester import ingest_documents
from app.repositories import project as project_repo
from app.repositories import source_asset as source_asset_repo
from app.services.fulltext_extractor import extract_text_from_asset
from app.services.fulltext_mapper import map_fulltext_raw_document

logger = structlog.get_logger(__name__)


def _allowed_domains() -> set[str]:
    """Parse optional domain allowlist for direct fetch safety checks."""
    raw = str(settings.FULLTEXT_ALLOWED_DOMAINS or "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


async def _download_asset(url: str) -> tuple[bytes, str]:
    """Download source bytes with bounded redirects/timeouts/size limits."""
    timeout = httpx.Timeout(settings.FULLTEXT_FETCH_TIMEOUT_SECONDS)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    allowed_domains = _allowed_domains()
    current_url = str(url)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False) as client:
        for _ in range(max(0, int(settings.FULLTEXT_MAX_REDIRECTS)) + 1):
            safe, reason = is_safe_public_url(current_url, allowed_domains=allowed_domains or None)
            if not safe:
                raise ValueError(f"unsafe_url:{reason}")

            async with client.stream(
                "GET",
                current_url,
                headers={"User-Agent": "googly-fulltext/1.0"},
            ) as resp:
                if resp.status_code in {301, 302, 303, 307, 308}:
                    location = str(resp.headers.get("location") or "").strip()
                    if not location:
                        raise ValueError("redirect_missing_location")
                    current_url = str(httpx.URL(current_url).join(location))
                    continue

                resp.raise_for_status()
                mime_type = str(resp.headers.get("content-type") or "application/octet-stream").split(";")[0]
                chunks: list[bytes] = []
                seen = 0
                async for part in resp.aiter_bytes():
                    seen += len(part)
                    if seen > settings.FULLTEXT_MAX_SOURCE_BYTES:
                        raise ValueError("source_too_large")
                    chunks.append(part)
                return b"".join(chunks), mime_type

    raise ValueError("too_many_redirects")


async def ingest_source_asset(ctx: dict, asset_id: str) -> None:
    """Fetch, extract, embed, and upsert one fulltext source asset."""
    if not settings.ENABLE_FULLTEXT_ENRICHMENT:
        return

    redis = ctx.get("redis")
    if redis is None:
        logger.warning("fulltext_asset.missing_redis", asset_id=asset_id)
        return

    lock_key = RedisKeys.FULLTEXT_ASSET_LOCK.format(asset_id=asset_id)
    acquired = await redis.set(lock_key, "1", nx=True, ex=RedisTTL.FULLTEXT_ASSET_LOCK.value)
    if not acquired:
        return

    pool = await get_db_pool()
    started = datetime.now(UTC)

    try:
        asset = await source_asset_repo.fetch_source_asset(pool, asset_id)
        if not asset:
            logger.warning("fulltext_asset.not_found", asset_id=asset_id)
            return

        attempts = await source_asset_repo.increment_attempt(pool, asset_id=asset_id)
        if attempts > settings.FULLTEXT_MAX_ENRICHMENT_ATTEMPTS:
            await source_asset_repo.mark_fetch_result(
                pool,
                asset_id=asset_id,
                fetch_status="failed",
                mime_type=str(asset.get("mime_type") or ""),
                byte_size=int(asset.get("byte_size") or 0),
                checksum_sha256=str(asset.get("checksum_sha256") or ""),
                blob_path=str(asset.get("blob_path") or ""),
                error_code="attempts_exhausted",
                error_message="max enrichment attempts exceeded",
            )
            return

        source = str(asset.get("source") or "paper")
        source_url = str(asset.get("resolved_url") or asset.get("source_url") or "")
        if not source_url:
            await source_asset_repo.mark_fetch_result(
                pool,
                asset_id=asset_id,
                fetch_status="failed",
                mime_type="",
                byte_size=0,
                checksum_sha256="",
                blob_path="",
                error_code="missing_url",
                error_message="resolved URL missing",
            )
            fulltext_fetch_total.labels(source=source, status="failed").inc()
            return

        try:
            body, mime_type = await _download_asset(source_url)
            sha = hashlib.sha256(body).hexdigest()
            await source_asset_repo.mark_fetch_result(
                pool,
                asset_id=asset_id,
                fetch_status="fetched",
                mime_type=mime_type,
                byte_size=len(body),
                checksum_sha256=sha,
                blob_path=f"inline://sha256/{sha}",
            )
            fulltext_fetch_total.labels(source=source, status="success").inc()
            fulltext_fetch_duration_seconds.labels(source=source).observe(
                max((datetime.now(UTC) - started).total_seconds(), 0.0)
            )
        except httpx.TimeoutException:
            should_retry = attempts < settings.FULLTEXT_MAX_ENRICHMENT_ATTEMPTS
            await source_asset_repo.mark_fetch_result(
                pool,
                asset_id=asset_id,
                fetch_status="retry" if should_retry else "failed",
                mime_type="",
                byte_size=0,
                checksum_sha256="",
                blob_path="",
                error_code="timeout",
                error_message="fulltext fetch timeout",
            )
            fulltext_fetch_total.labels(source=source, status="timeout").inc()
            if should_retry:
                retry_delay = min(300, 30 * attempts)
                await redis.enqueue_job(
                    "ingest_source_asset",
                    asset_id,
                    _job_id=f"fulltext:{asset_id}:retry:{attempts}",
                    _defer_by=timedelta(seconds=retry_delay),
                )
            return
        except Exception as exc:
            await source_asset_repo.mark_fetch_result(
                pool,
                asset_id=asset_id,
                fetch_status="failed",
                mime_type="",
                byte_size=0,
                checksum_sha256="",
                blob_path="",
                error_code="fetch_failed",
                error_message=str(exc),
            )
            fulltext_fetch_total.labels(source=source, status="failed").inc()
            return

        extract = extract_text_from_asset(body, mime_type)
        fulltext_extract_total.labels(source=source, status=extract.status).inc()
        if extract.status != "success":
            await source_asset_repo.mark_extract_result(
                pool,
                asset_id=asset_id,
                extract_status=extract.status,
                extracted_chars=extract.extracted_chars,
                extracted_pages=extract.page_count,
                error_code=extract.error_code,
                error_message=extract.error_message,
            )
            return

        metadata = dict(asset.get("metadata") or {})
        raw_doc = map_fulltext_raw_document(
            project_id=str(asset["project_id"]),
            user_id=str(asset["user_id"]),
            source=source,
            source_id=str(asset["source_id"]),
            title=str(asset.get("title") or ""),
            text=extract.text,
            base_metadata=metadata,
            asset_id=asset_id,
            source_fetcher=str(metadata.get("source_fetcher") or "resolver"),
            page_count=extract.page_count,
        )

        try:
            inserted = await ingest_documents([raw_doc])
            fulltext_embed_total.labels(source=source, status="success").inc()
            fulltext_upsert_total.labels(source=source, status="success").inc()
        except Exception:
            fulltext_embed_total.labels(source=source, status="failed").inc()
            fulltext_upsert_total.labels(source=source, status="failed").inc()
            raise

        await source_asset_repo.mark_extract_result(
            pool,
            asset_id=asset_id,
            extract_status="success",
            extracted_chars=extract.extracted_chars,
            extracted_pages=extract.page_count,
        )

        await project_repo.increment_project_kb_stats(
            pool,
            str(asset["project_id"]),
            int(inserted),
            datetime.now(UTC),
        )
        try:
            await bump_project_cache_version(redis, str(asset["project_id"]))
        except Exception:
            logger.warning(
                "fulltext_asset.cache_version_bump_failed",
                asset_id=asset_id,
                project_id=str(asset["project_id"]),
            )
        logger.info(
            "fulltext_asset.done",
            asset_id=asset_id,
            project_id=str(asset["project_id"]),
            source=source,
            chunks_inserted=inserted,
        )
    finally:
        await redis.delete(lock_key)
