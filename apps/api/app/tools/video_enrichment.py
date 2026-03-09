"""Video enrichment tool — calls Gemini 3.1 Flash Lite to extract transcript and signals from YouTube videos."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from google import genai
from google.genai.errors import ClientError
from pydantic import BaseModel
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.constants import RedisKeys, SourceType

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.kb.ingester import RawDocument

logger = structlog.get_logger(__name__)

_LUA_INCR_EXPIRE = """
local c = redis.call('INCR', KEYS[1])
if c == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return c
"""

_EXTRACTION_PROMPT = """
You are analysing a YouTube video.

Extract the following and return ONLY valid JSON:
- transcript: a clean, concise summary of the spoken content (max 800 words)
- products_mentioned: a list of product names or brands explicitly mentioned or shown
- key_claims: a list of factual or promotional claims made about products/ingredients/techniques
- enrichment_source: always the string "gemini"

Be precise. Do not invent claims not present in the video.
""".strip()


class VideoEnrichmentResult(BaseModel):
    """Structured output from Gemini video enrichment."""

    transcript: str
    products_mentioned: list[str]
    key_claims: list[str]
    enrichment_source: str

    def to_raw_document(
        self,
        *,
        source_id: str,
        project_id: str,
        original_description: str,
    ) -> RawDocument:
        """Build the enriched RawDocument for re-ingest into the knowledge base.

        Combines the Gemini transcript with the original description so existing
        keyword matches are preserved alongside the enriched content.
        """
        from app.kb.ingester import RawDocument

        enriched_content = f"{self.transcript}\n\n---\n{original_description}"
        return RawDocument(
            source=SourceType.SOCIAL_YOUTUBE,
            source_id=source_id,
            title="",  # preserved from existing DB row via upsert ON CONFLICT
            content=enriched_content,
            user_id="",  # worker context: superuser role bypasses RLS; project_id scopes the record
            metadata={
                "enrichment_status": "done",
                "products_mentioned": self.products_mentioned,
                "key_claims": self.key_claims,
                "enrichment_source": self.enrichment_source,
                "project_id": project_id,
            },
            project_id=project_id,
        )


async def _acquire_gemini_slot(redis: Redis) -> None:
    """Block until a Gemini API rate-limit slot is available (RPM + RPD), then claim it.

    Two counters are checked atomically via a Lua INCR script:
    - Per-minute window: keyed on epoch // window, expires after 2x window.
    - Per-day counter: keyed on UTC date (YYYY-MM-DD), expires after 25 hours.

    Both counters are shared across all worker pods via Redis, so limits are
    global. If either limit is exceeded:
    - RPM: sleep 1s and retry in the next minute window.
    - RPD: raise DailyQuotaExceeded immediately — no point retrying today.
    """
    window = settings.GEMINI_RATE_LIMIT_WINDOW
    rpm_limit = settings.GEMINI_MAX_CALLS_PER_MINUTE
    rpd_limit = settings.GEMINI_MAX_CALLS_PER_DAY

    # Check and claim daily quota first (cheapest check — no looping)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    day_key = RedisKeys.GEMINI_RATE_LIMIT_DAY.format(date=today)
    day_count = int(
        await redis.eval(_LUA_INCR_EXPIRE, 1, day_key, "90000")  # type: ignore[misc]
    )  # 25h TTL
    if day_count > rpd_limit:
        await redis.decr(day_key)
        raise RuntimeError(
            f"Gemini daily quota exhausted ({rpd_limit} RPD). "
            "Enrichment will resume tomorrow (UTC)."
        )

    # Check and claim per-minute slot — attempt in current window, then once after sleeping
    # to the next window boundary. If still over after 2 attempts, give up.
    for attempt in range(2):
        current_window = int(time.time()) // window
        key = RedisKeys.GEMINI_RATE_LIMIT.format(window=current_window)

        count = int(await redis.eval(_LUA_INCR_EXPIRE, 1, key, str(window * 2)))  # type: ignore[misc]

        if count <= rpm_limit:
            logger.debug(
                "gemini_rate_limit.slot_acquired",
                window=current_window,
                count=count,
                rpm_limit=rpm_limit,
                day_count=day_count,
                rpd_limit=rpd_limit,
            )
            return

        # Over RPM limit — release the per-minute slot and sleep until the next window boundary
        await redis.decr(key)
        seconds_until_next_window = window - (int(time.time()) % window)
        logger.info(
            "gemini_rate_limit.waiting",
            window=current_window,
            count=count - 1,
            rpm_limit=rpm_limit,
            attempt=attempt + 1,
            sleep_seconds=seconds_until_next_window,
        )
        await asyncio.sleep(seconds_until_next_window)

    # Also release the daily slot we claimed since we couldn't get an RPM slot
    await redis.decr(day_key)
    raise RuntimeError(f"Gemini rate limit ({rpm_limit} RPM) exceeded after 2 retries")


class VideoEnrichmentTool:
    """Enriches video content using Gemini 3.1 Flash Lite.

    Only YouTube is supported. TikTok and Instagram are explicitly out of scope
    due to CDN auth and expiring URL issues — raises NotImplementedError for those
    platforms so callers can skip gracefully.

    Pass a Redis client at construction time to enable global rate limiting across
    all worker pods. Without Redis, rate limiting is skipped (e.g. in unit tests).
    """

    def __init__(self, redis: Redis | None = None) -> None:
        self._redis = redis
        self._client = (
            genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None
        )

    async def enrich(self, *, platform: str, video_id: str) -> VideoEnrichmentResult:
        """Dispatch to the correct platform enrichment method.

        Raises NotImplementedError for any platform other than 'youtube'.
        """
        if platform == "youtube":
            return await self._enrich_youtube(video_id)
        raise NotImplementedError(f"Video enrichment not supported for platform: {platform}")

    async def _enrich_youtube(self, video_id: str) -> VideoEnrichmentResult:
        """Acquire a rate-limit slot once, then delegate to the retried inner method.

        The slot is claimed here — outside the @retry loop — so that transient Gemini
        failures do not cause multiple RPM + RPD counter increments per enrich() call.
        """
        if self._redis is not None:
            await _acquire_gemini_slot(self._redis)
        return await self._enrich_youtube_inner(video_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(ClientError),
        reraise=True,
    )
    async def _enrich_youtube_inner(self, video_id: str) -> VideoEnrichmentResult:
        """Call Gemini 3.1 Flash Lite to extract transcript and signals from a YouTube video.

        Only the Gemini API call is retried here. Rate-limit acquisition happens once
        in _enrich_youtube before this method is invoked, so retries do not burn extra
        RPM or RPD quota slots.
        """
        from google.genai import types

        if self._client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured — cannot enrich video")

        logger.debug(
            "video_enrichment.duration_guard_not_enforced",
            video_id=video_id,
            configured_max_seconds=settings.VIDEO_ENRICH_MAX_DURATION_SECONDS,
            note="VIDEO_ENRICH_MAX_DURATION_SECONDS is set but video duration is not checked; long videos may exhaust Gemini quota",
        )
        logger.info(
            "video_enrichment.youtube.start",
            video_id=video_id,
            model=settings.GEMINI_VIDEO_MODEL,
        )
        client = self._client
        response = await client.aio.models.generate_content(
            model=settings.GEMINI_VIDEO_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=f"https://www.youtube.com/watch?v={video_id}",
                                mime_type="video/mp4",
                            )
                        ),
                        types.Part(text=_EXTRACTION_PROMPT),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoEnrichmentResult,
                # LOW resolution samples fewer frames → cuts token usage significantly.
                # Sufficient for transcript/claim extraction which is audio-dominant.
                # Reduces risk of hitting the 1M token ceiling on long-form videos.
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            ),
        )
        result: VideoEnrichmentResult = response.parsed  # type: ignore[assignment]
        logger.info(
            "video_enrichment.youtube.success",
            video_id=video_id,
            products_count=len(result.products_mentioned),
            claims_count=len(result.key_claims),
        )
        return result
