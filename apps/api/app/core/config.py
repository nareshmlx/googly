from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=["../../.env", ".env"], extra="ignore")

    ENVIRONMENT: str = "local"
    LOG_LEVEL: str = "INFO"

    OPENAI_API_KEY: str | None = None

    # === Gemini video enrichment ===
    GEMINI_API_KEY: str | None = None
    GEMINI_VIDEO_MODEL: str = Field(
        default="gemini-3.1-flash-lite-preview",
        description="Gemini model used for YouTube video transcript and signal extraction.",
    )
    VIDEO_ENRICH_MAX_DURATION_SECONDS: int = Field(
        default=600,
        ge=10,
        le=3600,
        description="Maximum video duration (seconds) sent to Gemini for enrichment. Videos exceeding this limit are skipped.",
    )

    # === Embedding Model Configuration ===
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_MODEL_VERSION: str = "v3-small-2024"  # Update when OpenAI changes model

    # === LLM Model Configuration ===
    INTENT_MODEL: str = "gpt-4o-mini"
    ANALYZER_MODEL: str = "gpt-4o-mini"
    SUMMARY_MODEL: str = "gpt-4o-mini"

    DATABASE_URL: str = "postgresql+asyncpg://googly:googly@localhost:5432/googly"
    REDIS_URL: str = "redis://localhost:6379/0"

    CLERK_SECRET_KEY: str | None = None
    CLERK_PUBLISHABLE_KEY: str | None = None
    CLERK_JWKS_URL: str | None = None
    CLERK_EXPECTED_ISSUER: str | None = None
    CLERK_EXPECTED_AUDIENCE: str | None = None

    APIM_INTERNAL_TOKEN: str | None = None

    EXA_API_KEY: str | None = None
    NEWSAPI_KEY: str | None = None
    GNEWS_API_KEY: str | None = None

    SEMANTIC_SCHOLAR_API_KEY: str | None = None

    # === New Search Engine API Keys (Phase 1) ===
    PERIGON_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    # EXA_API_KEY already exists above
    APIFY_API_KEY: str | None = None  # For X/Twitter scraping
    PUBMED_API_KEY: str | None = None  # Optional - increases rate limit from 3 to 10 req/sec
    LENS_API_KEY: str | None = None  # Lens.org patent API - free tier 10k requests/month
    PATENTSVIEW_API_KEY: str | None = (
        None  # PatentsView Search API - register at patentsview.org/apis
    )

    INSTAGRAM_ACCESS_TOKEN: str | None = None
    TIKTOK_CLIENT_KEY: str | None = None
    X_BEARER_TOKEN: str | None = None

    AZURE_KEY_VAULT_URL: str | None = None
    AZURE_STORAGE_ACCOUNT_URL: str | None = None

    EMBEDDING_BATCH_SIZE: int = 100
    CHAT_RATE_LIMIT: int = 100
    CHAT_RATE_WINDOW_SECONDS: int = 60
    KB_SCORE_THRESHOLD: float = (
        0.40  # Lowered from 0.70 - use KB more aggressively before web fallback
    )
    KB_RETRIEVAL_TOP_K: int = 15  # Default top-k for internal KB retrieval (legacy, used when per-source quotas are disabled)

    # === Per-Source Retrieval Quotas ===
    # When KB_RETRIEVAL_PER_SOURCE_QUOTAS is True, retrieval runs one KNN scan per
    # source bucket and caps each bucket at its quota. This prevents high-volume sources
    # (social) from dominating the context and guarantees a floor for papers/patents/uploads.
    # Total effective top-k = sum of all quotas = 20.
    # Slack (underfilled buckets) is NOT redistributed — simpler and avoids over-representing
    # any single source when others have limited relevant content.
    KB_RETRIEVAL_PER_SOURCE_QUOTAS: bool = Field(
        default=True,
        description="Enable per-source bucket retrieval to prevent social dominance in KB context.",
    )
    KB_RETRIEVAL_TOTAL_K: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Total chunk cap when per-source quotas are enabled (sum of all bucket quotas).",
    )
    KB_RETRIEVAL_QUOTA_SOCIAL: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Max chunks from social sources (tiktok, instagram, x, reddit, youtube) per retrieval.",
    )
    KB_RETRIEVAL_QUOTA_PAPERS: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Max chunks from paper sources per retrieval.",
    )
    KB_RETRIEVAL_QUOTA_WEB_NEWS: int = Field(
        default=8,
        ge=0,
        le=50,
        description="Max chunks from web/news/search sources (web, news, search) per retrieval. 'search' = Tavily/Exa results. Raised to 8 to accommodate three merged source types.",
    )
    KB_RETRIEVAL_QUOTA_PATENTS: int = Field(
        default=3,
        ge=0,
        le=50,
        description="Max chunks from patent sources per retrieval.",
    )
    KB_RETRIEVAL_QUOTA_UPLOADS: int = Field(
        default=2,
        ge=0,
        le=50,
        description="Max chunks from user upload sources per retrieval.",
    )
    INGEST_MAX_ACCOUNTS: int = 10
    INGEST_INSTAGRAM_ACCOUNT_CANDIDATES: int = 15
    INGEST_INSTAGRAM_REELS_PER_ACCOUNT: int = 6
    INGEST_INSTAGRAM_DETAIL_ENRICH_PER_ACCOUNT: int = 3
    INGEST_INSTAGRAM_ACCOUNTS_TO_FETCH: int = 10
    INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT: int = 60  # Increased from 24 to get more volume
    INGEST_INSTAGRAM_MAX_REELS_PER_ACCOUNT: int = 2
    INGEST_INSTAGRAM_MIN_REEL_RELEVANCE: int = 1
    INGEST_INSTAGRAM_MIN_ACCOUNT_RELEVANCE: int = 1
    INGEST_INSTAGRAM_HASHTAG_QUERIES: int = 10  # Increased from 6 to fetch more hashtags
    INGEST_INSTAGRAM_HASHTAG_PAGES: int = 8  # Increased from 5 to fetch more pages
    INGEST_INSTAGRAM_VIDEO_ONLY: bool = Field(
        default=False,  # Changed from True - include both images and videos
        description="Only keep video posts from hashtag search (filter out images).",
    )
    INGEST_INSTAGRAM_USE_HASHTAG_ONLY: bool = Field(
        default=False,  # Changed from True - enable account reel expansion
        description="Skip expensive user lookup/reels and use hashtag posts only.",
    )
    SOCIAL_RELEVANCE_MIN_SIMILARITY: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for social items to pass strict relevance gating.",
    )
    INGEST_TIKTOK_KEYWORD_LIMIT: int = 8
    INGEST_TIKTOK_MIN_RELEVANT_RESULTS: int = 8
    INGEST_TIKTOK_MAX_RESULTS: int = 50

    AGNO_TELEMETRY: bool = False

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://app.googly.io"]
    CORS_MAX_AGE_SECONDS: int = 3600
    GZIP_MINIMUM_SIZE_BYTES: int = 1000
    KB_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB — keep low until Blob Storage is wired

    # === Research Tool Optimization (Phase 3) ===
    EXA_FALLBACK_THRESHOLD: int = 5  # Min academic papers before skipping Exa
    QUERY_ENTITY_MATCH_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum lexical entity coverage ratio for specific-query relevance gating.",
    )
    QUERY_ENABLE_PRE_INTENT_ENHANCER: bool = Field(
        default=True,
        description="Enable a lightweight LLM pass to normalize user query before intent extraction.",
    )

    ENSEMBLE_API_TOKEN: str | None = None
    ENSEMBLE_API_BASE_URL: str = "https://api.ensembledata.com"
    ENSEMBLE_X_CT0: str | None = None
    ENSEMBLE_X_AUTH_TOKEN: str | None = None
    ENSEMBLE_X_GUEST_ID: str | None = None

    OPENALEX_EMAIL: str = ""

    # === Search Engine Toggles (Phase 1) ===
    SEARCH_USE_PERIGON: bool = True
    SEARCH_USE_TAVILY: bool = True
    SEARCH_USE_EXA: bool = True
    SEARCH_USE_ARXIV: bool = True
    SEARCH_USE_PUBMED: bool = True

    # === API Rate Limits (requests per second) ===
    PERIGON_RATE_LIMIT: float = 4.0
    TAVILY_RATE_LIMIT: float = 10.0  # 10 req/sec (Tavily Growth plan verified 2026-02-20)
    EXA_RATE_LIMIT: float = 5.0
    SEMANTIC_SCHOLAR_RATE_LIMIT: float = 1.0
    OPENALEX_RATE_LIMIT: float = 1.0
    ARXIV_RATE_LIMIT: float = 1.0  # Use 1 req/sec (safer than 0.33 for AsyncLimiter token bucket)
    PATENTSVIEW_RATE_LIMIT: float = 5.0  # 5 req/sec (conservative, no official limit)
    PUBMED_RATE_LIMIT: float = 10.0  # 10 req/sec (with API key), 3 req/sec (without)
    LENS_RATE_LIMIT: float = 3.0  # 3 req/sec conservative (10k/month = ~330/day = ~0.23/min avg)

    # === Circuit Breaker Config ===
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60

    # === L1 In-Memory Cache Config ===
    L1_CACHE_SIZE: int = 10000
    L1_CACHE_TTL: int = 300

    # === Agent Configuration ===
    AGENT_TIMEOUT: float = Field(default=30.0, description="Timeout for agent LLM calls in seconds")
    PROJECT_CREATE_INTENT_TIMEOUT: float = Field(
        default=12.0,
        ge=1.0,
        le=120.0,
        description="Timeout for project-creation intent preparation steps (seconds).",
    )
    INGEST_SOURCE_TIMEOUT: float = Field(
        default=30.0,
        ge=5.0,
        le=180.0,
        description="Per-source timeout for ingest source calls (seconds).",
    )
    INGEST_RETRY_ATTEMPTS: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum retry attempts for retry-wrapped ingestion source calls.",
    )
    INGEST_SOCIAL_SOURCE_TIMEOUT: float = Field(
        default=150.0,
        ge=10.0,
        le=300.0,
        description="Per-source timeout for social ingest calls (seconds). Instagram needs more time due to Ensemble API + filtering.",
    )
    INGEST_PAPERS_TIMEOUT: float = Field(
        default=90.0,
        ge=30.0,
        le=240.0,
        description="Per-source timeout for paper ingest calls (OpenAlex, Semantic Scholar, PubMed, arXiv). Papers need more time for internal filtering and scoring.",
    )
    INGEST_PATENTS_TIMEOUT: float = Field(
        default=60.0,
        ge=30.0,
        le=180.0,
        description="Per-source timeout for patent ingest calls (PatentsView, Lens).",
    )
    INGEST_NEWS_TIMEOUT: float = Field(
        default=45.0,
        ge=20.0,
        le=120.0,
        description="Per-source timeout for news ingest calls (Perigon).",
    )
    INGEST_REDDIT_BASE_SUBREDDITS: str = (
        "SkincareAddiction,AsianBeauty,MakeupAddiction,beauty,femalefashionadvice,Fashion"
    )
    INGEST_REDDIT_MAX_DYNAMIC_SUBREDDITS: int = Field(
        default=6,  # Increased from 2 to fetch from more subreddits
        ge=0,
        le=10,
        description="Maximum number of dynamic subreddit candidates beyond curated base list.",
    )
    INGEST_REDDIT_MAX_PAGES_PER_SUBREDDIT: int = Field(
        default=5,  # Increased from 2 to fetch more pages per subreddit
        ge=1,
        le=10,
        description="Maximum pagination pages fetched per subreddit during ingest.",
    )
    INGEST_REDDIT_MAX_POSTS_PER_SUBREDDIT: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Maximum posts fetched per subreddit endpoint call.",
    )
    INGEST_YOUTUBE_SEARCH_DEPTH: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Depth/pages requested from Ensemble YouTube keyword search.",
    )
    SOCIAL_REDDIT_TIMEOUT_SECONDS: float = 20.0
    SOCIAL_X_TIMEOUT_SECONDS: float = 20.0
    SOCIAL_YOUTUBE_TIMEOUT_SECONDS: float = 45.0
    SOCIAL_INSTAGRAM_TIMEOUT_SECONDS: float = 10.0
    INGEST_SOCIAL_MIN_RELEVANCE_MATCHES: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Minimum lexical relevance term matches required for social items to be kept.",
    )
    INGEST_SOCIAL_FETCH_LIMIT_PER_SOURCE: int = Field(
        default=140,
        ge=20,
        le=200,
        description="Maximum raw social items fetched per source before filtering/reranking.",
    )
    INGEST_SOCIAL_KEEP_LIMIT_PER_SOURCE: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Maximum social items kept per source after filtering/reranking.",
    )
    INGEST_FILTER_STAGE1_KEEP_RATIO: float = Field(
        default=0.6,
        ge=0.1,
        le=1.0,
        description="Top ratio of candidates retained by stage-1 embedding relevance filter.",
    )
    INGEST_FILTER_STAGE2_MIN_SURVIVORS: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Minimum paper/patent candidates kept from stage 1 when the stage-2 LLM judge "
            "returns no survivors."
        ),
    )
    INGEST_SOCIAL_RECENCY_FALLBACK_SCORE: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Fallback recency score when source timestamp parsing fails.",
    )
    INGEST_SOCIAL_RECENCY_DECAY_DAYS_TIKTOK: float = Field(
        default=21.0,
        ge=1.0,
        le=365.0,
        description="Recency decay horizon (days) for TikTok social scoring.",
    )
    INGEST_SOCIAL_RECENCY_DECAY_DAYS_X: float = Field(
        default=14.0,
        ge=1.0,
        le=365.0,
        description="Recency decay horizon (days) for X social scoring.",
    )
    INGEST_SOCIAL_RECENCY_DECAY_DAYS_REDDIT: float = Field(
        default=28.0,
        ge=1.0,
        le=365.0,
        description="Recency decay horizon (days) for Reddit social scoring.",
    )
    INGEST_SOCIAL_RECENCY_DECAY_DAYS_YOUTUBE: float = Field(
        default=21.0,
        ge=1.0,
        le=365.0,
        description="Recency decay horizon (days) for YouTube social scoring.",
    )
    INGEST_SOCIAL_RECENCY_DECAY_DAYS_INSTAGRAM: float = Field(
        default=30.0,
        ge=1.0,
        le=365.0,
        description="Recency decay horizon (days) for Instagram social scoring.",
    )
    INGEST_INSTAGRAM_GLOBAL_REELS_MIN_FLOOR: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Minimum candidate reel floor before applying global Instagram reel cap.",
    )
    INGEST_SOCIAL_LLM_FILTER_ENABLED: bool = Field(
        default=True,
        description="Enable LLM stage-2 relevance filtering for social sources.",
    )
    INGEST_SOCIAL_LLM_FAIL_OPEN: bool = Field(
        default=False,
        description="If true, keep stage-1 social items when stage-2 LLM parsing/call fails.",
    )
    INGEST_SOCIAL_LLM_MAX_CANDIDATES: int = Field(
        default=60,
        ge=5,
        le=120,
        description="Max social candidates per source sent to stage-2 LLM relevance filtering.",
    )
    INGEST_QUERY_VARIANTS_PER_SOURCE: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Max number of query variants generated per source from base query + intent terms.",
    )
    INGEST_SOCIAL_QUERY_MAX_TERMS: int = Field(
        default=10,
        ge=4,
        le=20,
        description="Maximum number of social query terms to send to social source APIs.",
    )
    INGEST_SOCIAL_TIER1_MAX_VARIANTS: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Maximum number of query variants per social source in fast tier-1 mode.",
    )
    INGEST_SOCIAL_TIER1_PROBE_TERMS: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Number of top intent terms to probe as standalone social queries in tier-1.",
    )
    INGEST_SOCIAL_EXPANSION_MAX_VARIANTS: int = Field(
        default=10,
        ge=2,
        le=24,
        description="Maximum number of query variants per social source during expansion mode.",
    )
    INGEST_SOCIAL_MAX_AGE_DAYS: int = Field(
        default=30,
        ge=7,
        le=120,
        description="Maximum social post age in days for social ingest recency cutoff.",
    )
    INGEST_SOCIAL_EXPANSION_ENABLED: bool = Field(
        default=False,
        description="Enable second-pass social fetch expansion when raw social volume is low.",
    )
    INGEST_SOCIAL_EXPANSION_TIMEOUT: float = Field(
        default=180.0,
        ge=30.0,
        le=600.0,
        description="Per-source timeout for second-pass social expansion fetch (seconds).",
    )
    ARQ_WORKER_JOB_TIMEOUT: int = Field(
        default=1200,
        ge=120,
        le=3600,
        description="ARQ worker job timeout in seconds.",
    )
    ARQ_WORKER_MAX_JOBS: int = Field(
        default=40,
        ge=1,
        le=200,
        description=(
            "Max concurrent jobs per ARQ worker pod. I/O-bound tasks (LLM, HTTP, DB) are safe "
            "at high concurrency. Keep below the asyncpg pool size (50) to avoid connection "
            "contention. Default 40 leaves headroom for other job types sharing the pool."
        ),
    )
    ARQ_WORKER_HEALTH_CHECK_INTERVAL: int = 30
    GEMINI_MAX_CALLS_PER_MINUTE: int = Field(
        default=15,
        ge=1,
        le=1000,
        description=(
            "Maximum Gemini API calls allowed per minute across all worker pods. "
            "Enforced via a Redis sliding-window counter so the limit is global, "
            "not per-pod. Tune to match your Gemini quota tier. "
            "gemini-3.1-flash-lite-preview free tier: 15 RPM."
        ),
    )
    GEMINI_MAX_CALLS_PER_DAY: int = Field(
        default=500,
        ge=1,
        le=100000,
        description=(
            "Maximum Gemini API calls allowed per calendar day (UTC) across all worker pods. "
            "gemini-3.1-flash-lite-preview free tier: 500 RPD."
        ),
    )
    GEMINI_RATE_LIMIT_WINDOW: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Sliding window size in seconds for the Gemini per-minute rate limiter.",
    )
    ARQ_REFRESH_CRON_HOURS: tuple[int, ...] = (0, 6, 12, 18)
    CHAT_HISTORY_RETENTION_DAYS: int = 90
    INGEST_SOCIAL_RAW_TARGET_TOTAL: int = Field(
        default=120,
        ge=20,
        le=500,
        description="Target total raw social candidates across enabled social sources.",
    )
    INGEST_SOCIAL_RAW_MIN_PER_SOURCE: int = Field(
        default=30,
        ge=5,
        le=200,
        description="Minimum raw candidates per strong social source before expansion.",
    )
    INGEST_SOCIAL_RAW_MIN_PER_WEAK_SOURCE: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Minimum raw candidates for weaker social sources (Instagram/YouTube).",
    )
    INGEST_SOCIAL_EXPANDED_QUERY_MAX_TERMS: int = Field(
        default=16,
        ge=8,
        le=32,
        description="Maximum query terms used in second-pass social expansion fetch.",
    )
    INGEST_SOCIAL_EXPANDED_FETCH_LIMIT_PER_SOURCE: int = Field(
        default=120,
        ge=20,
        le=300,
        description="Max raw items fetched per source in second-pass social expansion.",
    )
    INGEST_SOCIAL_EXPANDED_TIKTOK_MAX_RESULTS: int = Field(
        default=100,
        ge=20,
        le=200,
        description="TikTok max_results used during second-pass expansion.",
    )
    INGEST_SOCIAL_EXPANDED_INSTAGRAM_HASHTAG_PAGES: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Instagram hashtag pages to fetch per tag during second-pass expansion.",
    )
    INGEST_SOCIAL_EXPANDED_MAX_AGE_DAYS: int = Field(
        default=90,
        ge=14,
        le=365,
        description="Maximum social post age in days used during second-pass expansion.",
    )
    ENABLE_FULLTEXT_ENRICHMENT: bool = Field(
        default=False,
        description="Enable ARQ fulltext enrichment scheduling for paper/patent sources.",
    )
    ENABLE_FULLTEXT_RETRIEVAL_PREFERENCE: bool = Field(
        default=False,
        description="Prefer fulltext chunks over abstract chunks when relevance is comparable.",
    )
    ENABLE_FULLTEXT_BACKFILL: bool = Field(
        default=False,
        description="Enable backfill task for existing metadata-only paper/patent chunks.",
    )
    FULLTEXT_MAX_SOURCE_BYTES: int = Field(
        default=25 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
        description="Maximum bytes downloaded for one fulltext source asset.",
    )
    FULLTEXT_MAX_PAGES: int = Field(
        default=300,
        ge=1,
        le=2000,
        description="Maximum pages extracted from one fulltext document.",
    )
    FULLTEXT_FETCH_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout for resolving/fetching one fulltext source asset.",
    )
    FULLTEXT_HTTP_MAX_KEEPALIVE: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Max keepalive HTTP connections for fulltext fetch client.",
    )
    FULLTEXT_HTTP_MAX_CONNECTIONS: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max total HTTP connections for fulltext fetch client.",
    )
    FULLTEXT_RETRY_BASE_DELAY_SECONDS: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Base retry delay for fulltext asset retries.",
    )
    FULLTEXT_RETRY_MAX_DELAY_SECONDS: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Max retry delay for fulltext asset retries.",
    )
    FULLTEXT_MAX_ENRICHMENT_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum bounded retries for one source asset enrichment.",
    )
    FULLTEXT_MAX_REDIRECTS: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Maximum redirects followed when fetching fulltext source URLs.",
    )
    FULLTEXT_ALLOWED_DOMAINS: str = Field(
        default="",
        description="Optional comma-separated allowlist of fulltext source domains.",
    )
    ENSEMBLE_TIKTOK_FULL_SEARCH_TIMEOUT: float = Field(
        default=180.0,
        ge=20.0,
        le=900.0,
        description="HTTP timeout for Ensemble TikTok full keyword search endpoint.",
    )
    PERIGON_DAYS_LOOKBACK: int = 30
    PERIGON_TIMEOUT_SECONDS: float = 15.0
    TAVILY_TIMEOUT_SECONDS: float = 15.0
    EXA_TIMEOUT_SECONDS: float = 15.0
    PAPERS_PUBMED_TIMEOUT_SECONDS: float = 15.0
    PAPERS_ARXIV_TIMEOUT_SECONDS: float = 15.0
    PAPERS_SEMANTIC_SCHOLAR_TIMEOUT_SECONDS: float = 15.0
    PAPERS_OPENALEX_TIMEOUT_SECONDS: float = 10.0
    PATENTS_PATENTSVIEW_TIMEOUT_SECONDS: float = 15.0
    PATENTS_LENS_TIMEOUT_SECONDS: float = 20.0
    INTENT_EXTRACT_TIMEOUT_SECONDS: float = 30.0
    TAVILY_MAX_RESULTS: int = 15
    TAVILY_MAX_RESULTS_DOMAIN: int = 7
    EXA_NUM_RESULTS: int = 10
    EXA_MAX_CHARACTERS: int = 1000
    EXA_MAX_CHARACTERS_DOMAIN: int = 2000
    PAPERS_MAX_RESULTS: int = 20
    OPENALEX_TARGET_PAPER_COUNT: int = 20
    OPENALEX_QUERY_VARIANT_LIMIT: int = 3
    OPENALEX_PER_PAGE: int = 50
    OPENALEX_STRICT_PAGES: tuple[int, ...] = (1, 2)
    OPENALEX_LATEST_LOOKBACK_YEARS: int = 7
    OPENALEX_LATEST_MIN_YEAR: int = 2018
    SOCIAL_X_MAX_RESULTS: int = 100
    SOCIAL_YOUTUBE_MAX_RESULTS: int = 100
    SOCIAL_REDDIT_DEFAULT_LIMIT: int = 24
    PAPER_TITLE_DEDUP_THRESHOLD: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Fuzzy threshold for cross-source paper title deduplication.",
    )
    FULLTEXT_BACKFILL_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Default batch size for fulltext backfill cursor scans.",
    )
    PROJECT_DISCOVER_LIMIT: int = 500
    PROJECT_UPLOAD_SUMMARIES_LIMIT: int = 40
    SEARCH_QUERY_MAX_LEN: int = 1000

    # === Insights / Clustering ===
    INSIGHTS_ENABLED: bool = True
    INSIGHTS_FULL_REPORT_ENABLED: bool = True
    INSIGHTS_FOLLOWUP_ENABLED: bool = True
    CLUSTER_MIN_CLUSTER_SIZE: int = Field(default=3, ge=2, le=100)
    CLUSTER_MIN_SAMPLES: int = Field(default=2, ge=1, le=100)
    CLUSTER_MAX_CLUSTERS: int = Field(default=30, ge=1, le=200)
    CLUSTER_MAX_CHUNKS: int = Field(default=3000, ge=10, le=10000)
    CLUSTER_MAX_CHUNKS_PER_DOCUMENT: int = Field(default=8, ge=1, le=100)
    CLUSTER_TARGET_CLUSTER_SIZE: int = Field(default=24, ge=5, le=500)
    CLUSTER_MAX_CLUSTER_SIZE: int = Field(default=32, ge=8, le=1000)
    CLUSTER_NOISE_ASSIGNMENT_MIN_SIMILARITY: float = Field(default=0.35, ge=0.0, le=1.0)
    CLUSTER_MERGE_DUPLICATE_TITLES: bool = True
    CLUSTER_DUPLICATE_TITLE_MIN_DOC_JACCARD: float = Field(default=0.15, ge=0.0, le=1.0)
    CLUSTER_DUPLICATE_TITLE_MIN_SUMMARY_JACCARD: float = Field(default=0.22, ge=0.0, le=1.0)
    CLUSTER_LOCK_TTL: int = Field(default=600, ge=30, le=7200)
    CLUSTER_LOCK_HEARTBEAT_SECONDS: int = Field(default=30, ge=5, le=600)
    INSIGHTS_RECLUSTER_COOLDOWN_SECONDS: int = Field(default=900, ge=30, le=86400)
    CLUSTER_CPU_WORKERS: int = Field(default=2, ge=1, le=16)
    CLUSTER_EXTRACTION_MAX_CONCURRENCY: int = Field(default=4, ge=1, le=64)
    BACKFILL_INSIGHTS_BATCH_SIZE: int = Field(default=10, ge=1, le=500)
    INSIGHT_REPORT_MODEL: str = "gpt-4o"
    INSIGHTS_PREGENERATE_REPORTS_ON_CLUSTER: bool = False
    INSIGHTS_REPORT_VERIFY_ENABLED: bool = True
    INSIGHT_REPORT_VERIFIER_MODEL: str = "gpt-4.1"
    INSIGHT_REPORT_VERIFY_MAX_TOKENS: int = Field(default=3500, ge=200, le=8000)
    INSIGHT_REPORT_VERIFY_EVIDENCE_MAX_CHUNKS: int = Field(default=24, ge=8, le=80)
    INSIGHTS_MAX_REPORTS_PER_PROJECT_PER_DAY: int = Field(default=30, ge=1, le=10000)
    FOLLOWUP_RATE_LIMIT: int = Field(default=20, ge=1, le=1000)
    FOLLOWUP_RATE_WINDOW_SECONDS: int = Field(default=60, ge=10, le=3600)
    FOLLOWUP_HISTORY_RETENTION_DAYS: int = Field(default=30, ge=1, le=3650)
    DOC_SUMMARIZE_ENABLED: bool = True
    DOC_SUMMARIZE_BATCH_SIZE: int = Field(default=20, ge=1, le=200)
    DOC_SUMMARIZE_TARGET_MIN_WORDS: int = Field(default=100, ge=20, le=300)
    DOC_SUMMARIZE_TARGET_MAX_WORDS: int = Field(default=120, ge=20, le=400)
    DOC_SUMMARIZE_MAX_TOKENS: int = Field(default=260, ge=64, le=1024)
    INSIGHTS_BACKFILL_SUMMARIES_ON_DETAIL_READ: bool = True
    CLUSTER_MAX_KEY_FINDINGS: int = Field(default=5, ge=1, le=20)
    CLUSTER_REPORT_LAMBDA: float = Field(default=0.6, ge=0.0, le=1.0)
    CLUSTER_REPORT_MAX_CHUNKS: int = Field(default=32, ge=1, le=200)
    CLUSTER_REPORT_MAX_TOKENS: int = Field(default=8000, ge=100, le=8000)

    # === Cache TTL Configuration ===
    CACHE_TTL_PAPERS: int = Field(
        default=86400, description="Cache TTL for research papers (24 hours)"
    )
    CACHE_TTL_PATENTS: int = Field(default=86400, description="Cache TTL for patents (24 hours)")
    CACHE_TTL_SEARCH: int = Field(default=3600, description="Cache TTL for search results (1 hour)")
    CACHE_TTL_NEWS: int = Field(default=1800, description="Cache TTL for news articles (30 min)")
    CACHE_TTL_SOCIAL: int = Field(
        default=3600,
        description="Cache TTL for social media API results (1 hour)",
    )
    CACHE_TTL_STALE: int = Field(
        default=86400 * 7,
        description="Cache TTL for stale fallback copies of API results (7 days). "
        "Used as a long-lived fallback when the primary API call fails.",
    )
    INTENT_CACHE_TTL: int = Field(
        default=3600,
        ge=1,
        description="Cache TTL for intent extraction results (1 hour). "
        "Identical queries skip the LLM entirely on cache hit.",
    )
    DEDUP_LOCK_TTL_SECONDS: int = Field(
        default=120,
        ge=1,
        le=3600,
        description="TTL for request deduplication lock keys.",
    )
    DEDUP_RESULT_TTL_SECONDS: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="TTL for deduplicated request result payloads.",
    )
    DEDUP_WAIT_MAX_ATTEMPTS: int = Field(
        default=600,
        ge=1,
        le=10000,
        description="Max polling attempts for secondary deduplicated requests.",
    )
    DEDUP_WAIT_POLL_SECONDS: float = Field(
        default=0.1,
        ge=0.01,
        le=5.0,
        description="Polling sleep interval for secondary deduplicated requests.",
    )

    # === Routing Config (Query Type → API List) ===
    SEARCH_ROUTING: dict[str, list[str]] = {
        "trend": ["perigon", "tavily", "exa"],
        "research": ["exa", "tavily"],
        "brand": ["perigon", "tavily"],
        "general": ["tavily"],
    }

    # === Validators (AGENTS.md Rule 8: validate all config at startup) ===

    @field_validator(
        "PERIGON_RATE_LIMIT",
        "TAVILY_RATE_LIMIT",
        "EXA_RATE_LIMIT",
        "SEMANTIC_SCHOLAR_RATE_LIMIT",
        "OPENALEX_RATE_LIMIT",
        "ARXIV_RATE_LIMIT",
        "PATENTSVIEW_RATE_LIMIT",
        "PUBMED_RATE_LIMIT",
        "LENS_RATE_LIMIT",
    )
    @classmethod
    def validate_rate_limits(cls, v: float) -> float:
        """Validate API rate limits (requests per second)."""
        if v < 0.1:
            raise ValueError("Rate limit must be >= 0.1 requests/sec (minimum reasonable)")
        if v > 1000.0:
            raise ValueError("Rate limit must be <= 1000 requests/sec (reasonable max)")
        return v

    @field_validator("L1_CACHE_TTL")
    @classmethod
    def validate_ttls(cls, v: int) -> int:
        """Validate cache TTL values."""
        if v < 60:
            raise ValueError("TTL must be >= 60 seconds (minimum reasonable cache)")
        if v > 86400:
            raise ValueError("TTL must be <= 86400 seconds (24 hours max)")
        return v

    @field_validator("KB_SCORE_THRESHOLD")
    @classmethod
    def validate_kb_threshold(cls, v: float) -> float:
        """Validate KB similarity score threshold (cosine similarity: 0.0-1.0)."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("KB_SCORE_THRESHOLD must be between 0.0 and 1.0")
        return v

    @field_validator("CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    @classmethod
    def validate_circuit_breaker_failure_threshold(cls, v: int) -> int:
        """Validate circuit breaker failure threshold."""
        if v < 1:
            raise ValueError("CIRCUIT_BREAKER_FAILURE_THRESHOLD must be >= 1")
        if v > 100:
            raise ValueError("CIRCUIT_BREAKER_FAILURE_THRESHOLD must be <= 100")
        return v

    @field_validator("CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    @classmethod
    def validate_circuit_breaker_recovery_timeout(cls, v: int) -> int:
        """Validate circuit breaker recovery timeout (seconds)."""
        if v < 5:
            raise ValueError("CIRCUIT_BREAKER_RECOVERY_TIMEOUT must be >= 5 seconds")
        if v > 600:
            raise ValueError("CIRCUIT_BREAKER_RECOVERY_TIMEOUT must be <= 600 seconds (10 min)")
        return v

    @field_validator("L1_CACHE_SIZE")
    @classmethod
    def validate_l1_cache_size(cls, v: int) -> int:
        """Validate L1 cache size (number of entries)."""
        if v < 100:
            raise ValueError("L1_CACHE_SIZE must be >= 100 entries (minimum useful size)")
        if v > 100000:
            raise ValueError("L1_CACHE_SIZE must be <= 100000 entries (memory safety)")
        return v

    @field_validator("ARQ_REFRESH_CRON_HOURS")
    @classmethod
    def validate_arq_refresh_cron_hours(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Validate worker refresh cron hours as unique values in [0, 23]."""
        if not v:
            raise ValueError("ARQ_REFRESH_CRON_HOURS must include at least one hour")
        normalized = tuple(sorted(set(v)))
        if any(hour < 0 or hour > 23 for hour in normalized):
            raise ValueError("ARQ_REFRESH_CRON_HOURS values must be between 0 and 23")
        return normalized


settings = Settings()
