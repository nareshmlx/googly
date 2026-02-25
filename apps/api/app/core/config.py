from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=["../../.env", ".env"], extra="ignore")

    ENVIRONMENT: str = "local"
    LOG_LEVEL: str = "INFO"

    OPENAI_API_KEY: str | None = None

    # === Embedding Model Configuration ===
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_MODEL_VERSION: str = "v3-small-2024"  # Update when OpenAI changes model

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
    KB_SCORE_THRESHOLD: float = 0.70
    INGEST_MAX_ACCOUNTS: int = 10
    INGEST_INSTAGRAM_ACCOUNT_CANDIDATES: int = 15
    INGEST_INSTAGRAM_REELS_PER_ACCOUNT: int = 6
    INGEST_INSTAGRAM_DETAIL_ENRICH_PER_ACCOUNT: int = 3
    INGEST_INSTAGRAM_ACCOUNTS_TO_FETCH: int = 10
    INGEST_INSTAGRAM_GLOBAL_REELS_LIMIT: int = 24
    INGEST_INSTAGRAM_MAX_REELS_PER_ACCOUNT: int = 2
    INGEST_INSTAGRAM_MIN_REEL_RELEVANCE: int = 1
    INGEST_INSTAGRAM_MIN_ACCOUNT_RELEVANCE: int = 1
    INGEST_INSTAGRAM_HASHTAG_QUERIES: int = 4
    INGEST_INSTAGRAM_HASHTAG_PAGES: int = 1

    AGNO_TELEMETRY: bool = False

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://app.googly.io"]
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

    # === Cache TTL Configuration ===
    CACHE_TTL_PAPERS: int = Field(
        default=86400, description="Cache TTL for research papers (24 hours)"
    )
    CACHE_TTL_PATENTS: int = Field(default=86400, description="Cache TTL for patents (24 hours)")
    CACHE_TTL_SEARCH: int = Field(default=3600, description="Cache TTL for search results (1 hour)")
    CACHE_TTL_NEWS: int = Field(default=1800, description="Cache TTL for news articles (30 min)")
    INTENT_CACHE_TTL: int = Field(
        default=3600,
        ge=1,
        description="Cache TTL for intent extraction results (1 hour). "
        "Identical queries skip the LLM entirely on cache hit.",
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


# === Cache Key Prefixes (AGENTS.md Rule 5: No hardcoded strings) ===
class CacheKeys:
    """Standardized cache key prefixes for all tools.

    Format: <category>:<provider>
    - Papers: academic papers and research
    - Patents: patent databases
    - Search: web search engines
    - News: news aggregators
    - Social: social media platforms
    """

    PAPERS_SEMANTIC_SCHOLAR = "papers:semantic_scholar"
    PAPERS_ARXIV = "papers:arxiv"
    PAPERS_PUBMED = "papers:pubmed"
    PATENTS_PATENTSVIEW = "patents:patentsview"
    PATENTS_LENS = "patents:lens"
    SEARCH_TAVILY = "search:tavily"
    SEARCH_EXA = "search:exa"
    NEWS_PERIGON = "news:perigon"
    SOCIAL_INSTAGRAM = "social:instagram"
    SOCIAL_TIKTOK = "social:tiktok"
    SOCIAL_X = "social:x"


settings = Settings()
