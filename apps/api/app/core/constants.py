from enum import Enum


class DatabasePool:
    MIN_SIZE = 5
    MAX_SIZE = 50
    POOL_SIZE = 20
    MAX_OVERFLOW = 30


class RedisKeys:
    ARQ_QUEUE = "arq:queue:{queue}"
    SEMANTIC_CACHE = "sem_cache:{project_id}:{hash}"
    SESSION = "session:{user_id}:{project_id}"
    RATE_LIMIT = "ratelimit:{user_id}:{window}"
    KB_HOT = "kb_hot:{project_id}:{hash}"
    EMBED_CACHE = (
        "embed:{model_version}:{hash}"  # Include model version to auto-invalidate on model change
    )
    CHAT_HISTORY = "chat_history:{user_id}:{project_id}:{session_id}"
    UPLOAD_STAGING = "upload:staging:{upload_id}"
    PROJECT_CACHE_VERSION = "cache_version:{project_id}"
    PROJECTS_SUMMARY = "projects_summary:{user_id}"
    PROJECT_INGEST_STATUS = "ingest_status:{project_id}"

    # Search API cache keys (project-scoped, deterministic SHA256)
    SEARCH_CACHE = "search:cache:{project_id}:{api}:{query_type}:{hash}"
    SEARCH_CACHE_STALE = "search:stale:{project_id}:{api}:{query_type}:{hash}"

    # Circuit breaker state tracking
    CIRCUIT_BREAKER_STATE = "circuit:{api}:state"
    CIRCUIT_BREAKER_FAILURE_COUNT = "circuit:{api}:failures"
    CIRCUIT_BREAKER_LAST_FAILURE = "circuit:{api}:last_failure"


class RedisTTL(Enum):
    SEMANTIC_CACHE_TRENDING = 3600
    SEMANTIC_CACHE_STABLE = 86400
    SESSION = 1800
    RATE_LIMIT = 60
    KB_HOT = 3600
    EMBED_CACHE = 3600
    CHAT_HISTORY = 604800  # 7 days
    UPLOAD_STAGING = 3600  # 1 hour
    PROJECTS_SUMMARY = 300  # 5 minutes
    PROJECT_INGEST_STATUS = 86400  # 24 hours

    # Search API cache TTLs (query type-specific for freshness)
    SEARCH_CACHE_NEWS = 3600  # 1 hour (news changes frequently)
    SEARCH_CACHE_RESEARCH = 86400  # 24 hours (papers stable)
    SEARCH_CACHE_BRAND = 21600  # 6 hours (brand monitoring balance)
    SEARCH_CACHE_GENERAL = 43200  # 12 hours (general queries balance)
    SEARCH_CACHE_STALE = 604800  # 7 days (stale fallback for all-APIs-down)

    # L1 in-memory cache TTL
    L1_CACHE = 300  # 5 minutes (hot cache for high-frequency queries)

    # Circuit breaker recovery timeout
    CIRCUIT_BREAKER_RECOVERY = 60  # 60 seconds open before half-open retry


class EmbeddingBatchSize:
    DEFAULT = 100


class KBUpload:
    ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"pdf", "docx", "txt", "md"})


class ProjectRefresh:
    VALID_STRATEGIES: frozenset[str] = frozenset({"once", "daily", "weekly", "on_demand"})
