"""Prometheus metrics for monitoring API integration performance.

Provides counters and histograms for tracking:
- Cache hit/miss rates (L1 and L2)
- API call success/failure rates
- Circuit breaker state changes
- API response times
"""

from prometheus_client import Counter, Histogram

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_tier", "cache_type"],  # L1/L2, search/kb/embed
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_tier", "cache_type"],
)

# API metrics
api_calls_total = Counter(
    "api_calls_total",
    "Total external API calls",
    ["api_name", "status"],  # success/failure/timeout
)

api_call_duration_seconds = Histogram(
    "api_call_duration_seconds",
    "API call duration in seconds",
    ["api_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Circuit breaker metrics
circuit_breaker_state_changes_total = Counter(
    "circuit_breaker_state_changes_total",
    "Circuit breaker state transitions",
    ["api_name", "from_state", "to_state"],
)

circuit_breaker_open_total = Counter(
    "circuit_breaker_open_total",
    "Total times circuit breaker opened",
    ["api_name"],
)

# Rate limiter metrics
rate_limiter_throttled_total = Counter(
    "rate_limiter_throttled_total",
    "Total requests throttled by rate limiter",
    ["api_name"],
)

# Chat system metrics
chat_requests_total = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["status"],  # success/error
)

chat_response_duration_seconds = Histogram(
    "chat_response_duration_seconds",
    "Chat response generation duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

chat_messages_per_request = Histogram(
    "chat_messages_per_request",
    "Number of history messages included in chat context",
    buckets=[0, 5, 10, 20, 30, 40, 50],
)

chat_stream_tokens_total = Counter(
    "chat_stream_tokens_total",
    "Total tokens streamed in chat responses",
)

# Orchestrator stage metrics (Phase 3)
orchestrator_stage_duration_seconds = Histogram(
    "orchestrator_stage_duration_seconds",
    "Duration of orchestrator pipeline stages in seconds",
    ["stage"],  # intent, relevance, kb_retrieve, research, synthesis
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# Research tool metrics (Phase 3)
research_tool_calls_total = Counter(
    "research_tool_calls_total",
    "Total research tool calls",
    ["tool"],  # semantic_scholar, arxiv, pubmed, exa
)

research_deduplication_rate = Histogram(
    "research_deduplication_rate",
    "Percentage of duplicate papers removed (0.0-1.0)",
    buckets=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
)

exa_usage_rate = Histogram(
    "exa_usage_rate",
    "Percentage of research queries that triggered Exa (0.0-1.0)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Fulltext enrichment metrics
fulltext_resolve_total = Counter(
    "fulltext_resolve_total",
    "Total fulltext URL resolution attempts",
    ["source", "status"],  # source: paper|patent, status: success|blocked|failed
)

fulltext_fetch_total = Counter(
    "fulltext_fetch_total",
    "Total fulltext source fetch attempts",
    ["source", "status"],  # success|timeout|blocked|failed
)

fulltext_fetch_duration_seconds = Histogram(
    "fulltext_fetch_duration_seconds",
    "Fulltext source fetch duration in seconds",
    ["source"],
    buckets=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

fulltext_extract_total = Counter(
    "fulltext_extract_total",
    "Total fulltext extraction attempts",
    ["source", "status"],  # success|empty|unsupported|failed
)

fulltext_embed_total = Counter(
    "fulltext_embed_total",
    "Total fulltext embedding pipeline attempts",
    ["source", "status"],  # success|failed
)

fulltext_upsert_total = Counter(
    "fulltext_upsert_total",
    "Total fulltext chunk upsert attempts",
    ["source", "status"],  # success|failed
)
