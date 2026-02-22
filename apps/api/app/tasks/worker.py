"""ARQ WorkerSettings — registers all background tasks and cron jobs.

Rule: Every new ARQ task MUST be added to `functions` here or it will never run.
Every new cron job MUST be added to `cron_jobs` here.

Worker is started with: uv run arq app.tasks.worker.WorkerSettings
"""

from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.tasks.cleanup import cleanup_old_chat_messages
from app.tasks.ingest_document import ingest_document
from app.tasks.ingest_project import ingest_project
from app.tasks.refresh_project import refresh_due_projects, refresh_project


class WorkerSettings:
    """
    ARQ worker configuration.

    max_jobs: max concurrent jobs per worker pod. 20 is safe for I/O-bound
    tasks (LLM calls, HTTP, DB) — the event loop handles concurrency.
    job_timeout: 300s (5 min) — enough for large document ingestion + embedding.
    health_check_interval: 30s — used by KEDA to confirm worker is alive.
    """

    functions = [
        ingest_project,
        ingest_document,
        refresh_project,
        refresh_due_projects,
        cleanup_old_chat_messages,
    ]

    cron_jobs = [
        # refresh_due_projects runs every 6 hours — checks which projects are
        # overdue for daily/weekly refresh and enqueues refresh_project for each
        cron(refresh_due_projects, hour={0, 6, 12, 18}, minute=0),
        # cleanup_old_chat_messages runs daily at 3am UTC — deletes messages
        # older than 90 days from both Postgres and Redis
        cron(cleanup_old_chat_messages, hour=3, minute=0),
    ]

    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

    max_jobs = 20
    job_timeout = 300
    health_check_interval = 30
    # queue_name is intentionally omitted — ARQ default "arq:queue" matches create_pool default.
