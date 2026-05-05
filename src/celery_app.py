"""
Aletheia Celery Application Factory
------------------------------------
Configures the Celery broker, result backend, serializer, and task
autodiscovery.  All settings fall back to sane defaults for local
development (localhost Redis) and are overridden via environment
variables in Docker.
"""

import os
from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery(
    "aletheia",
    broker=broker_url,
    backend=result_backend,
)

app.conf.update(
    # Serialization — strict JSON for Pydantic compatibility
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    # Result expiry (24 h)
    result_expires=86400,
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Global time-limit safety net — task-level decorators take precedence.
    # Override via env vars for environments with different hardware budgets.
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "14460")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "14400")),
)

# Autodiscover tasks in src/tasks/
app.autodiscover_tasks(["src.tasks"])
