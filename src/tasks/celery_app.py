"""
Celery application instance for GlucoSense AI.

Broker  : Redis  (CELERY_BROKER_URL in .env)
Backend : Redis  (CELERY_RESULT_BACKEND in .env)

Usage (start worker):

    macOS (ML libraries crash with fork — use solo pool):
        OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
        celery -A src.tasks.celery_app worker --pool=solo --concurrency=1 -Q glucosense --loglevel=info

    Linux / Docker:
        celery -A src.tasks.celery_app worker --concurrency=2 -Q glucosense --loglevel=info

    Prerequisites: redis-server must be running on localhost:6379
"""

import os

from celery import Celery

BROKER_URL  = os.environ.get("CELERY_BROKER_URL",    "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# How often the Junction safety-net pull runs (seconds). Keep it under the failover
# staleness window so xDRIP fallback is detected promptly.
CGM_POLL_INTERVAL_SEC = int(os.environ.get("CGM_POLL_INTERVAL_SEC", "600"))

celery_app = Celery(
    "glucosense",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["src.tasks.retrain", "src.tasks.cgm_poll"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,            # re-queue on worker crash
    worker_prefetch_multiplier=1,   # one task at a time (training is heavy)
    task_routes={
        "src.tasks.retrain.*": {"queue": "glucosense"},
        "src.tasks.cgm_poll.*": {"queue": "glucosense"},
    },
    beat_schedule={
        "junction-glucose-pull": {
            "task": "src.tasks.cgm_poll.poll_junction_glucose",
            "schedule": float(CGM_POLL_INTERVAL_SEC),
            "options": {"queue": "glucosense"},
        },
    },
)
