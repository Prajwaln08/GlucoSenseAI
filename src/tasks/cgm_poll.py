"""
Scheduled Junction glucose pull (the safety-net half of "webhook + scheduled pull").

Junction webhooks are the real-time primary path, but if they go quiet the failover
detector would have no fresh signal. This periodic task pulls recent glucose for every
linked user, ingests it (deduped), refreshes the Junction heartbeat on success, and
re-evaluates each user's active CGM source — so xDRIP fallback engages reliably.

Runs on Celery beat (see celery_app.beat_schedule). Idempotent: dedup means re-pulling
the same window stores nothing new.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.models import User
from src.db.session import SessionLocal
from src.integrations import cgm_router
from src.integrations.ingest import correct_junction_clock, ingest_cgm_readings
from src.integrations.junction import JunctionClient
from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Pull a small trailing window each tick; dedup makes overlap free.
PULL_WINDOW_DAYS = 1


@celery_app.task(name="src.tasks.cgm_poll.poll_junction_glucose")
def poll_junction_glucose() -> dict:
    """Pull recent Junction glucose for all linked users; update failover state."""
    db = SessionLocal()
    client = JunctionClient()
    users_polled = 0
    readings_saved = 0
    try:
        users = (
            db.query(User)
            .filter(User.junction_user_id.isnot(None), User.is_active.is_(True))
            .all()
        )
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=PULL_WINDOW_DAYS)

        for user in users:
            users_polled += 1
            try:
                readings = client.fetch_glucose(
                    user.junction_user_id, user.id, start_date, end_date, ingested_via="poll"
                )
                readings = correct_junction_clock(db, user, readings)
                saved = ingest_cgm_readings(db, readings, commit=False)
                readings_saved += saved
                # Heartbeat if we got data, OR if a health check confirms the link is alive.
                if readings or client.health_check(user.junction_user_id):
                    cgm_router.record_junction_ok(db, user)
                else:
                    cgm_router.refresh_active_source(db, user)  # may flip to xdrip if stale
                db.commit()
            except Exception as exc:  # noqa: BLE001 — one user's failure must not stop the sweep
                db.rollback()
                logger.warning("Junction poll failed for user %s: %s", user.id, exc)
    finally:
        db.close()

    logger.info("Junction poll: %s users, %s new readings", users_polled, readings_saved)
    return {"users_polled": users_polled, "readings_saved": readings_saved}
