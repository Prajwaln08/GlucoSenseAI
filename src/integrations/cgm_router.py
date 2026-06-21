"""
CGM source precedence + failover: Junction PRIMARY → xDRIP FALLBACK.

The unified ingest layer records a per-user heartbeat each time a source delivers a
reading (or a Junction health-check passes). This module turns those heartbeats into a
single observable `cgm_active_source` the UI and operators can read.

Decision: Junction is primary. If Junction has produced no reading / passed no health
check within CGM_STALENESS_MINUTES, and xDRIP has, the active source flips to "xdrip".
When Junction recovers, it flips back. Overlapping same-timestamp readings never
double-count (the ingest layer dedups on (user_id, timestamp)).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.db.models import User

# How long Junction may be silent before we consider it stale and fall back to xDRIP.
# ~2 sensor intervals (a Libre/Dexcom reports every 5 min) with headroom.
CGM_STALENESS_MINUTES = int(os.environ.get("CGM_STALENESS_MINUTES", "20"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fresh(ts: datetime | None, *, minutes: int = CGM_STALENESS_MINUTES) -> bool:
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now() - ts) <= timedelta(minutes=minutes)


def evaluate_active_source(user: User) -> str:
    """Decide which CGM source is currently authoritative for this user."""
    if _fresh(user.cgm_last_junction_ok_at):
        return "junction"            # primary is healthy
    if _fresh(user.cgm_last_xdrip_at):
        return "xdrip"               # primary stale, fallback delivering
    # Neither is fresh — keep the last known source, defaulting to junction (primary).
    return user.cgm_active_source or "junction"


def record_junction_ok(db: Session, user: User, *, commit: bool = False) -> None:
    """Mark that Junction just delivered data / passed a health check, and re-evaluate."""
    user.cgm_last_junction_ok_at = _now()
    user.cgm_active_source = evaluate_active_source(user)
    if commit:
        db.commit()


def record_xdrip(db: Session, user: User, *, commit: bool = False) -> None:
    """Mark that xDRIP just pushed a reading, and re-evaluate the active source."""
    user.cgm_last_xdrip_at = _now()
    user.cgm_active_source = evaluate_active_source(user)
    if commit:
        db.commit()


def refresh_active_source(db: Session, user: User, *, commit: bool = False) -> str:
    """Re-evaluate without recording a new heartbeat (e.g. on a status read)."""
    user.cgm_active_source = evaluate_active_source(user)
    if commit:
        db.commit()
    return user.cgm_active_source


def status(user: User) -> dict:
    """Compact failover status for the UI / API."""
    return {
        "active_source": evaluate_active_source(user),
        "junction_fresh": _fresh(user.cgm_last_junction_ok_at),
        "xdrip_fresh": _fresh(user.cgm_last_xdrip_at),
        "last_junction_ok_at": user.cgm_last_junction_ok_at,
        "last_xdrip_at": user.cgm_last_xdrip_at,
        "staleness_minutes": CGM_STALENESS_MINUTES,
    }
