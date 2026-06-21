"""
The single DB write-path for ALL CGM + activity data.

Every source — Junction (primary), xDRIP+ (fallback), Google Fit (watch) — funnels
through here. This is the one place that:
  - dedups CGM readings by (user_id, timestamp), so the same instant reported by both
    Junction and xDRIP collapses to one row (no double-counting during failover),
  - upserts daily activity by (user_id, calendar_date, provider),
  - writes provenance (source / source_device_id / device_type / ingested_via).

No external API calls live here — callers pass already-normalised dataclasses.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.db.models import CgmReading, WearableActivity
from src.integrations.schemas import ACTIVITY_FIELDS, ActivityIngest, CgmReadingIngest


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ingest_cgm_readings(
    db: Session,
    readings: list[CgmReadingIngest],
    *,
    commit: bool = True,
) -> int:
    """Insert new CGM readings, deduped by (user_id, timestamp). Returns # inserted.

    Dedup is intentionally source-agnostic on (user_id, timestamp): a single CGM stream
    has one true value per instant, so a Junction reading and an xDRIP reading for the
    same timestamp must not both be stored. First writer wins; Junction (primary) is
    typically first via webhook/poll, xDRIP only fills genuine gaps.
    """
    saved = 0
    for r in readings:
        exists = (
            db.query(CgmReading.id)
            .filter(CgmReading.user_id == r.user_id, CgmReading.timestamp == r.timestamp)
            .first()
        )
        if exists:
            continue
        db.add(CgmReading(
            id=str(uuid.uuid4()),
            user_id=r.user_id,
            timestamp=r.timestamp,
            glucose_mgdl=r.glucose_mgdl,
            glucose_mmol=r.glucose_mmol,
            direction=r.direction,
            source_device_id=r.source_device_id,
            source=r.source,
            device_type=r.device_type,
            ingested_via=r.ingested_via,
            created_at=_now(),
        ))
        saved += 1
    if commit:
        db.commit()
    return saved


def upsert_activity(
    db: Session,
    days: list[ActivityIngest],
    *,
    commit: bool = True,
) -> int:
    """Upsert daily activity by (user_id, calendar_date, provider). Returns # newly inserted.

    Existing rows are updated in place; only non-None incoming fields overwrite (so a
    partial update from one source never wipes another field).
    """
    new = 0
    for a in days:
        existing = (
            db.query(WearableActivity)
            .filter(
                WearableActivity.user_id_fk == a.user_id,
                WearableActivity.calendar_date == a.calendar_date,
                WearableActivity.provider == a.provider,
            )
            .first()
        )
        target = existing or WearableActivity(
            id=str(uuid.uuid4()),
            user_id_fk=a.user_id,
            calendar_date=a.calendar_date,
            provider=a.provider,
            created_at=_now(),
        )
        for field in ACTIVITY_FIELDS:
            val = getattr(a, field)
            if val is not None:
                setattr(target, field, val)
        if existing is None:
            db.add(target)
            new += 1
    if commit:
        db.commit()
    return new
