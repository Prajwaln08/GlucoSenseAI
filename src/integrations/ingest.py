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
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from src.db.models import CgmReading, WearableActivity
from src.integrations.schemas import ACTIVITY_FIELDS, ActivityIngest, CgmReadingIngest
from src.utils.metrics import activity_days_upserted, cgm_readings_ingested


def _now() -> datetime:
    return datetime.now(timezone.utc)


def correct_junction_clock(db: Session, user, readings: list) -> list:
    """Apply the user's learned Junction clock offset; ratchet it up when the batch
    proves it's too small (readings stamped in the future are physically impossible).

    Persisting the offset on the user keeps every ingest path — webhook, poll,
    manual sync — on ONE correction basis, so timestamp-dedup stays stable.
    Caller is responsible for the surrounding commit.
    """
    import math
    from dataclasses import replace

    if not readings:
        return readings
    tolerance, quarter = timedelta(minutes=5), 15
    stored_min = int(user.junction_clock_offset_min or 0)
    newest = max(r.timestamp for r in readings)
    lead = (newest - timedelta(minutes=stored_min)) - _now()
    if lead > tolerance:
        stored_min += math.ceil((lead - tolerance) / timedelta(minutes=quarter)) * quarter
        user.junction_clock_offset_min = stored_min
        db.add(user)
    if not stored_min:
        return readings
    off = timedelta(minutes=stored_min)
    return [replace(r, timestamp=r.timestamp - off) for r in readings]


def with_serialization_retry(fn, db, *args, attempts: int = 3, **kwargs):
    """Run an ingest function, retrying CockroachDB serialization aborts (pgcode 40001).

    Cockroach runs SERIALIZABLE and aborts one side of a write-write conflict with a
    retryable error — e.g. a Junction webhook and a Health Connect sync upserting the
    same activity row. The ingest functions are idempotent, so re-running is safe.
    """
    import time as _time

    from sqlalchemy.exc import DBAPIError

    for attempt in range(attempts):
        try:
            return fn(db, *args, **kwargs)
        except DBAPIError as exc:
            retryable = getattr(getattr(exc, "orig", None), "pgcode", None) == "40001"
            db.rollback()
            if retryable and attempt < attempts - 1:
                _time.sleep(0.1 * (attempt + 1))
                continue
            raise


def _ts_key(ts: datetime) -> datetime:
    """Normalise a timestamp for dedup-dict keys: naive-UTC. Postgres returns
    tz-aware rows while SQLite returns naive — without this, comparing incoming
    (aware) stamps against stored ones would silently never match."""
    return ts if ts.tzinfo is None else ts.astimezone(timezone.utc).replace(tzinfo=None)


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
    inserted_by_user: dict[str, list] = {}

    # ONE existence query per user covering the batch window — per-reading SELECTs
    # are fatal against a remote DB (a backfill = thousands of round-trips).
    seen: set[tuple[str, datetime]] = set()
    by_user: dict[str, list] = {}
    for r in readings:
        by_user.setdefault(r.user_id, []).append(r.timestamp)
    for uid, tss in by_user.items():
        existing = (db.query(CgmReading.timestamp)
                    .filter(CgmReading.user_id == uid,
                            CgmReading.timestamp >= min(tss),
                            CgmReading.timestamp <= max(tss)).all())
        seen.update((uid, _ts_key(t[0])) for t in existing)

    for r in readings:
        key = (r.user_id, _ts_key(r.timestamp))
        if key in seen:
            continue
        seen.add(key)                    # dedup within the batch too (first writer wins)
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
        inserted_by_user.setdefault(r.user_id, []).append(r.timestamp)
        cgm_readings_ingested.labels(source=r.source).inc()

    # Maintain the per-user CGM sensor-session lifecycle (drives personalization phases).
    # Non-fatal: a session-tracking hiccup must never block glucose ingestion.
    if inserted_by_user:
        try:
            from src.personalization.lifecycle import update_session_on_ingest
            for uid, tss in inserted_by_user.items():
                update_session_on_ingest(db, uid, tss, commit=False)
        except Exception:  # noqa: BLE001
            pass

    if commit:
        db.commit()
    return saved


def ingest_wearable_samples(
    db: Session,
    user_id: str,
    rows: list[dict],
    *,
    commit: bool = True,
) -> int:
    """Upsert intraday wearable samples by (user_id, timestamp). Returns # new rows.

    Each row is a dict: {"timestamp": datetime, "hr_bpm"?, "spo2_pct"?, "steps"?,
    "calories_active"?, "distance_m"?, "provider"?}. Coinciding metrics merge into the
    same instant; re-syncing the same window is idempotent.
    """
    from src.db.models import WearableSample
    rows = [r for r in rows if r.get("timestamp") is not None]
    if not rows:
        if commit:
            db.commit()
        return 0

    # ONE round-trip fetches every existing row in the batch window — per-sample
    # SELECTs made a 48h sync take minutes against a remote DB (client timed out).
    tss = [r["timestamp"] for r in rows]
    existing_rows = (db.query(WearableSample)
                     .filter(WearableSample.user_id_fk == user_id,
                             WearableSample.timestamp >= min(tss),
                             WearableSample.timestamp <= max(tss)).all())
    by_ts = {_ts_key(e.timestamp): e for e in existing_rows}

    new = 0
    for r in rows:
        key = _ts_key(r["timestamp"])
        existing = by_ts.get(key)
        target = existing or WearableSample(
            id=str(uuid.uuid4()), user_id_fk=user_id, timestamp=r["timestamp"],
            provider=r.get("provider", "health_connect"), created_at=_now(),
        )
        for field in ("hr_bpm", "spo2_pct", "steps", "calories_active", "distance_m"):
            val = r.get(field)
            if val is not None:
                setattr(target, field, val)
        if existing is None:
            db.add(target)
            by_ts[key] = target          # merge duplicate instants within the batch
            new += 1
    if commit:
        db.commit()
    return new


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
            activity_days_upserted.labels(provider=a.provider).inc()
    if commit:
        db.commit()
    return new
