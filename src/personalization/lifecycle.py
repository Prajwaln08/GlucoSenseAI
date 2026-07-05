"""
CGM sensor-session lifecycle + personalization phase logic.

A user's forecast **phase** is driven by their CGM stream:

    CGM_COLLECTING ──▶ CGM_PERSONAL_TRAINING ──▶ CGM_PERSONAL
                                                     │ day 13 (pre-emptive)
                                                     ▼
    POST_CGM_PERSONAL ◀── POST_CGM_TRAINING

Population models are the BASE model (no-CGM / Basic mode) only — never served in the CGM
flow. Until a user's personal model exists, the graph shows ACTUAL CGM values, no prediction.
See docs/personalization_lifecycle_plan.md.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import CgmSession

# ── Lifecycle thresholds (days, unless noted) ────────────────────────────────
SESSION_MAX_DAYS      = 14.0   # sensor lifespan → session hard-ends
SENSOR_SILENCE_HOURS  = 2.0    # gap between readings → the session is considered ended
WHILE_ON_CGM_MIN_DAYS = 8.0    # train the personal while_on_cgm model at/after this many days
POST_CGM_PRETRAIN_DAY = 13.0   # pre-emptively train the personal post_cgm model here (zero-gap)
POST_CGM_MIN_DAYS     = 12.0   # minimum data to train a post_cgm model on a short journey

# ── Watch-gate: watch (HR) data is MANDATORY for any forecast + any training ──
# Serving gate (recent window). Mirrors the HR-continuity gate in serving/inference.py.
WATCH_MIN_HR_READINGS  = 8      # min HR points in the recent window to allow a forecast
WATCH_FULL_HR_READINGS = 16     # full-fidelity; 3h window at the watch's ~10-min HC cadence
                                # tops out at ~18 points, so 24 was unreachable
WATCH_HR_GAP_LIMIT_MIN = 30.0   # max gap between consecutive HR readings → watch is "flowing"
WATCH_WINDOW_HOURS     = 3.0    # recent lookback window (matches the glucose n_lags depth)
# Training gate reuses eligibility.WATCH_MIN_COVERAGE (0.30) over the journey.

# Phase labels (also used by the API/serving/scheduler).
NO_SOURCE             = "NO_SOURCE"
CGM_COLLECTING        = "CGM_COLLECTING"
CGM_PERSONAL_TRAINING = "CGM_PERSONAL_TRAINING"
CGM_PERSONAL          = "CGM_PERSONAL"
POST_CGM_TRAINING     = "POST_CGM_TRAINING"
POST_CGM_PERSONAL     = "POST_CGM_PERSONAL"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Coerce a possibly-naive datetime (SQLite returns naive) to aware UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _span_days(a: datetime, b: datetime) -> float:
    return max(0.0, (_as_utc(b) - _as_utc(a)).total_seconds() / 86400.0)


def active_session(db: Session, user_id: str) -> Optional[CgmSession]:
    return (
        db.query(CgmSession)
        .filter(CgmSession.user_id_fk == user_id, CgmSession.status == "active")
        .order_by(CgmSession.last_reading_at.desc())
        .first()
    )


def latest_session(db: Session, user_id: str) -> Optional[CgmSession]:
    return (
        db.query(CgmSession)
        .filter(CgmSession.user_id_fk == user_id)
        .order_by(CgmSession.last_reading_at.desc())
        .first()
    )


def _end(session: CgmSession, at: datetime, reason: str) -> None:
    session.status = "ended"
    session.ended_at = _as_utc(at)
    session.end_reason = reason


def update_session_on_ingest(
    db: Session,
    user_id: str,
    timestamps: list[datetime],
    *,
    commit: bool = True,
) -> Optional[CgmSession]:
    """
    Maintain the user's CGM session after new readings land.

    - Close the active session if it went silent (> SENSOR_SILENCE_HOURS) before these readings.
    - Start a new session if none is active.
    - Otherwise extend the active session (last_reading_at, n_readings, n_days).
    - Hard-close the session once it spans SESSION_MAX_DAYS.

    Call with the *newly-inserted* timestamps of a sync batch. Returns the current
    (active or just-ended) session, or None if no timestamps.
    """
    ts = sorted(t for t in (_as_utc(x) for x in timestamps) if t is not None)
    if not ts:
        return None
    # Split at internal silence gaps: only the most-recent contiguous run belongs to the
    # CURRENT sensor session (earlier runs are a prior/historical session), so a
    # gap-straddling backfill batch doesn't inflate this session's span/n_days.
    seg_start = 0
    for i in range(1, len(ts)):
        if (ts[i] - ts[i - 1]).total_seconds() / 3600.0 > SENSOR_SILENCE_HOURS:
            seg_start = i
    ts = ts[seg_start:]
    earliest, latest = ts[0], ts[-1]

    sess = active_session(db, user_id)

    if sess is not None:
        gap_h = (earliest - _as_utc(sess.last_reading_at)).total_seconds() / 3600.0
        if gap_h > SENSOR_SILENCE_HOURS:
            _end(sess, sess.last_reading_at, "silent")
            sess = None

    if sess is None:
        sess = CgmSession(
            id=str(uuid.uuid4()),
            user_id_fk=user_id,
            started_at=earliest,
            first_reading_at=earliest,
            last_reading_at=latest,
            n_readings=len(ts),
            n_days=_span_days(earliest, latest),
            status="active",
            created_at=_now(),
        )
        db.add(sess)
    else:
        if latest > _as_utc(sess.last_reading_at):
            sess.last_reading_at = latest
        if earliest < _as_utc(sess.first_reading_at):   # backfill earlier than session start
            sess.first_reading_at = earliest
        sess.n_readings = (sess.n_readings or 0) + len(ts)
        sess.n_days = _span_days(sess.first_reading_at, sess.last_reading_at)

    if sess.n_days >= SESSION_MAX_DAYS and sess.status == "active":
        _end(sess, sess.last_reading_at, "expired")

    if commit:
        db.commit()
    return sess


def end_stale_sessions(db: Session, *, now: Optional[datetime] = None, commit: bool = True) -> int:
    """Close active sessions whose last reading is older than the silence window.

    Called by the periodic scheduler so a sensor that simply stops reporting (no new ingest
    to trigger ``update_session_on_ingest``) still transitions out of the CGM flow.
    Returns the number of sessions closed.
    """
    ref = _as_utc(now) or _now()
    closed = 0
    for sess in db.query(CgmSession).filter(CgmSession.status == "active").all():
        gap_h = (ref - _as_utc(sess.last_reading_at)).total_seconds() / 3600.0
        if gap_h > SENSOR_SILENCE_HOURS or (sess.n_days or 0.0) >= SESSION_MAX_DAYS:
            _end(sess, sess.last_reading_at,
                 "silent" if gap_h > SENSOR_SILENCE_HOURS else "expired")
            closed += 1
    if commit and closed:
        db.commit()
    return closed


def watch_readiness(db: Session, user_id: str, *, window_hours: float = WATCH_WINDOW_HOURS,
                    now: Optional[datetime] = None) -> dict:
    """Is watch (HR) data flowing for the recent window? Gates ALL forecasts (product rule).

    Ready requires ≥ WATCH_MIN_HR_READINGS intraday HR readings within ``window_hours``, with no
    gap > WATCH_HR_GAP_LIMIT_MIN AND a fresh latest reading. Returns
    ``{ready, full, have, need, gap_ok, last_at}``. Non-fatal if the samples table is unmigrated.
    """
    from src.db.models import WearableSample
    ref = _as_utc(now) or _now()
    cutoff = ref - timedelta(hours=window_hours)
    try:
        rows = (db.query(WearableSample.timestamp)
                .filter(WearableSample.user_id_fk == user_id,
                        WearableSample.hr_bpm.isnot(None),
                        WearableSample.timestamp >= cutoff)
                .order_by(WearableSample.timestamp).all())
    except Exception:                                    # noqa: BLE001 - table may be unmigrated
        db.rollback()
        rows = []
    ts = [_as_utc(r[0]) for r in rows]
    have = len(ts)
    # No gap > limit between consecutive readings...
    gap_ok = all((b - a).total_seconds() / 60.0 <= WATCH_HR_GAP_LIMIT_MIN
                 for a, b in zip(ts, ts[1:]))
    # ...and the latest reading is fresh (watch is actually still on).
    fresh = have >= 1 and (ref - ts[-1]).total_seconds() / 60.0 <= WATCH_HR_GAP_LIMIT_MIN
    ready = have >= WATCH_MIN_HR_READINGS and gap_ok and fresh
    return {"ready": bool(ready), "full": have >= WATCH_FULL_HR_READINGS,
            "have": have, "need": WATCH_MIN_HR_READINGS,
            "gap_ok": bool(gap_ok and fresh),
            "last_at": ts[-1].isoformat() if ts else None}


def resolve_phase(db: Session, user, has_personal) -> str:
    """Current forecast phase for a real user.

    ``has_personal`` is a callable ``(phase_name) -> bool`` that checks whether a servable
    personal model exists (reads the ``registry["personal"]`` namespace). Phase drives what
    the Home graph shows (actual CGM only vs a personal forecast) — never a population guess.
    """
    sess = latest_session(db, user.id)
    if sess is None:
        return NO_SOURCE
    if sess.status == "active":
        if has_personal("while_on_cgm"):
            return CGM_PERSONAL
        if (sess.n_days or 0.0) >= WHILE_ON_CGM_MIN_DAYS:
            return CGM_PERSONAL_TRAINING
        return CGM_COLLECTING
    # Session ended → post-CGM gap.
    if has_personal("post_cgm"):
        return POST_CGM_PERSONAL
    return POST_CGM_TRAINING


def post_cgm_split(n_days: float) -> Optional[tuple[float, float, float]]:
    """(train, val, test) day split for a post_cgm model given the journey length.

    Full journey → 10/2/2; shorter journeys scale ~75/12.5/12.5 (e.g. 12d → 9/1.5/1.5).
    Returns None if there isn't enough data (< POST_CGM_MIN_DAYS) to train.
    """
    if n_days < POST_CGM_MIN_DAYS:
        return None
    if n_days >= SESSION_MAX_DAYS:
        return (10.0, 2.0, 2.0)
    return (round(n_days * 0.75, 2), round(n_days * 0.125, 2), round(n_days * 0.125, 2))
