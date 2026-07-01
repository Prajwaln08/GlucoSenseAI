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
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import CgmSession

# ── Lifecycle thresholds (days, unless noted) ────────────────────────────────
SESSION_MAX_DAYS      = 14.0   # sensor lifespan → session hard-ends
SENSOR_SILENCE_HOURS  = 2.0    # gap between readings → the session is considered ended
WHILE_ON_CGM_MIN_DAYS = 8.0    # train the personal while_on_cgm model at/after this many days
POST_CGM_PRETRAIN_DAY = 13.0   # pre-emptively train the personal post_cgm model here (zero-gap)
POST_CGM_MIN_DAYS     = 12.0   # minimum data to train a post_cgm model on a short journey

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
