"""
Per-phase personal model training (runs on the Celery worker).

Trains a user's personal ``while_on_cgm`` / ``post_cgm`` model on their LIVE DB data
using the *exact same* feature build that ``live_inference`` serves — so a personal
model trains and serves on identical features (zero train/serve skew). Uses a
chronological FRACTION split so fractional-day windows (e.g. 9/1.5/1.5) work.

Models are saved under ``tiers[phase]["user/<pk>"]`` (same convention as the existing
research per-user models), so ``live_inference`` picks them up and the user flips from
``collecting`` → ``ready``. Serving is decoupled from training: the graph keeps showing
the prior model / actual CGM the whole time.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.models.tier_trainer import TierTrainer
from src.personalization import lifecycle as lc
from src.serving import live_inference as li
from src.utils import get_logger

log = get_logger(__name__)

# Small, fast, diverse zoo for per-user background training (limited data → keep it lean).
PERSONAL_MODELS = ["lightgbm", "extratrees", "ridge"]
HORIZONS = (30, 60, 90, 120)


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PersonalTierTrainer(TierTrainer):
    """TierTrainer with a chronological fraction split (one continuous user)."""

    def __init__(self, tier: str, horizon_min: int, fractions: tuple[float, float, float], **kw):
        super().__init__(tier, horizon_min, **kw)
        self._fractions = fractions   # (train, val, test), summing ≤ 1.0

    def _split(self, table: pd.DataFrame):
        df = table.sort_index()
        n = len(df)
        ftr, fva, fte = self._fractions
        i_tr, i_va = int(n * ftr), int(n * (ftr + fva))
        train, val = df.iloc[:i_tr], df.iloc[i_tr:i_va]
        test = df.iloc[i_va:] if fte > 0 else df.iloc[i_va:i_va]   # empty test → eval on val
        return train, val, test


def _fractions_for(phase: str, n_days: Optional[float]) -> tuple[float, float, float]:
    if phase == "while_on_cgm":
        return (0.75, 0.25, 0.0)          # no held-out test; live CGM is the test
    split = lc.post_cgm_split(n_days if n_days is not None else 0.0)
    if split is None:
        raise RuntimeError(
            f"insufficient data for post_cgm personal model "
            f"({n_days}d < {lc.POST_CGM_MIN_DAYS}d minimum)")
    tr, va, te = split
    tot = tr + va + te
    return (tr / tot, va / tot, te / tot)


def train_personal(db, user, phase: str, *, n_days: Optional[float] = None,
                   models_dir=None, reports_dir=None,
                   model_names=None, horizons=HORIZONS) -> dict:
    """Train + register the user's personal model for a phase. Returns a summary dict.

    Raises if there's no data or every horizon fails. Writes to
    ``tiers[phase]["user/<user.id>"]`` (+ per-slot leaderboards).
    """
    model_names = model_names or PERSONAL_MODELS
    scope = li.personal_scope(user.id)
    mode = "cgm_active" if phase == "while_on_cgm" else "post_cgm"

    frame = li.build_live_frame(db, user)
    if frame is None:
        raise RuntimeError(f"user {user.id}: no CGM data to train on")
    table = li.featurize(frame, mode=mode)

    if n_days is None and len(table):
        n_days = (table.index.max() - table.index.min()).total_seconds() / 86400.0
    fractions = _fractions_for(phase, n_days)

    kw = {}
    if models_dir is not None:
        kw["models_dir"] = models_dir
    if reports_dir is not None:
        kw["reports_dir"] = reports_dir

    trained: dict[int, dict] = {}
    for h in horizons:
        try:
            trainer = PersonalTierTrainer(phase, h, fractions, **kw)
            winner, _ = trainer.select_best(table, model_names, scope=scope, save=True)
            if winner is not None:
                trained[h] = {"model": winner.model_name,
                              "val_rmse": round(winner.val_rmse, 2),
                              "clarke_a": round(winner.clarke_a, 1)}
        except Exception as exc:                        # noqa: BLE001
            log.warning(f"[personal {phase}/{scope}/{h}min] failed: {exc}")

    if not trained:
        raise RuntimeError(f"user {user.id}: all horizons failed to train ({phase})")

    log.info(f"personal model trained: {scope} {phase} horizons={list(trained)} "
             f"(split {tuple(round(f,3) for f in fractions)})")
    return {"scope": scope, "phase": phase, "n_days": round(n_days or 0, 1), "horizons": trained}


# ── Celery task + scheduler ──────────────────────────────────────────────────

def _set(db, job_id: str, **kw) -> None:
    from src.db.models import RetrainJob
    job = db.query(RetrainJob).filter(RetrainJob.id == job_id).first()
    if job is not None:
        for k, v in kw.items():
            setattr(job, k, v)
        db.commit()


try:
    from src.tasks.celery_app import celery_app

    @celery_app.task(bind=True, name="src.personalization.training.train_personal_task",
                     max_retries=1, default_retry_delay=120)
    def train_personal_task(self, job_id: str, user_pk: str, phase: str) -> dict:
        from src.db.session import SessionLocal
        from src.db.models import User
        db = SessionLocal()
        try:
            _set(db, job_id, status="running", started_at=_utcnow(), celery_id=self.request.id)
            user = db.query(User).filter(User.id == user_pk).first()
            if user is None:
                raise RuntimeError(f"user {user_pk} not found")
            sess = lc.latest_session(db, user_pk)
            summary = train_personal(db, user, phase, n_days=(sess.n_days if sess else None))
            _set(db, job_id, status="done", completed_at=_utcnow())
            return summary
        except Exception as exc:                        # noqa: BLE001
            log.error(f"personal-train job {job_id} failed: {exc}")
            _set(db, job_id, status="failed", completed_at=_utcnow(), error_msg=str(exc)[:500])
            raise
        finally:
            db.close()

    @celery_app.task(name="src.personalization.training.personalization_tick")
    def personalization_tick() -> dict:
        """Periodic: close stale sessions + enqueue personal training at the thresholds."""
        from src.db.session import SessionLocal
        from src.db.models import User
        db = SessionLocal()
        enq = 0
        try:
            lc.end_stale_sessions(db)
            for user in db.query(User).all():
                enq += _maybe_enqueue(db, user)
        finally:
            db.close()
        return {"enqueued": enq}

except Exception as _exc:                               # noqa: BLE001 - Celery optional at import
    log.debug(f"Celery tasks not registered (broker/app unavailable): {_exc}")
    train_personal_task = None                          # type: ignore
    personalization_tick = None                         # type: ignore


def _pending(db, user_id: str, phase: str) -> bool:
    from src.db.models import RetrainJob
    return db.query(RetrainJob).filter(
        RetrainJob.user_id_fk == user_id, RetrainJob.phase == phase,
        RetrainJob.status.in_(["pending", "running"])).first() is not None


def enqueue_personal(db, user, phase: str, session) -> None:
    """Create a pending RetrainJob + dispatch the training task (idempotent per phase)."""
    from src.db.models import RetrainJob
    job = RetrainJob(id=_uuid(), user_id_fk=user.id, dataset=(user.dataset or "live"),
                     horizon="all", phase=phase, session_id=(session.id if session else None),
                     status="pending", triggered_by="lifecycle")
    db.add(job)
    db.commit()
    if train_personal_task is not None:
        try:
            train_personal_task.delay(job.id, user.id, phase)
        except Exception:                               # noqa: BLE001 - broker down; worker sweeps later
            pass


def _has_journey_watch(db, user_id: str, sess) -> bool:
    """Did the user wear the watch enough during the CGM journey? (training gate — product rule).

    Coverage = fraction of journey days with ≥1 HR sample ≥ eligibility.WATCH_MIN_COVERAGE (0.30).
    No watch during the journey → no personal model is ever trained.
    """
    import math
    from src.db.models import WearableSample
    from src.data.eligibility import WATCH_MIN_COVERAGE
    try:
        rows = (db.query(WearableSample.timestamp)
                .filter(WearableSample.user_id_fk == user_id,
                        WearableSample.hr_bpm.isnot(None),
                        WearableSample.timestamp >= sess.started_at).all())
    except Exception:                                    # noqa: BLE001 - table may be unmigrated
        db.rollback()
        return False
    if not rows:
        return False
    days_with = {lc._as_utc(r[0]).date() for r in rows}
    journey_days = max(1, math.ceil(sess.n_days or 1))
    return len(days_with) / journey_days >= WATCH_MIN_COVERAGE


def _maybe_enqueue(db, user) -> int:
    """Enqueue any personal training this user is now due for. Returns # enqueued."""
    sess = lc.latest_session(db, user.id)
    if sess is None:
        return 0
    # WATCH GATE (product rule): no watch worn during the journey → no personal training, ever.
    if not _has_journey_watch(db, user.id, sess):
        return 0
    n = 0
    if sess.status == "active":
        # Day ~8 → personal while_on_cgm.
        if (sess.n_days >= lc.WHILE_ON_CGM_MIN_DAYS
                and not li.has_personal_model(user.id, "while_on_cgm")
                and not _pending(db, user.id, "while_on_cgm")):
            enqueue_personal(db, user, "while_on_cgm", sess); n += 1
        # Day ~13 → pre-emptive personal post_cgm (zero-gap at sensor end).
        if (sess.n_days >= lc.POST_CGM_PRETRAIN_DAY
                and not li.has_personal_model(user.id, "post_cgm")
                and not _pending(db, user.id, "post_cgm")):
            enqueue_personal(db, user, "post_cgm", sess); n += 1
    else:
        # Ended short journey (≥ ~12d) with no post_cgm yet → train from what we have.
        if (sess.n_days >= lc.POST_CGM_MIN_DAYS
                and not li.has_personal_model(user.id, "post_cgm")
                and not _pending(db, user.id, "post_cgm")):
            enqueue_personal(db, user, "post_cgm", sess); n += 1
    return n
