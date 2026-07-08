"""
Mobile-facing endpoints for the React Native app: profile, the dashboard glucose
timeseries, and vitals logging. Auth via the same JWT bearer as the rest of the API.
"""

from datetime import date, datetime, time, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db import crud
from src.db.models import CgmReading, FoodLog, User, Vitals, WearableActivity
from src.integrations.ingest import ingest_cgm_readings, upsert_activity
from src.integrations.schemas import ActivityIngest, CgmReadingIngest
from src.utils import get_logger

router = APIRouter(tags=["mobile"])
log = get_logger(__name__)


# ── Profile ───────────────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    hba1c: Optional[float] = None
    diabetes_type: Optional[str] = None
    medical_history: Optional[str] = None
    medications: Optional[str] = None


def _profile_dict(u: User) -> dict:
    return {
        "email": u.email, "name": u.name,
        "first_name": u.first_name, "last_name": u.last_name,
        "date_of_birth": u.date_of_birth.isoformat() if u.date_of_birth else None,
        "age": u.age, "gender": u.gender,
        "height_cm": u.height_cm, "weight_kg": u.weight_kg, "bmi": u.bmi,
        "bp_systolic": u.bp_systolic, "bp_diastolic": u.bp_diastolic,
        "bp_recorded_at": u.bp_recorded_at.isoformat() if u.bp_recorded_at else None,
        "hba1c": u.hba1c, "diabetes_type": u.diabetes_type,
        "medical_history": u.medical_history, "medications": u.medications,
        "onboarding_complete": u.onboarding_complete,
    }


def _age_from_dob(dob: date) -> int:
    t = date.today()
    return t.year - dob.year - ((t.month, t.day) < (dob.month, dob.day))


@router.get("/me/profile")
def get_profile(user: User = Depends(get_current_user)):
    return _profile_dict(user)


@router.put("/me/profile")
def update_profile(body: ProfileUpdate, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(user, k, v)
    # keep derived fields in sync: name = first + last, age = from DOB
    if body.first_name is not None or body.last_name is not None:
        full = f"{(user.first_name or '').strip()} {(user.last_name or '').strip()}".strip()
        if full:
            user.name = full
    if body.date_of_birth is not None:
        user.age = _age_from_dob(body.date_of_birth)
    if body.bp_systolic is not None or body.bp_diastolic is not None:
        user.bp_recorded_at = datetime.now(timezone.utc)
    if user.height_cm and user.weight_kg:
        user.bmi = round(user.weight_kg / (user.height_cm / 100) ** 2, 1)
    user.onboarding_complete = True
    db.commit()
    db.refresh(user)
    return _profile_dict(user)


# ── Glucose timeseries (dashboard graph) ──────────────────────────────────────

@router.get("/glucose/timeseries")
def glucose_timeseries(hours: int = 6, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    # ALL accounts → real, watch-gated serving from live DB data. No dataset/replay shortcut:
    # predictions only ever come from the user's actual CGM + watch, subject to the watch-gate.
    # (Dataset linkage is used for model TRAINING/research only, never for serving canned data.)
    from src.serving.live_inference import timeseries_for_user
    try:
        return timeseries_for_user(db, user, hours=hours)
    except Exception as exc:                           # noqa: BLE001
        log.warning(f"live timeseries failed for user={user.id}: {exc}")
        db.rollback()   # clear any poisoned txn (e.g. cgm_sessions not migrated) before fallback
        # Safe fallback: raw CGM only, never a fake number.
        readings = sorted(crud.get_recent_cgm_readings(db, user.id, limit=hours * 12),
                          key=lambda r: r.timestamp)
        now = readings[-1].timestamp.isoformat() if readings else None
        return {"now": now, "range": {"low": 70, "high": 180},
                "raw": [{"t": r.timestamp.isoformat(), "mgdl": r.glucose_mgdl} for r in readings],
                "predicted": [], "status": "no_data" if readings else "no_source"}


# ── Vitals logging ────────────────────────────────────────────────────────────

class VitalIn(BaseModel):
    kind: str                       # "bp" | "weight" | "glucose" | "hba1c"
    value: Optional[float] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    source: str = "home"            # "home" | "chat"
    recorded_at: Optional[datetime] = None   # omit → server now()


@router.post("/vitals", status_code=201)
def add_vital(body: VitalIn, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    v = Vitals(user_id_fk=user.id, kind=body.kind, value=body.value,
               bp_systolic=body.bp_systolic, bp_diastolic=body.bp_diastolic,
               source=body.source)
    if body.recorded_at is not None:
        v.recorded_at = body.recorded_at
    db.add(v)
    db.commit()
    db.refresh(v)
    return {"id": v.id, "kind": v.kind, "recorded_at": v.recorded_at.isoformat()}


@router.get("/vitals")
def list_vitals(limit: int = 20, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    rows = (db.query(Vitals).filter(Vitals.user_id_fk == user.id)
            .order_by(Vitals.recorded_at.desc()).limit(limit).all())
    return [{
        "id": v.id, "kind": v.kind, "value": v.value,
        "bp_systolic": v.bp_systolic, "bp_diastolic": v.bp_diastolic,
        "source": v.source, "recorded_at": v.recorded_at.isoformat(),
    } for v in rows]


# ── Daily logs (Profile → Logs calendar) ──────────────────────────────────────

@router.get("/me/logs")
def my_logs(date_str: str, user: User = Depends(get_current_user),
            db: Session = Depends(get_db)):
    """Food + vitals + activity for one calendar day (date_str = YYYY-MM-DD, UTC)."""
    day = datetime.strptime(date_str, "%Y-%m-%d").date()
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=timezone.utc)

    foods = (db.query(FoodLog)
             .filter(FoodLog.user_id_fk == user.id, FoodLog.logged_at >= start, FoodLog.logged_at <= end)
             .order_by(FoodLog.logged_at.desc()).all())
    vitals = (db.query(Vitals)
              .filter(Vitals.user_id_fk == user.id, Vitals.recorded_at >= start, Vitals.recorded_at <= end)
              .order_by(Vitals.recorded_at.desc()).all())
    act = (db.query(WearableActivity)
           .filter(WearableActivity.user_id_fk == user.id, WearableActivity.calendar_date == date_str)
           .first())

    # CGM glucose for the day → compact summary (a day can be ~288 readings)
    gvals = [r.glucose_mgdl for r in db.query(CgmReading)
             .filter(CgmReading.user_id == user.id, CgmReading.timestamp >= start, CgmReading.timestamp <= end)
             .all() if r.glucose_mgdl is not None]
    glucose = None if not gvals else {
        "count": len(gvals), "avg": round(sum(gvals) / len(gvals), 1),
        "min": round(min(gvals), 1), "max": round(max(gvals), 1),
        "time_in_range_pct": round(100 * sum(1 for g in gvals if 70 <= g <= 180) / len(gvals)),
    }

    return {
        "date": date_str,
        "glucose": glucose,
        "food": [{
            "id": f.id, "meal_type": f.meal_type, "description": f.description,
            "quantity": f.quantity, "portion_size": f.portion_size,
            "logged_at": f.logged_at.isoformat() if f.logged_at else None,
        } for f in foods],
        "vitals": [{
            "id": v.id, "kind": v.kind, "value": v.value,
            "bp_systolic": v.bp_systolic, "bp_diastolic": v.bp_diastolic,
            "recorded_at": v.recorded_at.isoformat() if v.recorded_at else None,
        } for v in vitals],
        "activity": None if act is None else {
            "steps": act.steps, "hr_avg_bpm": act.hr_avg_bpm,
            "calories_active": act.calories_active,
            "sleep_hours": act.sleep_hours, "spo2_avg": act.spo2_avg,
            "distance_m": act.distance_m,
        },
    }


# ── Health Connect sync (Android on-device → backend) ─────────────────────────

class HCGlucose(BaseModel):
    t: str                          # ISO 8601 instant
    mgdl: float

class HCActivity(BaseModel):
    date: str                       # "YYYY-MM-DD"
    steps: Optional[int] = None
    calories_active: Optional[float] = None
    distance_m: Optional[float] = None
    hr_avg_bpm: Optional[float] = None
    hr_resting_bpm: Optional[float] = None
    spo2_avg: Optional[float] = None
    sleep_hours: Optional[float] = None

class HCSample(BaseModel):
    t: str                          # ISO 8601 instant
    hr_bpm: Optional[float] = None
    spo2_pct: Optional[float] = None
    steps: Optional[int] = None
    calories_active: Optional[float] = None
    distance_m: Optional[float] = None

class HealthConnectSync(BaseModel):
    glucose: list[HCGlucose] = []
    activity: list[HCActivity] = []
    samples: list[HCSample] = []    # intraday (realtime) watch samples → wearable_samples


def _parse_instant(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@router.post("/health-connect/sync")
def health_connect_sync(body: HealthConnectSync,
                        user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Ingest glucose + daily activity the app read from Android Health Connect.

    Goes through the shared dedup ingest path, so re-syncing the same window is
    idempotent (CGM deduped by (user, timestamp); activity upserted by day).
    """
    from src.integrations.ingest import with_serialization_retry

    readings = [CgmReadingIngest(
        user_id=user.id, timestamp=_parse_instant(g.t), glucose_mgdl=float(g.mgdl),
        source="health_connect", ingested_via="push", device_type="cgm",
    ) for g in body.glucose]
    n_cgm = with_serialization_retry(ingest_cgm_readings, db, readings)

    days = [ActivityIngest(
        user_id=user.id, calendar_date=a.date, provider="health_connect",
        steps=a.steps, calories_active=a.calories_active, distance_m=a.distance_m,
        hr_avg_bpm=a.hr_avg_bpm, hr_resting_bpm=a.hr_resting_bpm,
        spo2_avg=a.spo2_avg, sleep_hours=a.sleep_hours,
    ) for a in body.activity]
    n_act = with_serialization_retry(upsert_activity, db, days)

    # Intraday (realtime) samples → wearable_samples (drives real HR features).
    # Non-fatal: if the table isn't migrated yet, the daily activity sync above still succeeds.
    n_samples = 0
    try:
        from src.integrations.ingest import ingest_wearable_samples
        sample_rows = [{
            "timestamp": _parse_instant(s.t), "hr_bpm": s.hr_bpm, "spo2_pct": s.spo2_pct,
            "steps": s.steps, "calories_active": s.calories_active, "distance_m": s.distance_m,
        } for s in body.samples]
        n_samples = with_serialization_retry(ingest_wearable_samples, db, user.id, sample_rows)
    except Exception as exc:                           # noqa: BLE001
        db.rollback()
        log.warning(f"wearable samples ingest skipped (table not migrated?): {exc}")

    log.info(f"health-connect sync user={user.id}: +{n_cgm} cgm, {n_act} activity days, +{n_samples} samples")
    return {"cgm_inserted": n_cgm, "activity_days": n_act, "samples_inserted": n_samples}


@router.get("/wearable/recent")
def wearable_recent(limit: int = 5, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """Latest intraday watch samples — proves realtime data is flowing to the DB."""
    from src.db.models import WearableSample
    try:
        rows = (db.query(WearableSample)
                .filter(WearableSample.user_id_fk == user.id)
                .order_by(WearableSample.timestamp.desc()).limit(min(limit, 50)).all())
    except Exception:                                  # noqa: BLE001 - table may not be migrated yet
        db.rollback()
        return []
    return [{
        "t": r.timestamp.isoformat(), "hr_bpm": r.hr_bpm, "spo2_pct": r.spo2_pct,
        "steps": r.steps, "calories_active": r.calories_active, "distance_m": r.distance_m,
    } for r in rows]


# The raw signal columns of the model's 10-min input grid, in display order.
_MODEL_INPUT_COLS = [
    "glucose_mg_dl", "hr", "calories_burned", "mets",
    "calorie", "total_carb", "protein", "total_fat", "dietary_fiber", "sugar",
    "meal_type_encoded",
]


@router.get("/me/model-inputs")
def model_inputs(hours: int = 3, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """Transparency view: the per-timestamp raw signals on the model's 10-min grid.

    Exactly what enters the feature pipeline (before lags/rolls/imputation), so the
    user can see which of their data feeds each forecast step.
    """
    import math

    import pandas as pd

    from src.data.step1_loader import LoadedUser
    from src.data.step2_clinical import build_grid_table
    from src.serving.live_inference import build_live_frame

    frame = build_live_frame(db, user, require_cgm=False)
    if frame is None:
        return {"step_minutes": 10, "columns": [], "rows": []}

    lu = LoadedUser(uid="live", dataset="cgmacros", participant_id="live",
                    demographics={}, frame=frame, base_dir=None)
    try:
        grid = build_grid_table([lu])
    except Exception as exc:                           # noqa: BLE001 - not enough data to grid
        log.warning(f"model-inputs grid failed for user={user.id}: {exc}")
        return {"step_minutes": 10, "columns": [], "rows": []}

    cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=min(max(hours, 1), 48))
    idx = grid.index
    if getattr(idx, "tz", None) is None:
        cutoff = cutoff.replace(tzinfo=None)
    grid = grid[idx >= cutoff]

    cols = [c for c in _MODEL_INPUT_COLS if c in grid.columns]

    def _clean(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else round(f, 2)

    rows = [{"t": ts.isoformat(), **{c: _clean(row[c]) for c in cols}}
            for ts, row in grid[cols].iterrows()]
    return {"step_minutes": 10, "columns": cols, "rows": rows}
