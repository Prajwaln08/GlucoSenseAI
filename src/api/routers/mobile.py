"""
Mobile-facing endpoints for the React Native app: profile, the dashboard glucose
timeseries, and vitals logging. Auth via the same JWT bearer as the rest of the API.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db import crud
from src.db.models import User, Vitals
from src.integrations.ingest import ingest_cgm_readings, upsert_activity
from src.integrations.schemas import ActivityIngest, CgmReadingIngest
from src.utils import get_logger

router = APIRouter(tags=["mobile"])
log = get_logger(__name__)


# ── Profile ───────────────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
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
        "email": u.email, "name": u.name, "age": u.age, "gender": u.gender,
        "height_cm": u.height_cm, "weight_kg": u.weight_kg, "bmi": u.bmi,
        "bp_systolic": u.bp_systolic, "bp_diastolic": u.bp_diastolic,
        "bp_recorded_at": u.bp_recorded_at.isoformat() if u.bp_recorded_at else None,
        "hba1c": u.hba1c, "diabetes_type": u.diabetes_type,
        "medical_history": u.medical_history, "medications": u.medications,
        "onboarding_complete": u.onboarding_complete,
    }


@router.get("/me/profile")
def get_profile(user: User = Depends(get_current_user)):
    return _profile_dict(user)


@router.put("/me/profile")
def update_profile(body: ProfileUpdate, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(user, k, v)
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
    # Demo/replay accounts (linked to a research participant) → model-backed series.
    if user.dataset and user.user_id:
        from src.serving.tier_inference import timeseries
        try:
            return timeseries(user.dataset, user.user_id, hours=hours)
        except Exception as exc:                       # noqa: BLE001
            log.warning(f"timeseries failed for {user.dataset}/{user.user_id}: {exc}")
    # Real users → raw from their own CGM stream (predictions arrive once Health
    # Connect / a CGM is connected in Phase 3).
    readings = crud.get_recent_cgm_readings(db, user.id, limit=hours * 6)
    readings = sorted(readings, key=lambda r: r.timestamp)   # chronological for the chart
    now = readings[-1].timestamp.isoformat() if readings else None
    return {"now": now, "range": {"low": 70, "high": 180},
            "raw": [{"t": r.timestamp.isoformat(), "mgdl": r.glucose_mgdl} for r in readings],
            "predicted": []}


# ── Vitals logging ────────────────────────────────────────────────────────────

class VitalIn(BaseModel):
    kind: str                       # "bp" | "weight" | "glucose" | "hba1c"
    value: Optional[float] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    source: str = "home"            # "home" | "chat"


@router.post("/vitals", status_code=201)
def add_vital(body: VitalIn, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    v = Vitals(user_id_fk=user.id, kind=body.kind, value=body.value,
               bp_systolic=body.bp_systolic, bp_diastolic=body.bp_diastolic,
               source=body.source)
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

class HealthConnectSync(BaseModel):
    glucose: list[HCGlucose] = []
    activity: list[HCActivity] = []


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
    readings = [CgmReadingIngest(
        user_id=user.id, timestamp=_parse_instant(g.t), glucose_mgdl=float(g.mgdl),
        source="health_connect", ingested_via="push", device_type="cgm",
    ) for g in body.glucose]
    n_cgm = ingest_cgm_readings(db, readings)

    days = [ActivityIngest(
        user_id=user.id, calendar_date=a.date, provider="health_connect",
        steps=a.steps, calories_active=a.calories_active, distance_m=a.distance_m,
        hr_avg_bpm=a.hr_avg_bpm, hr_resting_bpm=a.hr_resting_bpm,
        spo2_avg=a.spo2_avg, sleep_hours=a.sleep_hours,
    ) for a in body.activity]
    n_act = upsert_activity(db, days)

    log.info(f"health-connect sync user={user.id}: +{n_cgm} cgm, {n_act} activity days")
    return {"cgm_inserted": n_cgm, "activity_days": n_act}
