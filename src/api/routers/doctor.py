"""
Doctor portal endpoints — all routes require a valid JWT from a doctor account.

GET  /doctor/patients
GET  /doctor/patients/{pk}
GET  /doctor/patients/{pk}/predictions
GET  /doctor/patients/{pk}/retrain-jobs
POST /doctor/patients/{pk}/retrain
POST /doctor/patients/{pk}/assign
DELETE /doctor/patients/{pk}/assign
GET  /doctor/patients/{pk}/messages
POST /doctor/patients/{pk}/messages
GET  /doctor/patients/{pk}/model-info     (NEW — tier, readiness, retrain history)
POST /doctor/patients/{pk}/restore-model  (NEW — revert to a past RetrainJob artefact)
GET  /doctor/population-stats             (NEW — cohort risk + all population/virtual slots)
GET  /doctor/retrain-jobs
GET  /doctor/retrain-jobs/{job_id}
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db.models import FoodLog, Message, PredictionLog, RetrainJob, User, WearableActivity

router = APIRouter(prefix="/doctor", tags=["doctor"])


def _require_doctor(user: User) -> User:
    if not user.is_doctor:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor access required.",
        )
    return user


# ── Response schemas ──────────────────────────────────────────────────────────

class PatientSummary(BaseModel):
    id: str
    email: str
    user_id: str
    dataset: str
    name: Optional[str]
    gender: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    diabetes_type: Optional[str]
    medical_history: Optional[str]
    age: Optional[float]
    bmi: Optional[float]
    hba1c: Optional[float]
    assigned_doctor_id: Optional[str]
    is_active: bool
    created_at: datetime
    prediction_count: int
    latest_glucose: Optional[float]
    latest_hypo_risk: Optional[bool]
    latest_hyper_risk: Optional[bool]
    latest_clarke_zone: Optional[str]
    pending_retrain_jobs: int

    model_config = {"from_attributes": True}


class PredictionLogOut(BaseModel):
    id: str
    dataset: str
    horizon: str
    model_type: str
    current_glucose: float
    horizon_glucose: float
    clarke_zone: str
    is_hypo_risk: bool
    is_hyper_risk: bool
    n_readings: int
    predicted_at: datetime
    actual_glucose: Optional[float]
    absolute_error: Optional[float]

    model_config = {"from_attributes": True}


class RetrainJobOut(BaseModel):
    id: str
    celery_id: Optional[str]
    patient_email: str
    patient_user_id: str
    dataset: str
    horizon: str
    status: str
    triggered_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    test_rmse: Optional[float]
    prev_rmse: Optional[float]
    improved: Optional[bool]
    error_msg: Optional[str]
    artefact_dir: Optional[str] = None
    triggered_by: Optional[str] = None
    notes: Optional[str] = None
    version_tag: Optional[str] = None


class ModelSlotInfo(BaseModel):
    dataset: str
    horizon: str
    tier: str
    model_type: str
    test_rmse: Optional[float]
    val_rmse: Optional[float]
    clarke_a_pct: Optional[float]
    trained_at: Optional[str]


class PopulationStatsResponse(BaseModel):
    total_patients: int
    patients_with_predictions: int
    population_models: list[ModelSlotInfo]
    virtual_models: list[ModelSlotInfo]
    individual_model_count: int
    tier_distribution: dict[str, int]
    hypo_risk_count: int
    hyper_risk_count: int


class PatientModelInfoResponse(BaseModel):
    patient_id: str
    user_id: str
    dataset: str
    horizon: str
    current_tier: str
    model_type: str
    test_rmse: Optional[float]
    val_rmse: Optional[float]
    clarke_a_pct: Optional[float]
    trained_at: Optional[str]
    has_individual_model: bool
    individual_beats_population: bool
    individual_test_rmse: Optional[float]
    latest_cgm_mode: Optional[str]
    latest_predicted_at: Optional[datetime]
    latest_retrain_job_id: Optional[str]
    latest_retrain_status: Optional[str]
    latest_retrain_at: Optional[datetime]


class RestoreModelRequest(BaseModel):
    job_id: str = Field(..., description="RetrainJob ID whose artefact_dir to restore into registry")
    horizon: str = "2h"


class RetrainRequest(BaseModel):
    horizon: str = "2h"
    notes: Optional[str] = Field(None, description="Doctor's note on why retrain was triggered")


class RetrainResponse(BaseModel):
    job_id: str
    celery_id: Optional[str]
    status: str


class ChatMessageOut(BaseModel):
    id: str
    sender_id: str
    receiver_id: str
    body: str
    sent_at: datetime
    read_at: Optional[datetime]
    attachment_url: Optional[str]
    attachment_type: Optional[str]
    attachment_name: Optional[str]

    model_config = {"from_attributes": True}


class SendChatRequest(BaseModel):
    body: str = Field(..., min_length=1, max_length=4000)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _patient_summary(patient: User, db: Session) -> PatientSummary:
    count = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id_fk == patient.id)
        .count()
    )
    latest = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id_fk == patient.id)
        .order_by(PredictionLog.predicted_at.desc())
        .first()
    )
    pending_jobs = (
        db.query(RetrainJob)
        .filter(
            RetrainJob.user_id_fk == patient.id,
            RetrainJob.status.in_(["pending", "running"]),
        )
        .count()
    )
    return PatientSummary(
        id=patient.id,
        email=patient.email,
        user_id=patient.user_id,
        dataset=patient.dataset,
        name=patient.name,
        gender=patient.gender,
        height_cm=patient.height_cm,
        weight_kg=patient.weight_kg,
        diabetes_type=patient.diabetes_type,
        medical_history=patient.medical_history,
        age=patient.age,
        bmi=patient.bmi,
        hba1c=patient.hba1c,
        assigned_doctor_id=patient.assigned_doctor_id,
        is_active=patient.is_active,
        created_at=patient.created_at,
        prediction_count=count,
        latest_glucose=latest.current_glucose if latest else None,
        latest_hypo_risk=latest.is_hypo_risk if latest else None,
        latest_hyper_risk=latest.is_hyper_risk if latest else None,
        latest_clarke_zone=latest.clarke_zone if latest else None,
        pending_retrain_jobs=pending_jobs,
    )


def _get_patient_or_404(patient_pk: str, db: Session) -> User:
    patient = db.query(User).filter(User.id == patient_pk).first()
    if not patient or patient.is_doctor:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/patients", response_model=list[PatientSummary])
def list_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PatientSummary]:
    _require_doctor(current_user)
    patients = (
        db.query(User)
        .filter(User.is_doctor.is_(False))
        .order_by(User.created_at.desc())
        .all()
    )
    return [_patient_summary(p, db) for p in patients]


@router.get("/patients/{patient_pk}", response_model=PatientSummary)
def get_patient(
    patient_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PatientSummary:
    _require_doctor(current_user)
    return _patient_summary(_get_patient_or_404(patient_pk, db), db)


@router.post("/patients/{patient_pk}/assign", response_model=PatientSummary)
def assign_doctor(
    patient_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PatientSummary:
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)
    patient.assigned_doctor_id = current_user.id
    db.commit()
    db.refresh(patient)
    return _patient_summary(patient, db)


@router.delete("/patients/{patient_pk}/assign")
def unassign_doctor(
    patient_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)
    patient.assigned_doctor_id = None
    db.commit()
    return Response(status_code=204)


@router.get("/patients/{patient_pk}/predictions", response_model=list[PredictionLogOut])
def patient_predictions(
    patient_pk: str,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PredictionLog]:
    _require_doctor(current_user)
    _get_patient_or_404(patient_pk, db)
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id_fk == patient_pk)
        .order_by(PredictionLog.predicted_at.desc())
        .limit(limit)
        .all()
    )


@router.post("/patients/{patient_pk}/retrain", response_model=RetrainResponse)
def trigger_patient_retrain(
    patient_pk: str,
    req: RetrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RetrainResponse:
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)
    if req.horizon not in ("2h", "3h"):
        raise HTTPException(status_code=422, detail="horizon must be '2h' or '3h'.")

    from src.db.crud import create_retrain_job, update_retrain_job
    from src.tasks.retrain import trigger_retrain

    job = create_retrain_job(
        db, patient.id, patient.dataset, req.horizon,
        triggered_by="doctor",
        notes=req.notes,
    )
    db.commit()
    db.refresh(job)

    celery_id = None
    try:
        task = trigger_retrain.delay(job.id, patient.user_id, patient.dataset, req.horizon)
        celery_id = task.id
        update_retrain_job(db, job.id, status="pending", celery_id=celery_id)
        db.commit()
    except Exception:
        # Redis/broker unavailable — job is persisted in DB and will be picked up
        # when the Celery worker reconnects.
        pass

    return RetrainResponse(job_id=job.id, celery_id=celery_id, status="pending")


@router.get("/patients/{patient_pk}/retrain-jobs", response_model=list[RetrainJobOut])
def patient_retrain_jobs(
    patient_pk: str,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[RetrainJobOut]:
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)
    jobs = (
        db.query(RetrainJob)
        .filter(RetrainJob.user_id_fk == patient_pk)
        .order_by(RetrainJob.triggered_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrainJobOut(
            **{c: getattr(j, c) for c in [
                "id", "celery_id", "dataset", "horizon", "status",
                "triggered_at", "started_at", "completed_at",
                "test_rmse", "prev_rmse", "improved", "error_msg",
                "artefact_dir", "triggered_by", "notes", "version_tag",
            ]},
            patient_email=patient.email,
            patient_user_id=patient.user_id,
        )
        for j in jobs
    ]


@router.get("/patients/{patient_pk}/messages", response_model=list[ChatMessageOut])
def get_patient_messages(
    patient_pk: str,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Message]:
    _require_doctor(current_user)
    _get_patient_or_404(patient_pk, db)
    return (
        db.query(Message)
        .filter(
            (Message.sender_id == patient_pk) | (Message.receiver_id == patient_pk)
        )
        .order_by(Message.sent_at.asc())
        .limit(limit)
        .all()
    )


@router.post("/patients/{patient_pk}/messages", response_model=ChatMessageOut, status_code=201)
def send_patient_message(
    patient_pk: str,
    req: SendChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Message:
    _require_doctor(current_user)
    _get_patient_or_404(patient_pk, db)
    from src.db.crud import send_message
    msg = send_message(db, sender_id=current_user.id, receiver_id=patient_pk, body=req.body.strip())
    db.commit()
    db.refresh(msg)
    return msg


@router.get("/retrain-jobs", response_model=list[RetrainJobOut])
def list_all_retrain_jobs(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[RetrainJobOut]:
    _require_doctor(current_user)
    rows = (
        db.query(RetrainJob, User.email, User.user_id)
        .join(User, User.id == RetrainJob.user_id_fk)
        .order_by(RetrainJob.triggered_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrainJobOut(
            **{c: getattr(j, c) for c in [
                "id", "celery_id", "dataset", "horizon", "status",
                "triggered_at", "started_at", "completed_at",
                "test_rmse", "prev_rmse", "improved", "error_msg",
                "artefact_dir", "triggered_by", "notes", "version_tag",
            ]},
            patient_email=email,
            patient_user_id=uid,
        )
        for j, email, uid in rows
    ]


@router.get("/retrain-jobs/{job_id}", response_model=RetrainJobOut)
def get_retrain_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RetrainJobOut:
    _require_doctor(current_user)
    row = (
        db.query(RetrainJob, User.email, User.user_id)
        .join(User, User.id == RetrainJob.user_id_fk)
        .filter(RetrainJob.id == job_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    j, email, uid = row
    return RetrainJobOut(
        **{c: getattr(j, c) for c in [
            "id", "celery_id", "dataset", "horizon", "status",
            "triggered_at", "started_at", "completed_at",
            "test_rmse", "prev_rmse", "improved", "error_msg",
        ]},
        patient_email=email,
        patient_user_id=uid,
    )


# ── Admin assignment & doctor list ────────────────────────────────────────────

class DoctorInfo(BaseModel):
    id: str
    email: str
    model_config = {"from_attributes": True}


class AssignToRequest(BaseModel):
    doctor_id: str


@router.get("/doctors", response_model=list[DoctorInfo])
def list_doctors(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DoctorInfo]:
    _require_doctor(current_user)
    return db.query(User).filter(User.is_doctor.is_(True)).order_by(User.email).all()


@router.post("/patients/{patient_pk}/assign-to", response_model=PatientSummary)
def assign_doctor_to(
    patient_pk: str,
    req: AssignToRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PatientSummary:
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)
    doctor = db.query(User).filter(
        User.id == req.doctor_id,
        User.is_doctor.is_(True),
    ).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found.")
    patient.assigned_doctor_id = req.doctor_id
    db.commit()
    db.refresh(patient)
    return _patient_summary(patient, db)


# ── Patient food logs & activity (doctor view) ────────────────────────────────

class FoodLogOut(BaseModel):
    id: str
    meal_type: str
    description: Optional[str] = None
    calories: Optional[float] = None
    carbs_g: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sugar_g: Optional[float] = None
    gi_proxy: Optional[float] = None
    quantity: Optional[float] = None
    portion_size: Optional[str] = None
    notes: Optional[str] = None
    logged_at: datetime
    model_config = {"from_attributes": True}


class ActivityOut(BaseModel):
    id: str
    calendar_date: str
    steps: Optional[int] = None
    calories_total: Optional[float] = None
    calories_active: Optional[float] = None
    distance_m: Optional[float] = None
    hr_avg_bpm: Optional[float] = None
    hr_min_bpm: Optional[float] = None
    hr_max_bpm: Optional[float] = None
    hr_resting_bpm: Optional[float] = None
    provider: Optional[str] = None
    model_config = {"from_attributes": True}


@router.get("/patients/{patient_pk}/food-logs", response_model=list[FoodLogOut])
def patient_food_logs(
    patient_pk: str,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[FoodLog]:
    _require_doctor(current_user)
    _get_patient_or_404(patient_pk, db)
    return (
        db.query(FoodLog)
        .filter(FoodLog.user_id_fk == patient_pk)
        .order_by(FoodLog.logged_at.desc())
        .limit(min(limit, 200))
        .all()
    )


@router.get("/patients/{patient_pk}/activity", response_model=list[ActivityOut])
def patient_activity(
    patient_pk: str,
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[WearableActivity]:
    _require_doctor(current_user)
    _get_patient_or_404(patient_pk, db)
    return (
        db.query(WearableActivity)
        .filter(WearableActivity.user_id_fk == patient_pk)
        .order_by(WearableActivity.calendar_date.desc())
        .limit(min(days, 90))
        .all()
    )


# ── Population & model management endpoints ───────────────────────────────────

@router.get("/population-stats", response_model=PopulationStatsResponse)
def population_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PopulationStatsResponse:
    """
    Aggregated population and virtual model metrics from the registry,
    plus cohort-level risk distribution from the latest prediction per patient.
    """
    _require_doctor(current_user)

    from sqlalchemy import func
    from src.serving.model_loader import _read_registry

    reg = _read_registry()

    def _slot_info(dataset: str, horizon: str, tier: str) -> Optional[ModelSlotInfo]:
        slot = reg.get(tier, {}).get(dataset, {}).get(horizon)
        if not slot:
            return None
        return ModelSlotInfo(
            dataset=dataset,
            horizon=horizon,
            tier=tier,
            model_type=slot.get("best_model_type") or slot.get("model_type", "unknown"),
            test_rmse=slot.get("test_rmse"),
            val_rmse=slot.get("val_rmse"),
            clarke_a_pct=slot.get("clarke_a_pct"),
            trained_at=slot.get("trained_at"),
        )

    population_models = [
        s for s in [
            _slot_info(d, h, "population")
            for d in ("cgmacros", "nature_paper") for h in ("2h", "3h")
        ] if s
    ]
    virtual_models = [
        s for s in [
            _slot_info(d, h, "virtual")
            for d in ("cgmacros", "nature_paper") for h in ("2h", "3h")
        ] if s
    ]

    # Count only DB patients that have an individual model in the registry
    db_patients = db.query(User).filter(User.is_doctor.is_(False)).all()
    individual_model_count = sum(
        1
        for patient in db_patients
        for h in ("2h", "3h")
        if reg.get("individual", {})
              .get(patient.dataset, {})
              .get(patient.user_id, {})
              .get(h)
    )

    total_patients = db.query(User).filter(User.is_doctor.is_(False)).count()

    # Latest prediction per patient — for cohort risk and tier distribution
    max_ts_subq = (
        db.query(
            PredictionLog.user_id_fk,
            func.max(PredictionLog.predicted_at).label("max_ts"),
        )
        .group_by(PredictionLog.user_id_fk)
        .subquery()
    )
    latest_preds = (
        db.query(PredictionLog)
        .join(
            max_ts_subq,
            (PredictionLog.user_id_fk == max_ts_subq.c.user_id_fk)
            & (PredictionLog.predicted_at == max_ts_subq.c.max_ts),
        )
        .all()
    )

    tier_distribution: dict[str, int] = {"individual": 0, "virtual": 0, "population": 0}
    hypo_risk_count = 0
    hyper_risk_count = 0
    for pred in latest_preds:
        key = pred.model_tier or "population"
        tier_distribution[key] = tier_distribution.get(key, 0) + 1
        if pred.is_hypo_risk:
            hypo_risk_count += 1
        if pred.is_hyper_risk:
            hyper_risk_count += 1

    return PopulationStatsResponse(
        total_patients=total_patients,
        patients_with_predictions=len(latest_preds),
        population_models=population_models,
        virtual_models=virtual_models,
        individual_model_count=individual_model_count,
        tier_distribution=tier_distribution,
        hypo_risk_count=hypo_risk_count,
        hyper_risk_count=hyper_risk_count,
    )


@router.get("/patients/{patient_pk}/model-info", response_model=PatientModelInfoResponse)
def patient_model_info(
    patient_pk: str,
    horizon: str = "2h",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PatientModelInfoResponse:
    """
    Current model tier for this patient, individual model readiness,
    and the latest retrain/prediction metadata.
    """
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)

    if horizon not in ("2h", "3h"):
        raise HTTPException(status_code=400, detail="horizon must be '2h' or '3h'")

    from src.serving.model_loader import _read_registry, _resolve_slot

    reg = _read_registry()
    dataset = patient.dataset
    user_id = patient.user_id

    _, _, current_tier = _resolve_slot(reg, dataset, horizon, user_id, "cgm_active")

    if current_tier == "individual":
        slot = reg["individual"][dataset][user_id][horizon]
    elif current_tier == "virtual":
        slot = reg.get("virtual", {}).get(dataset, {}).get(horizon, {})
    else:
        slot = reg["population"][dataset][horizon]

    ind_slot  = reg.get("individual", {}).get(dataset, {}).get(user_id, {}).get(horizon)
    pop_rmse  = reg["population"][dataset][horizon]["test_rmse"]
    has_ind   = ind_slot is not None
    ind_beats = has_ind and ind_slot["test_rmse"] < pop_rmse

    latest_job = (
        db.query(RetrainJob)
        .filter(RetrainJob.user_id_fk == patient_pk, RetrainJob.horizon == horizon)
        .order_by(RetrainJob.triggered_at.desc())
        .first()
    )
    latest_pred = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id_fk == patient_pk)
        .order_by(PredictionLog.predicted_at.desc())
        .first()
    )

    return PatientModelInfoResponse(
        patient_id=patient.id,
        user_id=user_id,
        dataset=dataset,
        horizon=horizon,
        current_tier=current_tier,
        model_type=slot.get("best_model_type") or slot.get("model_type", "unknown"),
        test_rmse=slot.get("test_rmse"),
        val_rmse=slot.get("val_rmse"),
        clarke_a_pct=slot.get("clarke_a_pct"),
        trained_at=slot.get("trained_at"),
        has_individual_model=has_ind,
        individual_beats_population=ind_beats,
        individual_test_rmse=ind_slot["test_rmse"] if ind_slot else None,
        latest_cgm_mode=latest_pred.cgm_mode if latest_pred else None,
        latest_predicted_at=latest_pred.predicted_at if latest_pred else None,
        latest_retrain_job_id=latest_job.id if latest_job else None,
        latest_retrain_status=latest_job.status if latest_job else None,
        latest_retrain_at=latest_job.triggered_at if latest_job else None,
    )


@router.post("/patients/{patient_pk}/restore-model")
def restore_patient_model(
    patient_pk: str,
    req: RestoreModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Revert a patient's individual model to a previous version by re-pointing
    the registry to a completed RetrainJob's artefact_dir.
    Invalidates the model loader cache so the next prediction picks up the restored version.
    """
    _require_doctor(current_user)
    patient = _get_patient_or_404(patient_pk, db)

    if req.horizon not in ("2h", "3h"):
        raise HTTPException(status_code=400, detail="horizon must be '2h' or '3h'")

    job = (
        db.query(RetrainJob)
        .filter(
            RetrainJob.id == req.job_id,
            RetrainJob.user_id_fk == patient.id,
            RetrainJob.horizon == req.horizon,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Retrain job not found for this patient and horizon.")
    if job.status != "completed":
        raise HTTPException(
            status_code=422,
            detail=f"Cannot restore from a job with status='{job.status}'. Must be 'completed'.",
        )
    if job.artefact_dir is None:
        raise HTTPException(status_code=422, detail="This retrain job has no artefact_dir — cannot restore.")

    import json
    from pathlib import Path
    from src.serving.model_loader import REGISTRY_PATH, _read_registry, invalidate_user_cache

    artefact_path = Path(job.artefact_dir)
    if not artefact_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Artefact directory not found on disk: {job.artefact_dir}",
        )

    metrics_path = artefact_path / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=422, detail="metrics.json missing from artefact directory.")

    with open(metrics_path) as f:
        metrics = json.load(f)

    trained_at = None
    config_path = artefact_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        trained_at = cfg.get("trained_at")

    # Nested-format metrics (val.rmse / test.rmse) or flat fallback
    val_rmse  = (metrics.get("val") or {}).get("rmse") or metrics.get("val_rmse")
    test_rmse = (metrics.get("test") or {}).get("rmse") or metrics.get("test_rmse")
    clarke_a  = (metrics.get("test") or {}).get("clarke_a_pct") or metrics.get("clarke_a_pct")
    model_type = (metrics.get("config") or {}).get("model_name") or \
                 (metrics.get("config") or {}).get("model_type") or "unknown"

    reg = _read_registry()
    dataset = patient.dataset
    user_id = patient.user_id

    reg.setdefault("individual", {})
    reg["individual"].setdefault(dataset, {})
    reg["individual"][dataset].setdefault(user_id, {})

    reg["individual"][dataset][user_id][req.horizon] = {
        "best_model_type": model_type,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "clarke_a_pct": clarke_a,
        "trained_at": trained_at,
        "artefact_dir": job.artefact_dir,
        "version_tag": job.version_tag,
        "restored_from_job": job.id,
        "min_hr_readings": 8,
        "full_hr_readings": 24,
        "hr_gap_tolerance_min": 30,
    }

    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

    invalidate_user_cache(dataset, req.horizon, user_id)

    return {
        "restored": True,
        "patient_user_id": user_id,
        "dataset": dataset,
        "horizon": req.horizon,
        "artefact_dir": job.artefact_dir,
        "version_tag": job.version_tag,
        "test_rmse": test_rmse,
    }
