"""
CRUD helpers for GlucoSense AI.

All functions accept a SQLAlchemy Session and return ORM objects.
Callers (routers, Celery tasks) own the transaction — call db.commit() after writes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import FoodLog, Message, PredictionLog, RetrainJob, User


# ── Users ─────────────────────────────────────────────────────────────────────

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(
    db: Session,
    email: str,
    hashed_password: str,
    user_id: str,
    dataset: str,
    name: Optional[str] = None,
    gender: Optional[str] = None,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    diabetes_type: Optional[str] = None,
    medical_history: Optional[str] = None,
    age: Optional[float] = None,
    bmi: Optional[float] = None,
    hba1c: Optional[float] = None,
) -> User:
    user = User(
        email=email,
        hashed_password=hashed_password,
        user_id=user_id,
        dataset=dataset,
        name=name,
        gender=gender,
        height_cm=height_cm,
        weight_kg=weight_kg,
        diabetes_type=diabetes_type,
        medical_history=medical_history,
        age=age,
        bmi=bmi,
        hba1c=hba1c,
    )
    db.add(user)
    db.flush()   # assign PK without committing — caller commits
    return user


# ── Prediction logs ───────────────────────────────────────────────────────────

def log_prediction(
    db: Session,
    user_id_fk: str,
    dataset: str,
    horizon: str,
    model_type: str,
    current_glucose: float,
    horizon_glucose: float,
    clarke_zone: str,
    is_hypo_risk: bool,
    is_hyper_risk: bool,
    n_readings: int,
    model_tier: Optional[str] = None,
    cgm_mode: Optional[str] = None,
    data_quality: Optional[str] = None,
    readings_used: Optional[int] = None,
) -> PredictionLog:
    entry = PredictionLog(
        user_id_fk=user_id_fk,
        dataset=dataset,
        horizon=horizon,
        model_type=model_type,
        current_glucose=current_glucose,
        horizon_glucose=horizon_glucose,
        clarke_zone=clarke_zone,
        is_hypo_risk=is_hypo_risk,
        is_hyper_risk=is_hyper_risk,
        n_readings=n_readings,
        model_tier=model_tier,
        cgm_mode=cgm_mode,
        data_quality=data_quality,
        readings_used=readings_used,
    )
    db.add(entry)
    db.flush()
    return entry


def update_actual_glucose(
    db: Session,
    log_id: str,
    actual_glucose: float,
) -> Optional[PredictionLog]:
    entry = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if entry:
        entry.actual_glucose = actual_glucose
        entry.absolute_error = abs(actual_glucose - entry.horizon_glucose)
    return entry


def recent_prediction_rmse(
    db: Session,
    user_id_fk: str,
    horizon: str,
    last_n: int = 50,
) -> Optional[float]:
    """Mean absolute error over the last N predictions that have actuals."""
    logs = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.user_id_fk == user_id_fk,
            PredictionLog.horizon == horizon,
            PredictionLog.actual_glucose.isnot(None),
        )
        .order_by(PredictionLog.predicted_at.desc())
        .limit(last_n)
        .all()
    )
    if not logs:
        return None
    errors = [(log.actual_glucose - log.horizon_glucose) ** 2 for log in logs]
    return (sum(errors) / len(errors)) ** 0.5


# ── Retrain jobs ──────────────────────────────────────────────────────────────

def create_retrain_job(
    db: Session,
    user_id_fk: str,
    dataset: str,
    horizon: str,
    triggered_by: Optional[str] = None,
    notes: Optional[str] = None,
) -> RetrainJob:
    job = RetrainJob(
        user_id_fk=user_id_fk,
        dataset=dataset,
        horizon=horizon,
        triggered_by=triggered_by,
        notes=notes,
    )
    db.add(job)
    db.flush()
    return job


def update_retrain_job(
    db: Session,
    job_id: str,
    status: str,
    celery_id: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    test_rmse: Optional[float] = None,
    prev_rmse: Optional[float] = None,
    improved: Optional[bool] = None,
    error_msg: Optional[str] = None,
    artefact_dir: Optional[str] = None,
    version_tag: Optional[str] = None,
) -> Optional[RetrainJob]:
    job = db.query(RetrainJob).filter(RetrainJob.id == job_id).first()
    if not job:
        return None
    job.status = status
    if celery_id is not None:
        job.celery_id = celery_id
    if started_at is not None:
        job.started_at = started_at
    if completed_at is not None:
        job.completed_at = completed_at
    if test_rmse is not None:
        job.test_rmse = test_rmse
    if prev_rmse is not None:
        job.prev_rmse = prev_rmse
    if improved is not None:
        job.improved = improved
    if error_msg is not None:
        job.error_msg = error_msg
    if artefact_dir is not None:
        job.artefact_dir = artefact_dir
    if version_tag is not None:
        job.version_tag = version_tag
    return job


def get_pending_retrain_jobs(db: Session, limit: int = 5) -> list[RetrainJob]:
    return (
        db.query(RetrainJob)
        .filter(RetrainJob.status == "pending")
        .order_by(RetrainJob.triggered_at)
        .limit(limit)
        .all()
    )


# ── Food logs ─────────────────────────────────────────────────────────────────

def create_food_log(
    db: Session,
    user_id_fk: str,
    meal_type: str,
    description: Optional[str] = None,
    calories: Optional[float] = None,
    carbs_g: Optional[float] = None,
    protein_g: Optional[float] = None,
    fat_g: Optional[float] = None,
    fiber_g: Optional[float] = None,
    sugar_g: Optional[float] = None,
    gi_proxy: Optional[float] = None,
    quantity: Optional[float] = None,
    portion_size: Optional[str] = None,
    notes: Optional[str] = None,
    logged_at: Optional[datetime] = None,
) -> FoodLog:
    entry = FoodLog(
        user_id_fk=user_id_fk,
        meal_type=meal_type,
        description=description,
        calories=calories,
        carbs_g=carbs_g,
        protein_g=protein_g,
        fat_g=fat_g,
        fiber_g=fiber_g,
        sugar_g=sugar_g,
        gi_proxy=gi_proxy,
        quantity=quantity,
        portion_size=portion_size,
        notes=notes,
        logged_at=logged_at or datetime.now(timezone.utc),
    )
    db.add(entry)
    db.flush()
    return entry


def get_food_logs(
    db: Session,
    user_id_fk: str,
    limit: int = 50,
) -> list[FoodLog]:
    return (
        db.query(FoodLog)
        .filter(FoodLog.user_id_fk == user_id_fk)
        .order_by(FoodLog.logged_at.desc())
        .limit(limit)
        .all()
    )


# ── Messages ──────────────────────────────────────────────────────────────────

def send_message(
    db: Session,
    sender_id: str,
    receiver_id: str,
    body: str,
    attachment_url: Optional[str] = None,
    attachment_type: Optional[str] = None,
    attachment_name: Optional[str] = None,
) -> Message:
    msg = Message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        body=body,
        attachment_url=attachment_url,
        attachment_type=attachment_type,
        attachment_name=attachment_name,
    )
    db.add(msg)
    db.flush()
    return msg


def get_conversation(
    db: Session,
    user_a_id: str,
    user_b_id: str,
    limit: int = 100,
) -> list[Message]:
    return (
        db.query(Message)
        .filter(
            (
                (Message.sender_id == user_a_id) & (Message.receiver_id == user_b_id)
            ) | (
                (Message.sender_id == user_b_id) & (Message.receiver_id == user_a_id)
            )
        )
        .order_by(Message.sent_at.asc())
        .limit(limit)
        .all()
    )


def mark_messages_read(
    db: Session,
    receiver_id: str,
    sender_id: str,
) -> None:
    now = datetime.now(timezone.utc)
    db.query(Message).filter(
        Message.receiver_id == receiver_id,
        Message.sender_id == sender_id,
        Message.read_at.is_(None),
    ).update({"read_at": now})


def unread_count(db: Session, receiver_id: str) -> int:
    return (
        db.query(Message)
        .filter(Message.receiver_id == receiver_id, Message.read_at.is_(None))
        .count()
    )
