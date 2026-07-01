"""
SQLAlchemy ORM models for GlucoSense AI.

Tables:
    users           — registered app users (cgmacros or nature_paper cohort)
    prediction_logs — every /predict call is recorded here for drift monitoring
    retrain_jobs    — Celery retrain task status tracking
    food_logs       — meal entries logged by patients
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, String, Text,
)
from sqlalchemy.orm import relationship

from src.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id              = Column(String, primary_key=True, default=_uuid)
    email           = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    # Research-participant linkage — NULL for real app users (only demo/replay accounts set these)
    user_id         = Column(String, nullable=True, index=True)  # "003", "019", etc.
    dataset         = Column(String, nullable=True)              # "cgmacros" | "nature_paper"
    # Profile
    name            = Column(String,  nullable=True)   # full display name (first + last)
    first_name      = Column(String,  nullable=True)
    last_name       = Column(String,  nullable=True)
    date_of_birth   = Column(Date,    nullable=True)
    gender          = Column(String,  nullable=True)   # male|female|other|prefer_not_to_say
    height_cm       = Column(Float,   nullable=True)
    weight_kg       = Column(Float,   nullable=True)
    diabetes_type   = Column(String,  nullable=True)   # type1|type2|prediabetes|gestational|other
    medical_history = Column(Text,    nullable=True)
    medications     = Column(Text,    nullable=True)   # free-text / JSON list of meds
    # Clinical demographics
    age             = Column(Float,   nullable=True)
    bmi             = Column(Float,   nullable=True)
    hba1c           = Column(Float,   nullable=True)
    bp_systolic     = Column(Integer, nullable=True)
    bp_diastolic    = Column(Integer, nullable=True)
    bp_recorded_at  = Column(DateTime(timezone=True), nullable=True)
    onboarding_complete = Column(Boolean, default=False, nullable=False)
    is_active       = Column(Boolean, default=True,  nullable=False)
    created_at      = Column(DateTime(timezone=True), default=_now, nullable=False)
    # Junction wearable integration
    junction_user_id = Column(String, nullable=True, index=True)
    # Multi-source CGM failover (Junction primary → xDRIP fallback)
    cgm_active_source       = Column(String, nullable=True)   # "junction" | "xdrip"
    cgm_last_junction_ok_at = Column(DateTime(timezone=True), nullable=True)
    cgm_last_xdrip_at       = Column(DateTime(timezone=True), nullable=True)
    cgm_api_key             = Column(String, nullable=True, index=True)  # per-user xDRIP push key
    # Google Fit (sole source for all Huawei-watch data)
    google_fit_user_id       = Column(String, nullable=True, index=True)
    google_fit_refresh_token = Column(Text, nullable=True)   # encrypt at rest in Phase 5
    google_fit_token_expiry  = Column(DateTime(timezone=True), nullable=True)
    google_fit_scopes        = Column(Text, nullable=True)
    google_fit_last_sync_at  = Column(DateTime(timezone=True), nullable=True)

    prediction_logs = relationship("PredictionLog", back_populates="user",
                                   cascade="all, delete-orphan",
                                   foreign_keys="PredictionLog.user_id_fk")
    retrain_jobs    = relationship("RetrainJob", back_populates="user",
                                   cascade="all, delete-orphan",
                                   foreign_keys="RetrainJob.user_id_fk")
    food_logs       = relationship("FoodLog", back_populates="user",
                                   cascade="all, delete-orphan",
                                   foreign_keys="FoodLog.user_id_fk")


class PredictionLog(Base):
    """One row per /predict call — used for drift monitoring and RMSE tracking."""
    __tablename__ = "prediction_logs"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id_fk      = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    dataset         = Column(String, nullable=False)
    horizon         = Column(String, nullable=False)
    model_type      = Column(String, nullable=False)
    current_glucose = Column(Float,  nullable=False)
    horizon_glucose = Column(Float,  nullable=False)
    clarke_zone     = Column(String, nullable=False)
    is_hypo_risk    = Column(Boolean, nullable=False)
    is_hyper_risk   = Column(Boolean, nullable=False)
    n_readings      = Column(Integer, nullable=False)
    predicted_at    = Column(DateTime(timezone=True), default=_now, nullable=False)
    # Tier + mode context
    model_tier      = Column(String,  nullable=True)   # "population" | "virtual" | "individual"
    cgm_mode        = Column(String,  nullable=True)   # "cgm_active" | "post_cgm"
    data_quality    = Column(String,  nullable=True)   # "full" | "warming_up"
    readings_used   = Column(Integer, nullable=True)   # HR reading count in the request window
    # Filled in later when the actual reading arrives (for drift calculation)
    actual_glucose  = Column(Float, nullable=True)
    absolute_error  = Column(Float, nullable=True)

    user = relationship("User", back_populates="prediction_logs")


class RetrainJob(Base):
    """Tracks Celery individual-model retraining jobs."""
    __tablename__ = "retrain_jobs"

    id           = Column(String, primary_key=True, default=_uuid)
    celery_id    = Column(String, nullable=True, index=True)   # Celery task ID
    user_id_fk   = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    dataset      = Column(String, nullable=False)
    horizon      = Column(String, nullable=False)
    status       = Column(String, nullable=False, default="pending")
    # "pending" | "running" | "done" | "failed"
    triggered_at  = Column(DateTime(timezone=True), default=_now, nullable=False)
    started_at    = Column(DateTime(timezone=True), nullable=True)
    completed_at  = Column(DateTime(timezone=True), nullable=True)
    test_rmse     = Column(Float,   nullable=True)
    prev_rmse     = Column(Float,   nullable=True)
    improved      = Column(Boolean, nullable=True)
    error_msg     = Column(Text,    nullable=True)
    # Versioning + audit
    artefact_dir  = Column(Text,    nullable=True)   # path to artifacts; enables model restore
    triggered_by  = Column(String,  nullable=True)   # "auto_drift" | "patient_request" | "lifecycle"
    version_tag   = Column(String,  nullable=True)   # "v1", "v2", … per user/dataset/horizon
    # Personalization lifecycle (real-user per-phase models)
    phase        = Column(String,  nullable=True)   # "while_on_cgm" | "post_cgm"
    session_id   = Column(String,  nullable=True)   # CgmSession this model was trained from

    user = relationship("User", back_populates="retrain_jobs", foreign_keys=[user_id_fk])


class FoodLog(Base):
    """Patient meal entry — used to enrich glucose predictions with meal features."""
    __tablename__ = "food_logs"

    id          = Column(String, primary_key=True, default=_uuid)
    user_id_fk  = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    logged_at   = Column(DateTime(timezone=True), default=_now, nullable=False, index=True)
    meal_type   = Column(String, nullable=False)   # breakfast|lunch|dinner|snack|other
    description = Column(Text,   nullable=True)    # free-text description
    calories    = Column(Float,  nullable=True)
    carbs_g     = Column(Float,  nullable=True)
    protein_g   = Column(Float,  nullable=True)
    fat_g       = Column(Float,  nullable=True)
    fiber_g     = Column(Float,  nullable=True)
    sugar_g     = Column(Float,  nullable=True)
    gi_proxy     = Column(Float,  nullable=True)    # estimated glycemic index
    quantity     = Column(Float,  nullable=True)    # numeric quantity (e.g. 2)
    portion_size = Column(String, nullable=True)    # cup|bowl|katori|glass|plate|piece|serving
    notes        = Column(Text,   nullable=True)

    user = relationship("User", back_populates="food_logs", foreign_keys=[user_id_fk])


class CgmReading(Base):
    """CGM glucose reading — from xDRIP webhook or Junction wearable integration."""
    __tablename__ = "cgm_readings"

    id                     = Column(String, primary_key=True, default=_uuid)
    user_id                = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    timestamp              = Column(DateTime(timezone=True), nullable=False, index=True)
    glucose_mgdl           = Column(Float, nullable=False)
    glucose_rate_of_change = Column(Float, nullable=True)
    transmitter_time       = Column(String, nullable=True)
    source_device_id       = Column(String, nullable=True)   # provider slug from Junction / device from xDRIP
    glucose_mmol           = Column(Float, nullable=True)
    source                 = Column(String, nullable=True, default="junction")  # "xdrip" | "junction"
    created_at             = Column(DateTime(timezone=True), nullable=True)
    direction              = Column(String, nullable=True)
    device_type            = Column(String, nullable=True)   # "cgm" (provenance / future-proofing)
    ingested_via           = Column(String, nullable=True)   # "webhook" | "poll" | "manual_sync" | "push"


class WearableActivity(Base):
    """Daily activity summary ingested from Junction."""
    __tablename__ = "wearable_activity"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id_fk      = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    calendar_date   = Column(String, nullable=False, index=True)  # YYYY-MM-DD
    steps           = Column(Integer, nullable=True)
    calories_total  = Column(Float,   nullable=True)
    calories_active = Column(Float,   nullable=True)
    distance_m      = Column(Float,   nullable=True)
    hr_avg_bpm      = Column(Float,   nullable=True)
    hr_min_bpm      = Column(Float,   nullable=True)
    hr_max_bpm      = Column(Float,   nullable=True)
    hr_resting_bpm  = Column(Float,   nullable=True)
    spo2_avg        = Column(Float,   nullable=True)  # average blood oxygen %
    sleep_hours     = Column(Float,   nullable=True)  # total sleep duration in hours
    sleep_score     = Column(Integer, nullable=True)  # sleep quality score (0–100, if available)
    provider        = Column(String,  nullable=True)
    created_at      = Column(DateTime(timezone=True), default=_now, nullable=False)


class WearableSample(Base):
    """Intraday (realtime) watch reading from Health Connect — timestamped, NOT daily.

    One row per observation: HR samples, SpO₂ readings, and interval-binned steps/calories/
    distance land here with their real timestamp. This is what gives the model real
    ``hr_roll_mean_30m`` features (vs the flat daily average in WearableActivity). Deduped
    by (user, timestamp); metrics are merged into the same instant when they coincide.
    """
    __tablename__ = "wearable_samples"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id_fk      = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    timestamp       = Column(DateTime(timezone=True), nullable=False, index=True)
    hr_bpm          = Column(Float,   nullable=True)
    spo2_pct        = Column(Float,   nullable=True)
    steps           = Column(Integer, nullable=True)
    calories_active = Column(Float,   nullable=True)
    distance_m      = Column(Float,   nullable=True)
    provider        = Column(String,  nullable=True)   # "health_connect"
    created_at      = Column(DateTime(timezone=True), default=_now, nullable=False)


class CgmSession(Base):
    """A single CGM sensor journey (~14 days) — drives the personalization lifecycle.

    Started when a CGM reading arrives with no active session; extended on every new
    reading; ended after ~14 days (sensor life) or after a silence gap. ``n_days`` (span of
    the session in days) is what the phase state machine reads to decide when to train the
    personal ``while_on_cgm`` model (~8 days) and pre-emptively the ``post_cgm`` model (day 13).
    """
    __tablename__ = "cgm_sessions"

    id               = Column(String, primary_key=True, default=_uuid)
    user_id_fk       = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    started_at       = Column(DateTime(timezone=True), nullable=False)
    first_reading_at = Column(DateTime(timezone=True), nullable=False)
    last_reading_at  = Column(DateTime(timezone=True), nullable=False, index=True)
    ended_at         = Column(DateTime(timezone=True), nullable=True)
    n_readings       = Column(Integer, nullable=False, default=0)
    n_days           = Column(Float,   nullable=False, default=0.0)   # span (last − first) in days
    status           = Column(String,  nullable=False, default="active")  # "active" | "ended"
    end_reason       = Column(String,  nullable=True)   # "expired" | "silent" | "manual"
    created_at       = Column(DateTime(timezone=True), default=_now, nullable=False)

    user = relationship("User", foreign_keys=[user_id_fk])


class Vitals(Base):
    """User-logged vitals (BP, weight, manual glucose, HbA1c) — from the Home CTA or chat."""
    __tablename__ = "vitals"

    id           = Column(String, primary_key=True, default=_uuid)
    user_id_fk   = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    kind         = Column(String, nullable=False)   # "bp" | "weight" | "glucose" | "hba1c"
    value        = Column(Float,  nullable=True)     # weight / glucose / hba1c / single number
    bp_systolic  = Column(Integer, nullable=True)
    bp_diastolic = Column(Integer, nullable=True)
    source       = Column(String, nullable=True)     # "home" | "chat"
    recorded_at  = Column(DateTime(timezone=True), default=_now, nullable=False)


class ChatMessage(Base):
    """One turn in a user's coach conversation (persisted for history + context)."""
    __tablename__ = "chat_messages"

    id          = Column(String, primary_key=True, default=_uuid)
    user_id_fk  = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    role        = Column(String, nullable=False)     # "user" | "assistant"
    content     = Column(Text,   nullable=False)
    created_at  = Column(DateTime(timezone=True), default=_now, nullable=False)


class Recommendation(Base):
    """Coach-generated diet/activity suggestion shown on the dashboard."""
    __tablename__ = "recommendations"

    id          = Column(String, primary_key=True, default=_uuid)
    user_id_fk  = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    kind        = Column(String, nullable=False)     # "diet" | "activity"
    title       = Column(String, nullable=False)
    body        = Column(Text,   nullable=False)
    active      = Column(Boolean, default=True, nullable=False)
    created_at  = Column(DateTime(timezone=True), default=_now, nullable=False)
