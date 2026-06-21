from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

_T0 = "2024-01-15T06:00:00"
_T1 = "2024-01-15T06:15:00"
_T2 = "2024-01-15T06:30:00"
_T3 = "2024-01-15T06:45:00"
_T4 = "2024-01-15T07:00:00"
_T5 = "2024-01-15T07:15:00"
_T6 = "2024-01-15T07:30:00"
_T7 = "2024-01-15T07:45:00"
_T8 = "2024-01-15T08:00:00"
_T9 = "2024-01-15T08:15:00"


class GlucoseReading(BaseModel):
    timestamp: datetime

    # CGM glucose — required for cgm_active mode; omit (or set null) for post_cgm mode.
    # Mode is auto-detected at inference time: if any reading carries a non-null
    # glucose_mg_dl the request is treated as cgm_active, otherwise post_cgm.
    glucose_mg_dl: Optional[float] = Field(
        None, ge=40.0, le=400.0,
        description=(
            "CGM reading in mg/dL (40–400). Required for CGM-active predictions. "
            "Omit or set null for post-CGM (virtual model) predictions."
        ),
    )

    # Heart rate — REQUIRED on every reading.
    # Predictions are a continuous output that requires continuous HR input.
    # A missing HR value or a gap > 30 min in consecutive readings will block
    # the prediction and return a 503 DataGapResponse.
    hr: float = Field(
        ..., ge=30.0, le=220.0,
        description="Heart rate (bpm). REQUIRED — no prediction without continuous HR.",
    )

    # Nature's Paper — Empatica E4
    eda: Optional[float] = Field(None, ge=0.0, le=30.0, description="Electrodermal activity (µS) — Nature's Paper only")
    ibi_mean: Optional[float] = Field(None, ge=0.25, le=2.5, description="Mean inter-beat interval (s) — Nature's Paper only")
    ibi_rmssd: Optional[float] = Field(None, ge=0.0, le=2.5, description="HRV RMSSD (s) — Nature's Paper only")
    bvp: Optional[float] = Field(None, description="Blood volume pulse — Nature's Paper only")
    acc_magnitude_mean: Optional[float] = Field(None, description="Mean accelerometer magnitude")
    acc_magnitude_std: Optional[float] = Field(None, description="Std accelerometer magnitude")
    temp: Optional[float] = Field(None, ge=20.0, le=42.0, description="Skin temperature (°C)")

    # CGMacros — FitBit Sense
    calories_burned: Optional[float] = Field(None, ge=0.0, description="Calories burned — CGMacros only")
    mets: Optional[float] = Field(None, ge=0.0, le=20.0, description="Metabolic equivalents — CGMacros only")
    steps: Optional[int] = Field(None, ge=0, description="Step count — CGMacros only")

    # Optional sleep / stress
    sleep_stage: Optional[float] = Field(None, description="Sleep stage code")
    sleep_duration_h: Optional[float] = Field(None, description="Sleep duration (hours)")
    stress_score: Optional[float] = Field(None, description="Stress score (0–100)")


class MealEvent(BaseModel):
    timestamp: datetime
    calorie: float = Field(0.0, ge=0.0, description="Total meal calories (kcal)")
    total_carb: float = Field(0.0, ge=0.0, description="Total carbohydrates (g)")
    protein: float = Field(0.0, ge=0.0, description="Protein (g)")
    total_fat: float = Field(0.0, ge=0.0, description="Total fat (g)")
    dietary_fiber: float = Field(0.0, ge=0.0, description="Dietary fiber (g)")
    sugar: float = Field(0.0, ge=0.0, description="Sugar (g)")
    gi_proxy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Glycemic index proxy (0–1)")
    amount_consumed_pct: float = Field(1.0, ge=0.0, le=1.0, description="Fraction consumed (0–1)")
    meal_type_encoded: int = Field(0, ge=0, le=2, description="0=breakfast, 1=lunch/dinner, 2=snack")


_CGMACROS_EXAMPLE = {
    "user_id": "019",
    "dataset": "cgmacros",
    "horizon": "2h",
    "age": 45.0,
    "bmi": 27.3,
    "hba1c": 6.8,
    "meals": [
        {
            "timestamp": _T0,
            "calorie": 420.0,
            "total_carb": 55.0,
            "protein": 20.0,
            "total_fat": 12.0,
            "dietary_fiber": 4.5,
            "sugar": 18.0,
            "gi_proxy": 0.62,
            "amount_consumed_pct": 1.0,
            "meal_type_encoded": 0,
        }
    ],
    "readings": [
        {"timestamp": _T0, "glucose_mg_dl": 142.0, "hr": 72.0, "calories_burned": 1.2, "mets": 1.5, "steps": 0},
        {"timestamp": _T1, "glucose_mg_dl": 148.0, "hr": 74.0, "calories_burned": 1.3, "mets": 1.6, "steps": 45},
        {"timestamp": _T2, "glucose_mg_dl": 155.0, "hr": 75.0, "calories_burned": 1.1, "mets": 1.5, "steps": 30},
        {"timestamp": _T3, "glucose_mg_dl": 161.0, "hr": 73.0, "calories_burned": 1.2, "mets": 1.4, "steps": 20},
        {"timestamp": _T4, "glucose_mg_dl": 158.0, "hr": 76.0, "calories_burned": 1.4, "mets": 1.7, "steps": 60},
        {"timestamp": _T5, "glucose_mg_dl": 153.0, "hr": 75.0, "calories_burned": 1.3, "mets": 1.5, "steps": 40},
        {"timestamp": _T6, "glucose_mg_dl": 149.0, "hr": 74.0, "calories_burned": 1.2, "mets": 1.5, "steps": 25},
        {"timestamp": _T7, "glucose_mg_dl": 145.0, "hr": 73.0, "calories_burned": 1.1, "mets": 1.4, "steps": 15},
        {"timestamp": _T8, "glucose_mg_dl": 143.0, "hr": 72.0, "calories_burned": 1.0, "mets": 1.3, "steps": 10},
        {"timestamp": _T9, "glucose_mg_dl": 140.0, "hr": 71.0, "calories_burned": 1.0, "mets": 1.3, "steps": 5},
    ],
}

_NP_EXAMPLE = {
    "user_id": "003",
    "dataset": "nature_paper",
    "horizon": "2h",
    "meals": [],
    "readings": [
        {"timestamp": _T0, "glucose_mg_dl": 118.0, "hr": 68.0, "eda": 2.1, "ibi_mean": 0.88, "ibi_rmssd": 0.045, "temp": 34.2},
        {"timestamp": _T1, "glucose_mg_dl": 121.0, "hr": 69.0, "eda": 2.3, "ibi_mean": 0.87, "ibi_rmssd": 0.042, "temp": 34.3},
        {"timestamp": _T2, "glucose_mg_dl": 125.0, "hr": 70.0, "eda": 2.0, "ibi_mean": 0.86, "ibi_rmssd": 0.044, "temp": 34.4},
        {"timestamp": _T3, "glucose_mg_dl": 128.0, "hr": 71.0, "eda": 2.2, "ibi_mean": 0.85, "ibi_rmssd": 0.046, "temp": 34.3},
        {"timestamp": _T4, "glucose_mg_dl": 130.0, "hr": 70.0, "eda": 2.1, "ibi_mean": 0.86, "ibi_rmssd": 0.043, "temp": 34.2},
        {"timestamp": _T5, "glucose_mg_dl": 127.0, "hr": 69.0, "eda": 1.9, "ibi_mean": 0.87, "ibi_rmssd": 0.041, "temp": 34.1},
        {"timestamp": _T6, "glucose_mg_dl": 124.0, "hr": 68.0, "eda": 2.0, "ibi_mean": 0.88, "ibi_rmssd": 0.040, "temp": 34.0},
        {"timestamp": _T7, "glucose_mg_dl": 122.0, "hr": 67.0, "eda": 1.8, "ibi_mean": 0.89, "ibi_rmssd": 0.038, "temp": 33.9},
        {"timestamp": _T8, "glucose_mg_dl": 120.0, "hr": 67.0, "eda": 1.7, "ibi_mean": 0.89, "ibi_rmssd": 0.037, "temp": 33.8},
        {"timestamp": _T9, "glucose_mg_dl": 119.0, "hr": 66.0, "eda": 1.6, "ibi_mean": 0.90, "ibi_rmssd": 0.036, "temp": 33.7},
    ],
}

_VIRTUAL_EXAMPLE = {
    "user_id": "019",
    "dataset": "cgmacros",
    "horizon": "2h",
    "meals": [
        {
            "timestamp": _T0,
            "calorie": 350.0,
            "total_carb": 45.0,
            "protein": 18.0,
            "total_fat": 10.0,
            "dietary_fiber": 3.0,
            "sugar": 12.0,
            "gi_proxy": 0.55,
            "amount_consumed_pct": 1.0,
            "meal_type_encoded": 1,
        }
    ],
    "readings": [
        # No glucose_mg_dl — post-CGM mode; hr is still required on every reading
        {"timestamp": _T0, "hr": 72.0, "calories_burned": 1.2, "mets": 1.5, "steps": 0},
        {"timestamp": _T1, "hr": 74.0, "calories_burned": 1.3, "mets": 1.6, "steps": 45},
        {"timestamp": _T2, "hr": 75.0, "calories_burned": 1.1, "mets": 1.5, "steps": 30},
        {"timestamp": _T3, "hr": 73.0, "calories_burned": 1.2, "mets": 1.4, "steps": 20},
        {"timestamp": _T4, "hr": 76.0, "calories_burned": 1.4, "mets": 1.7, "steps": 60},
        {"timestamp": _T5, "hr": 75.0, "calories_burned": 1.3, "mets": 1.5, "steps": 40},
        {"timestamp": _T6, "hr": 74.0, "calories_burned": 1.2, "mets": 1.5, "steps": 25},
        {"timestamp": _T7, "hr": 73.0, "calories_burned": 1.1, "mets": 1.4, "steps": 15},
        {"timestamp": _T8, "hr": 72.0, "calories_burned": 1.0, "mets": 1.3, "steps": 10},
        {"timestamp": _T9, "hr": 71.0, "calories_burned": 1.0, "mets": 1.3, "steps": 5},
    ],
}


class PredictRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="Dataset participant ID (e.g. '019' for CGMacros, '003' for Nature's Paper). "
                    "Used for 3-tier model resolution: individual → virtual → population.",
    )
    dataset: Literal["cgmacros", "nature_paper"] = Field(
        ..., description="Which dataset this user belongs to.",
    )
    horizon: Literal["2h", "3h"] = Field("2h", description="Prediction horizon.")

    readings: List[GlucoseReading] = Field(
        ...,
        min_length=1,
        description=(
            "Chronological wearable + CGM readings at 15-min intervals. "
            "Minimum 8 readings required (enforced by the HR gate, which returns HTTP 503 "
            "with wait_minutes if insufficient). 24+ for full fidelity. "
            "hr is REQUIRED on every reading — missing HR or a gap > 30 min blocks the prediction. "
            "Omit glucose_mg_dl for post-CGM (virtual) predictions."
        ),
    )
    meals: List[MealEvent] = Field(
        default_factory=list,
        description="Meal events within the reading window.",
    )
    age: Optional[float] = Field(None, ge=0.0, le=120.0, description="Age (years) — CGMacros only")
    bmi: Optional[float] = Field(None, ge=10.0, le=70.0, description="BMI — CGMacros only")
    hba1c: Optional[float] = Field(None, ge=3.0, le=20.0, description="HbA1c (%) — CGMacros only")

    model_config = {
        "json_schema_extra": {
            "example": _CGMACROS_EXAMPLE,
        }
    }


PREDICT_OPENAPI_EXAMPLES = {
    "CGMacros — CGM active (user 019)": {
        "summary": "CGM-active: glucose + HR readings, 2h horizon",
        "value": _CGMACROS_EXAMPLE,
    },
    "Nature's Paper — CGM active (user 003)": {
        "summary": "Nature's Paper population model, 2h horizon, Empatica E4",
        "value": _NP_EXAMPLE,
    },
    "CGMacros — Post-CGM / virtual (user 019)": {
        "summary": "Post-CGM: HR-only readings, virtual model, no glucose signal",
        "value": _VIRTUAL_EXAMPLE,
    },
}


# ── Watch / HR data gap response ──────────────────────────────────────────────

class DataGapResponse(BaseModel):
    """
    Returned (HTTP 503) when the HR continuity gate blocks a prediction.

    reason values:
        "hr_missing"        — one or more readings have no hr value
        "insufficient_data" — fewer than 8 HR readings in the window
        "data_gap"          — consecutive reading gap > 30 min detected
    """
    reason:              Literal["hr_missing", "insufficient_data", "data_gap"]
    message:             str    # human-readable, safe to show in the app UI
    readings_available:  int
    readings_required:   int    # always 8
    wait_minutes:        int    # (readings_required - readings_available) × 15; 0 for data_gap
    last_hr_at:          Optional[datetime]  # timestamp of last valid HR reading
    gap_at:              Optional[datetime]  # first timestamp where the gap was detected
    gap_minutes:         Optional[int]       # length of the gap in minutes


# ── Prediction output ─────────────────────────────────────────────────────────

class GlucosePrediction(BaseModel):
    step:          int   = Field(..., description="Prediction step index (1-based)")
    minutes_ahead: int   = Field(..., description="Minutes into the future")
    glucose_mg_dl: float = Field(..., description="Predicted glucose (mg/dL)")
    delta_mg_dl:   float = Field(..., description="Change from current glucose (mg/dL). 0 in post_cgm mode.")


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    user_id:         str
    dataset:         str
    horizon:         str
    model_type:      str   = Field(..., description="Underlying algorithm: lightgbm, xgboost, random_forest, gru, …")
    model_tier:      str   = Field(..., description="Tier served: 'population' | 'virtual' | 'individual'")
    cgm_mode:        str   = Field(..., description="'cgm_active' — glucose signal present; 'post_cgm' — virtual model used")
    data_quality:    str   = Field(..., description="'full' (≥24 HR readings) | 'warming_up' (8–23 readings)")
    readings_used:   int   = Field(..., description="Number of HR readings present in the request window")

    current_glucose: float = Field(..., description="Most recent CGM reading (mg/dL). 0 in post_cgm mode.")
    predictions:     List[GlucosePrediction]
    horizon_glucose: float = Field(..., description="Predicted glucose at horizon end (mg/dL)")
    clarke_zone:     str   = Field(..., description="Clarke Error Grid zone (A–E). A/B = clinically safe.")
    is_hypo_risk:    bool  = Field(..., description="True if any predicted value < 70 mg/dL")
    is_hyper_risk:   bool  = Field(..., description="True if any predicted value > 180 mg/dL")
    n_readings_used: int   = Field(..., description="Total readings used for feature engineering")
    warning:         Optional[str] = Field(None, description="Non-fatal warning (e.g. fallback tier used)")


# ── Model info (GET /predict/model-info) ──────────────────────────────────────

class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    user_id:       str
    dataset:       str
    horizon:       str
    model_tier:    str            # "population" | "virtual" | "individual"
    model_type:    str
    test_rmse:     float
    val_rmse:      float
    clarke_a_pct:  float
    trained_at:    Optional[str]
    # Individual upgrade readiness
    has_individual_model: bool
    individual_beats_population: bool   # False if individual exists but didn't improve
    # HR gate thresholds (from registry)
    min_hr_readings:      int
    full_hr_readings:     int
    hr_gap_tolerance_min: int


# ── Retrain status (GET /predict/retrain-status) ──────────────────────────────

class RetrainStatusResponse(BaseModel):
    has_active_job:  bool
    job_id:          Optional[str]
    status:          Optional[str]   # "pending" | "running" | "done" | "failed"
    triggered_by:    Optional[str]
    triggered_at:    Optional[datetime]
    completed_at:    Optional[datetime]
    test_rmse:       Optional[float]
    prev_rmse:       Optional[float]
    improved:        Optional[bool]
    version_tag:     Optional[str]
    error_msg:       Optional[str]
