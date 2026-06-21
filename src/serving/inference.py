"""
End-to-end inference pipeline.

Flow:
    PredictRequest
      └─ validate_hr_continuity()  → HR gate (blocks on gap / missing / insufficient)
      └─ _detect_mode()            → "cgm_active" | "post_cgm"
      └─ load_model()              → 3-tier: individual → virtual → population
      └─ _readings_to_df()         → 15-min resampled DataFrame
      └─ build_feature_matrix()    → full feature matrix (same pipeline as training)
      └─ last row → X              → filter to model feature_cols + scale
      └─ model.predict(X)          → delta array (cgm_active) or absolute array (post_cgm)
      └─ PredictResponse           → includes model_tier, cgm_mode, data_quality
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.api.schemas import (
    DataGapResponse,
    GlucosePrediction,
    GlucoseReading,
    MealEvent,
    PredictRequest,
    PredictResponse,
)
from src.config import HORIZON_2H_STEPS, HORIZON_3H_STEPS
from src.features.pipeline import build_feature_matrix, build_virtual_feature_matrix
from src.serving.model_loader import load_model
from src.utils import get_logger

log = get_logger(__name__)

_HORIZON_STEPS = {"2h": HORIZON_2H_STEPS, "3h": HORIZON_3H_STEPS}

# HR continuity gate constants — must match registry HR_GATE values
HR_MIN_READINGS    = 8    # minimum to start predicting (2h at 15-min intervals)
HR_FULL_READINGS   = 24   # full fidelity (all rolling windows saturated, 6h)
HR_GAP_LIMIT_MIN   = 30   # max gap (minutes) between consecutive readings before blocked

# Wearable signal columns — resampled as mean (forward-filled up to 4 h)
_WEARABLE_COLS = [
    "hr", "eda", "ibi_mean", "ibi_rmssd", "bvp",
    "acc_magnitude_mean", "acc_magnitude_std", "temp",
    "calories_burned", "mets",
    "sleep_stage", "sleep_duration_h", "stress_score",
]


# ── HR continuity gate ────────────────────────────────────────────────────────

def validate_hr_continuity(readings: list[GlucoseReading]) -> DataGapResponse | None:
    """
    Three-check HR gate — runs before any model loading or feature engineering.

    Returns None if all checks pass.
    Returns a DataGapResponse (caller should raise HTTP 503) if blocked.

    Check 1 — HR presence: every reading must carry a non-null hr value.
    Check 2 — Minimum window: at least HR_MIN_READINGS (8) readings required.
    Check 3 — Gap continuity: no gap > HR_GAP_LIMIT_MIN between consecutive readings.
    """
    # Check 1: hr is now a required field in GlucoseReading, but guard defensively
    missing_hr = [r for r in readings if r.hr is None]
    if missing_hr:
        last_valid = next(
            (r.timestamp for r in reversed(readings) if r.hr is not None), None
        )
        return DataGapResponse(
            reason="hr_missing",
            message=(
                f"Heart rate is missing from {len(missing_hr)} reading(s). "
                "Ensure your watch is syncing before requesting a prediction."
            ),
            readings_available=len(readings) - len(missing_hr),
            readings_required=HR_MIN_READINGS,
            wait_minutes=max(0, (HR_MIN_READINGS - (len(readings) - len(missing_hr))) * 15),
            last_hr_at=last_valid,
            gap_at=None,
            gap_minutes=None,
        )

    # Check 2: minimum window
    n = len(readings)
    if n < HR_MIN_READINGS:
        wait = (HR_MIN_READINGS - n) * 15
        last_valid = readings[-1].timestamp if readings else None
        return DataGapResponse(
            reason="insufficient_data",
            message=(
                f"Collecting watch data — {n}/{HR_MIN_READINGS} readings available "
                f"({wait} min remaining before first prediction)."
            ),
            readings_available=n,
            readings_required=HR_MIN_READINGS,
            wait_minutes=wait,
            last_hr_at=last_valid,
            gap_at=None,
            gap_minutes=None,
        )

    # Check 3: gap continuity — walk consecutive pairs
    sorted_readings = sorted(readings, key=lambda r: r.timestamp)
    for i in range(1, len(sorted_readings)):
        prev_ts = sorted_readings[i - 1].timestamp
        curr_ts = sorted_readings[i].timestamp
        # Normalise to UTC-aware for safe subtraction
        if hasattr(prev_ts, "tzinfo") and prev_ts.tzinfo is None:
            from datetime import timezone
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
            curr_ts = curr_ts.replace(tzinfo=timezone.utc)
        gap_min = int((curr_ts - prev_ts).total_seconds() / 60)
        if gap_min > HR_GAP_LIMIT_MIN:
            return DataGapResponse(
                reason="data_gap",
                message=(
                    f"Watch data gap of {gap_min} minutes detected at "
                    f"{sorted_readings[i].timestamp.strftime('%H:%M')}. "
                    "Predictions paused — reconnect your watch to resume."
                ),
                readings_available=n,
                readings_required=HR_MIN_READINGS,
                wait_minutes=0,
                last_hr_at=sorted_readings[i - 1].timestamp,
                gap_at=sorted_readings[i].timestamp,
                gap_minutes=gap_min,
            )

    return None  # all checks passed


# ── Mode detection ─────────────────────────────────────────────────────────────

def _detect_mode(readings: list[GlucoseReading]) -> str:
    """
    Detect cgm_active vs post_cgm from whether any reading carries glucose_mg_dl.
    Presence of at least one non-null glucose value → cgm_active.
    """
    has_glucose = any(r.glucose_mg_dl is not None for r in readings)
    return "cgm_active" if has_glucose else "post_cgm"


def _data_quality(n_readings: int) -> str:
    return "full" if n_readings >= HR_FULL_READINGS else "warming_up"


# ── Main inference entry point ────────────────────────────────────────────────

def run_inference(
    request:  PredictRequest,
    user_id:  str | None = None,
) -> PredictResponse:
    """
    Convert a PredictRequest to a PredictResponse.

    The HR gate must be called BEFORE this function (done in the predict router).
    user_id is the authenticated user's participant ID (e.g. "019"); it enables
    tier-1 individual model lookup.  Falls back to request.user_id if not supplied.
    """
    effective_user = user_id or request.user_id
    mode           = _detect_mode(request.readings)
    n_hr           = len(request.readings)
    quality        = _data_quality(n_hr)

    loaded = load_model(
        request.dataset,
        request.horizon,
        user_id=effective_user,
        mode=mode,
    )

    warnings: list[str] = []

    if quality == "warming_up":
        warnings.append(
            f"Watch data window has {n_hr}/{HR_FULL_READINGS} readings — "
            "predictions will improve as more watch data arrives (full fidelity at 6h)."
        )

    if request.dataset == "cgmacros":
        missing_demo = [
            f for f, v in [("age", request.age), ("bmi", request.bmi), ("hba1c", request.hba1c)]
            if v is None
        ]
        if missing_demo:
            warnings.append(
                f"Demographics not provided {missing_demo} — using 0.0; accuracy may be reduced."
            )

    # Build DataFrame — post_cgm path skips glucose interpolation
    df = _readings_to_df(
        request.readings,
        request.meals,
        age=request.age,
        bmi=request.bmi,
        hba1c=request.hba1c,
        mode=mode,
    )

    if len(df) < 4:
        raise ValueError(
            f"Only {len(df)} rows after 15-min resampling. "
            "Send at least 8 readings spread over ≥1 h."
        )

    # Build feature matrix — uses the correct pipeline for the mode
    if mode == "post_cgm":
        feat_df = build_virtual_feature_matrix(df, user_id=effective_user, drop_target_nans=False)
        current_glucose = 0.0  # no live glucose in post-CGM mode
    else:
        feat_df = build_feature_matrix(df, user_id=effective_user, drop_target_nans=False)
        feat_df = feat_df.dropna(subset=["glucose_mg_dl"])
        if feat_df.empty:
            raise ValueError("Feature matrix empty after pipeline — check input readings.")
        current_glucose = float(feat_df["glucose_mg_dl"].iloc[-1])

    if feat_df.empty:
        raise ValueError("Feature matrix is empty after pipeline. Check input readings.")

    last_row     = feat_df.iloc[[-1]]
    feature_cols = loaded.feature_cols

    # Align X to training feature columns — zero-fill any that are absent
    X = pd.DataFrame(index=last_row.index, columns=feature_cols, dtype=float)
    missing_feats: list[str] = []
    for col in feature_cols:
        if col in last_row.columns:
            X[col] = last_row[col].values
        else:
            X[col] = 0.0
            missing_feats.append(col)

    if missing_feats:
        warnings.append(
            f"{len(missing_feats)} feature(s) not in input (zeroed): "
            f"{missing_feats[:3]}{'...' if len(missing_feats) > 3 else ''}"
        )

    X_scaled = pd.DataFrame(
        loaded.scaler.transform(X),
        columns=feature_cols,
    )

    # Predict
    raw    = loaded.model.predict(X_scaled)       # (1, n_steps) or (n_steps,)
    values = np.asarray(raw).flatten()

    n_steps     = _HORIZON_STEPS[request.horizon]
    predictions: list[GlucosePrediction] = []

    for i, val in enumerate(values[:n_steps]):
        if mode == "post_cgm":
            # Virtual model predicts absolute glucose
            abs_glucose = round(float(val), 1)
            delta       = 0.0
        else:
            # Population / individual predict glucose delta from current
            delta       = float(val)
            abs_glucose = round(current_glucose + delta, 1)

        predictions.append(
            GlucosePrediction(
                step=i + 1,
                minutes_ahead=(i + 1) * 15,
                glucose_mg_dl=abs_glucose,
                delta_mg_dl=round(delta, 1),
            )
        )

    horizon_glucose = predictions[-1].glucose_mg_dl

    return PredictResponse(
        user_id=request.user_id,
        dataset=request.dataset,
        horizon=request.horizon,
        model_type=loaded.model_type,
        model_tier=loaded.model_tier,
        cgm_mode=mode,
        data_quality=quality,
        readings_used=n_hr,
        current_glucose=round(current_glucose, 1),
        predictions=predictions,
        horizon_glucose=horizon_glucose,
        clarke_zone=_clarke_zone(horizon_glucose, current_glucose),
        is_hypo_risk=any(p.glucose_mg_dl < 70.0 for p in predictions),
        is_hyper_risk=any(p.glucose_mg_dl > 180.0 for p in predictions),
        n_readings_used=len(df),
        warning="; ".join(warnings) if warnings else None,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _readings_to_df(
    readings: list[GlucoseReading],
    meals:    list[MealEvent],
    age:      Optional[float],
    bmi:      Optional[float],
    hba1c:    Optional[float],
    mode:     str = "cgm_active",
) -> pd.DataFrame:
    """
    Convert raw readings + meals into a 15-min resampled DataFrame matching
    the training schema. In post_cgm mode, glucose columns are left as NaN/0
    (they will be stripped by build_virtual_feature_matrix).
    """
    rows = []
    for r in readings:
        rows.append({
            "timestamp":          r.timestamp,
            "glucose_mg_dl":      r.glucose_mg_dl,   # None in post_cgm mode
            "hr":                 r.hr,
            "eda":                r.eda,
            "ibi_mean":           r.ibi_mean,
            "ibi_rmssd":          r.ibi_rmssd,
            "bvp":                r.bvp,
            "acc_magnitude_mean": r.acc_magnitude_mean,
            "acc_magnitude_std":  r.acc_magnitude_std,
            "temp":               r.temp,
            "calories_burned":    r.calories_burned,
            "mets":               r.mets,
            "steps":              r.steps,
            "sleep_stage":        r.sleep_stage,
            "sleep_duration_h":   r.sleep_duration_h,
            "stress_score":       r.stress_score,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    steps_series = None
    if "steps" in df.columns:
        steps_series = df[["steps"]].resample("15min").sum()

    df = df.drop(columns=["steps"], errors="ignore").resample("15min").mean()

    if mode == "cgm_active":
        # Linear interpolation for glucose gaps ≤ 45 min (3 steps)
        df["glucose_mg_dl"] = df["glucose_mg_dl"].interpolate(method="linear", limit=3)
    else:
        # In post_cgm mode we still need the column present for target computation
        # (the pipeline uses it to build absolute targets and then drops it).
        # Keep any values that came through; NaN rows will be filled to 0.
        if "glucose_mg_dl" not in df.columns:
            df["glucose_mg_dl"] = 0.0
        df["glucose_mg_dl"] = df["glucose_mg_dl"].fillna(0.0)

    # Forward-fill wearable signals for gaps ≤ 4 h (16 steps)
    for col in _WEARABLE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float).ffill(limit=16)

    if steps_series is not None:
        df = df.join(steps_series, how="left")
        df["steps"] = df["steps"].astype(float).fillna(0.0)

    if mode == "cgm_active":
        df = df.dropna(subset=["glucose_mg_dl"])

    # ── Meals ─────────────────────────────────────────────────────────────────
    meal_cols = ["calorie", "total_carb", "protein", "total_fat",
                 "dietary_fiber", "sugar", "gi_proxy",
                 "amount_consumed_pct", "meal_type_encoded"]

    if meals:
        meal_rows = []
        for m in meals:
            ts = pd.Timestamp(m.timestamp)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            meal_rows.append({
                "timestamp":           ts,
                "calorie":             m.calorie,
                "total_carb":          m.total_carb,
                "protein":             m.protein,
                "total_fat":           m.total_fat,
                "dietary_fiber":       m.dietary_fiber,
                "sugar":               m.sugar,
                "gi_proxy":            m.gi_proxy if m.gi_proxy is not None else (
                                           m.sugar / m.total_carb if m.total_carb > 0 else 0.0),
                "amount_consumed_pct": m.amount_consumed_pct,
                "meal_type_encoded":   m.meal_type_encoded,
            })
        meal_df = pd.DataFrame(meal_rows).set_index("timestamp")
        meal_df.index = meal_df.index.round("15min")
        meal_df = meal_df.groupby(meal_df.index).agg({
            "calorie": "sum", "total_carb": "sum", "protein": "sum",
            "total_fat": "sum", "dietary_fiber": "sum", "sugar": "sum",
            "gi_proxy": "mean", "amount_consumed_pct": "mean",
            "meal_type_encoded": "max",
        })
        df = df.join(meal_df, how="left")
        for col in ["calorie", "total_carb", "protein", "total_fat", "dietary_fiber", "sugar"]:
            df[col] = df[col].fillna(0.0)
        df["gi_proxy"] = df["gi_proxy"].fillna(0.0)
        df["amount_consumed_pct"] = df["amount_consumed_pct"].fillna(0.0)
        df["meal_type_encoded"] = df["meal_type_encoded"].fillna(0.0)
    else:
        for col in meal_cols:
            df[col] = 0.0

    # Static demographics (CGMacros)
    if age is not None:
        df["age"] = age
    if bmi is not None:
        df["bmi"] = bmi
    if hba1c is not None:
        df["hba1c"] = hba1c

    df = df.fillna(0.0)
    return df


def _clarke_zone(predicted: float, current: float) -> str:
    """Simplified Clarke Error Grid zone for a single predicted value."""
    if 70.0 <= predicted <= 180.0:
        return "A"
    if predicted < 70.0:
        return "D" if current >= 70.0 and predicted < 58.0 else "B"
    return "D" if current <= 180.0 and predicted > 300.0 else "B"
