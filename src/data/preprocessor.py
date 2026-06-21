"""
Preprocessing pipeline: outlier handling + missing value imputation.

Applied AFTER merge — operates on the unified 15-min DataFrame per user.

Two stages:
  1. Outlier handling  — clip physiologically impossible values before imputation
                         so that forward-fill / interpolation does not propagate
                         extreme noise.
  2. Imputation        — fill NaN per the blueprint rules:
       CGM glucose     → forward fill (limit = 3 windows = 45 min)
       HR              → linear interpolation (limit = 4 windows = 60 min)
       EDA, IBI, TEMP, BVP, ACC → linear interpolation (limit = 4 windows)
       Food / carbs / calories → zero-fill (no event = no intake/burn)
       Categorical / flags     → zero-fill

Row dropping:
  Any row where glucose_mg_dl is still NaN after forward-fill is dropped —
  the CGM target requires a valid glucose reading.
"""

import numpy as np
import pandas as pd

from src.config import (
    GLUCOSE_MIN, GLUCOSE_MAX,
    HR_MIN, HR_MAX,
    EDA_MIN, EDA_MAX,
    TEMP_MIN, TEMP_MAX,
    BVP_PERCENTILE_CLIP,
    CARBS_MAX,
    CALORIES_FOOD_MAX,
    CALORIES_BURNED_MAX,
    METS_MAX,
    MAX_CGM_GAP_WINDOWS,
    MAX_HR_GAP_WINDOWS,
    MAX_WATCH_FFILL_WINDOWS,
)
from src.utils import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Outlier handling
# ══════════════════════════════════════════════════════════════════════════════

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip physiologically impossible values to valid ranges.

    Strategy: hard clip to known physiological bounds (from config.py).
    Values outside the bounds are set to NaN (NOT the boundary value) so that
    the imputation stage treats them as missing data rather than valid readings.

    This is better than clipping to the boundary because a glucose reading of
    401 mg/dL is more likely a sensor artefact than a true extreme value.
    """
    df = df.copy()

    # ── CGM glucose ───────────────────────────────────────────────────────────
    if "glucose_mg_dl" in df.columns:
        mask = (df["glucose_mg_dl"] < GLUCOSE_MIN) | (df["glucose_mg_dl"] > GLUCOSE_MAX)
        n_bad = mask.sum()
        if n_bad:
            log.warning(f"Glucose outliers → NaN: {n_bad} rows outside [{GLUCOSE_MIN}, {GLUCOSE_MAX}]")
        df.loc[mask, "glucose_mg_dl"] = np.nan

    # Rate of change: clip to physiologically plausible range (-5 to +5 mg/dL/min)
    if "glucose_rate_of_change" in df.columns:
        df["glucose_rate_of_change"] = df["glucose_rate_of_change"].clip(-5.0, 5.0)

    # ── Heart rate ────────────────────────────────────────────────────────────
    if "hr" in df.columns:
        mask = (df["hr"] < HR_MIN) | (df["hr"] > HR_MAX)
        df.loc[mask, "hr"] = np.nan

    # ── EDA ───────────────────────────────────────────────────────────────────
    if "eda" in df.columns:
        mask = (df["eda"] < EDA_MIN) | (df["eda"] > EDA_MAX)
        df.loc[mask, "eda"] = np.nan

    # ── IBI ───────────────────────────────────────────────────────────────────
    for col in ["ibi_mean", "ibi_rmssd"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)    # RMSSD/mean can't be negative

    # ── Skin temperature ──────────────────────────────────────────────────────
    if "temp" in df.columns:
        mask = (df["temp"] < TEMP_MIN) | (df["temp"] > TEMP_MAX)
        df.loc[mask, "temp"] = np.nan

    # ── BVP — percentile clip (arbitrary signal, no known hard bounds) ────────
    if "bvp" in df.columns and df["bvp"].notna().sum() > 10:
        p_low  = df["bvp"].quantile(1 - BVP_PERCENTILE_CLIP / 100)
        p_high = df["bvp"].quantile(BVP_PERCENTILE_CLIP / 100)
        df["bvp"] = df["bvp"].clip(p_low, p_high)

    # ── ACC magnitude — remove impossibly high values ─────────────────────────
    for col in ["acc_magnitude_mean", "acc_magnitude_std"]:
        if col in df.columns:
            # Empatica E4 ACC range is ±2g, magnitude ≤ ~3.5g ≈ 343 units
            df[col] = df[col].clip(lower=0, upper=500)

    # ── Food / nutrition ──────────────────────────────────────────────────────
    for col, cap in [
        ("total_carb",     CARBS_MAX),
        ("calorie",        CALORIES_FOOD_MAX),
        ("protein",        300.0),
        ("total_fat",      300.0),
        ("dietary_fiber",  100.0),
        ("sugar",          CARBS_MAX),
    ]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=cap)

    # ── CGMacros-specific ─────────────────────────────────────────────────────
    if "calories_burned" in df.columns:
        df["calories_burned"] = df["calories_burned"].clip(lower=0, upper=CALORIES_BURNED_MAX)

    if "mets" in df.columns:
        df["mets"] = df["mets"].clip(lower=0, upper=METS_MAX)

    if "amount_consumed_pct" in df.columns:
        df["amount_consumed_pct"] = df["amount_consumed_pct"].clip(lower=0, upper=100)

    log.debug("Outlier clipping complete.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Imputation
# ══════════════════════════════════════════════════════════════════════════════

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values according to the blueprint's per-feature rules.

    Forward fill  — CGM glucose (max 3 windows = 45 min).
                    Physiologically valid: glucose changes slowly.
    Interpolation — HR, EDA, IBI, TEMP, BVP, ACC (max 4 windows = 60 min).
                    Smooth signals that change gradually.
    Zero fill     — food/carbs/calories/steps (absence = no event).
    Zero fill     — meal_flag, activity_flag, meal_type_encoded, gi_proxy.

    After imputation: rows where glucose is still NaN are dropped —
    we cannot train without a valid glucose value.
    """
    df = df.copy()

    # ── CGM glucose: forward fill up to 45 min ────────────────────────────────
    if "glucose_mg_dl" in df.columns:
        df["glucose_mg_dl"] = df["glucose_mg_dl"].ffill(limit=MAX_CGM_GAP_WINDOWS)

    # ── Rate of change: derive from filled glucose if still NaN ──────────────
    if "glucose_rate_of_change" in df.columns and df["glucose_rate_of_change"].isna().any():
        if "glucose_mg_dl" in df.columns:
            # 15-min delta / 15 min → mg/dL/min
            df["glucose_rate_of_change"] = df["glucose_rate_of_change"].fillna(
                df["glucose_mg_dl"].diff() / 15.0
            )

    # ── Wearable / activity signals: ffill 4h → bfill 1h → session median ────
    # Using forward-fill (carry last known value) is more realistic than
    # linear interpolation or zero-fill for physiological signals: a flat line
    # from the last reading is a better guess than 0 when the sensor has a gap.
    watch_cols = [
        "hr",
        "eda", "ibi_mean", "ibi_rmssd", "temp", "bvp",
        "acc_magnitude_mean", "acc_magnitude_std",
        "calories_burned", "mets",
    ]
    for col in watch_cols:
        if col not in df.columns:
            continue
        s = df[col]
        s = s.ffill(limit=MAX_WATCH_FFILL_WINDOWS)   # carry forward up to 4 h
        s = s.bfill(limit=MAX_HR_GAP_WINDOWS)         # backfill session start up to 1 h
        if s.isna().any():
            # Long gaps (device removed): fill with the session median so the
            # model sees a plausible resting value rather than 0.
            med = s.median()
            s = s.fillna(med if not pd.isna(med) else 0.0)
        df[col] = s

    # ── Zero-fill: event-based columns (absence = zero) ──────────────────────
    zero_fill_cols = [
        # Food / nutrition
        "calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat",
        "gi_proxy",
        # Flags / categoricals
        "meal_flag", "meal_type_encoded", "amount_consumed_pct",
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ── Drop rows where glucose is still NaN after forward fill ──────────────
    if "glucose_mg_dl" in df.columns:
        n_before = len(df)
        df = df[df["glucose_mg_dl"].notna()]
        n_dropped = n_before - len(df)
        if n_dropped:
            log.warning(
                f"Dropped {n_dropped} rows with no recoverable CGM reading "
                f"(gap > {MAX_CGM_GAP_WINDOWS * 15} min or at data boundary)."
            )

    # ── Fill any remaining NaN with 0 (optional features not in this dataset) -
    df = df.fillna(0.0)

    log.debug(
        f"Imputation complete. Remaining NaN: "
        f"{df.isnull().sum().sum()} total cells."
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Combined entry point
# ══════════════════════════════════════════════════════════════════════════════

def _derive_stable_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace raw continuous demographic values that drift over months with
    stable categorical/bucketed equivalents.

    hba1c → hba1c_bucket (4-level ordinal):
        0 = < 6.5%  (normal / non-diabetic)
        1 = 6.5–7.0 (at-threshold / well-controlled)
        2 = 7.0–8.0 (moderate)
        3 = ≥ 8.0   (elevated)
        NaN → 0 (NP users have no hba1c record; treat as normal)

    Raw hba1c is dropped — its exact value will be stale at inference time
    (it is a 3-month average re-measured quarterly), and the model should
    not learn on a signal that will shift 0.2–0.5% between training and serving.
    The bucket is stable: a user's clinical category rarely changes in < 6 months.

    age and bmi pass through unchanged — age changes by at most 1 year over
    a deployment window, and bmi is a valid stable descriptor.
    """
    if "hba1c" not in df.columns:
        return df

    hba1c = df["hba1c"].fillna(0.0)
    df["hba1c_bucket"] = pd.cut(
        hba1c,
        bins=[-float("inf"), 6.5, 7.0, 8.0, float("inf")],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype(float).fillna(0.0)

    df = df.drop(columns=["hba1c"])
    return df


def preprocess_user(df: pd.DataFrame, user_id: str = "") -> pd.DataFrame:
    """
    Full preprocessing for a single user's merged 15-min DataFrame.

    Order: outlier clipping → imputation → stable demographic derivation.
    This function is safe to call on both NP and CGMacros DataFrames.

    Args:
        df:      Merged 15-min DataFrame from merger.py.
        user_id: Used only for log messages.

    Returns:
        Clean DataFrame ready for feature engineering.
    """
    label = f"user {user_id}" if user_id else "user"
    rows_in = len(df)

    df = clip_outliers(df)
    df = impute(df)
    df = _derive_stable_demographics(df)

    rows_out = len(df)
    pct_kept = 100 * rows_out / rows_in if rows_in else 0
    log.info(
        f"Preprocessing {label}: {rows_in} → {rows_out} rows "
        f"({pct_kept:.1f}% retained). "
        f"Span: {df.index.min()} → {df.index.max()}"
    )
    return df
