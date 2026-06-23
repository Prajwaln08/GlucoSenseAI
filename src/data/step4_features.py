"""
Step 4 — Feature engineering (per user) on the 10-min grid.

A single, self-contained builder for the unified pipeline.  Everything is a
LAG, a FIXED-WINDOW rolling statistic, or a point-in-time signal — there are NO
reservoir, exponential-decay, or cumulative running-total features (by design).

Windows are expressed in MINUTES and converted to step counts with config.steps,
so the same code works on any grid.

Anti-leakage
------------
Every lag / rolling feature is shifted by one step before its window, so a row
never sees its own (or future) value.  Targets are shifted into the future only.

Modes
-----
"cgm_active"  glucose lag/rolling/delta features included; used while a CGM is live.
"post_cgm"    glucose-derived features are NOT built and are forbidden in the
              feature set (virtual tiers).  Targets are absolute glucose.

Targets (one per forecast horizon, 30/60/90/120 min)
  target_abs_<m>   = glucose at +<m> min            (used by virtual tiers)
  target_delta_<m> = glucose at +<m> min − current  (used by cgm_active tier)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.config import HORIZONS_MIN, RESAMPLE_MIN, BASELINE_STEPS, steps
from src.utils import get_logger

log = get_logger(__name__)

Mode = Literal["cgm_active", "post_cgm"]

# Identity / non-feature columns.
META_COLS = ["uid", "dataset", "participant_id", "gender"]

# Glucose-derived columns — forbidden in post_cgm feature sets.
GLUCOSE_DERIVED = {
    "glucose_mg_dl", "glucose_rate_of_change",
    "glucose_lag_1", "glucose_lag_3", "glucose_lag_6", "glucose_lag_12", "glucose_lag_18",
    "glucose_roll_mean_6", "glucose_roll_mean_12", "glucose_roll_mean_18",
    "glucose_roll_std_6", "glucose_roll_std_12",
    "glucose_delta_1", "glucose_delta_6", "glucose_delta_12", "glucose_accel",
}


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_user_features(df: pd.DataFrame, mode: Mode = "cgm_active") -> pd.DataFrame:
    """Engineer features (+ targets) for one user's imputed 10-min frame."""
    if mode not in ("cgm_active", "post_cgm"):
        raise ValueError(f"mode must be 'cgm_active' or 'post_cgm', got {mode!r}")

    df = df.sort_index().copy()

    if mode == "cgm_active":
        df = _add_glucose_features(df)
    df = _add_meal_features(df)
    df = _add_watch_features(df)
    df = _add_time_features(df)
    df = _add_interaction_features(df)
    df = _add_demographic_features(df)
    df = _add_targets(df)
    return df


def build_features_table(table: pd.DataFrame, mode: Mode = "cgm_active",
                         uid_col: str = "uid") -> pd.DataFrame:
    """Apply build_user_features() per user across the unified table."""
    if uid_col not in table.columns:
        raise ValueError(f"build_features_table: missing '{uid_col}' column.")
    out = [build_user_features(g, mode=mode) for _, g in table.groupby(uid_col, sort=False)]
    result = pd.concat(out, axis=0)
    result.index.name = table.index.name or "timestamp"
    feats = get_feature_cols(result, mode=mode)
    log.info(f"Built features for {result[uid_col].nunique()} users "
             f"({len(feats)} feature cols, mode={mode}).")
    return result


def get_target_cols(mode: Mode) -> list[str]:
    """Target column names for a mode, ordered by horizon."""
    kind = "delta" if mode == "cgm_active" else "abs"
    return [f"target_{kind}_{m}" for m in HORIZONS_MIN]


def get_feature_cols(df: pd.DataFrame, mode: Mode) -> list[str]:
    """Feature columns = all columns minus meta, all targets, and (post_cgm) glucose-derived."""
    exclude = set(META_COLS)
    exclude |= {c for c in df.columns if c.startswith("target_")}
    if mode == "post_cgm":
        exclude |= GLUCOSE_DERIVED
    return [c for c in df.columns if c not in exclude]


def get_xy(df: pd.DataFrame, horizon_min: int, mode: Mode) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for one horizon.  Rows whose target is NaN are dropped, so a
    missing-target row never reaches a model (NaN policy).  X may still contain
    structural NaN — handled per model tier.
    """
    kind = "delta" if mode == "cgm_active" else "abs"
    tcol = f"target_{kind}_{horizon_min}"
    if tcol not in df.columns:
        raise ValueError(f"target column {tcol!r} not found — run build_user_features first.")

    feat = get_feature_cols(df, mode=mode)
    sub  = df.dropna(subset=[tcol])
    return sub[feat].copy(), sub[tcol].copy()


# ══════════════════════════════════════════════════════════════════════════════
# Feature builders (operate on one sorted user frame)
# ══════════════════════════════════════════════════════════════════════════════

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column if present, else an all-NaN series (preserves structural absence)."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, name=name, dtype="float64")


def _add_glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df["glucose_mg_dl"]
    for n in (1, 3, 6, 12, 18):                 # 10/30/60/120/180 min
        df[f"glucose_lag_{n}"] = g.shift(n)
    gs = g.shift(1)                              # shift before rolling (anti-leakage)
    for w in (6, 12, 18):
        df[f"glucose_roll_mean_{w}"] = gs.rolling(w, min_periods=1).mean()
    for w in (6, 12):
        df[f"glucose_roll_std_{w}"] = gs.rolling(w, min_periods=2).std()
    df["glucose_delta_1"]  = g.diff(1).shift(1)
    df["glucose_delta_6"]  = g.diff(6).shift(1)
    df["glucose_delta_12"] = g.diff(12).shift(1)
    df["glucose_accel"]    = df["glucose_delta_1"].diff(1)

    roc = _col(df, "glucose_rate_of_change")
    if roc.notna().any():
        df["glucose_rate_of_change"] = roc.shift(1)
    else:                                        # derive: mg/dL per minute
        df["glucose_rate_of_change"] = g.diff(1).shift(1) / RESAMPLE_MIN
    return df


def _add_meal_features(df: pd.DataFrame) -> pd.DataFrame:
    carb  = _col(df, "total_carb").shift(1)
    cal   = _col(df, "calorie").shift(1)
    prot  = _col(df, "protein").shift(1)
    fat   = _col(df, "total_fat").shift(1)
    fiber = _col(df, "dietary_fiber").shift(1)
    sugar = _col(df, "sugar").shift(1)

    for label, s in [("carbs", carb), ("calories", cal), ("protein", prot),
                     ("fat", fat), ("fiber", fiber), ("sugar", sugar)]:
        df[f"{label}_window_1h"] = s.rolling(steps(60),  min_periods=1).sum()
        df[f"{label}_window_2h"] = s.rolling(steps(120), min_periods=1).sum()
        df[f"{label}_window_3h"] = s.rolling(steps(180), min_periods=1).sum()

    # GI-weighted carb load over 1 h (NP has gi_proxy; CGMacros → NaN preserved)
    gi = _col(df, "gi_proxy").shift(1)
    carb_1h = carb.rolling(steps(60), min_periods=1).sum()
    gi_x    = (gi * carb).rolling(steps(60), min_periods=1).sum()
    df["gi_weighted_1h"] = gi_x / carb_1h.replace(0, np.nan)

    # Meal event signals
    raw_carb = _col(df, "total_carb")
    df["meal_flag"] = (carb > 0).astype("float64")
    df["carb_missing"] = raw_carb.isna().astype("float64")
    df["time_since_last_meal"] = _time_since(carb > 0)
    df["meal_size_category"] = pd.cut(
        carb, bins=[-1, 0, 30, 80, np.inf], labels=[0, 1, 2, 3]
    ).astype("float64")
    df["meal_type_encoded"]   = _col(df, "meal_type_encoded").shift(1)
    df["amount_consumed_pct"] = _col(df, "amount_consumed_pct").shift(1)
    return df


def _add_watch_features(df: pd.DataFrame) -> pd.DataFrame:
    hr   = _col(df, "hr").shift(1)
    eda  = _col(df, "eda").shift(1)
    ibim = _col(df, "ibi_mean").shift(1)
    ibir = _col(df, "ibi_rmssd").shift(1)
    temp = _col(df, "temp").shift(1)
    accm = _col(df, "acc_magnitude_mean").shift(1)
    mets = _col(df, "mets").shift(1)
    calb = _col(df, "calories_burned").shift(1)

    # Heart rate
    for w_min, w in [(30, steps(30)), (60, steps(60)), (120, steps(120)),
                     (240, steps(240)), (360, steps(360))]:
        df[f"hr_roll_mean_{w_min}m"] = hr.rolling(w, min_periods=1).mean()
    df["hr_roll_std_60m"]  = hr.rolling(steps(60),  min_periods=2).std()
    df["hr_roll_std_120m"] = hr.rolling(steps(120), min_periods=2).std()

    # EDA / IBI(HRV) / TEMP (NP) — NaN preserved for CGMacros
    df["eda_roll_mean_30m"]  = eda.rolling(steps(30),  min_periods=1).mean()
    df["eda_roll_mean_120m"] = eda.rolling(steps(120), min_periods=1).mean()
    df["ibi_roll_mean_30m"]  = ibim.rolling(steps(30), min_periods=1).mean()
    df["hrv_rmssd_30m"]      = ibir.rolling(steps(30), min_periods=1).mean()
    df["temp_roll_mean_30m"]  = temp.rolling(steps(30),  min_periods=1).mean()
    df["temp_roll_mean_120m"] = temp.rolling(steps(120), min_periods=1).mean()

    # Activity from ACC magnitude (NP) — windowed means (not cumulative)
    df["acc_roll_mean_30m"]  = accm.rolling(steps(30),  min_periods=1).mean()
    df["acc_roll_mean_120m"] = accm.rolling(steps(120), min_periods=1).mean()
    df["activity_flag"] = (accm > 10).astype("float64")

    # METs / calories burned (CGMacros) — windowed sums
    df["mets_window_1h"]  = mets.rolling(steps(60),  min_periods=1).sum()
    df["mets_window_3h"]  = mets.rolling(steps(180), min_periods=1).sum()
    df["calories_burned_1h"] = calb.rolling(steps(60),  min_periods=1).sum()
    df["calories_burned_3h"] = calb.rolling(steps(180), min_periods=1).sum()

    # Relative-to-baseline (7-day backward mean) — stable across deployment drift
    hr_base   = hr.rolling(BASELINE_STEPS, min_periods=1).mean()
    eda_base  = eda.rolling(BASELINE_STEPS, min_periods=1).mean()
    temp_base = temp.rolling(BASELINE_STEPS, min_periods=1).mean()
    df["hr_vs_baseline"]   = df["hr_roll_mean_60m"]  - hr_base
    df["eda_vs_baseline"]  = df["eda_roll_mean_30m"] - eda_base
    df["temp_vs_baseline"] = df["temp_roll_mean_30m"] - temp_base
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    dow  = idx.dayofweek
    week = idx.isocalendar().week.astype(float).values
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    df["is_weekend"] = (dow >= 5).astype("float64")
    df["is_night"]   = ((idx.hour < 6) | (idx.hour >= 22)).astype("float64")
    df["is_morning"] = ((idx.hour >= 6) & (idx.hour < 12)).astype("float64")
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["carbs_x_mets_1h"]  = _col(df, "carbs_window_1h") * _col(df, "mets_window_1h")
    df["carbs_x_acc_1h"]   = _col(df, "carbs_window_1h") * _col(df, "acc_roll_mean_30m")
    df["gi_x_carbs_1h"]    = _col(df, "gi_weighted_1h")  * _col(df, "carbs_window_1h")
    df["hr_x_meal_flag"]   = _col(df, "hr_roll_mean_60m") * _col(df, "meal_flag")
    df["meal_x_hour_sin"]  = _col(df, "meal_flag")       * _col(df, "hour_sin")
    return df


def _add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    # gender (string, varied casing across datasets) → 0/1; unknown stays NaN
    gmap = {"m": 1.0, "male": 1.0, "f": 0.0, "female": 0.0}
    if "gender" in df.columns:
        df["gender_encoded"] = (df["gender"].astype(str).str.strip().str.lower().map(gmap))
    else:
        df["gender_encoded"] = np.nan

    # hba1c → 4-level clinical bucket (stable); NaN stays NaN
    hba1c = _col(df, "hba1c")
    df["hba1c_bucket"] = pd.cut(
        hba1c, bins=[-np.inf, 6.5, 7.0, 8.0, np.inf], labels=[0, 1, 2, 3], right=False
    ).astype("float64")
    # age, bmi pass through unchanged (NaN allowed)
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df["glucose_mg_dl"]
    for m in HORIZONS_MIN:
        h = steps(m)
        df[f"target_abs_{m}"]   = g.shift(-h)
        df[f"target_delta_{m}"] = g.shift(-h) - g
    return df


# ── helper ────────────────────────────────────────────────────────────────────

def _time_since(bool_series: pd.Series) -> pd.Series:
    """Steps since the last True event; NaN before the first event."""
    out = np.full(len(bool_series), np.nan)
    counter, seen = 0, False
    for i, v in enumerate(bool_series.values):
        if bool(v):
            counter, seen = 0, True
        elif seen:
            counter += 1
        if seen:
            out[i] = counter
    return pd.Series(out, index=bool_series.index)
