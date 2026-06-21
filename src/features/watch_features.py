"""
Wearable / watch features — heart rate, steps, activity, EDA, IBI, temperature.

Nature's Paper wearable: Empatica E4 — provides HR, EDA, IBI, BVP, TEMP, ACC.
CGMacros wearable: FitBit Sense — provides HR, METs, Calories (burned).

Rolling window sizes used:
    2  steps =  30 min
    4  steps =   1 h
    8  steps =   2 h
    16 steps =   4 h
    24 steps =   6 h

All dataset-specific features use safe_get_feature so the same code works
for both datasets without any conditional logic.
"""

import pandas as pd

from src.utils import safe_get_feature, get_logger

log = get_logger(__name__)


def add_watch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add heart rate, step/activity, EDA, IBI, temperature, METs, and
    calories-burned features with rolling windows from 30 min up to 6 h.

    CGMacros-only features (METs, Calories burned) are zero-filled for NP.
    NP-only features (EDA, IBI, TEMP, BVP, ACC) are zero-filled for CGMacros.

    Args:
        df: preprocessed user DataFrame (after imputation — watch signals are
            forward-filled, not zero, so rolling means are meaningful).

    Returns:
        df with watch feature columns appended.
    """
    df = df.copy()

    # Shift sources by 1 before rolling (anti-leakage rule)
    hr_s            = safe_get_feature(df, "hr").shift(1)
    eda_s           = safe_get_feature(df, "eda").shift(1)
    ibi_mean_s      = safe_get_feature(df, "ibi_mean").shift(1)
    ibi_rmssd_s     = safe_get_feature(df, "ibi_rmssd").shift(1)
    temp_s          = safe_get_feature(df, "temp").shift(1)
    acc_mean_s      = safe_get_feature(df, "acc_magnitude_mean").shift(1)
    calories_burn_s = safe_get_feature(df, "calories_burned").shift(1)
    mets_s          = safe_get_feature(df, "mets").shift(1)

    # ── Heart rate rolling mean (30 min → 6 h) ───────────────────────────────
    df["hr_roll_mean_2"]  = hr_s.rolling(window=2,  min_periods=1).mean()
    df["hr_roll_mean_4"]  = hr_s.rolling(window=4,  min_periods=1).mean()
    df["hr_roll_mean_8"]  = hr_s.rolling(window=8,  min_periods=1).mean()
    df["hr_roll_mean_16"] = hr_s.rolling(window=16, min_periods=1).mean()   # 4 h
    df["hr_roll_mean_24"] = hr_s.rolling(window=24, min_periods=1).mean()   # 6 h

    # HR rolling std (variability) — 1 h and 2 h
    df["hr_roll_std_4"] = hr_s.rolling(window=4,  min_periods=2).std().fillna(0.0)
    df["hr_roll_std_8"] = hr_s.rolling(window=8,  min_periods=2).std().fillna(0.0)

    # ── EDA (stress proxy — NP only) ─────────────────────────────────────────
    df["eda_roll_mean_2"] = eda_s.rolling(window=2,  min_periods=0).mean().fillna(0.0)
    df["eda_roll_mean_4"] = eda_s.rolling(window=4,  min_periods=0).mean().fillna(0.0)
    df["eda_roll_mean_8"] = eda_s.rolling(window=8,  min_periods=0).mean().fillna(0.0)  # 2 h

    # ── IBI / HRV (NP only) ───────────────────────────────────────────────────
    df["ibi_roll_mean_2"] = ibi_mean_s.rolling(window=2, min_periods=0).mean().fillna(0.0)
    df["ibi_roll_mean_4"] = ibi_mean_s.rolling(window=4, min_periods=0).mean().fillna(0.0)
    df["hrv_rmssd"]       = ibi_rmssd_s.rolling(window=4, min_periods=0).mean().fillna(0.0)

    # ── Skin temperature (NP only) ────────────────────────────────────────────
    df["temp_roll_mean_2"] = temp_s.rolling(window=2,  min_periods=0).mean().fillna(0.0)
    df["temp_roll_mean_4"] = temp_s.rolling(window=4,  min_periods=0).mean().fillna(0.0)
    df["temp_roll_mean_8"] = temp_s.rolling(window=8,  min_periods=0).mean().fillna(0.0)  # 2 h

    # ── Activity / steps proxy using ACC magnitude (NP) ──────────────────────
    df["steps_window_30min"] = acc_mean_s.rolling(window=2,  min_periods=0).sum().fillna(0.0)
    df["steps_window_1h"]    = acc_mean_s.rolling(window=4,  min_periods=0).sum().fillna(0.0)
    df["steps_window_2h"]    = acc_mean_s.rolling(window=8,  min_periods=0).sum().fillna(0.0)
    df["steps_window_4h"]    = acc_mean_s.rolling(window=16, min_periods=0).sum().fillna(0.0)  # 4 h
    df["steps_window_6h"]    = acc_mean_s.rolling(window=24, min_periods=0).sum().fillna(0.0)  # 6 h
    df["activity_flag"]      = (acc_mean_s > 10).astype(int).fillna(0)

    # ── METs (CGMacros FitBit) ────────────────────────────────────────────────
    df["mets_roll_mean_4"] = mets_s.rolling(window=4,  min_periods=0).mean().fillna(0.0)
    df["mets_window_1h"]   = mets_s.rolling(window=4,  min_periods=0).sum().fillna(0.0)
    df["mets_window_3h"]   = mets_s.rolling(window=12, min_periods=0).sum().fillna(0.0)  # 3 h
    df["mets_window_6h"]   = mets_s.rolling(window=24, min_periods=0).sum().fillna(0.0)  # 6 h

    # ── Calories burned (CGMacros only) ──────────────────────────────────────
    df["calories_burned_1h"] = calories_burn_s.rolling(window=4,  min_periods=0).sum().fillna(0.0)
    df["calories_burned_3h"] = calories_burn_s.rolling(window=12, min_periods=0).sum().fillna(0.0)
    df["calories_burned_6h"] = calories_burn_s.rolling(window=24, min_periods=0).sum().fillna(0.0)

    # ── Future reserved slots — zero-filled until Phase 3 ────────────────────
    for col in ["sleep_duration_h", "sleep_stage", "stress_score",
                "steps_window_15min", "steps_total_today"]:
        if col not in df.columns:
            df[col] = 0.0

    # ── Relative-to-baseline features (temporal stability) ───────────────────
    # Absolute HR/EDA/TEMP values drift over months: resting HR shifts with
    # fitness level, EDA drifts with electrode wear and ambient humidity, skin
    # temperature varies seasonally.  The DEVIATION from the user's own recent
    # baseline is physiologically meaningful and stable regardless of when
    # inference runs.
    #
    # Baseline = 7-day backward rolling mean of the same shifted source series
    # (anti-leakage consistent with the rest of the module).  At inference
    # time the 7-day window automatically tracks the user's current state, so
    # the feature never goes out of distribution due to seasonal or lifestyle
    # drift.  min_periods=1 prevents NaN on the first few rows of a new user.
    _7D = 7 * 24 * 4   # 672 steps × 15 min = 7 days

    hr_baseline   = hr_s.rolling(window=_7D,   min_periods=1).mean()
    eda_baseline  = eda_s.rolling(window=_7D,  min_periods=1).mean()
    temp_baseline = temp_s.rolling(window=_7D, min_periods=1).mean()

    df["hr_vs_baseline"]   = (df["hr_roll_mean_4"]   - hr_baseline).fillna(0.0)
    df["eda_vs_baseline"]  = (df["eda_roll_mean_4"]  - eda_baseline).fillna(0.0)
    df["temp_vs_baseline"] = (df["temp_roll_mean_4"] - temp_baseline).fillna(0.0)

    log.debug("Added watch features (incl. relative-to-baseline).")
    return df
