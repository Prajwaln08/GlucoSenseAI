"""
Glucose-derived features — all past-looking (no forward leakage).

Every lag and rolling computation uses shift(n) with n >= 1 before
applying the window, so the current observation is never part of its
own feature computation.

At 15-min intervals: lag_1 = 15 min ago, lag_4 = 1 h ago, lag_8 = 2 h ago.

IMPORTANT — CGM dependency
--------------------------
All features in this module require live or historical CGM glucose readings.
They must NOT be computed for post-CGM inference rows.

Use build_cgm_features(df, mode) as the safe entry point from the pipeline:
it raises immediately if mode == "post_cgm", preventing silent feature leakage.
"""

from typing import Literal

import pandas as pd

from src.utils import safe_get_feature, get_logger

log = get_logger(__name__)

# Canonical set of column names produced by this module.
# Kept in sync with feature_groups.CGM_GROUP — used for assertions and tests.
CGM_OUTPUT_COLS: frozenset[str] = frozenset({
    "glucose_lag_1", "glucose_lag_2", "glucose_lag_4", "glucose_lag_8",
    "glucose_lag_12", "glucose_lag_20", "glucose_lag_24", "glucose_lag_28",
    "glucose_roll_mean_4", "glucose_roll_mean_8", "glucose_roll_mean_12",
    "glucose_roll_std_4", "glucose_roll_std_8",
    "glucose_delta_1", "glucose_delta_4", "glucose_delta_8",
    "glucose_accel",
    "glucose_rate_of_change",
})


def add_glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all glucose lag, rolling, rate-of-change, and acceleration features.

    Operates on df['glucose_mg_dl'] — must exist and have no NaN after preprocessing.
    All new columns are added in-place and returned.

    Args:
        df: preprocessed user DataFrame with a sorted DatetimeIndex.

    Returns:
        df with new glucose feature columns appended.
    """
    df = df.copy()
    g = df["glucose_mg_dl"]

    # ── Lag features ──────────────────────────────────────────────────────────
    # Each shift(n) looks n timesteps (n × 15 min) into the past.
    for n in [1, 2, 4, 8, 12, 20, 24, 28]:
        df[f"glucose_lag_{n}"] = g.shift(n)

    # ── Rolling mean (shifted by 1 so current is excluded) ───────────────────
    g_shifted = g.shift(1)    # shift first, then roll — prevents current-row leakage
    for window in [4, 8, 12]:
        df[f"glucose_roll_mean_{window}"] = (
            g_shifted.rolling(window=window, min_periods=1).mean()
        )

    # ── Rolling std ──────────────────────────────────────────────────────────
    for window in [4, 8]:
        df[f"glucose_roll_std_{window}"] = (
            g_shifted.rolling(window=window, min_periods=2).std()
        )

    # ── Rate of change (delta) ────────────────────────────────────────────────
    # delta_n = glucose - glucose n windows ago (mg/dL per n×15 min)
    df["glucose_delta_1"] = g.diff(1)       # 15-min change
    df["glucose_delta_4"] = g.diff(4)       # 1-hour change
    df["glucose_delta_8"] = g.diff(8)       # 2-hour change

    # Shift deltas by 1 so they represent PAST change, not current
    df["glucose_delta_1"] = df["glucose_delta_1"].shift(1)
    df["glucose_delta_4"] = df["glucose_delta_4"].shift(1)
    df["glucose_delta_8"] = df["glucose_delta_8"].shift(1)

    # ── Acceleration (second derivative) ─────────────────────────────────────
    # Diff of the 1-lag delta — both already shifted, so no leakage.
    df["glucose_accel"] = df["glucose_delta_1"].diff(1)

    log.debug(f"Added {len([c for c in df.columns if 'glucose_' in c and c != 'glucose_mg_dl'])} glucose features")
    return df


def build_cgm_features(
    df: pd.DataFrame,
    mode: Literal["cgm_active", "post_cgm"],
) -> pd.DataFrame:
    """
    Safe pipeline entry point for CGM feature construction.

    Raises RuntimeError immediately when mode == "post_cgm" so that the
    post-CGM path can never accidentally compute glucose lags, deltas, or
    rolling statistics from stale CGM values.

    Args:
        df:   Preprocessed user DataFrame with glucose_mg_dl column.
        mode: "cgm_active" to proceed normally; "post_cgm" raises.

    Returns:
        df with all CGM features added (cgm_active mode only).
    """
    if mode == "post_cgm":
        raise RuntimeError(
            "build_cgm_features() called with mode='post_cgm'. "
            "CGM-derived features (lags, deltas, rolling means) must not be "
            "computed for post-CGM rows. Use the virtual feature pipeline instead."
        )
    df = add_glucose_features(df)
    df = add_glucose_rate_of_change(df)
    return df


def add_glucose_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the Dexcom-provided rate-of-change column if available,
    otherwise derive it from 15-min glucose differences.
    This is a dataset-specific feature (NP only) — CGMacros uses derived version.
    """
    df = df.copy()
    roc = safe_get_feature(df, "glucose_rate_of_change")

    if (roc != 0).any():
        # Dexcom-provided column present — just shift by 1 to make it past-only
        df["glucose_rate_of_change"] = roc.shift(1)
    else:
        # Derive from glucose differences (CGMacros / fallback)
        df["glucose_rate_of_change"] = df["glucose_mg_dl"].diff(1).shift(1) / 15.0

    return df
