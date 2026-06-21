"""
Temporal / cyclical features — derived from the timestamp index.

These features are always computable from the timestamp alone — no past data
required.  They are also valid FUTURE regressors for NeuralProphet (since the
prediction horizon timestamp is known at inference time).

Cyclical encoding with sin/cos handles the wraparound discontinuity:
  23:59 and 00:01 should be "similar" — raw hour 23 vs 0 are not.
"""

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical hour/day-of-week features, binary flags, and monotonic trends.

    All features are derived from df.index (the 15-min DatetimeIndex).
    No past data is used — these are non-leaking temporal features.

    Args:
        df: DataFrame with UTC-aware DatetimeIndex.

    Returns:
        df with time feature columns appended.
    """
    df = df.copy()
    idx = df.index

    # Convert UTC to numeric components (hour, day_of_week)
    hour = idx.hour + idx.minute / 60.0          # 0–24 (fractional)
    dow  = idx.dayofweek                          # 0 = Monday … 6 = Sunday

    # ── Cyclical encodings ────────────────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dow  /  7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow  /  7)

    # ── Binary temporal flags ─────────────────────────────────────────────────
    df["is_weekend"]   = (dow >= 5).astype(int)     # Saturday or Sunday

    # Night: 22:00 – 06:00 (risk window for nocturnal hypoglycemia)
    df["is_night"]     = ((idx.hour >= 22) | (idx.hour < 6)).astype(int)

    # Morning: dawn phenomenon risk window
    df["is_morning"]   = ((idx.hour >= 6) & (idx.hour < 10)).astype(int)

    # Post-meal risk window: 08:00–10:00, 12:00–14:00, 18:00–20:00
    df["is_post_meal_window"] = (
        ((idx.hour >= 8)  & (idx.hour < 10)) |
        ((idx.hour >= 12) & (idx.hour < 14)) |
        ((idx.hour >= 18) & (idx.hour < 20))
    ).astype(int)

    log.debug("Added time features.")
    return df


def get_future_time_features(horizon_ts: pd.Timestamp) -> dict:
    """
    Compute time features for a future prediction horizon timestamp.
    Used as future regressors for NeuralProphet at inference time.

    Args:
        horizon_ts: the UTC timestamp we're predicting for.

    Returns:
        dict of {feature_name: value} for all time-based future regressors.
    """
    hour = horizon_ts.hour + horizon_ts.minute / 60.0
    dow  = horizon_ts.dayofweek

    return {
        "hour_sin":   float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos":   float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin":    float(np.sin(2 * np.pi * dow  /  7)),
        "dow_cos":    float(np.cos(2 * np.pi * dow  /  7)),
        "is_weekend": int(dow >= 5),
        "is_night":   int(horizon_ts.hour >= 22 or horizon_ts.hour < 6),
        "is_morning": int(6 <= horizon_ts.hour < 10),
    }
