"""
Resampling all data sources to a uniform 15-minute grid.

Rules per source:
  CGM (Dexcom 5-min)   → 15-min MEAN of glucose values
  HR (E4 ~1Hz / FitBit 1-min) → 15-min MEAN
  EDA (E4 4Hz)          → 15-min MEAN
  IBI (event-based)     → 15-min MEAN of intervals + RMSSD
  BVP (E4 64Hz)         → 15-min MEAN (optional — not a primary feature)
  ACC (E4 32Hz)         → 15-min MEAN of vector magnitude; STD as movement proxy
  TEMP (E4 4Hz)         → 15-min MEAN
  Food Log (event)      → 15-min SUM of carbs/calories/etc; GI = carb-weighted mean

After resampling all sources share the same 15-min DatetimeIndex,
which the merger then aligns with an outer join.
"""

import numpy as np
import pandas as pd

from src.config import RESAMPLE_INTERVAL
from src.utils import get_logger

log = get_logger(__name__)

# Alias for readability
FREQ = RESAMPLE_INTERVAL  # "15min"


# ══════════════════════════════════════════════════════════════════════════════
# Nature's Paper — per-source resamplers
# ══════════════════════════════════════════════════════════════════════════════

def resample_cgm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample Dexcom 5-min CGM to 15-min grid.
    Aggregation: mean (glucose changes slowly; averaging 3 readings is safe).
    """
    if df.empty:
        return df

    result = df.resample(FREQ).mean()
    result.index.name = "timestamp"
    log.debug(f"CGM resampled: {len(df)} → {len(result)} rows")
    return result


def resample_hr(df: pd.DataFrame, col: str = "hr") -> pd.DataFrame:
    """Resample heart rate (any cadence) to 15-min mean."""
    if df.empty or col not in df.columns:
        return df
    result = df[[col]].resample(FREQ).mean()
    result.index.name = "timestamp"
    log.debug(f"HR resampled: {len(df)} → {len(result)} rows")
    return result


def resample_eda(df: pd.DataFrame, col: str = "eda") -> pd.DataFrame:
    """Resample EDA (4 Hz) to 15-min mean."""
    if df.empty or col not in df.columns:
        return df
    result = df[[col]].resample(FREQ).mean()
    result.index.name = "timestamp"
    return result


def resample_temp(df: pd.DataFrame, col: str = "temp") -> pd.DataFrame:
    """Resample skin temperature (4 Hz) to 15-min mean."""
    if df.empty or col not in df.columns:
        return df
    result = df[[col]].resample(FREQ).mean()
    result.index.name = "timestamp"
    return result


def resample_bvp(df: pd.DataFrame, col: str = "bvp") -> pd.DataFrame:
    """Resample BVP (64 Hz) to 15-min mean. Optional signal — zero-fill if absent."""
    if df.empty or col not in df.columns:
        return df
    result = df[[col]].resample(FREQ).mean()
    result.index.name = "timestamp"
    return result


def resample_ibi(df: pd.DataFrame, col: str = "ibi") -> pd.DataFrame:
    """
    Resample IBI (event-based) to 15-min grid.
    Computes both mean IBI and RMSSD (HRV metric) within each window.

    RMSSD = sqrt( mean( (IBI[i+1] - IBI[i])² ) ) per window.
    A higher RMSSD means more heart rate variability (lower stress).
    """
    if df.empty or col not in df.columns:
        return df

    def rmssd(series: pd.Series) -> float:
        vals = series.dropna().values
        if len(vals) < 2:
            return np.nan
        diffs = np.diff(vals)
        return float(np.sqrt(np.mean(diffs ** 2)))

    mean_ibi  = df[[col]].resample(FREQ).mean().rename(columns={col: "ibi_mean"})
    rmssd_ibi = df[[col]].resample(FREQ).agg(rmssd).rename(columns={col: "ibi_rmssd"})

    result = pd.concat([mean_ibi, rmssd_ibi], axis=1)
    result.index.name = "timestamp"
    log.debug(f"IBI resampled: {len(df)} → {len(result)} rows (mean + RMSSD)")
    return result


def resample_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 3-axis ACC (32 Hz) to 15-min grid.
    Uses vector magnitude if available; otherwise computes it from x/y/z.
    Returns:
        acc_magnitude_mean  — mean activity level
        acc_magnitude_std   — movement variability (proxy for exercise intensity)
    """
    if df.empty:
        return df

    # Compute magnitude if not already present
    if "acc_magnitude" not in df.columns:
        for ax in ["acc_x", "acc_y", "acc_z"]:
            if ax not in df.columns:
                log.warning(f"ACC column {ax} missing — returning empty")
                return pd.DataFrame()
        df = df.copy()
        df["acc_magnitude"] = np.sqrt(
            df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
        )

    result = df[["acc_magnitude"]].resample(FREQ).agg(
        acc_magnitude_mean=("acc_magnitude", "mean"),
        acc_magnitude_std=("acc_magnitude", "std"),
    )
    result.index.name = "timestamp"
    log.debug(f"ACC resampled: {len(df)} → {len(result)} rows")
    return result


def resample_food_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample event-based food log to 15-min grid.

    Aggregation rules:
        calorie, total_carb, dietary_fiber, sugar, protein, total_fat → SUM
        gi_proxy (sugar / total_carb) → carb-weighted mean
        meal_flag → 1 if any meal in window else 0
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "calorie", "total_carb", "dietary_fiber", "sugar",
            "protein", "total_fat", "gi_proxy", "meal_flag",
        ])

    sum_cols = ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]
    present  = [c for c in sum_cols if c in df.columns]

    agg = df[present].resample(FREQ).sum()

    # GI proxy: sugar / total_carb, carb-weighted across rows in the window
    if "sugar" in df.columns and "total_carb" in df.columns:
        def _gi_proxy_agg(grp: pd.DataFrame) -> float:
            # gi_proxy = total_sugar / total_carbs in this window (range 0–1)
            carb_total = grp["total_carb"].sum()
            if carb_total <= 0:
                return 0.0
            return float(grp["sugar"].sum() / carb_total)

        gi_series = (
            df[["sugar", "total_carb"]]
            .resample(FREQ)
            .apply(_gi_proxy_agg)
        )
        agg["gi_proxy"] = gi_series.values

    # meal_flag: 1 if any row (meal event) exists in this 15-min window
    agg["meal_flag"] = (agg["total_carb"] > 0).astype(int) if "total_carb" in agg.columns else 0

    agg.index.name = "timestamp"
    log.debug(f"Food log resampled: {len(df)} events → {len(agg)} 15-min rows")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# High-level convenience functions
# ══════════════════════════════════════════════════════════════════════════════

def resample_np_user(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Resample all 8 Nature's Paper sources to 15-min for one user.

    Args:
        raw: dict returned by NaturePaperLoader.load()

    Returns:
        dict with same keys but all DataFrames on 15-min grid.
    """
    return {
        "cgm":  resample_cgm(raw.get("cgm", pd.DataFrame())),
        "hr":   resample_hr(raw.get("hr", pd.DataFrame())),
        "eda":  resample_eda(raw.get("eda", pd.DataFrame())),
        "ibi":  resample_ibi(raw.get("ibi", pd.DataFrame())),
        "bvp":  resample_bvp(raw.get("bvp", pd.DataFrame())),
        "acc":  resample_acc(raw.get("acc", pd.DataFrame())),
        "temp": resample_temp(raw.get("temp", pd.DataFrame())),
        "food": resample_food_log(raw.get("food", pd.DataFrame())),
    }


def resample_cgmacros_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    CGMacros data is natively at 15-min intervals (FreeStyle Libre).
    This function standardises the index to a clean 15-min grid and
    forward-fills tiny gaps without changing the underlying data.

    Returns the DataFrame unchanged except for a clean DatetimeIndex.
    """
    if df.empty:
        return df

    # Round timestamps to nearest 15-min boundary to snap any off-grid readings
    df = df.copy()
    df.index = df.index.round(FREQ)

    # Drop exact duplicates that arise from rounding
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df.index.name = "timestamp"

    log.debug(f"CGMacros: aligned to {FREQ} grid → {len(df)} rows")
    return df
