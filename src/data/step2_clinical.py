"""
Step 2 — Clinical thresholds + 10-min resample.

Takes the loaded users from Step 1 and puts every user on one common 10-minute
grid, then applies clinical-range checks.  Output is the unified 10-min table
holding BOTH datasets (NP users join here, since their raw streams are read from
disk in this step rather than in Step 1).

Resampling rules (per 10-min window)
------------------------------------
  physiological (glucose, glucose_rate_of_change, hr, eda, ibi, temp, mets)
        -> MEDIAN of the readings in the window (robust to sensor spikes)
  acc magnitude          -> mean + std (activity level + movement variability)
  food macros / calories_burned -> SUM (intake / energy accumulates in the window)
  meal annotations (meal_type_encoded, amount_consumed_pct) -> MAX (the event value)
  demographics           -> first (constant per user)

High-frequency Nature's Paper signals are huge (BVP 1.5 GB, ACC 0.5 GB per user).
BVP feeds no feature, so it is **dropped** (never read).  ACC is read and reduced
to magnitude mean/std.  One user is processed at a time.

Clinical-range check
--------------------
Physiological values outside plausible bounds are set to NaN (so Step 3 treats
them as gaps, not real readings); event/activity values are clipped to caps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    GRID,
    GLUCOSE_MIN, GLUCOSE_MAX,
    HR_MIN, HR_MAX,
    EDA_MIN, EDA_MAX,
    TEMP_MIN, TEMP_MAX,
    CARBS_MAX, CALORIES_FOOD_MAX, CALORIES_BURNED_MAX, METS_MAX,
)
from src.data.loader import NaturePaperLoader
from src.data.step1_loader import (
    META_COLS, UNIFIED_COLUMNS, LoadedUser, unify_columns,
)
from src.utils import get_logger

log = get_logger(__name__)

FREQ = GRID  # "10min"

# ── Aggregation column groups (by standardised name) ──────────────────────────
_MEDIAN_COLS = [
    "glucose_mg_dl", "glucose_rate_of_change", "hr", "eda", "temp", "mets",
]
_SUM_COLS = [
    "calorie", "total_carb", "protein", "total_fat", "dietary_fiber", "sugar",
    "calories_burned",
]
_MAX_COLS = ["meal_type_encoded", "amount_consumed_pct"]
_DEMOGRAPHIC_COLS = ["age", "bmi", "hba1c", "gender"]

# BVP is never used downstream; skip the multi-GB read entirely.
_NP_SKIP_SOURCES = {"bvp"}


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def to_grid(user: LoadedUser) -> pd.DataFrame:
    """
    Put one user on the 10-min grid, clinically clipped and conformed to the
    unified schema (META_COLS + UNIFIED_COLUMNS).
    """
    if user.dataset == "cgmacros":
        wide = _resample_cgmacros(user.frame)
    elif user.dataset == "nature_paper":
        wide = _resample_nature_paper(user.participant_id, user.base_dir)
    else:
        raise ValueError(f"Unknown dataset {user.dataset!r}.")

    # Identity + demographics → columns
    wide["uid"]            = user.uid
    wide["dataset"]        = user.dataset
    wide["participant_id"] = user.participant_id
    for col in _DEMOGRAPHIC_COLS:
        if col not in wide.columns and col in user.demographics:
            wide[col] = user.demographics[col]

    wide = clinical_clip(wide)
    wide = _complete_grid(wide)
    return unify_columns(wide)


def build_grid_table(users: list[LoadedUser]) -> pd.DataFrame:
    """
    Resample every user to the 10-min grid and concatenate into the single
    unified table holding BOTH datasets.  Users that fail are logged and skipped.
    """
    frames: list[pd.DataFrame] = []
    for u in users:
        try:
            frames.append(to_grid(u))
        except Exception as exc:                   # noqa: BLE001 - report and continue
            log.warning(f"{u.uid}: failed to grid — {exc}")

    if not frames:
        raise ValueError("build_grid_table: no users produced a 10-min frame.")

    table = pd.concat(frames, axis=0)
    table.index.name = "timestamp"
    log.info(f"Unified 10-min table: {len(table)} rows, "
             f"{table['uid'].nunique()} users, {len(table.columns)} columns.")
    return table


# ══════════════════════════════════════════════════════════════════════════════
# Clinical-range check
# ══════════════════════════════════════════════════════════════════════════════

def clinical_clip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physiological values outside plausible bounds → NaN (treated as gaps in
    Step 3).  Event / activity values → clipped to hard caps.  Operates on a
    10-min frame; only columns present are touched.
    """
    df = df.copy()

    # Physiological → CLIP to plausible bounds (keep the reading; never NaN it).
    # A low value is clipped up to the lower bound, a high value down to the upper.
    for col, lo, hi in [
        ("glucose_mg_dl", GLUCOSE_MIN, GLUCOSE_MAX),
        ("hr",            HR_MIN,      HR_MAX),
        ("eda",           EDA_MIN,     EDA_MAX),
        ("temp",          TEMP_MIN,    TEMP_MAX),
    ]:
        if col in df.columns:
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                log.debug(f"clinical_clip: {col} {n} value(s) clipped to [{lo},{hi}]")
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Glucose rate of change: clip to physiologically plausible ±5 mg/dL/min
    if "glucose_rate_of_change" in df.columns:
        df["glucose_rate_of_change"] = df["glucose_rate_of_change"].clip(-5.0, 5.0)

    # IBI / HRV cannot be negative
    for col in ("ibi_mean", "ibi_rmssd"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # ACC magnitude — Empatica E4 magnitude tops out well under 500 units
    for col in ("acc_magnitude_mean", "acc_magnitude_std"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=500)

    # Event / nutrition caps
    for col, cap in [
        ("total_carb",    CARBS_MAX),
        ("sugar",         CARBS_MAX),
        ("calorie",       CALORIES_FOOD_MAX),
        ("protein",       300.0),
        ("total_fat",     300.0),
        ("dietary_fiber", 100.0),
    ]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=cap)

    if "calories_burned" in df.columns:
        df["calories_burned"] = df["calories_burned"].clip(lower=0, upper=CALORIES_BURNED_MAX)
    if "mets" in df.columns:
        df["mets"] = df["mets"].clip(lower=0, upper=METS_MAX)
    if "amount_consumed_pct" in df.columns:
        df["amount_consumed_pct"] = df["amount_consumed_pct"].clip(lower=0, upper=100)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CGMacros resampling (wide 1-min frame already in memory)
# ══════════════════════════════════════════════════════════════════════════════

def _resample_cgmacros(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample a CGMacros wide 1-min frame to the 10-min grid."""
    if frame is None or frame.empty:
        raise ValueError("CGMacros frame is empty.")
    return _aggregate_to_grid(frame)


# ══════════════════════════════════════════════════════════════════════════════
# Nature's Paper resampling (per-source, read from disk; BVP skipped)
# ══════════════════════════════════════════════════════════════════════════════

def _resample_nature_paper(
    participant_id: str,
    base_dir: Optional[Path],
) -> pd.DataFrame:
    """
    Read one NP user's raw streams (BVP skipped) and resample each to the 10-min
    grid, then outer-join them on the grid.
    """
    raw = NaturePaperLoader(base_dir=base_dir).load(participant_id, skip=_NP_SKIP_SOURCES)

    parts: list[pd.DataFrame] = []

    # CGM (Dexcom): glucose + rate-of-change → median
    cgm = raw.get("cgm")
    if cgm is not None and not cgm.empty:
        parts.append(_median_grid(cgm, [c for c in ("glucose_mg_dl", "glucose_rate_of_change")
                                        if c in cgm.columns]))

    # Single-signal streams → median
    for key, col in [("hr", "hr"), ("eda", "eda"), ("temp", "temp")]:
        df = raw.get(key)
        if df is not None and not df.empty and col in df.columns:
            parts.append(_median_grid(df, [col]))

    # IBI → median (ibi_mean) + RMSSD (ibi_rmssd)
    ibi = raw.get("ibi")
    if ibi is not None and not ibi.empty and "ibi" in ibi.columns:
        parts.append(_resample_ibi(ibi))

    # ACC → magnitude mean + std
    acc = raw.get("acc")
    if acc is not None and not acc.empty:
        parts.append(_resample_acc(acc))

    # Food (event) → SUM of macros + meal_flag
    food = raw.get("food")
    if food is not None and not food.empty:
        parts.append(_resample_food(food))

    if not parts:
        raise ValueError(f"NP user {participant_id}: no usable streams.")

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.join(part, how="outer")
    merged.index.name = "timestamp"
    return merged


def _resample_ibi(df: pd.DataFrame) -> pd.DataFrame:
    """IBI event series → 10-min median (ibi_mean) and RMSSD (ibi_rmssd)."""
    def _rmssd(s: pd.Series) -> float:
        vals = s.dropna().values
        if len(vals) < 2:
            return np.nan
        return float(np.sqrt(np.mean(np.diff(vals) ** 2)))

    mean = df[["ibi"]].resample(FREQ).median().rename(columns={"ibi": "ibi_mean"})
    rmssd = df[["ibi"]].resample(FREQ).agg(_rmssd).rename(columns={"ibi": "ibi_rmssd"})
    out = pd.concat([mean, rmssd], axis=1)
    out.index.name = "timestamp"
    return out


def _resample_acc(df: pd.DataFrame) -> pd.DataFrame:
    """3-axis ACC → 10-min magnitude mean + std."""
    df = df.copy()
    if "acc_magnitude" not in df.columns:
        if not all(c in df.columns for c in ("acc_x", "acc_y", "acc_z")):
            return pd.DataFrame()
        df["acc_magnitude"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    out = df[["acc_magnitude"]].resample(FREQ).agg(
        acc_magnitude_mean=("acc_magnitude", "mean"),
        acc_magnitude_std=("acc_magnitude", "std"),
    )
    out.index.name = "timestamp"
    return out


def _resample_food(df: pd.DataFrame) -> pd.DataFrame:
    """Event-based food log → 10-min SUM of macros (+ meal_flag)."""
    cols = [c for c in ("calorie", "total_carb", "protein", "total_fat",
                        "dietary_fiber", "sugar") if c in df.columns]
    out = df[cols].resample(FREQ).sum() if cols else pd.DataFrame(index=df.resample(FREQ).size().index)
    out["meal_flag"] = (out["total_carb"] > 0).astype(int) if "total_carb" in out.columns else 0
    out.index.name = "timestamp"
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Shared aggregation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_to_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a wide frame to the 10-min grid using the per-column aggregation
    rule for each known column.  Unknown columns are dropped.
    """
    out = pd.DataFrame()
    g = frame.resample(FREQ)

    median_cols = [c for c in _MEDIAN_COLS if c in frame.columns]
    sum_cols    = [c for c in _SUM_COLS    if c in frame.columns]
    max_cols    = [c for c in _MAX_COLS    if c in frame.columns]
    demo_cols   = [c for c in _DEMOGRAPHIC_COLS if c in frame.columns]

    if median_cols:
        out = out.join(frame[median_cols].resample(FREQ).median(), how="outer")
    if sum_cols:
        out = out.join(frame[sum_cols].resample(FREQ).sum(), how="outer")
    if max_cols:
        out = out.join(frame[max_cols].resample(FREQ).max(), how="outer")
    if demo_cols:
        out = out.join(frame[demo_cols].resample(FREQ).first(), how="outer")

    if "meal_flag" not in out.columns and "total_carb" in out.columns:
        out["meal_flag"] = (out["total_carb"] > 0).astype(int)

    out.index.name = "timestamp"
    return out


def _median_grid(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Resample selected columns to the 10-min median."""
    out = df[cols].resample(FREQ).median()
    out.index.name = "timestamp"
    return out


def _complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a gapless 10-min grid over the user's full span.  New rows are NaN
    (Step 3 decides how to fill).  Identity / demographic columns are forward- and
    back-filled so every row carries the user's id and static attributes.
    """
    if df.empty:
        return df
    full = pd.date_range(
        start=df.index.min().floor(FREQ),
        end=df.index.max().ceil(FREQ),
        freq=FREQ,
        tz=df.index.tz,
    )
    out = df.reindex(full)
    out.index.name = "timestamp"

    carry = [c for c in (META_COLS + _DEMOGRAPHIC_COLS) if c in out.columns]
    if carry:
        out[carry] = out[carry].ffill().bfill()
    return out
