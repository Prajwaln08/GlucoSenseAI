"""
Step 3 — Missing-value imputation (per user).

Runs strictly per user (never across users, never across a split boundary).
The rule depends on what kind of signal a column is:

  Class A — slowly-varying physiological (hr, eda, ibi, temp, acc, mets,
            calories_burned, glucose):
      A real dropout in a signal the user HAS is filled by carrying the last
      value forward (and back-filling the session start) within a bounded gap.
      Beyond the bound the value stays NaN.  A signal the user NEVER produced
      (the whole column is NaN) is left entirely NaN — we do not invent it.

  Class B — event-based (food macros, meal annotations, meal_flag):
      Absence means "no event", so NaN → 0 within the wear window.

  Class C — static demographics (age, bmi, hba1c, gender):
      Left as-is.  Unknown stays NaN (resolved per model tier later).

For every Class A signal we add a per-user availability flag `<col>_available`
(1 if the user produced that signal at all, else 0).  This drives the population
availability cohorts (Step 7) and is dropped as zero-variance by Step 5 inside a
cohort, so it never leaks into a model as a constant.

What this step does NOT do
--------------------------
It does not drop rows with a missing glucose target (that happens once targets
exist, in Step 4 / prepare) and it does not eliminate structural NaN — tree
models read NaN as "sensor absent"; NaN-intolerant models get an imputer at
train time (model tier, later phase).
"""

from __future__ import annotations

import pandas as pd

from src.config import WATCH_FFILL_STEPS, WATCH_BFILL_STEPS
from src.utils import get_logger

log = get_logger(__name__)

# Class A — physiological signals filled by bounded ffill/bfill.
# glucose is handled separately (ffill only, no bfill — never fabricate history).
WATCH_SIGNALS = [
    "hr", "eda", "ibi_mean", "ibi_rmssd", "temp",
    "acc_magnitude_mean", "acc_magnitude_std",
    "mets", "calories_burned",
]

# Class A signals that get a per-user availability flag.
STRUCTURAL_SIGNALS = WATCH_SIGNALS + ["glucose_rate_of_change"]

# Class B — event-based columns zero-filled (absence = no event).
EVENT_COLS = [
    "calorie", "total_carb", "protein", "total_fat", "dietary_fiber", "sugar",
    "meal_type_encoded", "amount_consumed_pct", "meal_flag",
]

# Class C — static per-user demographics: carry the constant value to every row.
DEMOGRAPHIC_COLS = ["age", "bmi", "hba1c", "gender"]


def impute_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute one user's 10-min frame.  The frame must be sorted by timestamp and
    contain a single user's rows.
    """
    df = df.sort_index().copy()

    # ── Availability flags (computed before filling) ──────────────────────────
    for col in STRUCTURAL_SIGNALS:
        present = col in df.columns and df[col].notna().any()
        df[f"{col}_available"] = int(present)

    # ── TARGET: glucose is NEVER imputed ──────────────────────────────────────
    # The CGM glucose value is the target variable, so its real gaps stay NaN — no
    # fabricated reading can ever become a training target. Rows with a missing
    # target are dropped downstream (get_xy). (Glucose is already clipped to
    # physiological bounds in Step 2.)

    # ── Class A: watch signals — bounded ffill + bfill (only if user has them) ─
    for col in WATCH_SIGNALS:
        if col not in df.columns or df[col].isna().all():
            continue  # structurally absent → leave NaN
        df[col] = df[col].ffill(limit=WATCH_FFILL_STEPS).bfill(limit=WATCH_BFILL_STEPS)
        # gaps longer than the bound remain NaN

    # ── Class B: event columns — zero-fill (absence = no event) ───────────────
    for col in EVENT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ── Class C: static demographics — ffill + bfill (constant per user) ──────
    for col in DEMOGRAPHIC_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    # glucose_rate_of_change left as-is (derived in Step 4)
    return df


def impute_table(table: pd.DataFrame, uid_col: str = "uid") -> pd.DataFrame:
    """
    Apply impute_user() to every user in the unified 10-min table, keeping each
    user's timeline isolated (no cross-user fill).
    """
    if uid_col not in table.columns:
        raise ValueError(f"impute_table: missing '{uid_col}' column.")

    out = [impute_user(g) for _, g in table.groupby(uid_col, sort=False)]
    result = pd.concat(out, axis=0)
    result.index.name = table.index.name or "timestamp"

    n_users = result[uid_col].nunique()
    log.info(f"Imputed {n_users} users; "
             f"{int(result.isna().sum().sum())} NaN cells remain (structural / long gaps).")
    return result
