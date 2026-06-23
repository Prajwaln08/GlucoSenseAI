"""
Step 3 — Imputation checks.

Verifies bounded ffill/bfill for watch signals, ffill-only for glucose,
zero-fill for events, structural-absence left as NaN, availability flags, and
strict per-user isolation.

Run:
    conda activate glucosenseai
    pytest tests/check/test_step3_imputation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.config import WATCH_FFILL_STEPS
from src.data.step3_imputation import impute_user, impute_table


def _frame(values: dict, uid="cg-017", n=None) -> pd.DataFrame:
    n = n or len(next(iter(values.values())))
    idx = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    df = pd.DataFrame(values, index=idx)
    df["uid"] = uid
    return df


# ── Glucose: bounded ffill only, no bfill ─────────────────────────────────────

def test_glucose_ffill_bounded_no_bfill():
    # leading NaN must stay NaN (no back-fill of fabricated history);
    # a short gap after a real value is carried forward.
    g = [np.nan, 100.0, np.nan, np.nan, 130.0]
    out = impute_user(_frame({"glucose_mg_dl": g}))
    assert np.isnan(out["glucose_mg_dl"].iloc[0])      # no bfill
    assert out["glucose_mg_dl"].iloc[2] == 100.0       # ffill within bound
    assert out["glucose_mg_dl"].iloc[3] == 100.0


# ── Watch signals: bounded ffill + bfill ──────────────────────────────────────

def test_watch_signal_ffill_and_bfill():
    hr = [np.nan, 70.0, np.nan, 80.0, np.nan]
    out = impute_user(_frame({"hr": hr}))
    assert out["hr"].iloc[0] == 70.0     # bfill session start
    assert out["hr"].iloc[2] == 70.0     # ffill the gap
    assert out["hr"].iloc[4] == 80.0     # ffill tail


def test_watch_long_gap_stays_nan():
    n = WATCH_FFILL_STEPS + 5
    hr = [50.0] + [np.nan] * (n - 1)     # one value then a very long gap
    out = impute_user(_frame({"hr": hr}))
    # beyond the ffill bound the tail must remain NaN
    assert out["hr"].iloc[-1] != out["hr"].iloc[-1] or np.isnan(out["hr"].iloc[-1])
    assert np.isnan(out["hr"].iloc[-1])


# ── Structural absence: whole column NaN left untouched + flag 0 ──────────────

def test_structural_absence_left_nan_with_flag():
    out = impute_user(_frame({"hr": [70.0, 71.0], "eda": [np.nan, np.nan]}))
    assert out["eda"].isna().all()
    assert (out["eda_available"] == 0).all()
    assert (out["hr_available"] == 1).all()


# ── Events: zero-fill ─────────────────────────────────────────────────────────

def test_event_columns_zero_filled():
    out = impute_user(_frame({"total_carb": [np.nan, 30.0, np.nan]}))
    assert out["total_carb"].iloc[0] == 0.0
    assert out["total_carb"].iloc[1] == 30.0
    assert out["total_carb"].iloc[2] == 0.0


# ── Demographics: untouched ───────────────────────────────────────────────────

def test_demographics_left_as_is():
    out = impute_user(_frame({"hr": [70.0, 71.0], "bmi": [np.nan, np.nan]}))
    assert out["bmi"].isna().all()       # unknown demographic stays NaN


# ── Per-user isolation ────────────────────────────────────────────────────────

def test_impute_table_does_not_fill_across_users():
    a = _frame({"hr": [70.0, np.nan]}, uid="cg-001")
    b = _frame({"hr": [np.nan, 90.0]}, uid="cg-002")
    out = impute_table(pd.concat([a, b]))
    u2 = out[out["uid"] == "cg-002"]
    # cg-002's leading NaN is back-filled from its OWN 90, not carried from cg-001's 70
    assert u2["hr"].iloc[0] == 90.0
    assert out["uid"].nunique() == 2
