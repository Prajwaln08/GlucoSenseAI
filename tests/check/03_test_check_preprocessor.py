"""
CHECK 03 — Preprocessor

Tests outlier clipping and imputation on synthetic DataFrames.
Verifies physiological bounds, forward-fill limits, and row-dropping rules.

Run:
    conda activate glucosenseai
    pytest tests/check/03_test_check_preprocessor.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import clip_outliers, impute, preprocess_user
from src.config import (
    GLUCOSE_MIN, GLUCOSE_MAX,
    HR_MIN, HR_MAX,
    EDA_MIN, EDA_MAX,
    TEMP_MIN, TEMP_MAX,
    MAX_CGM_GAP_WINDOWS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_clean_df(n: int = 100, add_glucose: bool = True) -> pd.DataFrame:
    """Synthetic 15-min DataFrame with all key columns in physiological range."""
    idx = pd.date_range("2020-03-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(index=idx)
    if add_glucose:
        df["glucose_mg_dl"] = 100 + 20 * np.sin(np.arange(n) / 8)
    df["hr"]   = 75 + 5 * np.random.randn(n)
    df["eda"]  = np.abs(np.random.randn(n) * 0.5)
    df["temp"] = 35 + 0.2 * np.random.randn(n)
    df["total_carb"] = 0.0
    df["calorie"]    = 0.0
    df["meal_flag"]  = 0
    df["participant_id"] = "003"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Outlier clipping tests
# ══════════════════════════════════════════════════════════════════════════════

def test_clip_glucose_below_min_becomes_nan():
    df = _make_clean_df()
    df.loc[df.index[5], "glucose_mg_dl"] = 20.0   # below 40 mg/dL
    result = clip_outliers(df)
    assert np.isnan(result["glucose_mg_dl"].iloc[5]), \
        "Glucose below 40 must become NaN (not clipped to boundary)."


def test_clip_glucose_above_max_becomes_nan():
    df = _make_clean_df()
    df.loc[df.index[5], "glucose_mg_dl"] = 500.0  # above 400 mg/dL
    result = clip_outliers(df)
    assert np.isnan(result["glucose_mg_dl"].iloc[5]), \
        "Glucose above 400 must become NaN."


def test_clip_valid_glucose_unchanged():
    df = _make_clean_df()
    df["glucose_mg_dl"] = 120.0
    result = clip_outliers(df)
    assert (result["glucose_mg_dl"] == 120.0).all(), \
        "Valid glucose values must not be changed by clipping."


def test_clip_hr_below_30_becomes_nan():
    df = _make_clean_df()
    df.loc[df.index[3], "hr"] = 20.0
    result = clip_outliers(df)
    assert np.isnan(result["hr"].iloc[3])


def test_clip_hr_above_220_becomes_nan():
    df = _make_clean_df()
    df.loc[df.index[3], "hr"] = 280.0
    result = clip_outliers(df)
    assert np.isnan(result["hr"].iloc[3])


def test_clip_eda_negative_becomes_nan():
    df = _make_clean_df()
    df["eda"] = -0.5
    result = clip_outliers(df)
    assert result["eda"].isna().all(), "Negative EDA values must become NaN."


def test_clip_temp_below_20_becomes_nan():
    df = _make_clean_df()
    df.loc[df.index[0], "temp"] = 15.0
    result = clip_outliers(df)
    assert np.isnan(result["temp"].iloc[0])


def test_clip_carbs_above_500_capped():
    df = _make_clean_df()
    df.loc[df.index[0], "total_carb"] = 9999.0
    result = clip_outliers(df)
    assert result["total_carb"].iloc[0] == 500.0, \
        "Carbs above 500g should be clipped to 500."


def test_clip_does_not_modify_original():
    df = _make_clean_df()
    df.loc[df.index[0], "glucose_mg_dl"] = 999.0
    _ = clip_outliers(df)
    # Original must not change (clip_outliers uses df.copy())
    assert df.loc[df.index[0], "glucose_mg_dl"] == 999.0


# ══════════════════════════════════════════════════════════════════════════════
# Imputation tests
# ══════════════════════════════════════════════════════════════════════════════

def test_impute_glucose_forward_fill():
    """CGM gap ≤ 45 min (3 windows) must be forward-filled."""
    df = _make_clean_df(20)
    df.iloc[5, df.columns.get_loc("glucose_mg_dl")] = np.nan   # 1 gap → filled
    df.iloc[6, df.columns.get_loc("glucose_mg_dl")] = np.nan   # 2nd gap → filled
    result = impute(df)
    assert not result["glucose_mg_dl"].isna().any(), \
        "Short CGM gaps (≤ 45 min) must be forward-filled."


def test_impute_glucose_long_gap_drops_rows():
    """
    A gap of 4 consecutive NaN rows (> 45 min) must result in those rows
    being DROPPED (not left as NaN) after imputation.
    """
    df = _make_clean_df(30)
    # Make 4 consecutive NaN rows — exceeds the 3-window forward-fill limit
    for i in range(5, 9):
        df.iloc[i, df.columns.get_loc("glucose_mg_dl")] = np.nan

    result = impute(df)
    # Rows with unfillable NaN glucose should be dropped
    assert result["glucose_mg_dl"].isna().sum() == 0, \
        "No NaN glucose should remain after imputation + row dropping."
    assert len(result) < len(df), \
        "Rows with long CGM gaps must be dropped."


def test_impute_food_columns_zero_filled():
    """Food/carb NaN should be filled with 0 (no event = no intake)."""
    df = _make_clean_df(20)
    df["total_carb"] = np.nan
    df["calorie"]    = np.nan
    result = impute(df)
    assert (result["total_carb"] == 0).all(), "total_carb NaN must be zero-filled."
    assert (result["calorie"]    == 0).all(), "calorie NaN must be zero-filled."


def test_impute_hr_interpolated():
    """Short HR gaps should be linearly interpolated."""
    df = _make_clean_df(20)
    df.iloc[10, df.columns.get_loc("hr")] = np.nan
    result = impute(df)
    assert not result["hr"].isna().any(), "Short HR gaps must be interpolated."


def test_impute_does_not_modify_original():
    df = _make_clean_df(20)
    df.iloc[5, df.columns.get_loc("glucose_mg_dl")] = np.nan
    _ = impute(df)
    assert np.isnan(df.iloc[5]["glucose_mg_dl"]), \
        "impute() must use df.copy() — original must not be modified."


# ══════════════════════════════════════════════════════════════════════════════
# Full preprocess_user pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_preprocess_user_no_nan_after(tmp_path):
    """After full preprocessing, no NaN should remain in the output."""
    df = _make_clean_df(100)
    # Inject some outliers and gaps
    df.iloc[10, df.columns.get_loc("glucose_mg_dl")] = 999.0   # outlier → NaN → ffill
    df.iloc[20, df.columns.get_loc("hr")]            = np.nan   # gap → interpolate
    result = preprocess_user(df, user_id="003")

    assert result["glucose_mg_dl"].isna().sum() == 0
    assert result["hr"].isna().sum() == 0
    assert result.isnull().sum().sum() == 0, "No NaN should remain after preprocessing."


def test_preprocess_user_glucose_in_range():
    """After preprocessing, all glucose values must be in [40, 400]."""
    df = _make_clean_df(100)
    df.iloc[0, df.columns.get_loc("glucose_mg_dl")] = 10.0   # extreme outlier
    result = preprocess_user(df, user_id="003")

    glucose = result["glucose_mg_dl"]
    assert (glucose >= GLUCOSE_MIN).all() and (glucose <= GLUCOSE_MAX).all()


def test_preprocess_user_returns_fewer_or_equal_rows():
    """Preprocessing can only remove rows, never add them."""
    df = _make_clean_df(50)
    result = preprocess_user(df, user_id="003")
    assert len(result) <= len(df)


def test_preprocess_user_index_still_sorted():
    """Preprocessing must preserve timestamp ordering."""
    df = _make_clean_df(50)
    result = preprocess_user(df, user_id="003")
    assert result.index.is_monotonic_increasing, \
        "Index must remain sorted after preprocessing."
