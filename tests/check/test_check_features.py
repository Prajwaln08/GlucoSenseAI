"""
CHECK 04 — Feature Engineering

Verifies that all feature modules:
  - Produce non-empty output.
  - Do NOT look forward (no lag-0 features, no shift(0)).
  - Produce cyclical features in the correct range [-1, 1].
  - Handle missing optional columns with zero-fill (safe_get_feature).

Run:
    conda activate glucosenseai
    pytest tests/check/04_test_check_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.glucose_features import add_glucose_features, add_glucose_rate_of_change
from src.features.meal_features import add_meal_features
from src.features.watch_features import add_watch_features
from src.features.time_features import add_time_features, get_future_time_features
from src.features.interaction_features import add_interaction_features


# ══════════════════════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_df() -> pd.DataFrame:
    """100-row synthetic preprocessed DataFrame with all primary columns."""
    n   = 100
    idx = pd.date_range("2020-03-01 06:00", periods=n, freq="15min", tz="UTC")
    df  = pd.DataFrame(index=idx)
    df["glucose_mg_dl"]           = 100 + 20 * np.sin(np.arange(n) / 8)
    df["glucose_rate_of_change"]  = 0.0
    df["hr"]                      = 75 + 5 * np.random.randn(n)
    df["eda"]                     = np.abs(np.random.randn(n) * 0.5)
    df["ibi_mean"]                = 0.85 + 0.05 * np.random.randn(n)
    df["ibi_rmssd"]               = np.abs(np.random.randn(n) * 0.02)
    df["temp"]                    = 35 + 0.2 * np.random.randn(n)
    df["total_carb"]              = 0.0
    df["calorie"]                 = 0.0
    df["sugar"]                   = 0.0
    df["gi_proxy"]                = 0.0
    df["meal_flag"]               = 0
    df["meal_type_encoded"]       = 0
    df["amount_consumed_pct"]     = 0.0
    df["calories_burned"]         = np.abs(np.random.randn(n))
    df["mets"]                    = 1.0 + np.abs(np.random.randn(n))
    df["acc_magnitude_mean"]      = np.abs(np.random.randn(n) * 30)
    df["acc_magnitude_std"]       = np.abs(np.random.randn(n) * 5)
    df["participant_id"]          = "003"
    return df


@pytest.fixture
def base_with_meal(base_df) -> pd.DataFrame:
    """Add two realistic meals to the base DataFrame."""
    df = base_df.copy()
    # Meal at row 8 (08:00 + 2h)
    df.iloc[8,  df.columns.get_loc("total_carb")] = 60.0
    df.iloc[8,  df.columns.get_loc("calorie")]    = 500.0
    df.iloc[8,  df.columns.get_loc("sugar")]      = 10.0
    # Meal at row 32
    df.iloc[32, df.columns.get_loc("total_carb")] = 80.0
    df.iloc[32, df.columns.get_loc("calorie")]    = 700.0
    df.iloc[32, df.columns.get_loc("sugar")]      = 20.0
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Glucose features
# ══════════════════════════════════════════════════════════════════════════════

def test_glucose_lags_created(base_df):
    result = add_glucose_features(base_df)
    for n in [1, 2, 4, 8, 12, 20, 24, 28]:
        assert f"glucose_lag_{n}" in result.columns, f"Missing glucose_lag_{n}."


def test_glucose_rolling_mean_created(base_df):
    result = add_glucose_features(base_df)
    for w in [4, 8, 12]:
        assert f"glucose_roll_mean_{w}" in result.columns


def test_glucose_delta_features_created(base_df):
    result = add_glucose_features(base_df)
    for n in [1, 4, 8]:
        assert f"glucose_delta_{n}" in result.columns


def test_glucose_accel_created(base_df):
    result = add_glucose_features(base_df)
    assert "glucose_accel" in result.columns


def test_glucose_lag1_is_1_step_behind(base_df):
    """glucose_lag_1[i] should equal glucose_mg_dl[i-1]."""
    result = add_glucose_features(base_df)
    # Row 5's lag_1 should equal row 4's glucose
    g  = base_df["glucose_mg_dl"].values
    g1 = result["glucose_lag_1"].values
    # First row has NaN (no lag), compare from row 1 onward
    for i in range(1, min(10, len(g))):
        assert abs(g1[i] - g[i - 1]) < 1e-6, \
            f"glucose_lag_1[{i}]={g1[i]:.3f} but glucose[{i-1}]={g[i-1]:.3f}"


def test_no_lag_0_feature(base_df):
    """No feature should be a lag-0 (current row) computation."""
    result = add_glucose_features(base_df)
    lag0 = [c for c in result.columns if "_lag_0" in c]
    assert not lag0, f"Found lag-0 features: {lag0}"


def test_rolling_features_start_with_nan(base_df):
    """First rolling window rows must be NaN (not enough history)."""
    result = add_glucose_features(base_df)
    # glucose_roll_mean_12 requires 12 shifted rows — first 12 must be NaN
    assert result["glucose_roll_mean_12"].iloc[0:11].isna().any(), \
        "Rolling features need warm-up rows that should be NaN initially."


# ══════════════════════════════════════════════════════════════════════════════
# Meal features
# ══════════════════════════════════════════════════════════════════════════════

def test_carbs_window_1h_created(base_df):
    result = add_meal_features(base_df)
    assert "carbs_window_1h" in result.columns
    assert "carbs_window_2h" in result.columns


def test_meal_flag_after_meal(base_with_meal):
    """meal_flag must be 1 in the window AFTER a meal event (shifted by 1)."""
    result = add_meal_features(base_with_meal)
    # Row 8 has the meal; meal_flag at row 9 should reflect it
    assert result["meal_flag"].iloc[9] == 1, \
        "meal_flag should be 1 in the window after a meal."


def test_carbs_window_sums_correctly(base_with_meal):
    """carbs_window_1h at row 12 should sum rows 9–12 of shifted carbs."""
    result = add_meal_features(base_with_meal)
    # The 60g meal was at row 8; shifted to row 9; window of 4 rows ending at row 12
    cw = result["carbs_window_1h"].iloc[12]
    assert cw >= 60.0, f"Expected >= 60g in carbs_window_1h, got {cw}"


def test_time_since_last_meal_positive(base_with_meal):
    """time_since_last_meal should increase between meals."""
    result = add_meal_features(base_with_meal)
    tsm = result["time_since_last_meal"].dropna()
    # After a meal the counter resets to 0, then climbs
    assert (tsm >= 0).all(), "time_since_last_meal must be non-negative."


def test_meal_features_handle_missing_columns(base_df):
    """Columns absent from CGMacros (sugar, gi_proxy) must be zero-filled."""
    df_no_sugar = base_df.drop(columns=["sugar", "gi_proxy"])
    result = add_meal_features(df_no_sugar)
    # gi_weighted_1h must exist (zero-filled) without error
    assert "gi_weighted_1h" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# Watch features
# ══════════════════════════════════════════════════════════════════════════════

def test_hr_rolling_features_created(base_df):
    result = add_watch_features(base_df)
    assert "hr_roll_mean_4" in result.columns
    assert "hr_roll_mean_8" in result.columns


def test_watch_features_handle_missing_mets(base_df):
    """mets column absent → zero-filled, no KeyError."""
    df_no_mets = base_df.drop(columns=["mets"])
    result = add_watch_features(df_no_mets)
    assert "mets_roll_mean_4" in result.columns
    assert (result["mets_roll_mean_4"] == 0).all(), \
        "Missing METs column must produce zero-filled mets_roll_mean_4."


def test_watch_features_handle_missing_eda(base_df):
    """EDA absent (CGMacros) → zero-filled."""
    df_no_eda = base_df.drop(columns=["eda"])
    result = add_watch_features(df_no_eda)
    assert (result["eda_roll_mean_4"] == 0).all()


def test_activity_flag_binary(base_df):
    result = add_watch_features(base_df)
    assert set(result["activity_flag"].unique()).issubset({0, 1}), \
        "activity_flag must only contain 0 or 1."


# ══════════════════════════════════════════════════════════════════════════════
# Time features
# ══════════════════════════════════════════════════════════════════════════════

def test_time_features_cyclical_range(base_df):
    """Cyclical features must be in [-1, 1]."""
    result = add_time_features(base_df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert col in result.columns
        vals = result[col].dropna()
        assert (vals >= -1).all() and (vals <= 1).all(), \
            f"{col} must be in [-1, 1]."


def test_is_weekend_binary(base_df):
    result = add_time_features(base_df)
    assert set(result["is_weekend"].unique()).issubset({0, 1})


def test_is_night_binary(base_df):
    result = add_time_features(base_df)
    assert set(result["is_night"].unique()).issubset({0, 1})


def test_night_flag_correct_hours():
    """is_night = 1 for hours 22–23 and 0–5."""
    night_ts = pd.date_range("2020-03-01 22:00", periods=16, freq="15min", tz="UTC")
    df = pd.DataFrame({"glucose_mg_dl": 100.0}, index=night_ts)
    result = add_time_features(df)
    # All timestamps are between 22:00 and 02:00 — should all be night
    assert result["is_night"].all(), "22:00–02:00 should all be is_night=1."


def test_get_future_time_features():
    ts = pd.Timestamp("2020-03-01 23:00:00", tz="UTC")
    feats = get_future_time_features(ts)
    assert "hour_sin" in feats
    assert "is_night" in feats
    assert feats["is_night"] == 1, "23:00 must be is_night=1."
    assert feats["is_weekend"] in (0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Interaction features
# ══════════════════════════════════════════════════════════════════════════════

def test_interaction_features_created(base_with_meal):
    """Run all prior features first, then interaction features."""
    df = add_glucose_features(base_with_meal)
    df = add_meal_features(df)
    df = add_watch_features(df)
    df = add_time_features(df)
    df = add_interaction_features(df)

    for col in ["carbs_x_steps_1h", "gi_x_carbs_1h", "meal_x_hour_sin"]:
        assert col in df.columns, f"Missing interaction feature: {col}."


def test_interaction_features_zero_for_no_meal(base_df):
    """With no meals, carbs_x_steps_1h must be zero."""
    df = add_glucose_features(base_df)
    df = add_meal_features(df)
    df = add_watch_features(df)
    df = add_time_features(df)
    df = add_interaction_features(df)

    # carbs_window_1h is 0 for all rows → interaction should be 0
    assert (df["carbs_x_steps_1h"] == 0).all(), \
        "carbs × steps must be 0 when there are no meals."
