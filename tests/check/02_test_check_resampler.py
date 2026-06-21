"""
CHECK 02 — Resampler

Tests that all per-source resamplers correctly aggregate to a 15-min grid
with the right aggregation method (sum vs mean) and no data loss beyond
what is expected from averaging.

Run:
    conda activate glucosenseai
    pytest tests/check/02_test_check_resampler.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.resampler import (
    resample_cgm,
    resample_hr,
    resample_eda,
    resample_ibi,
    resample_acc,
    resample_food_log,
    resample_np_user,
    resample_cgmacros_user,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_5min_cgm(n_readings: int = 36) -> pd.DataFrame:
    """36 readings = 3 hours at 5-min intervals → should resample to 12 rows."""
    idx = pd.date_range("2020-03-01 06:00", periods=n_readings, freq="5min", tz="UTC")
    return pd.DataFrame({"glucose_mg_dl": 100 + np.arange(n_readings) * 0.5}, index=idx)


def _make_1hz_hr(n_seconds: int = 900) -> pd.DataFrame:
    """900 seconds = 15 minutes of 1Hz HR data → should resample to 1 row."""
    idx = pd.date_range("2020-03-01 06:00", periods=n_seconds, freq="1s", tz="UTC")
    return pd.DataFrame({"hr": 70 + np.random.randn(n_seconds) * 3}, index=idx)


def _make_4hz_eda(n_samples: int = 3600) -> pd.DataFrame:
    """3600 samples at 4Hz = 15 minutes → should resample to 1 row."""
    idx = pd.date_range("2020-03-01 06:00", periods=n_samples, freq="250ms", tz="UTC")
    return pd.DataFrame({"eda": np.abs(np.random.randn(n_samples) * 0.3)}, index=idx)


def _make_food_events() -> pd.DataFrame:
    """3 meal events that fall in 2 different 15-min windows."""
    idx = pd.DatetimeIndex([
        "2020-03-01 08:02:00+00:00",   # window 08:00
        "2020-03-01 08:10:00+00:00",   # window 08:00 (same window)
        "2020-03-01 08:18:00+00:00",   # window 08:15
    ])
    return pd.DataFrame({
        "total_carb": [30.0, 10.0, 50.0],
        "calorie":    [200.0, 80.0, 400.0],
        "sugar":      [5.0, 2.0, 20.0],
        "protein":    [10.0, 5.0, 15.0],
        "total_fat":  [8.0, 3.0, 12.0],
        "dietary_fiber": [2.0, 1.0, 3.0],
    }, index=idx)


# ══════════════════════════════════════════════════════════════════════════════
# CGM resampling
# ══════════════════════════════════════════════════════════════════════════════

def test_cgm_resample_row_count():
    """3 hours of 5-min data → 12 rows at 15-min."""
    cgm = _make_5min_cgm(36)
    result = resample_cgm(cgm)
    # 3 readings per 15-min window, 36 total → expect ~12 windows
    assert 10 <= len(result) <= 14, f"Expected ~12 rows, got {len(result)}"


def test_cgm_resample_uses_mean():
    """Verify mean aggregation: 3 readings of 100, 101, 102 → mean = 101."""
    idx = pd.date_range("2020-03-01 06:00", periods=3, freq="5min", tz="UTC")
    cgm = pd.DataFrame({"glucose_mg_dl": [100.0, 101.0, 102.0]}, index=idx)
    result = resample_cgm(cgm)
    expected_mean = 101.0
    assert abs(result["glucose_mg_dl"].iloc[0] - expected_mean) < 0.01


def test_cgm_resample_index_is_15min_grid():
    """Resampled index should be on 15-min boundaries."""
    result = resample_cgm(_make_5min_cgm(36))
    diffs = result.index.to_series().diff().dropna()
    expected = pd.Timedelta("15min")
    assert (diffs == expected).all(), "All intervals should be exactly 15 minutes."


def test_cgm_resample_empty_input():
    """Empty input returns empty output without error."""
    empty = pd.DataFrame(columns=["glucose_mg_dl"])
    result = resample_cgm(empty)
    assert result.empty


# ══════════════════════════════════════════════════════════════════════════════
# HR resampling
# ══════════════════════════════════════════════════════════════════════════════

def test_hr_resample_single_window():
    """900 seconds of HR → 1 row at 15-min grid."""
    hr  = _make_1hz_hr(900)
    res = resample_hr(hr)
    # All data falls within one 15-min window
    assert len(res) == 1


def test_hr_resample_mean_correct():
    """Mean of constant HR values should equal that constant."""
    idx = pd.date_range("2020-03-01 06:00", periods=60, freq="1s", tz="UTC")
    hr  = pd.DataFrame({"hr": [80.0] * 60}, index=idx)
    res = resample_hr(hr)
    assert abs(res["hr"].iloc[0] - 80.0) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# EDA resampling
# ══════════════════════════════════════════════════════════════════════════════

def test_eda_resample_column_preserved():
    eda = _make_4hz_eda(3600)
    res = resample_eda(eda)
    assert "eda" in res.columns


def test_eda_values_non_negative():
    eda = _make_4hz_eda(3600)
    res = resample_eda(eda)
    assert (res["eda"].dropna() >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# IBI resampling
# ══════════════════════════════════════════════════════════════════════════════

def test_ibi_resample_produces_mean_and_rmssd():
    """IBI resampler must output both ibi_mean and ibi_rmssd columns."""
    idx = pd.date_range("2020-03-01 06:00", periods=20, freq="850ms", tz="UTC")
    ibi = pd.DataFrame({"ibi": 0.85 + 0.05 * np.random.randn(20)}, index=idx)
    res = resample_ibi(ibi)
    assert "ibi_mean"  in res.columns, "Must have ibi_mean column."
    assert "ibi_rmssd" in res.columns, "Must have ibi_rmssd column."


def test_ibi_rmssd_non_negative():
    idx = pd.date_range("2020-03-01 06:00", periods=30, freq="850ms", tz="UTC")
    ibi = pd.DataFrame({"ibi": 0.85 + 0.05 * np.random.randn(30)}, index=idx)
    res = resample_ibi(ibi)
    rmssd_vals = res["ibi_rmssd"].dropna()
    assert (rmssd_vals >= 0).all(), "RMSSD must be non-negative."


# ══════════════════════════════════════════════════════════════════════════════
# ACC resampling
# ══════════════════════════════════════════════════════════════════════════════

def test_acc_resample_magnitude_columns():
    idx = pd.date_range("2020-03-01 06:00", periods=500, freq="31ms", tz="UTC")
    acc = pd.DataFrame({
        "acc_x": np.random.uniform(-60, 60, 500),
        "acc_y": np.random.uniform(-10, 10, 500),
        "acc_z": np.random.uniform(-30, 30, 500),
    }, index=idx)
    res = resample_acc(acc)
    assert "acc_magnitude_mean" in res.columns
    assert "acc_magnitude_std"  in res.columns


def test_acc_magnitude_non_negative():
    idx = pd.date_range("2020-03-01 06:00", periods=500, freq="31ms", tz="UTC")
    acc = pd.DataFrame({
        "acc_x": np.random.uniform(-60, 60, 500),
        "acc_y": np.random.uniform(-10, 10, 500),
        "acc_z": np.random.uniform(-30, 30, 500),
    }, index=idx)
    res = resample_acc(acc)
    assert (res["acc_magnitude_mean"].dropna() >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Food log resampling
# ══════════════════════════════════════════════════════════════════════════════

def test_food_resample_uses_sum():
    """
    Two events in the same 15-min window must be SUMMED, not averaged.
    Events at 08:02 and 08:10 both fall in the 08:00 window.
    """
    food = _make_food_events()
    res  = resample_food_log(food)

    # Window 08:00 should have carbs = 30 + 10 = 40
    window_800 = res.index[res.index.hour == 8].min()
    assert abs(res.loc[window_800, "total_carb"] - 40.0) < 0.01, \
        "Carbs in 08:00 window should be 30+10=40."


def test_food_resample_meal_flag():
    food = _make_food_events()
    res  = resample_food_log(food)
    assert "meal_flag" in res.columns
    # Both windows have meals → meal_flag = 1
    meal_windows = res[res["meal_flag"] == 1]
    assert len(meal_windows) >= 1, "At least one window should have meal_flag=1."


def test_food_resample_gi_proxy():
    food = _make_food_events()
    res  = resample_food_log(food)
    assert "gi_proxy" in res.columns
    # gi_proxy must be between 0 and 1 (sugar/total_carb)
    gi_vals = res["gi_proxy"].dropna()
    gi_vals = gi_vals[gi_vals > 0]
    assert (gi_vals <= 1.0).all(), "gi_proxy = sugar/total_carb must be ≤ 1."


# ══════════════════════════════════════════════════════════════════════════════
# CGMacros resampler (alignment only)
# ══════════════════════════════════════════════════════════════════════════════

def test_cgmacros_resample_aligns_to_grid():
    """CGMacros timestamps that are slightly off-grid should snap to 15-min."""
    # Create timestamps that are 1 minute off the 15-min boundary
    idx = pd.date_range("2020-03-01 06:01", periods=10, freq="15min", tz="UTC")
    df  = pd.DataFrame({"glucose_mg_dl": 100 + np.arange(10, dtype=float)}, index=idx)
    res = resample_cgmacros_user(df)

    # All timestamps should now be on the 15-min grid
    for ts in res.index:
        assert ts.minute % 15 == 0, f"Timestamp {ts} not on 15-min grid."


def test_cgmacros_resample_no_duplicates():
    """Rounding should not introduce duplicate timestamps."""
    idx = pd.date_range("2020-03-01 06:00", periods=10, freq="15min", tz="UTC")
    df  = pd.DataFrame({"glucose_mg_dl": 100.0 + np.arange(10)}, index=idx)
    res = resample_cgmacros_user(df)
    assert not res.index.duplicated().any(), "No duplicate timestamps after resampling."


# ══════════════════════════════════════════════════════════════════════════════
# Integration: full NP user resample dict
# ══════════════════════════════════════════════════════════════════════════════

def test_resample_np_user_returns_all_keys():
    """resample_np_user must accept a dict and return the same 8 keys."""
    raw = {
        "cgm":  _make_5min_cgm(36),
        "hr":   _make_1hz_hr(900),
        "eda":  _make_4hz_eda(3600),
        "ibi":  pd.DataFrame({"ibi": [0.85]},
                             index=pd.DatetimeIndex(["2020-03-01 06:05:00+00:00"])),
        "bvp":  pd.DataFrame({"bvp": [0.1]},
                             index=pd.DatetimeIndex(["2020-03-01 06:00:01+00:00"])),
        "acc":  pd.DataFrame({"acc_x": [0.], "acc_y": [0.], "acc_z": [0.]},
                             index=pd.DatetimeIndex(["2020-03-01 06:00:00+00:00"])),
        "temp": pd.DataFrame({"temp": [35.0]},
                             index=pd.DatetimeIndex(["2020-03-01 06:00:00+00:00"])),
        "food": _make_food_events(),
    }
    result = resample_np_user(raw)
    assert set(result.keys()) == {"cgm", "hr", "eda", "ibi", "bvp", "acc", "temp", "food"}
