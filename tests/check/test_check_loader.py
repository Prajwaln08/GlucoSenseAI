"""
CHECK 01 — Data Loader

Tests NaturePaperLoader and CGMacrosLoader with synthetic CSV files written
to a temp directory. No Google Drive access required.

Run:
    conda activate glucosenseai
    pytest tests/check/01_test_check_loader.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import NaturePaperLoader, CGMacrosLoader


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — synthetic CSV files
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_np_dir(tmp_path: Path):
    """Create a synthetic Nature's Paper user folder with all 8 CSV files."""
    user_id  = "003"
    user_dir = tmp_path / "nature_paper" / user_id
    user_dir.mkdir(parents=True)

    # Timestamps
    timestamps = pd.date_range("2020-03-01 06:00", periods=100, freq="5min")
    ts_str = timestamps.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Dexcom CGM — build with pandas to guarantee correct column count ──────
    _DEXCOM_COLS = [
        "Index", "Timestamp (YYYY-MM-DDThh:mm:ss)", "Event Type", "Event Subtype",
        "Patient Info", "Device Info", "Source Device ID", "Glucose Value (mg/dL)",
        "Insulin Value (u)", "Carb Value (grams)", "Duration (hh:mm:ss)",
        "Glucose Rate of Change (mg/dL/min)", "Transmitter Time (Long Integer)",
    ]
    meta_rows = [
        dict(zip(_DEXCOM_COLS, [1, "", "FirstName",       "", "Test",       "", "", "", "", "", "", "", ""])),
        dict(zip(_DEXCOM_COLS, [2, "", "LastName",        "", "User",       "", "", "", "", "", "", "", ""])),
        dict(zip(_DEXCOM_COLS, [3, "", "PatientIdentifier","","2020-003",  "", "", "", "", "", "", "", ""])),
        dict(zip(_DEXCOM_COLS, [4, "", "DateOfBirth",     "", "1990-01-01", "", "", "", "", "", "", "", ""])),
    ]
    egv_rows = [
        dict(zip(_DEXCOM_COLS, [
            i + 5, ts, "EGV", "", "", "", "",
            round(100 + 20 * np.sin(i / 10), 1),
            "", "", "",
            round(2 * np.cos(i / 10) / 15, 3),
            "",
        ]))
        for i, ts in enumerate(ts_str)
    ]
    pd.DataFrame(meta_rows + egv_rows).to_csv(
        user_dir / f"Dexcom_{user_id}.csv", index=False
    )

    # ── HR ────────────────────────────────────────────────────────────────────
    hr_ts = pd.date_range("2020-03-01 06:00", periods=300, freq="1s")
    pd.DataFrame({
        "datetime": hr_ts.strftime("%Y-%m-%d %H:%M:%S"),
        " hr":      (70 + 5 * np.random.randn(300)).round(1),
    }).to_csv(user_dir / f"HR_{user_id}.csv", index=False)

    # ── EDA (4 Hz) ────────────────────────────────────────────────────────────
    eda_ts = pd.date_range("2020-03-01 06:00", periods=1200, freq="250ms")
    pd.DataFrame({
        "datetime": eda_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        " eda":     np.abs(np.random.randn(1200) * 0.5).round(4),
    }).to_csv(user_dir / f"EDA_{user_id}.csv", index=False)

    # ── IBI (event) ───────────────────────────────────────────────────────────
    ibi_ts = pd.date_range("2020-03-01 06:00", periods=60, freq="850ms")
    pd.DataFrame({
        "datetime": ibi_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        " ibi":     (0.85 + 0.05 * np.random.randn(60)).round(4),
    }).to_csv(user_dir / f"IBI_{user_id}.csv", index=False)

    # ── BVP (64 Hz) ───────────────────────────────────────────────────────────
    bvp_ts = pd.date_range("2020-03-01 06:00", periods=1000, freq="16ms")
    pd.DataFrame({
        "datetime": bvp_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        " bvp":     np.random.randn(1000).round(3),
    }).to_csv(user_dir / f"BVP_{user_id}.csv", index=False)

    # ── ACC (32 Hz) ───────────────────────────────────────────────────────────
    acc_ts = pd.date_range("2020-03-01 06:00", periods=2000, freq="31ms")
    pd.DataFrame({
        "datetime": acc_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        " acc_x":   np.random.uniform(-60, 60, 2000).round(1),
        " acc_y":   np.random.uniform(-10, 10, 2000).round(1),
        " acc_z":   np.random.uniform(-30, 30, 2000).round(1),
    }).to_csv(user_dir / f"ACC_{user_id}.csv", index=False)

    # ── TEMP (4 Hz) ───────────────────────────────────────────────────────────
    temp_ts = pd.date_range("2020-03-01 06:00", periods=1200, freq="250ms")
    pd.DataFrame({
        "datetime": temp_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        " temp":    (35 + 0.2 * np.random.randn(1200)).round(2),
    }).to_csv(user_dir / f"TEMP_{user_id}.csv", index=False)

    # ── Food Log (no header, 11 columns) ─────────────────────────────────────
    food_rows = [
        "2020-03-01,08:00:00,2020-03-01 08:00:00,Oatmeal,1.0,cup,Oatmeal,350.0,62.0,8.0,1.0",
        "2020-03-01,12:00:00,2020-03-01 12:00:00,Rice with chicken,1.0,plate,Rice + chicken,650.0,75.0,3.0,12.0",
    ]
    (user_dir / f"Food_Log_{user_id}.csv").write_text("\n".join(food_rows))

    # ── Demographics ──────────────────────────────────────────────────────────
    demo = (tmp_path / "nature_paper" / "Demographics.csv")
    demo.write_text("ID,Gender,HbA1c\n3,FEMALE,5.9\n")

    return tmp_path / "nature_paper", user_id


@pytest.fixture
def tmp_cgmacros_dir(tmp_path: Path):
    """Create a synthetic CGMacros user CSV."""
    user_id  = "001"
    sub_dir  = tmp_path / "cgmacros" / f"CGMacros-{user_id}"
    sub_dir.mkdir(parents=True)

    timestamps = pd.date_range("2021-01-01", periods=200, freq="15min")
    n = len(timestamps)

    df = pd.DataFrame({
        "Timestamp":           timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "Libre GL":            (100 + 20 * np.sin(np.arange(n) / 8)).round(1),
        "Dexcom GL":           [np.nan] * n,
        "HR":                  (75 + 3 * np.random.randn(n)).round(1),
        "Calories (Activity)": np.abs(np.random.randn(n) * 2).round(2),
        "METs":                (1 + np.abs(np.random.randn(n))).round(2),
        "Meal Type":           [""] * n,
        "Calories (Food)":     [0.0] * n,
        "Carbs":               [0.0] * n,
        "Protein":             [0.0] * n,
        "Fat":                 [0.0] * n,
        "Fiber":               [0.0] * n,
        "Amount Consumed":     [0.0] * n,
        "Image Path":          [""] * n,
    })
    # Add a few meals
    df.loc[4,  ["Carbs", "Calories (Food)", "Meal Type", "Amount Consumed"]] = [60.0, 500.0, "breakfast", 100.0]
    df.loc[16, ["Carbs", "Calories (Food)", "Meal Type", "Amount Consumed"]] = [80.0, 700.0, "lunch",     90.0]

    csv_path = sub_dir / f"CGMacros-{user_id}.csv"
    df.to_csv(csv_path, index=False)

    # bio.csv
    bio_path = tmp_path / "cgmacros" / "bio.csv"
    bio_path.write_text("ID,Age,Gender,BMI,A1c PDL (Lab)\n1,35,M,24.5,5.8\n")

    return tmp_path / "cgmacros", user_id


# ══════════════════════════════════════════════════════════════════════════════
# Nature's Paper Loader tests
# ══════════════════════════════════════════════════════════════════════════════

def test_np_load_returns_all_sources(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    loader = NaturePaperLoader(base_dir=base_dir)
    data   = loader.load(user_id)

    assert set(data.keys()) == {"cgm", "hr", "eda", "ibi", "bvp", "acc", "temp", "food"}, \
        "Loader must return all 8 source keys."


def test_np_cgm_has_correct_columns(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    data = NaturePaperLoader(base_dir=base_dir).load(user_id)
    cgm  = data["cgm"]

    assert "glucose_mg_dl" in cgm.columns,          "CGM must have 'glucose_mg_dl'."
    assert "glucose_rate_of_change" in cgm.columns, "CGM must have 'glucose_rate_of_change'."
    assert len(cgm) > 0,                            "CGM DataFrame must not be empty."


def test_np_cgm_glucose_values_in_range(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    cgm = NaturePaperLoader(base_dir=base_dir).load(user_id)["cgm"]

    glucose = cgm["glucose_mg_dl"].dropna()
    assert (glucose >= 40).all() and (glucose <= 400).all(), \
        "Glucose values should be within 40–400 mg/dL range."


def test_np_cgm_index_is_utc_datetime(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    cgm = NaturePaperLoader(base_dir=base_dir).load(user_id)["cgm"]

    assert isinstance(cgm.index, pd.DatetimeIndex), "CGM index must be DatetimeIndex."
    assert cgm.index.tz is not None,               "CGM index must be UTC-aware."
    assert cgm.index.is_monotonic_increasing,       "CGM timestamps must be sorted."


def test_np_food_log_loads_numeric_columns(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    food = NaturePaperLoader(base_dir=base_dir).load(user_id)["food"]

    for col in ["calorie", "total_carb"]:
        assert col in food.columns,               f"Food log must have '{col}' column."
        assert pd.api.types.is_numeric_dtype(food[col]), f"'{col}' must be numeric."


def test_np_food_log_positive_values(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    food = NaturePaperLoader(base_dir=base_dir).load(user_id)["food"]
    assert (food["total_carb"] >= 0).all(), "Carbs must be non-negative."
    assert (food["calorie"] >= 0).all(),    "Calories must be non-negative."


def test_np_acc_has_magnitude(tmp_np_dir):
    base_dir, user_id = tmp_np_dir
    acc = NaturePaperLoader(base_dir=base_dir).load(user_id)["acc"]
    assert "acc_magnitude" in acc.columns, "ACC must have computed 'acc_magnitude'."
    assert (acc["acc_magnitude"] >= 0).all(), "ACC magnitude must be non-negative."


def test_np_demographics_loads(tmp_np_dir):
    base_dir, _user_id = tmp_np_dir
    demo = NaturePaperLoader(base_dir=base_dir).load_demographics()
    assert "participant_id" in demo.columns
    assert "hba1c" in demo.columns


def test_np_missing_file_raises(tmp_path):
    """Loader raises FileNotFoundError for non-existent user."""
    loader = NaturePaperLoader(base_dir=tmp_path / "nature_paper")
    with pytest.raises(FileNotFoundError):
        loader.load("999")


# ══════════════════════════════════════════════════════════════════════════════
# CGMacros Loader tests
# ══════════════════════════════════════════════════════════════════════════════

def test_cgmacros_load_returns_dataframe(tmp_cgmacros_dir):
    base_dir, user_id = tmp_cgmacros_dir
    df = CGMacrosLoader(base_dir=base_dir).load(user_id)
    assert isinstance(df, pd.DataFrame), "CGMacros loader must return a DataFrame."
    assert len(df) > 0,                  "CGMacros DataFrame must not be empty."


def test_cgmacros_has_glucose_column(tmp_cgmacros_dir):
    base_dir, user_id = tmp_cgmacros_dir
    df = CGMacrosLoader(base_dir=base_dir).load(user_id)
    assert "glucose_mg_dl" in df.columns, "'glucose_mg_dl' must be present."
    assert df["glucose_mg_dl"].notna().sum() > 0, "Glucose must have non-NaN values."


def test_cgmacros_index_is_utc_datetime(tmp_cgmacros_dir):
    base_dir, user_id = tmp_cgmacros_dir
    df = CGMacrosLoader(base_dir=base_dir).load(user_id)
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex."
    assert df.index.tz is not None,               "Index must be UTC-aware."


def test_cgmacros_meal_type_encoded(tmp_cgmacros_dir):
    base_dir, user_id = tmp_cgmacros_dir
    df = CGMacrosLoader(base_dir=base_dir).load(user_id)
    assert "meal_type_encoded" in df.columns, "meal_type_encoded must be computed."
    # Breakfast/lunch should be encoded as 2
    meal_rows = df[df["meal_type_encoded"] > 0]
    assert len(meal_rows) >= 2, "At least 2 meal rows should have encoding > 0."


def test_cgmacros_bio_merged(tmp_cgmacros_dir):
    base_dir, user_id = tmp_cgmacros_dir
    df = CGMacrosLoader(base_dir=base_dir).load(user_id)
    # At least one bio column should be present
    assert any(c in df.columns for c in ["hba1c", "age", "bmi", "gender"]), \
        "At least one bio column should be merged from bio.csv."


def test_cgmacros_missing_file_raises(tmp_path):
    loader = CGMacrosLoader(base_dir=tmp_path / "cgmacros")
    with pytest.raises(FileNotFoundError):
        loader.load("999")
