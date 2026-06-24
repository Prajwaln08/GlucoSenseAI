"""
Step 2 — Clinical + 10-min resample checks.

Synthetic frames exercise the median/sum aggregation, clinical clipping, and the
gapless grid; one guarded real CGMacros user confirms to_grid() end-to-end.

Run:
    conda activate glucosenseai
    pytest tests/check/test_step2_clinical.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.step1_loader import (
    META_COLS, UNIFIED_COLUMNS, LoadedUser, discover_users, load_user,
)
from src.data import step2_clinical as s2


def _wide_cgmacros(uid="cg-017", minutes=30) -> pd.DataFrame:
    """A CGMacros-shaped 1-min wide frame spanning `minutes` minutes."""
    idx = pd.date_range("2020-01-01 00:00", periods=minutes, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "glucose_mg_dl": np.arange(minutes, dtype=float) + 100.0,
            "hr": np.full(minutes, 70.0),
            "total_carb": np.where(np.arange(minutes) == 5, 30.0, 0.0),
            "calorie": np.where(np.arange(minutes) == 5, 200.0, 0.0),
            "mets": np.full(minutes, 1.0),
            "calories_burned": np.full(minutes, 0.5),
            "meal_type_encoded": np.where(np.arange(minutes) == 5, 2.0, 0.0),
            "age": 40, "bmi": 25.0, "hba1c": 5.5, "gender": "M",
        },
        index=idx,
    )
    df["uid"] = uid
    df["dataset"] = "cgmacros"
    df["participant_id"] = uid.split("-", 1)[1]
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def test_resample_is_on_10min_grid():
    out = s2._aggregate_to_grid(_wide_cgmacros(minutes=30))
    diffs = out.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta("10min")


def test_glucose_uses_window_median():
    out = s2._aggregate_to_grid(_wide_cgmacros(minutes=30))
    # window 00:00–00:09 holds glucose 100..109 -> median 104.5
    assert out["glucose_mg_dl"].iloc[0] == pytest.approx(104.5)


def test_food_macros_summed_in_window():
    out = s2._aggregate_to_grid(_wide_cgmacros(minutes=30))
    # the single 30 g carb event (minute 5) lands in the first window
    assert out["total_carb"].iloc[0] == pytest.approx(30.0)
    assert out["meal_flag"].iloc[0] == 1
    assert out["calories_burned"].iloc[0] == pytest.approx(5.0)  # 10 × 0.5 summed


def test_demographics_carried_constant():
    out = s2._aggregate_to_grid(_wide_cgmacros(minutes=30))
    assert (out["age"] == 40).all()
    assert (out["gender"] == "M").all()


# ── Clinical clipping ─────────────────────────────────────────────────────────

def test_clinical_clip_glucose_clipped_to_bounds():
    df = pd.DataFrame({"glucose_mg_dl": [500.0, 120.0, 10.0]})
    out = s2.clinical_clip(df)
    assert out["glucose_mg_dl"].iloc[0] == 400.0   # 500 -> upper bound (not NaN)
    assert out["glucose_mg_dl"].iloc[1] == 120.0   # in range
    assert out["glucose_mg_dl"].iloc[2] == 40.0    # 10 -> lower bound (not NaN)
    assert out["glucose_mg_dl"].notna().all()      # nothing nulled


def test_clinical_clip_food_capped_not_nulled():
    df = pd.DataFrame({"total_carb": [9999.0], "mets": [99.0]})
    out = s2.clinical_clip(df)
    assert out["total_carb"].iloc[0] == 500.0       # capped at CARBS_MAX
    assert out["mets"].iloc[0] == 20.0              # capped at METS_MAX


# ── to_grid + unified schema ──────────────────────────────────────────────────

def test_to_grid_conforms_and_marks_np_only_nan():
    user = LoadedUser("cg-017", "cgmacros", "017", frame=_wide_cgmacros())
    grid = s2.to_grid(user)
    assert list(grid.columns) == META_COLS + UNIFIED_COLUMNS
    # 10-min spacing
    diffs = grid.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta("10min")
    # NP-only signal absent for a CGMacros user
    assert grid["eda"].isna().all()
    assert grid["acc_magnitude_mean"].isna().all()
    # identity present on every row
    assert (grid["uid"] == "cg-017").all()


def test_complete_grid_fills_gaps_with_nan_and_carries_identity():
    # Two windows 30 min apart -> the middle windows must appear as NaN rows.
    idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:30"], tz="UTC")
    df = pd.DataFrame({"glucose_mg_dl": [100.0, 130.0], "uid": "cg-017"}, index=idx)
    out = s2._complete_grid(df)
    assert len(out) == 4                              # 00:00,10,20,30
    assert out["glucose_mg_dl"].isna().sum() == 2     # the two empty windows
    assert (out["uid"] == "cg-017").all()             # identity carried across gaps


def test_build_grid_table_concats_users():
    u1 = LoadedUser("cg-017", "cgmacros", "017", frame=_wide_cgmacros("cg-017"))
    u2 = LoadedUser("cg-018", "cgmacros", "018", frame=_wide_cgmacros("cg-018"))
    table = s2.build_grid_table([u1, u2])
    assert table["uid"].nunique() == 2
    assert list(table.columns) == META_COLS + UNIFIED_COLUMNS


# ── Real data (guarded) ───────────────────────────────────────────────────────

def test_real_cgmacros_to_grid():
    ids = discover_users("cgmacros")
    if not ids:
        pytest.skip("No CGMacros users on disk.")
    grid = s2.to_grid(load_user("cgmacros", ids[0]))
    assert grid["glucose_mg_dl"].notna().any()
    diffs = grid.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta("10min")
