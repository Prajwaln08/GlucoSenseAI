"""
Step 4 — Feature engineering checks.

Covers: glucose features only in cgm_active mode, no glucose leakage in post_cgm,
target alignment, the no-NaN-target guarantee in get_xy, structural-NaN
preservation, per-user isolation, and absence of reservoir/decay/cumulative cols.

Run:
    conda activate glucosenseai
    pytest tests/check/test_step4_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.config import steps
from src.data import step4_features as s4


def _user(uid="cg-017", n=40, with_eda=False) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    df = pd.DataFrame(
        {
            "glucose_mg_dl": 100.0 + np.arange(n),
            "hr": 70.0 + np.zeros(n),
            "total_carb": np.where(np.arange(n) % 10 == 0, 25.0, 0.0),
            "calorie": np.where(np.arange(n) % 10 == 0, 150.0, 0.0),
            "hba1c": 5.5, "age": 40.0, "bmi": 25.0, "gender": "F",
        },
        index=idx,
    )
    if with_eda:
        df["eda"] = 1.0
    else:
        df["eda"] = np.nan          # structurally absent (CGMacros-like)
    df["uid"] = uid
    df["dataset"] = "cgmacros"
    df["participant_id"] = uid.split("-", 1)[1]
    return df


# ── Mode separation ───────────────────────────────────────────────────────────

def test_cgm_active_builds_glucose_features():
    out = s4.build_user_features(_user(), mode="cgm_active")
    feats = s4.get_feature_cols(out, mode="cgm_active")
    assert "glucose_lag_6" in feats
    assert "glucose_roll_mean_12" in feats
    assert s4.get_target_cols("cgm_active") == ["target_delta_30", "target_delta_60",
                                                "target_delta_90", "target_delta_120"]


def test_post_cgm_has_no_glucose_features():
    out = s4.build_user_features(_user(), mode="post_cgm")
    feats = set(s4.get_feature_cols(out, mode="post_cgm"))
    assert not (feats & s4.GLUCOSE_DERIVED), "glucose-derived columns leaked into post_cgm"
    assert s4.get_target_cols("post_cgm")[0] == "target_abs_30"


# ── Target alignment + NaN policy ─────────────────────────────────────────────

def test_absolute_target_is_future_glucose():
    out = s4.build_user_features(_user(), mode="post_cgm")
    h = steps(30)  # 3 steps
    # target_abs_30 at row i must equal glucose at row i+3
    assert out["target_abs_30"].iloc[0] == out["glucose_mg_dl"].iloc[h]


def test_get_xy_drops_missing_target_rows():
    out = s4.build_user_features(_user(n=40), mode="post_cgm")
    X, y = s4.get_xy(out, horizon_min=30, mode="post_cgm")
    assert not y.isna().any()                 # NaN target never reaches the model
    assert len(X) == len(y) == 40 - steps(30)  # trailing 3 rows dropped


def test_delta_target_relation():
    out = s4.build_user_features(_user(), mode="cgm_active")
    h = steps(60)
    expected = out["glucose_mg_dl"].iloc[h] - out["glucose_mg_dl"].iloc[0]
    assert out["target_delta_60"].iloc[0] == pytest.approx(expected)


# ── Structural NaN + no banned feature families ───────────────────────────────

def test_structural_nan_preserved_for_absent_sensor():
    out = s4.build_user_features(_user(with_eda=False), mode="post_cgm")
    assert out["eda_roll_mean_30m"].isna().all()   # CGMacros user has no EDA


def test_no_reservoir_decay_or_cumulative_features():
    out = s4.build_user_features(_user(), mode="cgm_active")
    banned = [c for c in out.columns
              if "reservoir" in c or "_decay" in c or "total_today" in c or "cumulative" in c]
    assert banned == [], f"banned feature families present: {banned}"


# ── Per-user isolation ────────────────────────────────────────────────────────

def test_features_table_isolates_users():
    table = pd.concat([_user("cg-001"), _user("cg-002")])
    out = s4.build_features_table(table, mode="cgm_active")
    # first row of each user has no prior glucose -> lag is NaN (no bleed across users)
    first_rows = out.groupby("uid").head(1)
    assert first_rows["glucose_lag_6"].isna().all()
    assert out["uid"].nunique() == 2
