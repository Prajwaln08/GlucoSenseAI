"""
Phase 2 unit tests.

Covers:
- meal_features: new reservoir columns exist and decay correctly
- meal_features: extended nutrient windows zero-fill when raw columns absent
- medicine_features: all six effect columns zero when no dose data present
- medicine_features: effect rises then decays after a single dose event
- pipeline (post_cgm mode): zero glucose-derived columns in feature matrix
- pipeline (post_cgm mode): absolute targets are present
- pipeline (cgm_active mode): delta targets still work (regression guard)
- feature_groups: MEDICINE_OUTPUT_COLS ⊆ MEDICINE_GROUP.columns
"""

import pytest
import numpy as np
import pandas as pd

from src.features.meal_features import add_meal_features
from src.features.medicine_features import add_medicine_features, MEDICINE_OUTPUT_COLS
from src.features.pipeline import build_feature_matrix, get_feature_cols, get_target_cols
from src.features.feature_groups import CGM_FEATURES, MEDICINE_GROUP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_df(n: int = 60, with_glucose: bool = True) -> pd.DataFrame:
    """Minimal preprocessed DataFrame spanning n 15-min intervals."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    data: dict = {
        "hr": np.full(n, 70.0),
        "calories_burned": np.zeros(n),
        "mets": np.zeros(n),
        "meal_type_encoded": np.zeros(n),
        "amount_consumed_pct": np.zeros(n),
    }
    if with_glucose:
        data["glucose_mg_dl"] = np.linspace(90, 150, n)
    return pd.DataFrame(data, index=idx)


def _df_with_meal(n: int = 60) -> pd.DataFrame:
    """DataFrame where one meal (50 g carbs, 400 kcal) is logged at step 10."""
    df = _base_df(n)
    df["total_carb"] = 0.0
    df["calorie"]    = 0.0
    df["protein"]    = 0.0
    df["total_fat"]  = 0.0
    df["dietary_fiber"] = 0.0
    df["sugar"]      = 0.0
    df.iloc[10, df.columns.get_loc("total_carb")] = 50.0
    df.iloc[10, df.columns.get_loc("calorie")]    = 400.0
    return df


def _df_with_dose(category: str = "ip", n: int = 80) -> pd.DataFrame:
    """DataFrame with a single medicine dose at step 10."""
    df = _base_df(n)
    dose_col = f"med_{category}_dose"
    df[dose_col] = 0.0
    df.iloc[10, df.columns.get_loc(dose_col)] = 1.0   # single unit dose
    return df


# ── meal_features: named reservoirs ──────────────────────────────────────────

def test_carb_reservoir_present():
    df = add_meal_features(_df_with_meal())
    assert "carb_reservoir" in df.columns


def test_energy_reservoir_present():
    df = add_meal_features(_df_with_meal())
    assert "energy_reservoir" in df.columns


def test_reservoir_slope_present():
    df = add_meal_features(_df_with_meal())
    assert "reservoir_slope" in df.columns


def test_glycemic_risk_index_present():
    df = add_meal_features(_df_with_meal())
    assert "glycemic_risk_index" in df.columns


def test_carb_reservoir_rises_after_meal_then_decays():
    df = add_meal_features(_df_with_meal(n=80))
    r  = df["carb_reservoir"].values
    # Before the meal: reservoir should be essentially zero
    assert r[:11].max() < 0.01
    # After the meal: reservoir should rise
    peak_idx = np.argmax(r)
    assert peak_idx > 10, "Peak should occur after the meal step"
    # After peak: should decay monotonically
    tail = r[peak_idx:]
    assert all(tail[i] >= tail[i + 1] for i in range(len(tail) - 1)), (
        "Reservoir should decay monotonically after peak"
    )


def test_reservoir_slope_positive_after_meal():
    df   = add_meal_features(_df_with_meal(n=80))
    # Slope should be positive immediately after the meal injection
    slope = df["reservoir_slope"].values
    # Find the first step after the meal where the reservoir grew
    rising_steps = np.where(slope > 0)[0]
    assert len(rising_steps) > 0, "reservoir_slope should be positive after a meal"


# ── meal_features: extended nutrient columns ──────────────────────────────────

def test_extended_nutrients_zero_fill_when_absent():
    df   = add_meal_features(_base_df())
    for col in ["sodium_window_1h", "caffeine_window_1h",
                "vegetable_portions_window_2h", "liquid_window_2h"]:
        assert col in df.columns, f"Expected column {col!r}"
        assert df[col].sum() == 0.0, f"{col!r} should be all-zero when raw column absent"


def test_carb_missing_flag():
    df = add_meal_features(_base_df())   # no total_carb column → all missing
    assert "carb_missing" in df.columns
    assert df["carb_missing"].all(), "carb_missing should be 1 when total_carb is absent"


# ── medicine_features: zero when no dose data ─────────────────────────────────

def test_medicine_effects_zero_when_no_dose():
    df     = add_medicine_features(_base_df())
    effect_cols = [c for c in MEDICINE_OUTPUT_COLS if c.endswith("_effect")]
    for col in effect_cols:
        assert col in df.columns, f"Effect column {col!r} missing"
        assert df[col].abs().max() < 1e-9, (
            f"{col!r} should be zero when no dose data present"
        )


def test_med_any_recent_zero_when_no_dose():
    df = add_medicine_features(_base_df())
    assert "med_any_recent" in df.columns
    assert df["med_any_recent"].sum() == 0


def test_med_accumulated_effect_zero_when_no_dose():
    df = add_medicine_features(_base_df())
    assert df["med_accumulated_effect"].abs().max() < 1e-9


# ── medicine_features: single dose produces rise-then-decay ──────────────────

@pytest.mark.parametrize("category", ["ip", "is", "ie", "ic", "ge", "gr"])
def test_single_dose_produces_nonzero_effect(category: str):
    df     = add_medicine_features(_df_with_dose(category=category, n=200))
    effect = df[f"med_{category}_effect"].values
    assert effect.max() > 0.0, (
        f"med_{category}_effect should be non-zero after a dose"
    )


def test_single_dose_effect_decays_after_peak():
    df     = add_medicine_features(_df_with_dose(category="ip", n=200))
    effect = df["med_ip_effect"].values
    peak_i = int(np.argmax(effect))
    # After peak, effect must monotonically decrease (allow tiny numerical noise)
    tail   = effect[peak_i:]
    diffs  = np.diff(tail)
    assert all(d <= 1e-9 for d in diffs), (
        "med_ip_effect should decay monotonically after its peak"
    )


def test_medicine_output_cols_subset_of_medicine_group():
    undeclared = MEDICINE_OUTPUT_COLS - MEDICINE_GROUP.columns
    assert undeclared == frozenset(), (
        f"MEDICINE_OUTPUT_COLS contains columns not in MEDICINE_GROUP: {undeclared}"
    )


# ── pipeline: post_cgm mode ───────────────────────────────────────────────────

def _full_df(n: int = 200) -> pd.DataFrame:
    """Richer DataFrame suitable for full pipeline (has glucose for target calc)."""
    df = _df_with_meal(n)
    # Add a few more columns the pipeline expects
    df["gi_proxy"] = 0.0
    return df


def test_post_cgm_pipeline_no_cgm_features_in_X():
    df     = build_feature_matrix(_full_df(), mode="post_cgm")
    feat   = get_feature_cols(df, mode="post_cgm")
    leaked = set(feat) & CGM_FEATURES
    assert leaked == set(), (
        f"CGM features leaked into post_cgm feature matrix: {leaked}"
    )


def test_post_cgm_pipeline_absolute_targets_present():
    df = build_feature_matrix(_full_df(), mode="post_cgm")
    for col in get_target_cols("2h", mode="post_cgm"):
        assert col in df.columns, f"Absolute target {col!r} missing"


def test_post_cgm_pipeline_no_delta_targets():
    df = build_feature_matrix(_full_df(), mode="post_cgm")
    for col in get_target_cols("2h", mode="cgm_active"):
        assert col not in df.columns, (
            f"Delta target {col!r} should not be present in post_cgm mode"
        )


def test_post_cgm_pipeline_glucose_col_excluded_from_X():
    df   = build_feature_matrix(_full_df(), mode="post_cgm")
    feat = get_feature_cols(df, mode="post_cgm")
    assert "glucose_mg_dl" not in feat, (
        "glucose_mg_dl must not appear in the post_cgm feature list"
    )


# ── pipeline: cgm_active mode (regression guard) ─────────────────────────────

def test_cgm_active_pipeline_delta_targets_present():
    df = build_feature_matrix(_full_df(), mode="cgm_active")
    for col in get_target_cols("2h", mode="cgm_active"):
        assert col in df.columns, f"Delta target {col!r} missing in cgm_active mode"


def test_cgm_active_pipeline_glucose_features_present():
    df   = build_feature_matrix(_full_df(), mode="cgm_active")
    feat = set(get_feature_cols(df, mode="cgm_active"))
    assert "glucose_lag_4" in feat, "glucose_lag_4 should be in cgm_active feature set"


def test_cgm_active_pipeline_no_absolute_targets():
    df = build_feature_matrix(_full_df(), mode="cgm_active")
    for col in get_target_cols("2h", mode="post_cgm"):
        assert col not in df.columns, (
            f"Absolute target {col!r} should not appear in cgm_active mode"
        )
