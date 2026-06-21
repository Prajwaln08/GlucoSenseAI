"""
Unit tests for src/features/feature_groups.py and the glucose_features guard.

Covers:
- CGM_FEATURES / NON_CGM_FEATURES disjointness
- CGM_OUTPUT_COLS ⊆ CGM_FEATURES consistency
- FeatureContract.validate() in both modes
- build_cgm_features() raises in post_cgm mode
"""

import pytest
import pandas as pd

from src.features.feature_groups import (
    CGM_FEATURES,
    NON_CGM_FEATURES,
    ALL_KNOWN_FEATURES,
    FeatureContract,
)
from src.features.glucose_features import CGM_OUTPUT_COLS, build_cgm_features


# ── Group disjointness ────────────────────────────────────────────────────────

def test_cgm_and_non_cgm_are_disjoint():
    overlap = CGM_FEATURES & NON_CGM_FEATURES
    assert overlap == frozenset(), (
        f"CGM_FEATURES and NON_CGM_FEATURES must not overlap, but share: {overlap}"
    )


def test_all_known_features_covers_both():
    assert CGM_FEATURES <= ALL_KNOWN_FEATURES
    assert NON_CGM_FEATURES <= ALL_KNOWN_FEATURES


# ── CGM_OUTPUT_COLS consistency ───────────────────────────────────────────────

def test_cgm_output_cols_subset_of_cgm_features():
    unlisted = CGM_OUTPUT_COLS - CGM_FEATURES
    assert unlisted == frozenset(), (
        f"CGM_OUTPUT_COLS contains columns not in CGM_FEATURES: {unlisted}. "
        "Update feature_groups.CGM_GROUP.columns to include them."
    )


# ── FeatureContract — post_cgm mode ──────────────────────────────────────────

def test_contract_post_cgm_raises_on_cgm_column():
    cgm_col = next(iter(CGM_FEATURES))  # any one CGM column
    df = pd.DataFrame({cgm_col: [1.0], "hr_roll_mean_4": [70.0]})
    contract = FeatureContract(feature_cols=[cgm_col, "hr_roll_mean_4"], mode="post_cgm")
    with pytest.raises(ValueError, match="CGM-derived columns found"):
        contract.validate(df)


def test_contract_post_cgm_passes_clean_df():
    df = pd.DataFrame({"hr_roll_mean_4": [70.0], "hour_sin": [0.5]})
    contract = FeatureContract(feature_cols=["hr_roll_mean_4", "hour_sin"], mode="post_cgm")
    contract.validate(df)  # must not raise


def test_contract_raises_on_missing_expected_column():
    df = pd.DataFrame({"hr_roll_mean_4": [70.0]})
    contract = FeatureContract(
        feature_cols=["hr_roll_mean_4", "hour_sin"],  # hour_sin is absent
        mode="post_cgm",
    )
    with pytest.raises(ValueError, match="expected columns missing"):
        contract.validate(df)


# ── FeatureContract — cgm_active mode ────────────────────────────────────────

def test_contract_cgm_active_allows_cgm_columns():
    cgm_col = next(iter(CGM_FEATURES))
    df = pd.DataFrame({cgm_col: [120.0], "hr_roll_mean_4": [70.0]})
    contract = FeatureContract(feature_cols=[cgm_col, "hr_roll_mean_4"], mode="cgm_active")
    contract.validate(df)  # must not raise


# ── FeatureContract.select() ─────────────────────────────────────────────────

def test_contract_select_returns_ordered_subset():
    df = pd.DataFrame({
        "hr_roll_mean_4": [70.0],
        "hour_sin": [0.5],
        "extra_col": [99.0],
    })
    contract = FeatureContract(
        feature_cols=["hour_sin", "hr_roll_mean_4"],
        mode="post_cgm",
    )
    result = contract.select(df)
    assert list(result.columns) == ["hour_sin", "hr_roll_mean_4"]
    assert "extra_col" not in result.columns


# ── build_cgm_features guard ─────────────────────────────────────────────────

def test_build_cgm_features_raises_in_post_cgm_mode():
    df = pd.DataFrame(
        {"glucose_mg_dl": [100.0, 105.0, 110.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="15min"),
    )
    with pytest.raises(RuntimeError, match="post_cgm"):
        build_cgm_features(df, mode="post_cgm")


def test_build_cgm_features_runs_in_cgm_active_mode():
    idx = pd.date_range("2024-01-01", periods=30, freq="15min")
    df = pd.DataFrame({"glucose_mg_dl": range(80, 110)}, index=idx)
    result = build_cgm_features(df, mode="cgm_active")
    # All output columns must be present
    for col in CGM_OUTPUT_COLS:
        assert col in result.columns, f"Expected column {col!r} missing from result"


# ── FeatureContract invalid mode ─────────────────────────────────────────────

def test_contract_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        FeatureContract(feature_cols=[], mode="unknown")  # type: ignore[arg-type]
