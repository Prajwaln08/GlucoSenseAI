"""
CHECK 05 — Full Pipeline + Leakage Checks

End-to-end test: preprocessed DataFrame → feature matrix → split → validator.

Key assertions:
  1. build_feature_matrix produces features + targets.
  2. No target columns in the feature matrix (X).
  3. Temporal split has no timestamp overlap.
  4. train.max < val.min < test.min.
  5. Target values are future glucose (positive shift works correctly).
  6. get_X_y returns shapes that match expected feature count.

Run:
    conda activate glucosenseai
    pytest tests/check/05_test_check_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_feature_matrix, get_X_y, get_feature_cols, TARGET_COLS
from src.data.splitter import chronological_split, population_split
from src.data.validator import validate_no_leakage, validate_schema, validate_target_availability
from src.config import HORIZON_2H_STEPS, HORIZON_3H_STEPS


# ══════════════════════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def preprocessed_df() -> pd.DataFrame:
    """
    Simulate a fully preprocessed 15-min DataFrame for one user.
    300 rows = ~75 hours of data.
    """
    n   = 300
    idx = pd.date_range("2020-03-01", periods=n, freq="15min", tz="UTC")
    df  = pd.DataFrame(index=idx)

    df["glucose_mg_dl"]       = 100 + 20 * np.sin(np.arange(n) / 8)
    df["glucose_rate_of_change"] = 0.5 * np.cos(np.arange(n) / 8)
    df["hr"]                  = 75 + 3 * np.random.randn(n)
    df["eda"]                 = np.abs(np.random.randn(n) * 0.5)
    df["ibi_mean"]            = 0.85 + 0.05 * np.random.randn(n)
    df["ibi_rmssd"]           = np.abs(np.random.randn(n) * 0.02)
    df["temp"]                = 35 + 0.2 * np.random.randn(n)
    df["total_carb"]          = 0.0
    df["calorie"]             = 0.0
    df["sugar"]               = 0.0
    df["gi_proxy"]            = 0.0
    df["meal_flag"]           = 0
    df["meal_type_encoded"]   = 0
    df["amount_consumed_pct"] = 0.0
    df["calories_burned"]     = np.abs(np.random.randn(n))
    df["mets"]                = 1.0 + np.abs(np.random.randn(n))
    df["acc_magnitude_mean"]  = np.abs(np.random.randn(n) * 30)
    df["acc_magnitude_std"]   = np.abs(np.random.randn(n) * 5)
    df["participant_id"]      = "003"
    df["dataset"]             = "nature_paper"

    # Add 3 realistic meals
    df.iloc[20, df.columns.get_loc("total_carb")] = 60.0
    df.iloc[20, df.columns.get_loc("calorie")]    = 500.0
    df.iloc[60, df.columns.get_loc("total_carb")] = 80.0
    df.iloc[60, df.columns.get_loc("calorie")]    = 700.0
    df.iloc[100, df.columns.get_loc("total_carb")] = 40.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline tests
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_returns_dataframe(preprocessed_df):
    result = build_feature_matrix(preprocessed_df, user_id="003")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_pipeline_has_both_targets(preprocessed_df):
    result = build_feature_matrix(preprocessed_df, user_id="003")
    assert any(c.startswith("target_2h_step") for c in result.columns), \
        "target_2h_step* columns must be in output."
    assert any(c.startswith("target_3h_step") for c in result.columns), \
        "target_3h_step* columns must be in output."


def test_pipeline_no_nan_in_features(preprocessed_df):
    """After pipeline, no NaN should remain in any column."""
    result = build_feature_matrix(preprocessed_df, user_id="003")
    total_nan = result.isnull().sum().sum()
    assert total_nan == 0, f"Found {total_nan} NaN values after pipeline."


def test_pipeline_drops_trailing_target_nans(preprocessed_df):
    """
    Rows at the end of the series have no future glucose (target is NaN).
    These should be dropped by the pipeline.
    """
    n_in = len(preprocessed_df)
    result = build_feature_matrix(preprocessed_df, drop_target_nans=True)
    # At least HORIZON_3H_STEPS rows should be dropped (trailing)
    assert len(result) <= n_in - HORIZON_3H_STEPS, \
        "Trailing NaN target rows must be dropped."


def test_pipeline_target_2h_is_correct_future_glucose(preprocessed_df):
    """
    target_2h_step08 for row i should equal glucose_mg_dl for row i+8.
    Verify for a few rows to confirm correct shift direction.
    """
    result = build_feature_matrix(preprocessed_df, drop_target_nans=False)

    final_step_col = f"target_2h_step{HORIZON_2H_STEPS:02d}"
    assert final_step_col in result.columns, \
        f"{final_step_col} must be in build_feature_matrix output."

    glucose = preprocessed_df["glucose_mg_dl"].values
    target  = result[final_step_col].values

    # Targets are delta values: glucose[i+step] - glucose[i]
    for i in range(10):
        expected_delta = glucose[i + HORIZON_2H_STEPS] - glucose[i]
        actual         = target[i]
        assert abs(expected_delta - actual) < 1e-6, \
            f"{final_step_col}[{i}]={actual:.3f} but expected delta " \
            f"glucose[{i+HORIZON_2H_STEPS}]-glucose[{i}]={expected_delta:.3f}"


def test_get_feature_cols_excludes_targets(preprocessed_df):
    result = build_feature_matrix(preprocessed_df)
    feature_cols = get_feature_cols(result)
    for tc in TARGET_COLS:
        assert tc not in feature_cols, f"Target column {tc} must NOT be in feature_cols."


def test_get_X_y_shapes_consistent(preprocessed_df):
    result = build_feature_matrix(preprocessed_df)
    X, y   = get_X_y(result, horizon="2h")
    assert len(X) == len(y), "X and y must have the same number of rows."
    assert len(X.columns) > 0, "X must have at least one feature column."


def test_get_X_y_no_target_in_X(preprocessed_df):
    result = build_feature_matrix(preprocessed_df)
    X, _   = get_X_y(result, horizon="2h")
    for tc in TARGET_COLS:
        assert tc not in X.columns, f"Target {tc} must not appear in X."


def test_get_X_y_y_has_no_nan(preprocessed_df):
    result = build_feature_matrix(preprocessed_df)
    _, y   = get_X_y(result, horizon="2h")
    # y is a multi-output DataFrame (n_samples × n_steps)
    assert y.isna().sum().sum() == 0, "y must have no NaN values across all steps."


# ══════════════════════════════════════════════════════════════════════════════
# Splitter tests
# ══════════════════════════════════════════════════════════════════════════════

def test_split_temporal_order(preprocessed_df):
    fm     = build_feature_matrix(preprocessed_df)
    result = chronological_split(fm, user_id="003")

    assert result.train_end   < result.val_start,  "train must end before val starts."
    assert result.val_end     < result.test_start, "val must end before test starts."


def test_split_no_timestamp_overlap(preprocessed_df):
    fm     = build_feature_matrix(preprocessed_df)
    result = chronological_split(fm, user_id="003")

    train_idx = set(result.train.index.astype(str))
    val_idx   = set(result.val.index.astype(str))
    test_idx  = set(result.test.index.astype(str))

    assert len(train_idx & val_idx)  == 0, "No overlap between train and val."
    assert len(val_idx  & test_idx)  == 0, "No overlap between val and test."
    assert len(train_idx & test_idx) == 0, "No overlap between train and test."


def test_split_row_counts_sum_to_total(preprocessed_df):
    fm     = build_feature_matrix(preprocessed_df)
    result = chronological_split(fm, user_id="003")
    assert result.n_train + result.n_val + result.n_test == len(fm)


def test_split_approximately_60_20_20(preprocessed_df):
    fm     = build_feature_matrix(preprocessed_df)
    total  = len(fm)
    result = chronological_split(fm, user_id="003")

    train_pct = result.n_train / total
    val_pct   = result.n_val   / total
    test_pct  = result.n_test  / total

    assert 0.55 <= train_pct <= 0.65, f"Expected ~60% train, got {train_pct:.0%}"
    assert 0.15 <= val_pct   <= 0.25, f"Expected ~20% val, got {val_pct:.0%}"
    assert 0.15 <= test_pct  <= 0.25, f"Expected ~20% test, got {test_pct:.0%}"


# ══════════════════════════════════════════════════════════════════════════════
# Validator (leakage checks) tests
# ══════════════════════════════════════════════════════════════════════════════

def test_validator_passes_clean_split(preprocessed_df):
    """validate_no_leakage must not raise on a correctly split DataFrame."""
    fm     = build_feature_matrix(preprocessed_df)
    result = chronological_split(fm, user_id="003")
    feature_cols = get_feature_cols(fm)
    # Should not raise
    validate_no_leakage(result.train, result.val, result.test, feature_cols)


def test_validator_fails_on_target_in_features(preprocessed_df):
    """Validator must raise if target column is in feature list."""
    fm     = build_feature_matrix(preprocessed_df)
    result = chronological_split(fm, user_id="003")
    bad_features = get_feature_cols(fm) + ["target_2h_step08"]
    with pytest.raises(AssertionError, match="LEAKAGE"):
        validate_no_leakage(result.train, result.val, result.test, bad_features)


def test_validator_fails_on_shuffled_split(preprocessed_df):
    """If val timestamps precede train timestamps, validator must fail."""
    fm     = build_feature_matrix(preprocessed_df)
    # Deliberately create a bad split (reversed order)
    n = len(fm)
    bad_train = fm.iloc[n // 2 :].copy()  # later timestamps in train
    bad_val   = fm.iloc[: n // 4].copy()  # earlier timestamps in val
    bad_test  = fm.iloc[n // 4 : n // 2].copy()
    with pytest.raises(AssertionError, match="LEAKAGE"):
        validate_no_leakage(bad_train, bad_val, bad_test)


def test_validate_schema_passes(preprocessed_df):
    """validate_schema must not raise for a valid preprocessed DataFrame."""
    validate_schema(preprocessed_df, dataset="test_user", min_rows=100)


def test_validate_target_availability(preprocessed_df):
    fm = build_feature_matrix(preprocessed_df)
    validate_target_availability(fm, horizon="2h")
    validate_target_availability(fm, horizon="3h")


# ══════════════════════════════════════════════════════════════════════════════
# Population split
# ══════════════════════════════════════════════════════════════════════════════

def test_population_split_preserves_all_rows(preprocessed_df):
    """All rows from all users must appear in exactly one of train/val/test."""
    user_a = build_feature_matrix(preprocessed_df.copy())
    user_b = build_feature_matrix(preprocessed_df.copy().rename(
        index=lambda ts: ts + pd.Timedelta(days=30)
    ))
    user_b["participant_id"] = "004"

    result = population_split([user_a, user_b])
    total = result.n_train + result.n_val + result.n_test
    assert total == len(user_a) + len(user_b), \
        "All rows must be in exactly one split."
