"""
Leakage sanity assertions and schema validation.

Run these checks before EVERY training run.  If any assertion fails,
training is aborted — a leaking model is worse than no model at all.

Usage:
    from src.data.validator import validate_no_leakage, validate_schema
    validate_no_leakage(train, val, test, feature_cols)
    validate_schema(df, dataset="cgmacros")
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.config import HORIZON_2H_STEPS, HORIZON_3H_STEPS
from src.utils import get_logger

log = get_logger(__name__)

TARGET_COLS = ["target_2h", "target_3h"]  # legacy single-step names (kept for any callers)
_TARGET_PREFIX = "target_"               # multi-step targets follow target_<horizon>_step<NN>


# ══════════════════════════════════════════════════════════════════════════════
# Leakage checks
# ══════════════════════════════════════════════════════════════════════════════

def validate_no_leakage(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> None:
    """
    Run all anti-leakage assertions.  Raises AssertionError on any failure.

    Checks:
      1. Temporal ordering: train.max < val.min < test.min
      2. No timestamp overlap between splits
      3. Target columns not in feature matrix
      4. No lag-0 shift features (features computed from current timestep)
    """
    log.info("Running leakage sanity checks …")
    _check_temporal_order(train, val, test)
    _check_no_overlap(train, val, test)
    if feature_cols:
        _check_targets_not_in_features(feature_cols)
        _check_no_zero_lag_features(feature_cols)
    log.info("All leakage checks passed.")


def _check_temporal_order(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> None:
    train_max = train.index.max()
    val_min   = val.index.min()
    val_max   = val.index.max()
    test_min  = test.index.min()

    assert train_max < val_min, (
        f"LEAKAGE: train ends at {train_max} but val starts at {val_min}. "
        "Train timestamps must be strictly before val timestamps."
    )
    assert val_max < test_min, (
        f"LEAKAGE: val ends at {val_max} but test starts at {test_min}. "
        "Val timestamps must be strictly before test timestamps."
    )
    log.debug(
        f"Temporal order: train→{train_max} | val {val_min}→{val_max} | test {test_min}→"
    )


def _check_no_overlap(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> None:
    train_idx = set(train.index.astype(str))
    val_idx   = set(val.index.astype(str))
    test_idx  = set(test.index.astype(str))

    tv_overlap = train_idx & val_idx
    vt_overlap = val_idx & test_idx
    tt_overlap = train_idx & test_idx

    assert not tv_overlap, f"LEAKAGE: {len(tv_overlap)} timestamps in both train and val."
    assert not vt_overlap, f"LEAKAGE: {len(vt_overlap)} timestamps in both val and test."
    assert not tt_overlap, f"LEAKAGE: {len(tt_overlap)} timestamps in both train and test."
    log.debug("No timestamp overlap between splits.")


def _check_targets_not_in_features(feature_cols: list[str]) -> None:
    leaked = [c for c in feature_cols if c.startswith(_TARGET_PREFIX)]
    assert not leaked, (
        f"LEAKAGE: target columns in feature matrix: {leaked}. "
        "Remove them before training."
    )
    log.debug("Target columns absent from feature matrix.")


def _check_no_zero_lag_features(feature_cols: list[str]) -> None:
    """
    Warn if any feature column appears to be a lag-0 feature.
    We look for '_lag_0' patterns (conservative check).
    A full check would inspect the code, but this catches common mistakes.
    """
    lag0_suspects = [c for c in feature_cols if "_lag_0" in c.lower() or "_shift_0" in c.lower()]
    assert not lag0_suspects, (
        f"LEAKAGE: possible lag-0 features detected: {lag0_suspects}. "
        "All lag features must use shift(n) with n >= 1."
    )
    log.debug("No lag-0 feature names detected.")


# ══════════════════════════════════════════════════════════════════════════════
# Schema validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_schema(
    df: pd.DataFrame,
    dataset: str = "unknown",
    min_rows: int = 200,
) -> None:
    """
    Assert that a preprocessed DataFrame meets minimum quality requirements.

    Checks:
      1. DataFrame is not empty (minimum row count).
      2. glucose_mg_dl column exists and has no NaN.
      3. Index is a UTC-aware DatetimeIndex.
      4. Index is monotonically increasing.
      5. No NaN in glucose (after preprocessing).
    """
    log.info(f"Validating schema for {dataset} …")

    assert len(df) >= min_rows, (
        f"{dataset}: only {len(df)} rows — minimum {min_rows} required for training."
    )

    assert "glucose_mg_dl" in df.columns, (
        f"{dataset}: 'glucose_mg_dl' column missing. Cannot train without glucose target."
    )

    assert df["glucose_mg_dl"].isna().sum() == 0, (
        f"{dataset}: {df['glucose_mg_dl'].isna().sum()} NaN values in glucose_mg_dl "
        "after preprocessing. Check the imputation pipeline."
    )

    assert isinstance(df.index, pd.DatetimeIndex), (
        f"{dataset}: index is not a DatetimeIndex (got {type(df.index).__name__})."
    )

    assert df.index.tz is not None, (
        f"{dataset}: DatetimeIndex is timezone-naive. Convert to UTC before training."
    )

    assert df.index.is_monotonic_increasing, (
        f"{dataset}: index is not monotonically increasing. Sort by timestamp before training."
    )

    nan_pct = df.isnull().mean().mean() * 100
    if nan_pct > 5.0:
        warnings.warn(
            f"{dataset}: {nan_pct:.1f}% NaN across all columns after preprocessing. "
            "Consider reviewing imputation rules.",
            RuntimeWarning,
        )

    log.info(
        f"Schema valid: {len(df)} rows, {len(df.columns)} cols, "
        f"span {df.index.min()} → {df.index.max()}, NaN {nan_pct:.1f}%"
    )


def validate_target_availability(df: pd.DataFrame, horizon: str = "2h") -> None:
    """
    Verify that multi-step target columns for the given horizon exist and have data.
    Call this AFTER pipeline.py computes targets.
    """
    prefix = f"target_{horizon}_step"
    step_cols = [c for c in df.columns if c.startswith(prefix)]
    assert step_cols, (
        f"No target columns for horizon '{horizon}' found (expected '{prefix}NN'). "
        "Run the feature pipeline first."
    )
    first_col = step_cols[0]
    n_nan   = df[first_col].isna().sum()
    n_total = len(df)
    pct_valid = 100 * (n_total - n_nan) / n_total
    log.info(
        f"Targets {prefix}* ({len(step_cols)} steps): "
        f"{n_total - n_nan}/{n_total} valid rows ({pct_valid:.1f}%). "
        f"{n_nan} NaN rows will be dropped before training."
    )
