"""
Shared utility functions used across the entire GlucoSense AI codebase.

Key functions:
  safe_get_feature()        — zero-fill any optional column; never raises KeyError
  safe_get_future_feature() — same for future regressors (NeuralProphet inference)
  get_logger()              — returns a Loguru logger bound to a module name
"""

import sys
from typing import Any
import numpy as np
import pandas as pd
from loguru import logger as _root_logger


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str):
    """Return a Loguru logger with the module name bound as extra context."""
    return _root_logger.bind(module=name)


# ── Feature safety helpers ────────────────────────────────────────────────────

def safe_get_feature(
    df: pd.DataFrame,
    col: str,
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Return df[col] if it exists (filling NaN with fill_value).
    If the column is absent, return a zero-filled Series with the same index.

    This is the core resilience function for the dual-dataset pipeline.
    Nature's Paper has EDA/IBI/BVP; CGMacros has METs/Calories — they share
    the same code, so one dataset's columns appear as zeros in the other.
    """
    if col in df.columns:
        return df[col].fillna(fill_value)
    return pd.Series(fill_value, index=df.index, name=col, dtype=np.float64)


def safe_get_future_feature(
    feature_name: str,
    request: dict,
    fill_value: float = 0.0,
) -> float:
    """
    Return a future regressor value from an inference request dict.
    Zero-fills gracefully when the feature is absent (Phase 3 activation).

    Used in NeuralProphet inference to build the future_df without crashing
    when planned_meal_carbs_2h or similar Phase 3 features are not yet provided.
    """
    return float(request.get(feature_name, fill_value))


# ── DataFrame inspection helpers ─────────────────────────────────────────────

def describe_df(df: pd.DataFrame, label: str = "") -> None:
    """Print a concise summary of a DataFrame for quick inspection."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*60}")
    print(f"{prefix}shape: {df.shape}")
    print(f"{prefix}columns: {list(df.columns)}")
    print(f"{prefix}dtypes:\n{df.dtypes.to_string()}")
    print(f"{prefix}null counts:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    if not df.empty:
        print(f"{prefix}index range: {df.index.min()} → {df.index.max()}")
    print(f"{'='*60}\n")


def check_no_leakage(df: pd.DataFrame, target_cols: list[str]) -> None:
    """
    Assert that none of the target columns appear in a feature DataFrame.
    Raises AssertionError with a clear message on failure.
    """
    leaked = [c for c in target_cols if c in df.columns]
    assert not leaked, (
        f"Leakage detected! Target columns in feature matrix: {leaked}. "
        "Remove them before passing to any model."
    )
