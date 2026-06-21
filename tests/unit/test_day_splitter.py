"""
Unit tests for day_split() and population_day_split() in src/data/splitter.py.

Covers:
- Correct row assignment to train / val / test by calendar day
- No row overlap between splits
- Strict chronological ordering within each split
- Raises on empty DataFrame or insufficient days
- population_day_split skips short users rather than aborting
"""

import pytest
import pandas as pd
import numpy as np

from src.data.splitter import day_split, population_day_split


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n_days: int, rows_per_day: int = 4, user_id: str = "tst") -> pd.DataFrame:
    """Create a minimal DataFrame spanning n_days calendar days."""
    idx = pd.date_range("2024-01-01", periods=n_days * rows_per_day, freq="6h")
    df = pd.DataFrame(
        {"glucose_mg_dl": np.random.default_rng(0).integers(80, 180, len(idx)).astype(float),
         "participant_id": user_id},
        index=idx,
    )
    return df


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_day_split_row_counts():
    df = _make_df(n_days=14, rows_per_day=4)
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    assert result.n_train == 40   # 10 days × 4 rows
    assert result.n_val   == 8    # 2 days × 4 rows
    assert result.n_test  == 8    # 2 days × 4 rows


def test_day_split_total_rows_preserved():
    df = _make_df(n_days=14, rows_per_day=4)
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    assert result.n_train + result.n_val + result.n_test == len(df)


def test_day_split_no_row_overlap():
    df = _make_df(n_days=14, rows_per_day=4)
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    train_idx = set(result.train.index)
    val_idx   = set(result.val.index)
    test_idx  = set(result.test.index)
    assert train_idx.isdisjoint(val_idx),  "train and val share rows"
    assert train_idx.isdisjoint(test_idx), "train and test share rows"
    assert val_idx.isdisjoint(test_idx),   "val and test share rows"


def test_day_split_chronological_order():
    df = _make_df(n_days=14, rows_per_day=4)
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    assert result.train_end  < result.val_start,  "train end >= val start"
    assert result.val_end    < result.test_start, "val end >= test start"


def test_day_split_boundaries_align_with_days():
    """Last train row must be before day 10 midnight; first val row on day 10."""
    df = _make_df(n_days=14, rows_per_day=96)  # every 15 min
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    day_10_midnight = pd.Timestamp("2024-01-11")  # day index 10 starts here
    assert result.train_end   < day_10_midnight
    assert result.val_start  >= day_10_midnight


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_day_split_raises_on_empty_df():
    df = pd.DataFrame(
        {"glucose_mg_dl": []},
        index=pd.DatetimeIndex([]),
    )
    with pytest.raises(ValueError, match="empty"):
        day_split(df)


def test_day_split_raises_on_insufficient_days():
    df = _make_df(n_days=10, rows_per_day=4)  # only 10 days, need 14
    with pytest.raises(ValueError, match="at least 14 calendar days"):
        day_split(df, train_days=10, val_days=2, test_days=2)


def test_day_split_exactly_required_days():
    """A 14-day DataFrame should pass without error."""
    df = _make_df(n_days=14, rows_per_day=4)
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    assert result.n_train > 0
    assert result.n_val   > 0
    assert result.n_test  > 0


def test_day_split_more_than_required_uses_only_14():
    """Extra days beyond 14 are excluded (we only want the calibration window)."""
    df = _make_df(n_days=20, rows_per_day=4)
    result = day_split(df, train_days=10, val_days=2, test_days=2)
    assert result.n_train + result.n_val + result.n_test == 14 * 4


# ── population_day_split ──────────────────────────────────────────────────────

def test_population_day_split_aggregates_users():
    dfs = [_make_df(14, rows_per_day=4, user_id=f"{i:03d}") for i in range(3)]
    result = population_day_split(dfs, train_days=10, val_days=2, test_days=2)
    # Each user contributes 40 train + 8 val + 8 test rows
    assert result.n_train == 3 * 40
    assert result.n_val   == 3 * 8
    assert result.n_test  == 3 * 8


def test_population_day_split_skips_short_user():
    long_df  = _make_df(14, rows_per_day=4, user_id="001")
    short_df = _make_df(7,  rows_per_day=4, user_id="002")  # too few days
    # Should not raise; short_df user is skipped
    result = population_day_split([long_df, short_df], train_days=10, val_days=2, test_days=2)
    assert result.n_train == 40   # only the long user contributes


def test_population_day_split_raises_if_all_users_short():
    dfs = [_make_df(7, rows_per_day=4, user_id=f"{i:03d}") for i in range(3)]
    with pytest.raises(ValueError, match="no users had enough days"):
        population_day_split(dfs, train_days=10, val_days=2, test_days=2)
