"""
Chronological train/val/test splitters — two strategies:

1. chronological_split / population_split
   Fraction-based (60/20/20).  Used for the original CGM-active pipeline
   where the exact number of calendar days varies across users.

2. day_split / population_day_split
   Day-based (10 / 2 / 2 calendar days).  Required for the Stage A / Stage B
   CGM calibration pipeline where the study period is a fixed 14-day window.
   Each split boundary falls on a midnight — no row from day N ever appears
   in the split that starts at day N+1.

Key rules (anti-leakage charter):
  - All splits are by time — no shuffling ever.
  - Scaler is fitted on train only; applied to val/test.
  - Population split: each user is split individually, then concatenated.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.config import TRAIN_FRAC, VAL_FRAC, TRAIN_DAYS, VAL_DAYS, TEST_DAYS
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame
    train_end:   pd.Timestamp    # last timestamp in train
    val_start:   pd.Timestamp    # first timestamp in val
    val_end:     pd.Timestamp    # last timestamp in val
    test_start:  pd.Timestamp    # first timestamp in test
    n_train:     int
    n_val:       int
    n_test:      int

    def summary(self) -> str:
        return (
            f"train={self.n_train} rows (ends {self.train_end}) | "
            f"val={self.n_val} rows ({self.val_start}–{self.val_end}) | "
            f"test={self.n_test} rows (starts {self.test_start})"
        )


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
    user_id:    Optional[str] = None,
) -> SplitResult:
    """
    Split a single user's (or population) DataFrame into train/val/test by time.

    Args:
        df:          DataFrame with DatetimeIndex, sorted ascending.
        train_frac:  Fraction of rows for training (default 0.60).
        val_frac:    Fraction of rows for validation (default 0.20).
                     Remainder goes to test.
        user_id:     For log messages only.

    Returns:
        SplitResult with train, val, test DataFrames and boundary timestamps.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    # Ensure sorted by timestamp (critical — never assume sorted input)
    df = df.sort_index()

    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    # Test gets the remainder (avoids off-by-one from int rounding)
    n_test  = n - n_train - n_val

    if n_test <= 0:
        raise ValueError(
            f"Not enough rows for a 3-way split: {n} rows with "
            f"train={train_frac}, val={val_frac}. "
            "Minimum ~50 rows required."
        )

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val :]

    result = SplitResult(
        train=train,
        val=val,
        test=test,
        train_end=train.index[-1],
        val_start=val.index[0],
        val_end=val.index[-1],
        test_start=test.index[0],
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )

    label = f"user {user_id}" if user_id else "data"
    log.info(f"Split {label}: {result.summary()}")

    return result


def population_split(
    user_dfs: list[pd.DataFrame],
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
) -> SplitResult:
    """
    Split each user individually, then concatenate into population train/val/test.

    Why per-user split:
      If we split on the concatenated DataFrame, users with later collection dates
      would have their training data entirely in the test set of earlier users.
      Per-user splitting ensures every user contributes to all three sets.

    Returns:
        SplitResult where train/val/test contain rows from ALL users.
    """
    trains, vals, tests = [], [], []

    for df in user_dfs:
        uid = df["participant_id"].iloc[0] if "participant_id" in df.columns else "?"
        result = chronological_split(df, train_frac=train_frac, val_frac=val_frac, user_id=uid)
        trains.append(result.train)
        vals.append(result.val)
        tests.append(result.test)

    train_all = pd.concat(trains).sort_index()
    val_all   = pd.concat(vals).sort_index()
    test_all  = pd.concat(tests).sort_index()

    return SplitResult(
        train=train_all,
        val=val_all,
        test=test_all,
        train_end=train_all.index.max(),
        val_start=val_all.index.min(),
        val_end=val_all.index.max(),
        test_start=test_all.index.min(),
        n_train=len(train_all),
        n_val=len(val_all),
        n_test=len(test_all),
    )


def day_split(
    df: pd.DataFrame,
    train_days: int = TRAIN_DAYS,
    val_days:   int = VAL_DAYS,
    test_days:  int = TEST_DAYS,
    user_id:    Optional[str] = None,
) -> SplitResult:
    """
    Split a single user's CGM-period DataFrame into train/val/test by
    calendar day, not by row fraction.

    The split boundary falls exactly at midnight between day N and day N+1,
    so no row from a later day can appear in an earlier split.

    Day numbering starts at 0 from the user's first observed timestamp.
    The required window is train_days + val_days + test_days (default 14).

    Args:
        df:          DataFrame with DatetimeIndex, sorted ascending.
        train_days:  Number of calendar days in the training split (default 10).
        val_days:    Number of calendar days in the validation split (default 2).
        test_days:   Number of calendar days in the test split (default 2).
        user_id:     For log messages only.

    Returns:
        SplitResult with train, val, test DataFrames and boundary timestamps.

    Raises:
        ValueError: if the DataFrame is empty or contains fewer than
                    train_days + val_days + test_days unique calendar days.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    df = df.sort_index()

    required = train_days + val_days + test_days
    first_date = df.index.normalize().min()

    # Day index for each row (0-based from the first observed date)
    day_idx = (df.index.normalize() - first_date).days

    unique_days = int(day_idx.max()) + 1
    if unique_days < required:
        raise ValueError(
            f"day_split requires at least {required} calendar days "
            f"(train={train_days}, val={val_days}, test={test_days}), "
            f"but user {user_id!r} only has {unique_days} days of data."
        )

    train_mask = day_idx < train_days
    val_mask   = (day_idx >= train_days) & (day_idx < train_days + val_days)
    test_mask  = (day_idx >= train_days + val_days) & (day_idx < required)

    train = df[train_mask]
    val   = df[val_mask]
    test  = df[test_mask]

    result = SplitResult(
        train=train,
        val=val,
        test=test,
        train_end=train.index[-1],
        val_start=val.index[0],
        val_end=val.index[-1],
        test_start=test.index[0],
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
    )

    label = f"user {user_id}" if user_id else "data"
    log.info(
        f"Day split {label} ({train_days}/{val_days}/{test_days} days): "
        f"{result.summary()}"
    )
    return result


def population_day_split(
    user_dfs:   list[pd.DataFrame],
    train_days: int = TRAIN_DAYS,
    val_days:   int = VAL_DAYS,
    test_days:  int = TEST_DAYS,
) -> SplitResult:
    """
    Apply day_split to each user individually, then concatenate.

    Users with fewer than the required number of days are skipped with a
    warning rather than raising, so one short user does not abort a full
    population run.

    Returns:
        SplitResult where train/val/test contain rows from all eligible users.
    """
    trains, vals, tests = [], [], []
    skipped = 0

    for df in user_dfs:
        uid = df["participant_id"].iloc[0] if "participant_id" in df.columns else "?"
        try:
            result = day_split(
                df,
                train_days=train_days,
                val_days=val_days,
                test_days=test_days,
                user_id=uid,
            )
        except ValueError as exc:
            log.warning(f"Skipping user {uid} in population_day_split: {exc}")
            skipped += 1
            continue
        trains.append(result.train)
        vals.append(result.val)
        tests.append(result.test)

    if not trains:
        raise ValueError(
            "population_day_split: no users had enough days for a valid split."
        )

    if skipped:
        log.warning(f"population_day_split: skipped {skipped} user(s) with insufficient days.")

    train_all = pd.concat(trains).sort_index()
    val_all   = pd.concat(vals).sort_index()
    test_all  = pd.concat(tests).sort_index()

    return SplitResult(
        train=train_all,
        val=val_all,
        test=test_all,
        train_end=train_all.index.max(),
        val_start=val_all.index.min(),
        val_end=val_all.index.max(),
        test_start=test_all.index.min(),
        n_train=len(train_all),
        n_val=len(val_all),
        n_test=len(test_all),
    )
