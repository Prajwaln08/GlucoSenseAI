"""
Step 5 — Feature selection checks.

Covers low-variance drop, all-NaN drop, high-correlation drop, protected
columns, train-fitted transform applied to val, and save/load round-trip.

Run:
    conda activate glucosenseai
    pytest tests/check/test_step5_selection.py -v
"""

import numpy as np
import pandas as pd

from src.data.step5_selection import FeatureSelector, fit_feature_selector


def _train(n=50) -> pd.DataFrame:
    rng = np.arange(n, dtype=float)
    return pd.DataFrame({
        "good_a": rng + np.sin(rng),       # informative
        "good_b": np.cos(rng),             # informative, uncorrelated with good_a
        "constant": 3.0,                   # zero variance -> drop
        "all_nan": np.nan,                 # structural absence -> drop
        "dup_of_a": rng + np.sin(rng),     # perfectly correlated with good_a -> drop
    })


def test_low_variance_and_all_nan_dropped():
    sel = fit_feature_selector(_train())
    assert "constant" in sel.dropped_low_var
    assert "all_nan" in sel.dropped_low_var
    assert "constant" not in sel.kept
    assert "all_nan" not in sel.kept


def test_high_correlation_drops_one_of_pair():
    sel = fit_feature_selector(_train())
    assert "dup_of_a" in sel.dropped_high_corr
    assert "good_a" in sel.kept              # the earlier column survives


def test_protect_keeps_column_even_if_constant():
    sel = fit_feature_selector(_train(), protect=["constant"])
    assert "constant" in sel.kept
    assert "constant" not in sel.dropped_low_var


def test_transform_applies_keeplist_to_val():
    train = _train()
    sel = fit_feature_selector(train)
    val = train.iloc[:10].copy()
    out = sel.transform(val)
    assert list(out.columns) == sel.kept
    # a column missing at inference is added back as NaN (no crash)
    val_missing = val.drop(columns=[sel.kept[0]])
    out2 = sel.transform(val_missing)
    assert out2[sel.kept[0]].isna().all()


def test_save_load_round_trip(tmp_path):
    sel = fit_feature_selector(_train())
    path = tmp_path / "feature_cols.json"
    sel.save(path)
    loaded = FeatureSelector.load(path)
    assert loaded.kept == sel.kept
    assert loaded.dropped_high_corr == sel.dropped_high_corr
