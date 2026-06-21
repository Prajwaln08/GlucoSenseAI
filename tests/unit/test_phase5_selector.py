"""
Phase 5 unit tests — Unified Clarke-Based Virtual Model Selector.

Covers:
- config.SELECTOR_MIN_CLARKE_A and SELECTOR_MAX_VAL_TEST_GAP are defined
- SelectedModel: has required fields; to_dict() is JSON-serialisable
- VirtualModelSelector._passes_clarke / _passes_gap: correct True/False
- VirtualModelSelector.select: picks lowest val_rmse per horizon
- VirtualModelSelector.select: excludes skipped results
- VirtualModelSelector.select: all-skipped raises RuntimeError
- VirtualModelSelector.select: falls back when nothing passes Clarke filter
- VirtualModelSelector.select: falls back when nothing passes gap filter
- VirtualModelSelector.select: speed-rank tiebreak when val_rmse ties (rounded)
- VirtualModelSelector.select: returns one entry per horizon present
- VirtualModelSelector.select_from_file: round-trip with Phase 4 matrix
- VirtualModelSelector.save_selection: writes selected_models.json
- VirtualModelSelector.load_selection: round-trips selected_models.json
- End-to-end: run experiment matrix on tiny dfs, then select
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # prevent OpenMP segfault when lgbm + xgb coexist on macOS

import numpy as np
import pandas as pd
import pytest

from src.config import SELECTOR_MIN_CLARKE_A, SELECTOR_MAX_VAL_TEST_GAP
from src.experiments.matrix import (
    ExperimentConfig,
    ExperimentResult,
    build_experiment_grid,
    run_experiment_matrix,
    save_matrix_results,
)
from src.experiments.selector import (
    SelectedModel,
    VirtualModelSelector,
    _rows_to_experiment_results,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result(
    model_key:    str,
    horizon:      str,
    val_rmse:     float,
    test_rmse:    float,
    clarke_a_pct: float,
    skipped:      bool = False,
    input_window: int | None = None,
) -> ExperimentResult:
    cfg = ExperimentConfig(
        model_key    = model_key,
        horizon      = horizon,
        input_window = input_window,
        dataset      = "cgmacros",
    )
    return ExperimentResult(
        config       = cfg,
        val_metrics  = {"rmse": val_rmse},
        test_metrics = {"rmse": test_rmse, "clarke_a_pct": clarke_a_pct},
        elapsed_s    = 1.0,
        skipped      = skipped,
    )


def _make_df(n_days: int = 14) -> pd.DataFrame:
    n   = n_days * 96
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "hr":                  np.random.uniform(60, 100, n).astype(float),
            "calories_burned":     np.zeros(n),
            "mets":                np.zeros(n),
            "meal_type_encoded":   np.zeros(n),
            "amount_consumed_pct": np.zeros(n),
            "total_carb":          np.zeros(n),
            "calorie":             np.zeros(n),
            "protein":             np.zeros(n),
            "total_fat":           np.zeros(n),
            "dietary_fiber":       np.zeros(n),
            "sugar":               np.zeros(n),
            "gi_proxy":            np.zeros(n),
            "glucose_mg_dl":       np.random.uniform(80, 180, n).astype(float),
        },
        index=idx,
    )


def _tiny_dfs():
    from src.features.pipeline import build_virtual_feature_matrix
    dfs = [build_virtual_feature_matrix(_make_df()) for _ in range(3)]
    return dfs, ["003", "004", "005"]


# ── Config constants ──────────────────────────────────────────────────────────

def test_config_selector_min_clarke_defined():
    assert SELECTOR_MIN_CLARKE_A == 70.0


def test_config_selector_max_gap_defined():
    assert SELECTOR_MAX_VAL_TEST_GAP == 0.15


# ── SelectedModel ─────────────────────────────────────────────────────────────

def test_selected_model_fields():
    m = SelectedModel(
        model_key="virtual_lgbm", horizon="2h", input_window=None,
        dataset="cgmacros", config_key="virtual_lgbm__2h__winna",
        val_rmse=20.0, test_rmse=21.0, clarke_a_pct=82.0,
        passed_clarke=True, passed_gap=True,
    )
    assert m.model_key     == "virtual_lgbm"
    assert m.horizon       == "2h"
    assert m.input_window  is None
    assert m.passed_clarke is True
    assert m.passed_gap    is True


def test_selected_model_to_dict_is_json_serialisable():
    m = SelectedModel(
        model_key="virtual_lgbm", horizon="2h", input_window=None,
        dataset="cgmacros", config_key="virtual_lgbm__2h__winna",
        val_rmse=20.0, test_rmse=21.0, clarke_a_pct=82.0,
        passed_clarke=True, passed_gap=True,
    )
    d = m.to_dict()
    # must not raise
    json.dumps(d)
    assert d["model_key"] == "virtual_lgbm"


# ── _passes_clarke / _passes_gap ──────────────────────────────────────────────

def test_passes_clarke_true():
    sel = VirtualModelSelector(min_clarke_a=70.0)
    r   = _result("virtual_lgbm", "2h", 20.0, 21.0, clarke_a_pct=75.0)
    assert sel._passes_clarke(r) is True


def test_passes_clarke_false():
    sel = VirtualModelSelector(min_clarke_a=70.0)
    r   = _result("virtual_lgbm", "2h", 20.0, 21.0, clarke_a_pct=60.0)
    assert sel._passes_clarke(r) is False


def test_passes_clarke_exactly_at_threshold():
    sel = VirtualModelSelector(min_clarke_a=70.0)
    r   = _result("virtual_lgbm", "2h", 20.0, 21.0, clarke_a_pct=70.0)
    assert sel._passes_clarke(r) is True


def test_passes_gap_true():
    sel = VirtualModelSelector(max_val_test_gap=0.15)
    # gap = |20 - 22| / 20 = 0.10 — within 15%
    r   = _result("virtual_lgbm", "2h", 20.0, 22.0, clarke_a_pct=80.0)
    assert sel._passes_gap(r) is True


def test_passes_gap_false():
    sel = VirtualModelSelector(max_val_test_gap=0.15)
    # gap = |20 - 40| / 20 = 1.0 — way over 15%
    r   = _result("virtual_lgbm", "2h", 20.0, 40.0, clarke_a_pct=80.0)
    assert sel._passes_gap(r) is False


def test_passes_gap_zero_val_rmse_returns_false():
    sel = VirtualModelSelector(max_val_test_gap=0.15)
    r   = _result("virtual_lgbm", "2h", 0.0, 0.0, clarke_a_pct=80.0)
    assert sel._passes_gap(r) is False


# ── select: basic behaviour ───────────────────────────────────────────────────

def test_selector_picks_lowest_val_rmse():
    results = [
        _result("virtual_lgbm", "2h", 50.0, 52.0, 75.0),
        _result("virtual_xgb",  "2h", 30.0, 31.0, 80.0),  # best
    ]
    sel = VirtualModelSelector()
    best = sel.select(results)
    assert best["2h"].model_key == "virtual_xgb"
    assert best["2h"].val_rmse  == 30.0


def test_selector_returns_entry_per_horizon():
    results = [
        _result("virtual_lgbm", "2h", 50.0, 52.0, 75.0),
        _result("virtual_lgbm", "3h", 55.0, 57.0, 72.0),
    ]
    sel  = VirtualModelSelector()
    best = sel.select(results)
    assert set(best.keys()) == {"2h", "3h"}


def test_selector_excludes_skipped():
    results = [
        _result("virtual_lgbm", "2h", 50.0, 52.0, 75.0),
        _result("virtual_xgb",  "2h", 10.0,  9.0, 95.0, skipped=True),
    ]
    sel  = VirtualModelSelector()
    best = sel.select(results)
    # skipped xgb (val=10) must not win
    assert best["2h"].model_key == "virtual_lgbm"


def test_selector_all_skipped_raises():
    results = [
        _result("virtual_lgbm", "2h", 50.0, 52.0, 75.0, skipped=True),
        _result("virtual_xgb",  "2h", 30.0, 31.0, 80.0, skipped=True),
    ]
    sel = VirtualModelSelector()
    with pytest.raises(RuntimeError, match="skipped"):
        sel.select(results)


# ── select: filter fallback behaviour ────────────────────────────────────────

def test_selector_falls_back_when_no_clarke_pass():
    # Both fail Clarke gate but we should still get a winner
    results = [
        _result("virtual_lgbm", "2h", 50.0, 52.0, clarke_a_pct=40.0),
        _result("virtual_xgb",  "2h", 30.0, 31.0, clarke_a_pct=50.0),
    ]
    sel  = VirtualModelSelector(min_clarke_a=70.0)
    best = sel.select(results)
    assert "2h" in best
    # Fallback still picks lowest val_rmse
    assert best["2h"].model_key  == "virtual_xgb"
    assert best["2h"].passed_clarke is False


def test_selector_falls_back_when_no_gap_pass():
    # Both have huge val/test gap but we should still get a winner
    results = [
        _result("virtual_lgbm", "2h", 20.0, 40.0, 75.0),  # gap = 100%
        _result("virtual_xgb",  "2h", 30.0, 60.0, 80.0),  # gap = 100%
    ]
    sel  = VirtualModelSelector(max_val_test_gap=0.15)
    best = sel.select(results)
    assert "2h" in best
    assert best["2h"].passed_gap is False


# ── select: speed-rank tiebreak ───────────────────────────────────────────────

def test_selector_speed_rank_tiebreak():
    # Both have the same val_rmse (within 0.1 rounding) — lgbm is faster
    results = [
        _result("virtual_xgb",  "2h", 20.05, 21.0, 75.0),
        _result("virtual_lgbm", "2h", 20.00, 21.0, 75.0),
    ]
    sel  = VirtualModelSelector()
    best = sel.select(results)
    assert best["2h"].model_key == "virtual_lgbm"


# ── save / load selection ─────────────────────────────────────────────────────

def test_save_selection_creates_json(tmp_path):
    results = [_result("virtual_lgbm", "2h", 30.0, 31.0, 75.0)]
    sel      = VirtualModelSelector()
    selection = sel.select(results)
    VirtualModelSelector.save_selection(selection, tmp_path, dataset="cgmacros")
    assert (tmp_path / "selected_models.json").exists()


def test_save_selection_json_has_by_horizon(tmp_path):
    results = [
        _result("virtual_lgbm", "2h", 30.0, 31.0, 75.0),
        _result("virtual_lgbm", "3h", 35.0, 36.0, 72.0),
    ]
    sel       = VirtualModelSelector()
    selection = sel.select(results)
    VirtualModelSelector.save_selection(selection, tmp_path)
    with open(tmp_path / "selected_models.json") as f:
        payload = json.load(f)
    assert "by_horizon" in payload
    assert "2h" in payload["by_horizon"]
    assert "3h" in payload["by_horizon"]


def test_load_selection_roundtrips(tmp_path):
    results = [_result("virtual_lgbm", "2h", 30.0, 31.0, 75.0)]
    sel       = VirtualModelSelector()
    selection = sel.select(results)
    VirtualModelSelector.save_selection(selection, tmp_path)
    loaded = VirtualModelSelector.load_selection(tmp_path)
    assert "2h" in loaded
    assert loaded["2h"]["model_key"] == "virtual_lgbm"
    assert loaded["2h"]["val_rmse"]  == 30.0


# ── select_from_file (integration with Phase 4 save/load) ────────────────────

def test_select_from_file_roundtrip(tmp_path):
    dfs, ids = _tiny_dfs()
    configs  = build_experiment_grid(
        model_keys=["virtual_lgbm"],
        horizons=["2h"],
        input_windows=[24],
        dataset="cgmacros",
    )
    matrix_results = run_experiment_matrix(configs, dfs, ids, out_dir=None)
    save_matrix_results(matrix_results, tmp_path)

    sel       = VirtualModelSelector()
    selection = sel.select_from_file(tmp_path)
    assert "2h" in selection
    assert selection["2h"].model_key == "virtual_lgbm"


# ── _rows_to_experiment_results ───────────────────────────────────────────────

def test_rows_to_experiment_results_skipped_flag():
    rows = [
        {
            "model_key": "virtual_lgbm", "horizon": "2h",
            "input_window": None, "dataset": "cgmacros",
            "val_rmse": 30.0, "test_rmse": 31.0, "clarke_a_pct": 75.0,
            "elapsed_s": 1.0, "skipped": True, "error": "oops",
        }
    ]
    results = _rows_to_experiment_results(rows)
    assert len(results) == 1
    assert results[0].skipped is True
    assert results[0].error   == "oops"


# ── End-to-end: matrix → select ──────────────────────────────────────────────

def test_end_to_end_matrix_then_select():
    # Use a single model family to avoid lgbm+xgb OpenMP conflict in one process
    dfs, ids = _tiny_dfs()
    configs  = build_experiment_grid(
        model_keys=["virtual_lgbm"],
        horizons=["2h", "3h"],
        input_windows=[24],
        dataset="cgmacros",
    )
    matrix_results = run_experiment_matrix(configs, dfs, ids, out_dir=None)
    sel       = VirtualModelSelector()
    selection = sel.select(matrix_results)
    assert set(selection.keys()) == {"2h", "3h"}
    for horizon, winner in selection.items():
        assert winner.model_key == "virtual_lgbm"
        assert winner.val_rmse < 1000
        assert 0.0 <= winner.clarke_a_pct <= 100.0
