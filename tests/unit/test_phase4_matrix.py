"""
Phase 4 unit tests — Input-Window × Horizon Experiment Matrix.

Covers:
- build_experiment_grid: tabular models get input_window=None
- build_experiment_grid: RNN models produce all (horizon × input_window) combos
- build_experiment_grid: unique config keys
- build_experiment_grid: empty args raise ValueError
- ExperimentConfig.key: correct format and uniqueness
- ExperimentConfig.is_rnn: correct for each model family
- ExperimentConfig.horizon_steps: maps to HORIZON_2H_STEPS / HORIZON_3H_STEPS
- run_single_experiment: returns ExperimentResult with val/test metrics
- run_single_experiment: captures failure as skipped=True, not exception
- run_experiment_matrix: runs all configs; returns correct count
- run_experiment_matrix: empty configs raises ValueError
- save_matrix_results: writes results.json and summary.csv
- load_matrix_results: round-trips results.json
- find_best_per_horizon: picks lowest val_rmse per horizon
- find_best_per_horizon: excludes skipped results
- find_best_overall: picks global best
- find_best_overall: returns None when all skipped
- config.EXPERIMENT_INPUT_WINDOWS and EXPERIMENT_HORIZONS are set
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    EXPERIMENT_INPUT_WINDOWS,
    EXPERIMENT_HORIZONS,
    HORIZON_2H_STEPS,
    HORIZON_3H_STEPS,
)
from src.experiments.matrix import (
    ExperimentConfig,
    ExperimentResult,
    build_experiment_grid,
    find_best_overall,
    find_best_per_horizon,
    load_matrix_results,
    run_experiment_matrix,
    run_single_experiment,
    save_matrix_results,
    _RNN_KEYS,
    _TABULAR_KEYS,
)
from src.models.zoo import VIRTUAL_MODEL_KEYS


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _make_virtual_fm(n_days: int = 14) -> pd.DataFrame:
    from src.features.pipeline import build_virtual_feature_matrix
    return build_virtual_feature_matrix(_make_df(n_days=n_days))


def _tiny_dfs() -> tuple[list, list]:
    """Three-user virtual feature matrices for fast integration tests."""
    dfs = [_make_virtual_fm(14) for _ in range(3)]
    ids = ["003", "004", "005"]
    return dfs, ids


# ── config constants ──────────────────────────────────────────────────────────

def test_config_experiment_input_windows_defined():
    assert EXPERIMENT_INPUT_WINDOWS == [12, 24, 36]


def test_config_experiment_horizons_defined():
    assert EXPERIMENT_HORIZONS == ["2h", "3h"]


# ── build_experiment_grid: tabular models ─────────────────────────────────────

def test_grid_tabular_input_window_is_none():
    configs = build_experiment_grid(
        model_keys=list(_TABULAR_KEYS),
        horizons=["2h"],
        input_windows=[12, 24, 36],
        dataset="cgmacros",
    )
    for cfg in configs:
        assert cfg.input_window is None, (
            f"{cfg.model_key} should have input_window=None, got {cfg.input_window}"
        )


def test_grid_tabular_count():
    # 2 tabular models × 2 horizons = 4 configs
    configs = build_experiment_grid(
        model_keys=list(_TABULAR_KEYS),
        horizons=["2h", "3h"],
        input_windows=[12, 24, 36],
        dataset="cgmacros",
    )
    assert len(configs) == len(_TABULAR_KEYS) * 2


# ── build_experiment_grid: RNN models ────────────────────────────────────────

def test_grid_rnn_input_windows_all_present():
    configs = build_experiment_grid(
        model_keys=["virtual_gru"],
        horizons=["2h"],
        input_windows=[12, 24, 36],
        dataset="cgmacros",
    )
    windows = {cfg.input_window for cfg in configs}
    assert windows == {12, 24, 36}


def test_grid_rnn_count():
    # 2 RNN models × 2 horizons × 3 input_windows = 12
    configs = build_experiment_grid(
        model_keys=list(_RNN_KEYS),
        horizons=["2h", "3h"],
        input_windows=[12, 24, 36],
        dataset="cgmacros",
    )
    assert len(configs) == len(_RNN_KEYS) * 2 * 3


# ── build_experiment_grid: full grid ─────────────────────────────────────────

def test_grid_full_count():
    # 2 tabular × 2 horizons + 2 rnn × 2 horizons × 3 windows = 4 + 12 = 16
    configs = build_experiment_grid(
        model_keys=VIRTUAL_MODEL_KEYS,
        horizons=["2h", "3h"],
        input_windows=[12, 24, 36],
        dataset="cgmacros",
    )
    assert len(configs) == 4 + 12


def test_grid_unique_keys():
    configs = build_experiment_grid(
        model_keys=VIRTUAL_MODEL_KEYS,
        horizons=["2h", "3h"],
        input_windows=[12, 24, 36],
        dataset="cgmacros",
    )
    keys = [cfg.key for cfg in configs]
    assert len(keys) == len(set(keys)), "Duplicate config keys in grid"


def test_grid_empty_model_keys_raises():
    with pytest.raises(ValueError, match="empty"):
        build_experiment_grid(model_keys=[], horizons=["2h"], input_windows=[12])


def test_grid_empty_horizons_raises():
    with pytest.raises(ValueError, match="empty"):
        build_experiment_grid(model_keys=["virtual_lgbm"], horizons=[], input_windows=[12])


def test_grid_empty_input_windows_raises():
    with pytest.raises(ValueError, match="empty"):
        build_experiment_grid(model_keys=["virtual_lgbm"], horizons=["2h"], input_windows=[])


# ── ExperimentConfig properties ───────────────────────────────────────────────

def test_config_key_format_tabular():
    cfg = ExperimentConfig(
        model_key="virtual_lgbm", horizon="2h",
        input_window=None, dataset="cgmacros",
    )
    assert cfg.key == "virtual_lgbm__2h__winna"


def test_config_key_format_rnn():
    cfg = ExperimentConfig(
        model_key="virtual_gru", horizon="2h",
        input_window=24, dataset="cgmacros",
    )
    assert cfg.key == "virtual_gru__2h__win24"


@pytest.mark.parametrize("key", list(_RNN_KEYS))
def test_config_is_rnn_true(key):
    cfg = ExperimentConfig(model_key=key, horizon="2h", input_window=24, dataset="cgmacros")
    assert cfg.is_rnn is True


@pytest.mark.parametrize("key", list(_TABULAR_KEYS))
def test_config_is_rnn_false(key):
    cfg = ExperimentConfig(model_key=key, horizon="2h", input_window=None, dataset="cgmacros")
    assert cfg.is_rnn is False


def test_config_horizon_steps_2h():
    cfg = ExperimentConfig(model_key="virtual_lgbm", horizon="2h",
                           input_window=None, dataset="cgmacros")
    assert cfg.horizon_steps == HORIZON_2H_STEPS


def test_config_horizon_steps_3h():
    cfg = ExperimentConfig(model_key="virtual_lgbm", horizon="3h",
                           input_window=None, dataset="cgmacros")
    assert cfg.horizon_steps == HORIZON_3H_STEPS


# ── run_single_experiment ─────────────────────────────────────────────────────

def test_single_experiment_lgbm_succeeds():
    dfs, ids = _tiny_dfs()
    cfg = ExperimentConfig(
        model_key="virtual_lgbm", horizon="2h",
        input_window=None, dataset="cgmacros",
    )
    result = run_single_experiment(cfg, dfs, ids)
    assert not result.skipped
    assert result.val_rmse < 1000
    assert result.elapsed_s >= 0.0


def test_single_experiment_captures_failure_as_skipped():
    # Pass empty dfs — VirtualTrainer raises ValueError("empty")
    cfg = ExperimentConfig(
        model_key="virtual_lgbm", horizon="2h",
        input_window=None, dataset="cgmacros",
    )
    result = run_single_experiment(cfg, [], [])
    assert result.skipped is True
    assert result.error is not None


# ── run_experiment_matrix ─────────────────────────────────────────────────────

def test_matrix_runs_all_configs():
    dfs, ids = _tiny_dfs()
    configs = build_experiment_grid(
        model_keys=["virtual_lgbm"],
        horizons=["2h"],
        input_windows=[24],
        dataset="cgmacros",
    )
    results = run_experiment_matrix(configs, dfs, ids, out_dir=None)
    assert len(results) == len(configs)


def test_matrix_empty_configs_raises():
    with pytest.raises(ValueError, match="empty"):
        run_experiment_matrix([], [], [], out_dir=None)


# ── save / load round-trip ────────────────────────────────────────────────────

def test_save_load_round_trip(tmp_path):
    dfs, ids = _tiny_dfs()
    configs = build_experiment_grid(
        model_keys=["virtual_lgbm"],
        horizons=["2h"],
        input_windows=[24],
        dataset="cgmacros",
    )
    results = run_experiment_matrix(configs, dfs, ids, out_dir=None)
    save_matrix_results(results, tmp_path)

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.csv").exists()

    loaded = load_matrix_results(tmp_path)
    assert len(loaded) == len(results)
    assert loaded[0]["model_key"] == "virtual_lgbm"


def test_save_creates_summary_csv(tmp_path):
    dfs, ids = _tiny_dfs()
    configs = build_experiment_grid(
        model_keys=["virtual_lgbm"],
        horizons=["2h"],
        input_windows=[24],
        dataset="cgmacros",
    )
    results = run_experiment_matrix(configs, dfs, ids, out_dir=None)
    save_matrix_results(results, tmp_path)

    df = pd.read_csv(tmp_path / "summary.csv")
    assert "val_rmse" in df.columns
    assert "test_rmse" in df.columns
    assert "clarke_a_pct" in df.columns


# ── find_best_per_horizon / find_best_overall ─────────────────────────────────

def _fake_result(model_key, horizon, val_rmse, skipped=False) -> ExperimentResult:
    cfg = ExperimentConfig(
        model_key=model_key, horizon=horizon,
        input_window=None, dataset="cgmacros",
    )
    return ExperimentResult(
        config=cfg,
        val_metrics={"rmse": val_rmse},
        test_metrics={"rmse": val_rmse + 1.0},
        elapsed_s=1.0,
        skipped=skipped,
    )


def test_find_best_per_horizon_picks_lowest_val_rmse():
    results = [
        _fake_result("virtual_lgbm", "2h", val_rmse=50.0),
        _fake_result("virtual_xgb",  "2h", val_rmse=30.0),   # best 2h
        _fake_result("virtual_lgbm", "3h", val_rmse=55.0),
        _fake_result("virtual_xgb",  "3h", val_rmse=60.0),
    ]
    best = find_best_per_horizon(results)
    assert best["2h"].config.model_key == "virtual_xgb"
    assert best["2h"].val_rmse == 30.0
    assert best["3h"].config.model_key == "virtual_lgbm"


def test_find_best_per_horizon_excludes_skipped():
    results = [
        _fake_result("virtual_lgbm", "2h", val_rmse=50.0),
        _fake_result("virtual_xgb",  "2h", val_rmse=10.0, skipped=True),  # skipped
    ]
    best = find_best_per_horizon(results)
    assert best["2h"].config.model_key == "virtual_lgbm"


def test_find_best_overall():
    results = [
        _fake_result("virtual_lgbm", "2h", val_rmse=50.0),
        _fake_result("virtual_xgb",  "3h", val_rmse=20.0),
    ]
    best = find_best_overall(results)
    assert best.config.model_key == "virtual_xgb"
    assert best.val_rmse == 20.0


def test_find_best_overall_all_skipped_returns_none():
    results = [
        _fake_result("virtual_lgbm", "2h", val_rmse=50.0, skipped=True),
        _fake_result("virtual_xgb",  "3h", val_rmse=20.0, skipped=True),
    ]
    assert find_best_overall(results) is None
