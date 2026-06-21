"""
CHECK 06 — Model Layer

Tests BaseModel contract, all zoo models (LightGBM / XGBoost / RandomForest),
evaluator metrics, Clarke Error Grid, population trainer, and selector.
All tests use synthetic DataFrames — no Google Drive access or file downloads required.

Run:
    conda activate glucosenseai
    pytest tests/check/06_test_check_models.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.zoo import MODEL_REGISTRY, get_model
from src.models.zoo.lgbm_model import LightGBMModel
from src.models.zoo.xgb_model import XGBoostModel
from src.models.zoo.rf_model import RandomForestModel
from src.models.evaluator import (
    compute_metrics,
    clarke_error_grid,
    _classify_clarke_zone,
    plot_true_vs_pred,
    plot_scatter,
    plot_residuals,
    plot_clarke_grid,
    evaluate_and_plot,
)
from src.models.population.trainer import PopulationTrainer, _chronological_split
from src.models.individual.trainer import IndividualTrainer
from src.models.selector import ModelSelector


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_regression_data():
    """Synthetic regression dataset: 200 rows, 10 features, 8 multi-step targets."""
    rng    = np.random.default_rng(42)
    n      = 200
    n_steps = 8
    X = pd.DataFrame(rng.standard_normal((n, 10)),
                     columns=[f"feat_{i}" for i in range(10)])
    base  = 100 + 20 * np.sin(np.arange(n) / 5) + rng.standard_normal(n) * 5
    # Each step is the base glucose shifted slightly forward (simulate multi-step target)
    y = pd.DataFrame(
        {f"target_2h_step{s+1:02d}": base + s * 0.5 for s in range(n_steps)}
    )
    split    = int(n * 0.6)
    valsplit = int(n * 0.8)
    return {
        "X_train": X.iloc[:split],
        "y_train": y.iloc[:split],
        "X_val":   X.iloc[split:valsplit],
        "y_val":   y.iloc[split:valsplit],
        "X_test":  X.iloc[valsplit:],
        "y_test":  y.iloc[valsplit:],
        "n_steps": n_steps,
    }


@pytest.fixture
def user_feature_matrix() -> pd.DataFrame:
    """
    Synthetic 15-min feature matrix for one user (300 rows ≈ 75 h).
    Matches the schema produced by build_feature_matrix().
    """
    n   = 300
    idx = pd.date_range("2020-03-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)

    df = pd.DataFrame(index=idx)
    df["glucose_mg_dl"]            = 100 + 20 * np.sin(np.arange(n) / 8)
    df["glucose_rate_of_change"]   = 0.5 * np.cos(np.arange(n) / 8)
    df["hr"]                       = 75 + 3 * rng.standard_normal(n)
    df["eda"]                      = np.abs(rng.standard_normal(n) * 0.5)
    df["ibi_mean"]                 = 0.85 + 0.05 * rng.standard_normal(n)
    df["ibi_rmssd"]                = np.abs(rng.standard_normal(n) * 0.02)
    df["temp"]                     = 35 + 0.2 * rng.standard_normal(n)
    df["total_carb"]               = 0.0
    df["calorie"]                  = 0.0
    df["sugar"]                    = 0.0
    df["gi_proxy"]                 = 0.0
    df["meal_flag"]                = 0
    df["meal_type_encoded"]        = 0
    df["amount_consumed_pct"]      = 0.0
    df["calories_burned"]          = np.abs(rng.standard_normal(n))
    df["mets"]                     = 1.0 + np.abs(rng.standard_normal(n))
    df["acc_magnitude_mean"]       = np.abs(rng.standard_normal(n) * 30)
    df["acc_magnitude_std"]        = np.abs(rng.standard_normal(n) * 5)
    df["participant_id"]           = "003"
    df["dataset"]                  = "nature_paper"

    # Multi-step targets — each step i is glucose shifted forward by i rows (1-indexed)
    for step in range(1, 9):   # 8 steps = 2h at 15-min resolution
        df[f"target_2h_step{step:02d}"] = df["glucose_mg_dl"].shift(-step)
    for step in range(1, 13):  # 12 steps = 3h
        df[f"target_3h_step{step:02d}"] = df["glucose_mg_dl"].shift(-step)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    df = df.dropna(subset=target_cols)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Registry / zoo tests
# ══════════════════════════════════════════════════════════════════════════════

def test_registry_has_three_models():
    assert len(MODEL_REGISTRY) >= 3, "Registry must have at least 3 models."


def test_registry_contains_core_models():
    for name in ("lightgbm", "xgboost", "random_forest"):
        assert name in MODEL_REGISTRY, f"'{name}' must be in MODEL_REGISTRY."


def test_get_model_returns_instance():
    m = get_model("lightgbm")
    assert isinstance(m, LightGBMModel)


def test_get_model_unknown_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("nonexistent_model")


def test_model_name_attribute():
    for name, cls in MODEL_REGISTRY.items():
        assert hasattr(cls, "name"), f"{name} must have a 'name' class attribute."


# ══════════════════════════════════════════════════════════════════════════════
# BaseModel contract — all zoo models
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "random_forest"])
def test_model_fit_and_predict(small_regression_data, model_name):
    d = small_regression_data
    m = get_model(model_name)
    m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    preds = m.predict(d["X_test"])
    assert isinstance(preds, np.ndarray), "predict() must return ndarray."
    # Multi-output: shape is (n_test, n_steps)
    assert preds.shape == (len(d["X_test"]), d["n_steps"]), \
        f"predict() shape must be (n_test={len(d['X_test'])}, n_steps={d['n_steps']}), got {preds.shape}."


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "random_forest"])
def test_model_get_params(small_regression_data, model_name):
    m = get_model(model_name)
    params = m.get_params()
    assert isinstance(params, dict), "get_params() must return a dict."
    assert len(params) > 0, "get_params() must not return an empty dict."


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "random_forest"])
def test_model_save_load_round_trip(small_regression_data, model_name, tmp_path):
    d = small_regression_data
    m = get_model(model_name)
    m.fit(d["X_train"], d["y_train"])
    preds_before = m.predict(d["X_test"])  # shape (n_test, n_steps)

    path = tmp_path / f"{model_name}.pkl"
    m.save(path)
    assert path.exists(), "save() must create the file."

    m2 = type(m).load(path)
    preds_after = m2.predict(d["X_test"])

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5,
                               err_msg="Multi-step predictions must match after save/load.")


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "random_forest"])
def test_model_search_space(model_name):
    """get_search_space() must return a dict when given a mock trial."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    m = get_model(model_name)

    def _objective(trial):
        space = m.get_search_space(trial)
        assert isinstance(space, dict)
        return 0.0

    study = optuna.create_study()
    study.optimize(_objective, n_trials=1)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "random_forest"])
def test_predictions_in_plausible_glucose_range(small_regression_data, model_name):
    """Sanity check: predictions should be in a sensible glucose range after fit."""
    d  = small_regression_data
    m  = get_model(model_name)
    m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    p  = m.predict(d["X_test"])  # shape (n_test, n_steps)
    # The synthetic target is ~100 ± 30; predictions should not wildly diverge
    assert p.mean() > 0, "Mean prediction across all steps must be positive."
    assert p.std() < 200, "Prediction std across all steps must be plausible."


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator — metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_metrics_perfect_prediction():
    y = np.array([100.0, 120.0, 80.0, 150.0, 70.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["mae"]  == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_known_rmse():
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 90.0,  110.0, 90.0])
    m = compute_metrics(y_true, y_pred)
    assert m["rmse"] == pytest.approx(10.0, abs=1e-3)


def test_compute_metrics_tir_all_in_range():
    y = np.linspace(80, 170, 50)
    m = compute_metrics(y, y)
    assert m["tir"] == pytest.approx(100.0, abs=1e-3)


def test_compute_metrics_tir_none_in_range():
    y = np.array([200.0, 250.0, 300.0])
    m = compute_metrics(y, y)
    assert m["tir"] == pytest.approx(0.0, abs=1e-3)


def test_compute_metrics_returns_all_keys():
    y = np.linspace(80, 180, 50)
    m = compute_metrics(y, y + 5)
    for key in ("rmse", "mae", "mard", "tir", "tir_true", "clarke_a_pct",
                "n_samples", "clarke_zones"):
        assert key in m, f"Metric '{key}' must be present."


def test_compute_metrics_handles_empty():
    m = compute_metrics(np.array([]), np.array([]))
    assert m["n_samples"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Clarke Error Grid
# ══════════════════════════════════════════════════════════════════════════════

def test_clarke_perfect_predictions_zone_a():
    y = np.linspace(70, 300, 100)
    zones = clarke_error_grid(y, y)
    assert zones["A"] == pytest.approx(100.0, abs=1e-3), \
        "Perfect predictions must all be Zone A."


def test_clarke_zones_sum_to_100():
    rng    = np.random.default_rng(7)
    y_true = rng.uniform(60, 350, 200)
    y_pred = y_true + rng.normal(0, 30, 200)
    zones  = clarke_error_grid(y_true, y_pred)
    total  = sum(zones.values())
    assert total == pytest.approx(100.0, abs=1e-3), \
        "Clarke zone percentages must sum to 100."


def test_clarke_zones_returns_all_five():
    y = np.array([100.0] * 10)
    zones = clarke_error_grid(y, y)
    assert set(zones.keys()) == {"A", "B", "C", "D", "E"}


@pytest.mark.parametrize("ref,pred,expected_zone", [
    (100.0, 100.0, "A"),    # perfect match
    (100.0, 115.0, "A"),    # within 15% — still Zone A
    (100.0, 121.0, "B"),    # just outside 20%
    (50.0,  200.0, "E"),    # upper E: low ref, high pred
    (300.0, 60.0,  "E"),    # lower E: high ref, low pred
    (50.0,  100.0, "D"),    # upper D: low ref, normal pred
    (280.0, 130.0, "D"),    # lower D: high ref, normal pred
    (120.0, 240.0, "C"),    # upper C: normal ref, very high pred
])
def test_clarke_zone_classification(ref, pred, expected_zone):
    assert _classify_clarke_zone(ref, pred) == expected_zone, \
        f"Expected Zone {expected_zone} for ref={ref}, pred={pred}."


def test_clarke_good_predictions_mostly_zone_a():
    """Predictions within 10% of reference should be predominantly Zone A."""
    rng    = np.random.default_rng(99)
    y_true = rng.uniform(80, 200, 200)
    y_pred = y_true * rng.uniform(0.92, 1.08, 200)  # ±8%
    zones  = clarke_error_grid(y_true, y_pred)
    assert zones["A"] >= 80.0, \
        f"8% deviation predictions should be ≥80% Zone A, got {zones['A']:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# Plots (smoke tests — just ensure no exceptions and figure returned)
# ══════════════════════════════════════════════════════════════════════════════

def test_plot_true_vs_pred_returns_figure():
    y = np.linspace(80, 200, 50)
    fig = plot_true_vs_pred(y, y + 5)
    assert fig is not None


def test_plot_scatter_returns_figure():
    y = np.linspace(80, 200, 50)
    fig = plot_scatter(y, y + 5)
    assert fig is not None


def test_plot_residuals_returns_figure():
    y = np.linspace(80, 200, 50)
    fig = plot_residuals(y, y + 5)
    assert fig is not None


def test_plot_clarke_grid_returns_figure():
    y = np.linspace(80, 200, 50)
    fig = plot_clarke_grid(y, y + 5)
    assert fig is not None


def test_evaluate_and_plot_saves_files(tmp_path):
    n, n_steps = 50, 8
    # Multi-step: shape (n_samples, n_steps) — evaluate_and_plot expects 2D arrays
    base = np.linspace(80, 200, n)
    y_2d = np.column_stack([base + s * 0.5 for s in range(n_steps)])
    metrics = evaluate_and_plot(
        y_true_val=y_2d,   y_pred_val=y_2d + 3,
        y_true_test=y_2d,  y_pred_test=y_2d + 5,
        out_dir=tmp_path,
    )
    assert "val"  in metrics
    assert "test" in metrics
    # At least 3 plot files should be generated
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) >= 3, f"Expected ≥3 PNG files, found {len(pngs)}."


# ══════════════════════════════════════════════════════════════════════════════
# Chronological split helper
# ══════════════════════════════════════════════════════════════════════════════

def test_chronological_split_correct_fractions(user_feature_matrix):
    n     = len(user_feature_matrix)
    split = _chronological_split(user_feature_matrix)
    assert split is not None
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == n


def test_chronological_split_temporal_order(user_feature_matrix):
    split = _chronological_split(user_feature_matrix)
    assert split["train"].index.max() < split["val"].index.min()
    assert split["val"].index.max()   < split["test"].index.min()


def test_chronological_split_too_small():
    tiny = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    assert _chronological_split(tiny) is None


# ══════════════════════════════════════════════════════════════════════════════
# Population trainer (synthetic data, no file I/O)
# ══════════════════════════════════════════════════════════════════════════════

def test_population_trainer_runs(user_feature_matrix, tmp_path, monkeypatch):
    """PopulationTrainer must produce a TrainResult with finite RMSE."""
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.population.trainer.MODELS_DIR", tmp_path)

    model   = get_model("lightgbm", n_estimators=50)
    trainer = PopulationTrainer(model, dataset="nature_paper", horizon="2h")
    result  = trainer.run(
        [user_feature_matrix, user_feature_matrix],
        user_ids=["003", "004"],
        save=False,
        log_mlflow=False,
    )
    assert np.isfinite(result.val_rmse),  "val_rmse must be finite."
    assert np.isfinite(result.test_rmse), "test_rmse must be finite."
    assert result.val_rmse > 0,           "val_rmse must be positive."


def test_population_trainer_feature_cols(user_feature_matrix, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.population.trainer.MODELS_DIR", tmp_path)

    model   = get_model("random_forest", n_estimators=20)
    trainer = PopulationTrainer(model, dataset="nature_paper", horizon="2h")
    result  = trainer.run([user_feature_matrix], save=False, log_mlflow=False)
    assert len(result.feature_cols) > 0, "feature_cols must not be empty."
    for col in result.feature_cols:
        assert not col.startswith("target_"), \
            f"Target column '{col}' must not appear in feature_cols."


def test_population_trainer_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.population.trainer.MODELS_DIR", tmp_path)

    model   = get_model("lightgbm", n_estimators=10)
    trainer = PopulationTrainer(model, dataset="nature_paper", horizon="2h")
    with pytest.raises(ValueError, match="empty"):
        trainer.run([], save=False, log_mlflow=False)


# ══════════════════════════════════════════════════════════════════════════════
# Individual trainer (synthetic data, no file I/O)
# ══════════════════════════════════════════════════════════════════════════════

def test_individual_trainer_runs(user_feature_matrix, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.individual.trainer.MODELS_DIR", tmp_path)

    model   = get_model("lightgbm", n_estimators=50)
    trainer = IndividualTrainer(model, "nature_paper", "003", horizon="2h")
    result  = trainer.run(user_feature_matrix, save=False, log_mlflow=False)
    assert np.isfinite(result.val_rmse)
    assert np.isfinite(result.test_rmse)


def test_individual_trainer_too_few_rows_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.individual.trainer.MODELS_DIR", tmp_path)

    cols = {f"target_2h_step{s:02d}": [1.0]*10 for s in range(1, 9)}
    cols.update({f"target_3h_step{s:02d}": [1.0]*10 for s in range(1, 13)})
    cols["a"] = [0.0] * 10
    tiny  = pd.DataFrame(cols, index=pd.date_range("2020-01-01", periods=10, tz="UTC"))
    model   = get_model("lightgbm", n_estimators=10)
    trainer = IndividualTrainer(model, "nature_paper", "003", horizon="2h")
    with pytest.raises(ValueError, match="minimum"):
        trainer.run(tiny, save=False, log_mlflow=False)


# ══════════════════════════════════════════════════════════════════════════════
# Selector (synthetic, no file I/O)
# ══════════════════════════════════════════════════════════════════════════════

def test_selector_picks_a_winner(user_feature_matrix, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.population.trainer.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.selector.MODELS_DIR", tmp_path)

    # Use only RF (fast) for the test — LightGBM + XGBoost would take too long
    selector = ModelSelector("nature_paper", horizon="2h",
                             model_names=["random_forest"])
    selector.run(
        [user_feature_matrix, user_feature_matrix],
        user_ids=["003", "004"],
        save=False,
        log_mlflow=False,
    )
    assert selector.best_name == "random_forest"
    assert selector.best_result is not None
    assert np.isfinite(selector.best_result.val_rmse)


def test_selector_summary_table(user_feature_matrix, tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.population.trainer.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.selector.MODELS_DIR", tmp_path)

    selector = ModelSelector("nature_paper", "2h", model_names=["random_forest"])
    selector.run([user_feature_matrix], save=False, log_mlflow=False)
    tbl = selector.summary_table()
    assert "val_rmse" in tbl.columns
    assert len(tbl) >= 1
