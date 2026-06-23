"""
Phase 8–10 — glucose model zoo + TierTrainer (artifacts, versioning, selection).

Trains on a small synthetic two-user feature table so the test is fast and
deterministic, and checks the full artifact bundle, registry, NaN policy,
feature importance, and the Clarke-based selector + leaderboard.

Run:
    conda activate glucosenseai
    pytest tests/check/test_tier_models.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.step4_features import build_features_table
from src.models.glucose_models import GLUCOSE_MODELS, get_glucose_model
from src.models.tier_trainer import TierTrainer


def _imputed_user(uid, days=14, seed=0):
    """A clean imputed-style 10-min frame whose glucose depends on time-of-day."""
    rng = np.random.default_rng(seed)
    n = days * 144
    idx = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    hour = idx.hour + idx.minute / 60.0
    glucose = 120 + 35 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 4, n)
    df = pd.DataFrame({
        "glucose_mg_dl": glucose,
        "hr": 70 + 8 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 2, n),
        "total_carb": np.where(rng.random(n) < 0.02, rng.uniform(20, 60, n), 0.0),
        "calorie": 0.0, "protein": 0.0, "total_fat": 0.0, "dietary_fiber": 0.0,
        "sugar": 0.0, "age": 45.0, "bmi": 27.0, "hba1c": 5.6, "gender": "F",
    }, index=idx)
    df["uid"] = uid
    df["dataset"] = "cgmacros"
    df["participant_id"] = uid.split("-", 1)[1]
    return df


def _feature_table(mode, days=14):
    raw = pd.concat([_imputed_user("cg-101", days, seed=1),
                     _imputed_user("cg-102", days, seed=2)])
    return build_features_table(raw, mode=mode)


# ── Zoo ───────────────────────────────────────────────────────────────────────

def test_zoo_has_expected_models_including_catboost():
    for name in ("lightgbm", "xgboost", "histgbr", "extratrees", "ridge", "mlp"):
        assert name in GLUCOSE_MODELS
    assert "catboost" in GLUCOSE_MODELS   # installed in this env


def test_nan_intolerant_flag():
    assert get_glucose_model("lightgbm").handles_nan is True
    assert get_glucose_model("ridge").handles_nan is False


# ── Training: artifacts, versioning, NaN policy ───────────────────────────────

def test_train_cgm_active_lightgbm_saves_full_bundle(tmp_path):
    table = _feature_table("cgm_active", days=12)   # while_on_cgm needs 6+2+2
    trainer = TierTrainer("while_on_cgm", horizon_min=30,
                          models_dir=tmp_path / "models",
                          reports_dir=tmp_path / "reports", version="vTEST")
    res = trainer.train_model(table, "lightgbm", scope="population/test")

    assert np.isfinite(res.val_rmse) and np.isfinite(res.test_rmse)
    assert 0.0 <= res.clarke_a <= 100.0
    d = res.artefact_dir
    for f in ("model.pkl", "feature_cols.json", "metrics.json", "config.json", "importance.csv"):
        assert (d / f).exists(), f"missing artifact {f}"
    assert (d / "figures" / "test_clarke_grid.png").exists()
    # versioned path
    assert "vTEST" in str(d)


def test_registry_updated(tmp_path):
    table = _feature_table("cgm_active", days=12)
    trainer = TierTrainer("while_on_cgm", 30, models_dir=tmp_path / "models",
                          reports_dir=tmp_path / "reports", version="vTEST")
    trainer.train_model(table, "lightgbm", scope="population/test")
    reg = json.loads((tmp_path / "models" / "registry.json").read_text())
    assert "while_on_cgm" in reg["tiers"]
    assert "30min" in reg["tiers"]["while_on_cgm"]["population/test"]


def test_ridge_runs_through_imputer_no_nan_error(tmp_path):
    # post_cgm leaves structural NaN; Ridge is NaN-intolerant → trainer must impute.
    table = _feature_table("post_cgm", days=14)
    # inject a structurally-absent (all-NaN) column to mimic a missing sensor
    table["eda_roll_mean_30m"] = np.nan
    trainer = TierTrainer("without_cgm", 60, models_dir=tmp_path / "models",
                          reports_dir=tmp_path / "reports", version="vTEST")
    res = trainer.train_model(table, "ridge", scope="population/test")
    assert np.isfinite(res.test_rmse)


# ── Selector + leaderboard ────────────────────────────────────────────────────

def test_select_best_writes_leaderboard(tmp_path):
    table = _feature_table("cgm_active", days=12)
    trainer = TierTrainer("while_on_cgm", 30, models_dir=tmp_path / "models",
                          reports_dir=tmp_path / "reports", version="vTEST")
    winner, results = trainer.select_best(table, ["lightgbm", "ridge"], scope="population/test")
    assert winner is not None
    assert len(results) == 2
    lb = tmp_path / "reports" / "comparison" / "while_on_cgm" / "population/test" / "30min" / "leaderboard.csv"
    assert lb.exists()
    df = pd.read_csv(lb)
    assert df["winner"].sum() == 1
