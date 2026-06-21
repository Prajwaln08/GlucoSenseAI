"""
Phase 3 unit tests — Virtual Glucose (Stage-B) Training Pipeline.

Covers:
- Virtual model class attributes (requires_cgm=False, predicts_absolute_or_delta="absolute")
- MODEL_REGISTRY contains all four virtual model keys
- build_virtual_feature_matrix: no CGM features in output, absolute targets present
- VirtualTrainer: raises for CGM-active models (requires_cgm=True)
- VirtualTrainer: uses day_split (produces absolute targets, not delta)
- VirtualTrainer: skips users with insufficient days; raises if ALL users skip
- VirtualTrainer: artefact dir is under models/virtual/
- VirtualTrainer: config.json records requires_cgm=False and mode=post_cgm
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.models.zoo import MODEL_REGISTRY, VIRTUAL_MODEL_KEYS, get_model
from src.models.zoo.virtual_lgbm import VirtualLGBM
from src.models.zoo.virtual_xgb import VirtualXGB
from src.models.zoo.virtual_rnn import VirtualGRU, VirtualLSTM
from src.models.zoo.lgbm_model import LightGBMModel
from src.models.virtual.trainer import VirtualTrainer
from src.features.pipeline import build_virtual_feature_matrix, get_target_cols
from src.features.feature_groups import CGM_FEATURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n_days: int = 14, with_glucose: bool = True) -> pd.DataFrame:
    """
    Synthetic preprocessed DataFrame spanning n_days × 96 15-min intervals.
    Includes all columns the feature pipeline expects.
    """
    n = n_days * 96
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    data = {
        "hr":                np.random.uniform(60, 100, n).astype(float),
        "calories_burned":   np.zeros(n),
        "mets":              np.zeros(n),
        "meal_type_encoded": np.zeros(n),
        "amount_consumed_pct": np.zeros(n),
        "total_carb":        np.zeros(n),
        "calorie":           np.zeros(n),
        "protein":           np.zeros(n),
        "total_fat":         np.zeros(n),
        "dietary_fiber":     np.zeros(n),
        "sugar":             np.zeros(n),
        "gi_proxy":          np.zeros(n),
    }
    if with_glucose:
        data["glucose_mg_dl"] = np.random.uniform(80, 180, n).astype(float)
    return pd.DataFrame(data, index=idx)


def _make_virtual_fm(n_days: int = 14) -> pd.DataFrame:
    """Pre-built virtual feature matrix (post_cgm mode)."""
    raw = _make_df(n_days=n_days)
    return build_virtual_feature_matrix(raw)


# ── Virtual model class attributes ───────────────────────────────────────────

@pytest.mark.parametrize("cls", [VirtualLGBM, VirtualXGB, VirtualGRU, VirtualLSTM])
def test_virtual_model_requires_cgm_false(cls):
    assert cls.requires_cgm is False


@pytest.mark.parametrize("cls", [VirtualLGBM, VirtualXGB, VirtualGRU, VirtualLSTM])
def test_virtual_model_predicts_absolute(cls):
    assert cls.predicts_absolute_or_delta == "absolute"


@pytest.mark.parametrize("cls", [VirtualLGBM, VirtualXGB, VirtualGRU, VirtualLSTM])
def test_virtual_model_supported_feature_groups_no_cgm(cls):
    assert "cgm" not in (cls.supported_feature_groups or [])


# ── MODEL_REGISTRY contains virtual keys ─────────────────────────────────────

def test_virtual_keys_in_registry():
    for key in VIRTUAL_MODEL_KEYS:
        assert key in MODEL_REGISTRY, f"{key!r} missing from MODEL_REGISTRY"


def test_get_model_returns_virtual_instance():
    model = get_model("virtual_lgbm")
    assert isinstance(model, VirtualLGBM)
    assert model.requires_cgm is False


# ── build_virtual_feature_matrix ─────────────────────────────────────────────

def test_virtual_fm_no_cgm_features():
    fm     = _make_virtual_fm()
    cols   = set(fm.columns)
    leaked = cols & CGM_FEATURES
    assert leaked == set(), (
        f"CGM features leaked into virtual feature matrix: {leaked}"
    )


def test_virtual_fm_glucose_col_absent():
    fm = _make_virtual_fm()
    assert "glucose_mg_dl" not in fm.columns


def test_virtual_fm_absolute_targets_present():
    fm = _make_virtual_fm()
    for col in get_target_cols("2h", mode="post_cgm"):
        assert col in fm.columns, f"Absolute target {col!r} missing"


def test_virtual_fm_no_delta_targets():
    fm = _make_virtual_fm()
    for col in get_target_cols("2h", mode="cgm_active"):
        assert col not in fm.columns, f"Delta target {col!r} leaked into virtual FM"


# ── VirtualTrainer: rejects CGM-active models ────────────────────────────────

def test_virtual_trainer_rejects_cgm_model():
    cgm_model = LightGBMModel()
    with pytest.raises(ValueError, match="requires_cgm=False"):
        VirtualTrainer(cgm_model, dataset="cgmacros", horizon="2h")


# ── VirtualTrainer: fits and produces artefacts ───────────────────────────────

def test_virtual_trainer_fits_lgbm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.models.virtual.trainer.MODELS_DIR", tmp_path
    )
    dfs = [_make_virtual_fm(n_days=14) for _ in range(3)]
    ids = ["003", "004", "005"]

    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    result  = trainer.run(dfs, user_ids=ids, save=True, log_mlflow=False)

    assert result.val_rmse < 1000, "val_RMSE sanity check failed"
    assert result.artefact_dir.exists()
    assert (result.artefact_dir / "model.pkl").exists()
    assert (result.artefact_dir / "scaler.pkl").exists()
    assert (result.artefact_dir / "feature_cols.json").exists()
    assert (result.artefact_dir / "config.json").exists()
    assert (result.artefact_dir / "metrics.json").exists()


def test_virtual_trainer_artefact_path_under_virtual(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.models.virtual.trainer.MODELS_DIR", tmp_path
    )
    dfs  = [_make_virtual_fm(n_days=14)]
    ids  = ["003"]

    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    result  = trainer.run(dfs, user_ids=ids, save=True, log_mlflow=False)

    assert "virtual" in str(result.artefact_dir), (
        f"Expected 'virtual' in artefact path, got {result.artefact_dir}"
    )


def test_virtual_trainer_config_json_records_post_cgm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.models.virtual.trainer.MODELS_DIR", tmp_path
    )
    dfs = [_make_virtual_fm(n_days=14)]
    ids = ["003"]

    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    result  = trainer.run(dfs, user_ids=ids, save=True, log_mlflow=False)

    config = json.loads((result.artefact_dir / "config.json").read_text())
    assert config["mode"]         == "post_cgm"
    assert config["requires_cgm"] is False


def test_virtual_trainer_feature_cols_json_no_cgm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.models.virtual.trainer.MODELS_DIR", tmp_path
    )
    dfs = [_make_virtual_fm(n_days=14)]
    ids = ["003"]

    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    result  = trainer.run(dfs, user_ids=ids, save=True, log_mlflow=False)

    feature_cols = json.loads((result.artefact_dir / "feature_cols.json").read_text())
    leaked = set(feature_cols) & CGM_FEATURES
    assert leaked == set(), (
        f"CGM features in saved feature_cols.json: {leaked}"
    )


# ── VirtualTrainer: day-split behaviour ──────────────────────────────────────

def test_virtual_trainer_skips_short_user(tmp_path, monkeypatch):
    """Users with <14 days are skipped; run completes on remaining users."""
    monkeypatch.setattr(
        "src.models.virtual.trainer.MODELS_DIR", tmp_path
    )
    short_fm = _make_virtual_fm(n_days=10)   # only 10 days — below required 14
    good_fm  = _make_virtual_fm(n_days=14)

    dfs = [short_fm, good_fm, good_fm]
    ids = ["short_user", "good_001", "good_002"]

    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    result  = trainer.run(dfs, user_ids=ids, save=False, log_mlflow=False)

    # Short user must not appear in per-user metrics (was skipped)
    assert "short_user" not in result.per_user_test_rmse, (
        "short_user should have been skipped due to insufficient days"
    )
    # Remaining two users trained successfully
    assert result.val_rmse < 1000


def test_virtual_trainer_raises_if_all_users_too_short(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.models.virtual.trainer.MODELS_DIR", tmp_path
    )
    short_dfs = [_make_virtual_fm(n_days=5) for _ in range(3)]
    ids = ["a", "b", "c"]

    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    with pytest.raises(RuntimeError, match="all users were skipped"):
        trainer.run(short_dfs, user_ids=ids, save=False, log_mlflow=False)


def test_virtual_trainer_empty_input_raises():
    model   = VirtualLGBM()
    trainer = VirtualTrainer(model, dataset="cgmacros", horizon="2h")
    with pytest.raises(ValueError, match="empty"):
        trainer.run([], save=False, log_mlflow=False)
