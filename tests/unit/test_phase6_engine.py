"""
Phase 6 unit tests — Stateful Real-Time Inference Engine.

Covers:
- InferenceMode: enum values, str identity
- StepPrediction: correct fields
- PredictionResult: horizon_glucose property (normal and empty)
- StatefulInferenceEngine: default mode is CGM_ACTIVE
- StatefulInferenceEngine: POST_CGM at init requires selection_dir
- StatefulInferenceEngine: invalid horizon raises ValueError
- StatefulInferenceEngine.transition_to_post_cgm: switches mode
- StatefulInferenceEngine.transition_to_post_cgm: noop if already POST_CGM
- StatefulInferenceEngine.transition_to_post_cgm: releases CGM model ref
- StatefulInferenceEngine.predict (post_cgm): correct step count
- StatefulInferenceEngine.predict (post_cgm): result mode is POST_CGM
- StatefulInferenceEngine.predict (post_cgm): current_glucose is None
- StatefulInferenceEngine.predict (post_cgm): glucose values are plausible
- StatefulInferenceEngine.predict (cgm_active): raises without current_glucose
- StatefulInferenceEngine.predict (cgm_active): correct result shape (mocked)
- StatefulInferenceEngine.predict (cgm_active): result mode is CGM_ACTIVE (mocked)
- _align_and_scale: zero-fills missing columns
- _align_and_scale: preserves column order and applies scaler
- virtual_loader.load_virtual_model: raises FileNotFoundError for missing dir
- virtual_loader.load_virtual_model: raises FileNotFoundError for missing pkl
- virtual_loader.load_virtual_model: cache returns same object on second call
- virtual_loader.load_virtual_model_from_selection: raises for missing file
- virtual_loader.load_virtual_model_from_selection: raises for missing horizon
- virtual_loader.load_virtual_model_from_selection: loads correct model
"""

import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # prevent OpenMP segfault on macOS

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.config import HORIZON_2H_STEPS
from src.serving.stateful_engine import (
    InferenceMode,
    PredictionResult,
    StatefulInferenceEngine,
    StepPrediction,
    _align_and_scale,
)
from src.serving.virtual_loader import (
    clear_cache,
    load_virtual_model,
    load_virtual_model_from_selection,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_df(n_days: int = 14) -> pd.DataFrame:
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
    return build_virtual_feature_matrix(_make_raw_df(n_days))


@pytest.fixture(scope="module")
def trained_virtual_artefact(tmp_path_factory):
    """
    Train a tiny virtual LGBM model, save artefacts to a temp dir.

    Returns:
        (sel_dir, artefact_base) — sel_dir contains selected_models.json;
        artefact_base is the root under which artefacts are stored.
    """
    from src.models.base_model import BaseModel
    from src.models.zoo.virtual_lgbm import VirtualLGBM
    from src.models.virtual.trainer import VirtualTrainer
    from src.experiments.selector import SelectedModel, VirtualModelSelector

    tmp = tmp_path_factory.mktemp("virtual_artefact")

    # Build tiny feature matrices for 3 users
    dfs = [_make_virtual_fm(14) for _ in range(3)]
    user_ids = ["003", "004", "005"]

    trainer = VirtualTrainer(VirtualLGBM(), dataset="cgmacros", horizon="2h")
    result  = trainer.run(dfs, user_ids, save=False, log_mlflow=False)

    # Save artefacts manually to tmp dir
    artefact_dir = tmp / "cgmacros" / "2h" / "virtual_lgbm"
    artefact_dir.mkdir(parents=True)
    result.model.save(artefact_dir / "model.pkl")
    BaseModel._save_pickle(result.scaler, artefact_dir / "scaler.pkl")
    (artefact_dir / "feature_cols.json").write_text(
        json.dumps(result.feature_cols)
    )

    # Create selected_models.json in the same tmp dir
    sel = {
        "2h": SelectedModel(
            model_key     = "virtual_lgbm",
            horizon       = "2h",
            input_window  = None,
            dataset       = "cgmacros",
            config_key    = "virtual_lgbm__2h__winna",
            val_rmse      = result.val_rmse,
            test_rmse     = result.test_rmse,
            clarke_a_pct  = result.clarke_a_pct,
            passed_clarke = True,
            passed_gap    = True,
        )
    }
    VirtualModelSelector.save_selection(sel, tmp, dataset="cgmacros")

    clear_cache()   # ensure each test starts cold
    return tmp, tmp   # sel_dir == artefact_base == tmp


# ── InferenceMode ─────────────────────────────────────────────────────────────

def test_inference_mode_cgm_active_value():
    assert InferenceMode.CGM_ACTIVE == "cgm_active"


def test_inference_mode_post_cgm_value():
    assert InferenceMode.POST_CGM == "post_cgm"


def test_inference_mode_is_str():
    assert isinstance(InferenceMode.CGM_ACTIVE, str)


# ── StepPrediction ─────────────────────────────────────────────────────────────

def test_step_prediction_fields():
    sp = StepPrediction(step=1, minutes_ahead=15, glucose_mg_dl=112.5)
    assert sp.step          == 1
    assert sp.minutes_ahead == 15
    assert sp.glucose_mg_dl == 112.5


# ── PredictionResult ──────────────────────────────────────────────────────────

def test_prediction_result_horizon_glucose():
    preds = [
        StepPrediction(step=1, minutes_ahead=15, glucose_mg_dl=110.0),
        StepPrediction(step=2, minutes_ahead=30, glucose_mg_dl=115.0),
    ]
    r = PredictionResult(
        mode=InferenceMode.POST_CGM, horizon="2h", dataset="cgmacros",
        model_key="virtual_lgbm", current_glucose=None, predictions=preds,
    )
    assert r.horizon_glucose == 115.0


def test_prediction_result_horizon_glucose_empty():
    r = PredictionResult(
        mode=InferenceMode.POST_CGM, horizon="2h", dataset="cgmacros",
        model_key="virtual_lgbm", current_glucose=None,
    )
    assert r.horizon_glucose is None


# ── StatefulInferenceEngine construction ──────────────────────────────────────

def test_engine_default_mode_is_cgm_active():
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    assert engine.mode == InferenceMode.CGM_ACTIVE


def test_engine_invalid_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        StatefulInferenceEngine(dataset="cgmacros", horizon="5h")


def test_engine_post_cgm_without_selection_dir_raises():
    with pytest.raises(ValueError, match="selection_dir"):
        StatefulInferenceEngine(
            dataset="cgmacros", horizon="2h",
            mode=InferenceMode.POST_CGM,
            # selection_dir intentionally omitted
        )


def test_engine_post_cgm_at_init(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(
        dataset="cgmacros", horizon="2h",
        selection_dir=sel_dir,
        mode=InferenceMode.POST_CGM,
        artefact_base=art_base,
    )
    assert engine.mode == InferenceMode.POST_CGM
    assert engine._virtual_model is not None


# ── transition_to_post_cgm ────────────────────────────────────────────────────

def test_transition_switches_mode(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    assert engine.mode == InferenceMode.CGM_ACTIVE
    engine.transition_to_post_cgm(sel_dir, artefact_base=art_base)
    assert engine.mode == InferenceMode.POST_CGM


def test_transition_loads_virtual_model(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    engine.transition_to_post_cgm(sel_dir, artefact_base=art_base)
    assert engine._virtual_model is not None
    assert engine._virtual_model.model_key == "virtual_lgbm"


def test_transition_releases_cgm_model(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    engine._cgm_model = object()   # pretend CGM model was loaded
    engine.transition_to_post_cgm(sel_dir, artefact_base=art_base)
    assert engine._cgm_model is None


def test_transition_noop_if_already_post_cgm(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(
        dataset="cgmacros", horizon="2h",
        selection_dir=sel_dir,
        mode=InferenceMode.POST_CGM,
        artefact_base=art_base,
    )
    vm_before = engine._virtual_model
    engine.transition_to_post_cgm(sel_dir, artefact_base=art_base)
    assert engine._virtual_model is vm_before   # same object — no reload


# ── predict (POST_CGM) ────────────────────────────────────────────────────────

def test_predict_post_cgm_step_count(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(
        dataset="cgmacros", horizon="2h",
        selection_dir=sel_dir,
        mode=InferenceMode.POST_CGM,
        artefact_base=art_base,
    )
    fm     = _make_virtual_fm(14)
    result = engine.predict(fm)
    assert len(result.predictions) == HORIZON_2H_STEPS


def test_predict_post_cgm_result_mode(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(
        dataset="cgmacros", horizon="2h",
        selection_dir=sel_dir,
        mode=InferenceMode.POST_CGM,
        artefact_base=art_base,
    )
    result = engine.predict(_make_virtual_fm(14))
    assert result.mode == InferenceMode.POST_CGM


def test_predict_post_cgm_current_glucose_is_none(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(
        dataset="cgmacros", horizon="2h",
        selection_dir=sel_dir,
        mode=InferenceMode.POST_CGM,
        artefact_base=art_base,
    )
    result = engine.predict(_make_virtual_fm(14))
    assert result.current_glucose is None


def test_predict_post_cgm_glucose_values_plausible(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    engine = StatefulInferenceEngine(
        dataset="cgmacros", horizon="2h",
        selection_dir=sel_dir,
        mode=InferenceMode.POST_CGM,
        artefact_base=art_base,
    )
    result = engine.predict(_make_virtual_fm(14))
    for step in result.predictions:
        assert 30.0 <= step.glucose_mg_dl <= 500.0


# ── predict (CGM_ACTIVE, mocked) ──────────────────────────────────────────────

class _FakeLoadedModel:
    """Minimal stub that satisfies StatefulInferenceEngine._predict_cgm_active."""
    feature_cols = ["hr", "meal_feature"]
    model_type   = "lightgbm"

    class _Model:
        def predict(self, X):
            return np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])

    class _Scaler:
        def transform(self, X):
            return X.values

    model  = _Model()
    scaler = _Scaler()


def test_predict_cgm_active_raises_without_current_glucose():
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    with pytest.raises(ValueError, match="current_glucose"):
        engine.predict(pd.DataFrame({"hr": [70.0]}), current_glucose=None)


def test_predict_cgm_active_result_shape(monkeypatch):
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    monkeypatch.setattr(engine, "_get_cgm_model", lambda: _FakeLoadedModel())

    fm     = pd.DataFrame({"hr": [70.0], "meal_feature": [0.0]})
    result = engine.predict(fm, current_glucose=100.0)
    assert len(result.predictions) == HORIZON_2H_STEPS


def test_predict_cgm_active_result_mode(monkeypatch):
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    monkeypatch.setattr(engine, "_get_cgm_model", lambda: _FakeLoadedModel())

    fm     = pd.DataFrame({"hr": [70.0], "meal_feature": [0.0]})
    result = engine.predict(fm, current_glucose=100.0)
    assert result.mode == InferenceMode.CGM_ACTIVE


def test_predict_cgm_active_delta_added(monkeypatch):
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")
    monkeypatch.setattr(engine, "_get_cgm_model", lambda: _FakeLoadedModel())

    fm     = pd.DataFrame({"hr": [70.0], "meal_feature": [0.0]})
    result = engine.predict(fm, current_glucose=100.0)
    # First delta is 5.0 → should be 100.0 + 5.0 = 105.0
    assert result.predictions[0].glucose_mg_dl == 105.0


# ── _align_and_scale ──────────────────────────────────────────────────────────

def test_align_and_scale_zero_fills_missing():
    scaler = StandardScaler()
    # Fit on both features so the scaler knows about "b"
    data   = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    scaler.fit(data)

    # Row only has "a" — "b" should be zero-filled before scaling
    row    = pd.DataFrame({"a": [2.0]})
    result = _align_and_scale(row, feature_cols=["a", "b"], scaler=scaler)
    assert "b" in result.columns
    assert result.shape == (1, 2)
    # "b" was zero-filled (value = 0.0) which is below mean(b)=5, so scaled value < 0
    assert result["b"].iloc[0] < 0


def test_align_and_scale_column_order():
    scaler = StandardScaler()
    data   = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    scaler.fit(data)

    row    = pd.DataFrame({"y": [4.0], "x": [2.0]})  # intentionally reversed
    result = _align_and_scale(row, feature_cols=["x", "y"], scaler=scaler)
    assert list(result.columns) == ["x", "y"]


# ── virtual_loader ────────────────────────────────────────────────────────────

def test_load_virtual_model_raises_missing_dir(tmp_path):
    clear_cache()
    with pytest.raises(FileNotFoundError, match="not found"):
        load_virtual_model("virtual_lgbm", "2h", "cgmacros",
                           artefact_dir=tmp_path / "nonexistent")


def test_load_virtual_model_raises_missing_model_pkl(tmp_path):
    clear_cache()
    d = tmp_path / "virtual_lgbm"
    d.mkdir()
    # model.pkl is absent → should raise
    with pytest.raises(FileNotFoundError, match="model.pkl"):
        load_virtual_model("virtual_lgbm", "2h", "cgmacros", artefact_dir=d)


def test_load_virtual_model_cache_returns_same_object(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    art_dir = art_base / "cgmacros" / "2h" / "virtual_lgbm"
    vm1 = load_virtual_model("virtual_lgbm", "2h", "cgmacros", artefact_dir=art_dir)
    vm2 = load_virtual_model("virtual_lgbm", "2h", "cgmacros", artefact_dir=art_dir)
    assert vm1 is vm2


def test_load_virtual_model_from_selection_raises_missing_file(tmp_path):
    clear_cache()
    with pytest.raises(FileNotFoundError, match="selected_models.json"):
        load_virtual_model_from_selection(tmp_path, horizon="2h")


def test_load_virtual_model_from_selection_raises_missing_horizon(tmp_path):
    clear_cache()
    payload = {"by_horizon": {"3h": {"model_key": "virtual_lgbm", "dataset": "cgmacros"}}}
    (tmp_path / "selected_models.json").write_text(json.dumps(payload))
    with pytest.raises(KeyError, match="2h"):
        load_virtual_model_from_selection(tmp_path, horizon="2h")


def test_load_virtual_model_from_selection_loads_correct_model(trained_virtual_artefact):
    sel_dir, art_base = trained_virtual_artefact
    clear_cache()
    vm = load_virtual_model_from_selection(
        selection_dir = sel_dir,
        horizon       = "2h",
        artefact_base = art_base,
    )
    assert vm.model_key == "virtual_lgbm"
    assert vm.horizon   == "2h"
    assert vm.dataset   == "cgmacros"
    assert len(vm.feature_cols) > 0
