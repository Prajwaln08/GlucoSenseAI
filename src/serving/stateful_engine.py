"""
Phase 6 — Stateful Real-Time Inference Engine.

Manages mode switching between CGM-active and post-CGM inference for a single
(dataset, horizon) slot.

Modes
-----
CGM_ACTIVE  ("cgm_active")
    CGM sensor is present. The population model trained on CGM + watch + food
    features predicts *glucose deltas*. The engine adds the delta to
    current_glucose to produce absolute predictions.

POST_CGM    ("post_cgm")
    CGM sensor has been removed. The virtual model selected by Phase 5
    predicts *absolute glucose* directly from watch / food / time features —
    no CGM signal required.

Lifecycle example
-----------------
    engine = StatefulInferenceEngine(dataset="cgmacros", horizon="2h")

    # During 14-day CGM calibration:
    result = engine.predict(feature_df, current_glucose=112.5)

    # When the sensor is removed:
    engine.transition_to_post_cgm(
        selection_dir=Path("reports/experiment_matrix/cgmacros")
    )

    # Ongoing virtual prediction:
    result = engine.predict(virtual_feature_df)
    for step in result.predictions:
        print(step.minutes_ahead, step.glucose_mg_dl)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import HORIZON_2H_STEPS, HORIZON_3H_STEPS
from src.serving.virtual_loader import (
    LoadedVirtualModel,
    load_virtual_model_from_selection,
)
from src.utils import get_logger

log = get_logger(__name__)

_HORIZON_STEPS: dict[str, int] = {"2h": HORIZON_2H_STEPS, "3h": HORIZON_3H_STEPS}


# ── Value types ───────────────────────────────────────────────────────────────

class InferenceMode(str, Enum):
    CGM_ACTIVE = "cgm_active"
    POST_CGM   = "post_cgm"


@dataclass
class StepPrediction:
    step:          int
    minutes_ahead: int
    glucose_mg_dl: float   # absolute mg/dL


@dataclass
class PredictionResult:
    mode:            InferenceMode
    horizon:         str
    dataset:         str
    model_key:       str
    current_glucose: Optional[float]    # None in POST_CGM (no CGM signal)
    predictions:     list[StepPrediction] = field(default_factory=list)

    @property
    def horizon_glucose(self) -> Optional[float]:
        """Absolute glucose at the end of the prediction horizon."""
        return self.predictions[-1].glucose_mg_dl if self.predictions else None


# ── Engine ────────────────────────────────────────────────────────────────────

class StatefulInferenceEngine:
    """
    Mode-switching glucose inference engine.

    Args:
        dataset:       Dataset label ("cgmacros" or "nature_paper").
        horizon:       Prediction horizon ("2h" or "3h").
        selection_dir: Path to directory containing selected_models.json.
                       Required when starting in POST_CGM mode or calling
                       transition_to_post_cgm().
        mode:          Starting InferenceMode (default: CGM_ACTIVE).
        artefact_base: Override for the virtual artefact base directory.
                       Default: MODELS_DIR/virtual.  Primarily used in tests.
    """

    def __init__(
        self,
        dataset:       str,
        horizon:       str,
        selection_dir: Optional[Path] = None,
        mode:          InferenceMode  = InferenceMode.CGM_ACTIVE,
        artefact_base: Optional[Path] = None,
    ):
        if horizon not in _HORIZON_STEPS:
            raise ValueError(
                f"horizon must be one of {list(_HORIZON_STEPS)}, got {horizon!r}"
            )

        self.dataset        = dataset
        self.horizon        = horizon
        self._mode          = mode
        self._selection_dir = Path(selection_dir) if selection_dir else None
        self._artefact_base = Path(artefact_base) if artefact_base else None
        self._virtual_model: Optional[LoadedVirtualModel] = None
        self._cgm_model     = None   # lazy-loaded from model_loader

        if mode == InferenceMode.POST_CGM:
            if selection_dir is None:
                raise ValueError(
                    "selection_dir is required when starting in POST_CGM mode."
                )
            self._load_virtual()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def mode(self) -> InferenceMode:
        return self._mode

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        feature_df:      pd.DataFrame,
        current_glucose: Optional[float] = None,
    ) -> PredictionResult:
        """
        Generate multi-step glucose predictions for the current mode.

        CGM_ACTIVE
            feature_df must be the output of
            build_feature_matrix(df, mode="cgm_active").
            current_glucose is required (used to convert delta → absolute).

        POST_CGM
            feature_df must be the output of
            build_virtual_feature_matrix(df) or
            build_feature_matrix(df, mode="post_cgm").
            current_glucose is ignored (model predicts absolute glucose directly).

        Args:
            feature_df:      DataFrame with at least one row.
                             Only the last row is used for prediction.
            current_glucose: Current CGM reading in mg/dL (CGM_ACTIVE only).

        Returns:
            PredictionResult with n_steps StepPredictions.
        """
        if self._mode == InferenceMode.CGM_ACTIVE:
            return self._predict_cgm_active(feature_df, current_glucose)
        return self._predict_post_cgm(feature_df)

    def transition_to_post_cgm(
        self,
        selection_dir: Path,
        artefact_base: Optional[Path] = None,
    ) -> None:
        """
        Switch from CGM_ACTIVE to POST_CGM mode.

        Loads the virtual model selected by Phase 5, then releases the
        CGM-active model reference so it can be garbage-collected.

        Args:
            selection_dir: Directory containing selected_models.json.
            artefact_base: Override for the virtual artefact base directory.

        Raises:
            FileNotFoundError: if selected_models.json is missing.
            KeyError: if the horizon is not in the selection file.
        """
        if self._mode == InferenceMode.POST_CGM:
            log.warning("Engine is already in POST_CGM mode — transition is a no-op.")
            return

        self._selection_dir = Path(selection_dir)
        if artefact_base is not None:
            self._artefact_base = Path(artefact_base)

        self._load_virtual()
        self._cgm_model = None   # release CGM model reference
        self._mode      = InferenceMode.POST_CGM

        log.info(
            f"Mode switch → POST_CGM | "
            f"model: {self._virtual_model.model_key} | "
            f"dataset: {self.dataset} | horizon: {self.horizon}"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_virtual(self) -> None:
        assert self._selection_dir is not None
        self._virtual_model = load_virtual_model_from_selection(
            selection_dir = self._selection_dir,
            horizon       = self.horizon,
            artefact_base = self._artefact_base,
        )

    def _predict_cgm_active(
        self,
        feature_df:      pd.DataFrame,
        current_glucose: Optional[float],
    ) -> PredictionResult:
        if current_glucose is None:
            raise ValueError("current_glucose is required in CGM_ACTIVE mode.")

        loaded = self._get_cgm_model()
        row    = self._last_row(feature_df)
        X      = _align_and_scale(row, loaded.feature_cols, loaded.scaler)
        deltas = np.asarray(loaded.model.predict(X)).flatten()

        n_steps = _HORIZON_STEPS[self.horizon]
        preds = [
            StepPrediction(
                step          = i + 1,
                minutes_ahead = (i + 1) * 15,
                glucose_mg_dl = round(float(current_glucose + deltas[i]), 1),
            )
            for i in range(min(n_steps, len(deltas)))
        ]
        return PredictionResult(
            mode            = InferenceMode.CGM_ACTIVE,
            horizon         = self.horizon,
            dataset         = self.dataset,
            model_key       = loaded.model_type,
            current_glucose = round(current_glucose, 1),
            predictions     = preds,
        )

    def _predict_post_cgm(self, feature_df: pd.DataFrame) -> PredictionResult:
        assert self._virtual_model is not None, "Virtual model not loaded."
        vm  = self._virtual_model
        row = self._last_row(feature_df)
        X   = _align_and_scale(row, vm.feature_cols, vm.scaler)
        abs_vals = np.asarray(vm.model.predict(X)).flatten()

        n_steps = _HORIZON_STEPS[self.horizon]
        preds = [
            StepPrediction(
                step          = i + 1,
                minutes_ahead = (i + 1) * 15,
                glucose_mg_dl = round(float(abs_vals[i]), 1),
            )
            for i in range(min(n_steps, len(abs_vals)))
        ]
        return PredictionResult(
            mode            = InferenceMode.POST_CGM,
            horizon         = self.horizon,
            dataset         = self.dataset,
            model_key       = vm.model_key,
            current_glucose = None,
            predictions     = preds,
        )

    def _get_cgm_model(self):
        """Lazy-load the CGM-active population model from model_loader."""
        if self._cgm_model is None:
            from src.serving.model_loader import load_model
            try:
                self._cgm_model = load_model(self.dataset, self.horizon)
            except (FileNotFoundError, KeyError) as exc:
                raise RuntimeError(
                    f"CGM-active model for {self.dataset}/{self.horizon} not found. "
                    f"Run 'make train-{self.dataset.replace('_', '')}' first. "
                    f"Underlying error: {exc}"
                ) from exc
        return self._cgm_model

    @staticmethod
    def _last_row(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[[-1]]


# ── Module-level helper (also used in tests) ──────────────────────────────────

def _align_and_scale(
    row:          pd.DataFrame,
    feature_cols: list[str],
    scaler,
) -> pd.DataFrame:
    """
    Align a single-row DataFrame to the model's expected feature_cols,
    zero-filling any column not present in row, then apply scaler.transform().

    Returns a DataFrame with the same column order as at training time.
    """
    X = pd.DataFrame(index=row.index, columns=feature_cols, dtype=float)
    for col in feature_cols:
        X[col] = row[col].values if col in row.columns else 0.0
    return pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols,
        index=row.index,
    )
