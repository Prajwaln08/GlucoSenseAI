"""
Abstract base class that every model in the zoo must implement.

Any model registered in MODEL_REGISTRY must subclass BaseModel and
implement all abstract methods. The training pipeline, evaluator,
selector, and hyperparameter tuner all work through this interface —
they know nothing about which specific model they are running.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Plugin interface for GlucoSense AI models."""

    # ── Identity ──────────────────────────────────────────────────────────────
    # Human-readable name used for logging, registry keys, and artifact dirs.
    # Subclasses must override this.
    name: str = "base"

    # ── Required metadata — every subclass must declare these ─────────────────
    #
    # requires_cgm
    #   True  → model uses glucose lags / rolling stats; only valid during
    #           the CGM-active period.
    #   False → model uses only watch/food/medicine/time features; valid
    #           post-CGM (Virtual Glucose models).
    requires_cgm: bool = True

    # predicts_absolute_or_delta
    #   "delta"    → model output is (future_glucose − current_glucose)
    #   "absolute" → model output is absolute glucose (mg/dL)
    predicts_absolute_or_delta: str = "delta"

    # input_window
    #   Number of 15-min timesteps of history the model consumes.
    #   None means the model does not use a sliding window (tree models).
    input_window: int | None = None

    # prediction_horizon
    #   Number of 15-min steps ahead the model predicts.
    #   None until set by the experiment matrix (Phase 4).
    prediction_horizon: int | None = None

    # supported_feature_groups
    #   Names of FeatureGroupDef groups this model accepts as input.
    #   None means the model is not yet registered in the typed group system.
    supported_feature_groups: list[str] | None = None

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X_train:  Feature matrix, shape (n_train, n_features).
            y_train:  Targets — Series for single-step, DataFrame (n_train, n_steps)
                      for multi-step horizons.
            X_val:    Validation features for early stopping / monitoring.
            y_val:    Validation targets (same shape convention as y_train).

        Returns:
            self (enables chaining).
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix, shape (n, n_features).

        Returns:
            Array of shape (n, n_steps) for multi-step models.
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Return the hyperparameter dict used by this model instance."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Serialise the trained model to *path*.

        Typically saves a .pkl file, but subclasses may choose any format
        (e.g. LightGBM native .txt, PyTorch .pt).  The file or directory
        at *path* must exist after this call.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        Deserialise a model from *path* and return a ready instance.

        Args:
            path: Path to the file written by save().

        Returns:
            Loaded, ready-to-predict model instance.
        """

    # ── Optional override ─────────────────────────────────────────────────────

    def get_search_space(self, trial) -> dict:
        """
        Return a hyperparameter dict sampled by an Optuna trial object.

        Override in each subclass to expose model-specific search bounds.
        The base implementation returns an empty dict (no tuning).

        Example (LightGBM subclass):
            return {
                "n_estimators":  trial.suggest_int("n_estimators", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            }
        """
        return {}

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _save_pickle(obj, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_pickle(path: Path):
        # Shim: scalers pickled under numpy 2.x reference numpy._core.{multiarray,numeric}
        # which don't exist as sub-modules in numpy 1.x.  Register aliases before unpickling.
        import sys
        import numpy.core.multiarray as _nma
        import numpy.core.numeric as _ncn
        sys.modules.setdefault("numpy._core.multiarray", _nma)
        sys.modules.setdefault("numpy._core.numeric", _ncn)
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_params()})"
