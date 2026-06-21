"""
Individual-level trainer for GlucoSense AI.

Trains one model on a single user's data.  Used for:
  - Personal model generation (production: Phase 2 / Phase 3)
  - Best-individual-model selection from each training pool

The interface mirrors PopulationTrainer.run() but accepts a single DataFrame
instead of a list.

Artefact layout:
    models/individual/<dataset>/<user_id>/<horizon>/<model_name>/
        model.pkl
        scaler.pkl
        feature_cols.json
        metrics.json
        config.json
        figures/
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR
from src.features.pipeline import get_X_y
from src.models.base_model import BaseModel
from src.models.evaluator import evaluate_and_plot
from src.models.population.trainer import TrainResult, _chronological_split, _try_mlflow_log
from src.utils import get_logger

log = get_logger(__name__)

# Minimum rows required to train a meaningful individual model
# 14 days × 96 intervals/day = 1,344 rows; we use a relaxed floor for robustness
MIN_ROWS_INDIVIDUAL = 200


class IndividualTrainer:
    """
    Train a personal model for one user.

    Args:
        model:    A ready-to-fit BaseModel instance.
        dataset:  Dataset label ("cgmacros" or "nature_paper").
        user_id:  User identifier string (zero-padded, e.g. "003").
        horizon:  "2h" or "3h".
        version:  Optional version tag (e.g. "v2_20260529"). Appended as a
                  subdirectory so multiple pipeline versions can coexist:
                  models/individual/<dataset>/<user>/<horizon>/<model>/<version>/
                  Omit (None) for legacy single-version layout.
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: str,
        user_id: str,
        horizon: str = "2h",
        version: str | None = None,
    ):
        self.model   = model
        self.dataset = dataset
        self.user_id = user_id
        self.horizon = horizon
        self.version = version

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        save: bool = True,
        log_mlflow: bool = True,
    ) -> TrainResult:
        """
        Train and evaluate an individual model.

        Args:
            df:          Feature-matrix DataFrame from build_feature_matrix(),
                         UTC DatetimeIndex, target columns present.
            save:        If True, persist artefacts to disk.
            log_mlflow:  If True, attempt to log to MLflow.

        Returns:
            TrainResult with trained model, scaler, metrics, and artefact path.
        """
        if len(df) < MIN_ROWS_INDIVIDUAL:
            raise ValueError(
                f"User {self.user_id}: only {len(df)} rows — minimum is "
                f"{MIN_ROWS_INDIVIDUAL} for a meaningful individual model."
            )

        log.info(
            f"IndividualTrainer: user={self.user_id} dataset={self.dataset} "
            f"horizon={self.horizon} model={self.model.name} rows={len(df)}"
        )

        # ── 1. Split ──────────────────────────────────────────────────────────
        split = _chronological_split(df)
        if split is None:
            raise RuntimeError(f"User {self.user_id}: DataFrame too small to split.")

        df_train, df_val, df_test = split["train"], split["val"], split["test"]
        log.info(
            f"User {self.user_id} split — train: {len(df_train)}, "
            f"val: {len(df_val)}, test: {len(df_test)}"
        )

        # ── 2. Build X / y ────────────────────────────────────────────────────
        X_train, y_train = get_X_y(df_train, horizon=self.horizon)
        X_val,   y_val   = get_X_y(df_val,   horizon=self.horizon)
        X_test,  y_test  = get_X_y(df_test,  horizon=self.horizon)

        # ── 2b. Drop zero-variance features (computed on train only) ─────────
        var_train     = X_train.var()
        zero_var_cols = var_train[var_train < 1e-8].index.tolist()
        if zero_var_cols:
            log.info(
                f"User {self.user_id}: dropping {len(zero_var_cols)} zero-variance features: "
                f"{zero_var_cols[:6]}{'…' if len(zero_var_cols) > 6 else ''}"
            )
            X_train = X_train.drop(columns=zero_var_cols)
            X_val   = X_val.drop(columns=zero_var_cols)
            X_test  = X_test.drop(columns=zero_var_cols)

        # ── 2c. Drop high-correlation features (train-only, threshold=0.97) ──
        # glucose_mg_dl is always kept — needed for delta→absolute conversion.
        corr_abs  = X_train.corr().abs()
        upper_tri = corr_abs.where(np.triu(np.ones(corr_abs.shape, dtype=bool), k=1))
        high_corr = [c for c in upper_tri.columns
                     if (upper_tri[c] > 0.97).any() and c != "glucose_mg_dl"]
        if high_corr:
            log.info(
                f"User {self.user_id}: dropping {len(high_corr)} high-correlation features: "
                f"{high_corr[:6]}{'…' if len(high_corr) > 6 else ''}"
            )
            X_train = X_train.drop(columns=high_corr)
            X_val   = X_val.drop(columns=high_corr)
            X_test  = X_test.drop(columns=high_corr)

        feature_cols = list(X_train.columns)

        # ── 2d. Save raw current glucose for delta → absolute conversion ──────
        # Targets are deltas (future - current). After prediction we reconstruct
        # absolute glucose for Clarke Error Grid evaluation.
        glucose_val  = (X_val["glucose_mg_dl"].values.reshape(-1, 1)
                        if "glucose_mg_dl" in X_val.columns else None)
        glucose_test = (X_test["glucose_mg_dl"].values.reshape(-1, 1)
                        if "glucose_mg_dl" in X_test.columns else None)

        # ── 3. Scale ──────────────────────────────────────────────────────────
        scaler  = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
        X_val   = pd.DataFrame(scaler.transform(X_val),
                               columns=X_val.columns, index=X_val.index)
        X_test  = pd.DataFrame(scaler.transform(X_test),
                               columns=X_test.columns, index=X_test.index)

        # ── 4. Train ──────────────────────────────────────────────────────────
        self.model.fit(X_train, y_train, X_val, y_val)

        # ── 5. Predict & convert deltas to absolute glucose for evaluation ────
        y_pred_val_delta  = self.model.predict(X_val)
        y_pred_test_delta = self.model.predict(X_test)

        if glucose_val is not None and glucose_test is not None:
            y_pred_val  = y_pred_val_delta  + glucose_val
            y_pred_test = y_pred_test_delta + glucose_test
            y_true_val  = y_val.values  + glucose_val
            y_true_test = y_test.values + glucose_test
        else:
            y_pred_val,  y_pred_test  = y_pred_val_delta,  y_pred_test_delta
            y_true_val,  y_true_test  = y_val.values,      y_test.values

        # ── 6. Save artefacts + plots ─────────────────────────────────────────
        artefact_dir = self._get_artefact_dir()
        fig_dir      = artefact_dir / "figures"

        full_metrics = evaluate_and_plot(
            y_true_val=y_true_val,   y_pred_val=y_pred_val,
            y_true_test=y_true_test, y_pred_test=y_pred_test,
            out_dir=fig_dir,
            horizon=self.horizon,
            title_prefix=f"{self.model.name} | {self.dataset} | user {self.user_id} | {self.horizon}",
            timestamps_val=X_val.index,
            timestamps_test=X_test.index,
        )

        if save:
            self._save_artefacts(artefact_dir, scaler, feature_cols, full_metrics)

        # ── 7. MLflow ─────────────────────────────────────────────────────────
        if log_mlflow:
            _try_mlflow_log(
                experiment=f"glucosense_individual_{self.user_id}_{self.horizon}",
                run_name=f"{self.dataset}_{self.model.name}",
                params={**self.model.get_params(), "dataset": self.dataset,
                        "user_id": self.user_id, "horizon": self.horizon},
                metrics=full_metrics,
                artefact_dir=artefact_dir,
            )

        val_m  = full_metrics["val"]
        test_m = full_metrics["test"]

        result = TrainResult(
            model=self.model,
            scaler=scaler,
            feature_cols=feature_cols,
            val_metrics=val_m,
            test_metrics=test_m,
            artefact_dir=artefact_dir,
        )

        log.info(
            f"Individual training done — user={self.user_id} "
            f"val_RMSE={result.val_rmse:.2f} test_RMSE={result.test_rmse:.2f} "
            f"ClarkeA={result.clarke_a_pct:.1f}%"
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_artefact_dir(self) -> Path:
        base = (MODELS_DIR / "individual" / self.dataset
                / self.user_id / self.horizon / self.model.name)
        d = base / self.version if self.version else base
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_artefacts(
        self,
        artefact_dir: Path,
        scaler: StandardScaler,
        feature_cols: list[str],
        metrics: dict,
    ) -> None:
        self.model.save(artefact_dir / "model.pkl")
        BaseModel._save_pickle(scaler, artefact_dir / "scaler.pkl")

        with open(artefact_dir / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        with open(artefact_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        config = {
            "model_name":   self.model.name,
            "dataset":      self.dataset,
            "user_id":      self.user_id,
            "horizon":      self.horizon,
            "version":      self.version,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
            "model_params": self.model.get_params(),
        }
        with open(artefact_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        log.info(f"Individual artefacts saved → {artefact_dir}")
