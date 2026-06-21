"""
NeuralProphet individual trainer for blood glucose level prediction.

Why NeuralProphet addresses the peak/valley miss problem:
  1. AR-Net  — uses the last N_LAGS_AR actual glucose readings as a sequence,
               giving the model a real trajectory instead of scalar lag features.
  2. Daily seasonality — captures dawn phenomenon, post-meal timing patterns.
  3. Trend — handles gradual baseline drift across the study period.
  4. Lagged regressors — HR, EDA, carbs, activity as time-aware inputs
               (NeuralProphet handles its own temporal alignment internally).

Input:  preprocessed user DataFrame (output of preprocess_user()).
        NOT the tabular feature matrix — NeuralProphet builds its own structure.

Evaluation strategy (two-fit for honest out-of-sample metrics):
  Fit 1: train-only → val predictions (out-of-sample)
  Fit 2: train+val  → test predictions (out-of-sample)

Artefact layout:
    models/individual/<dataset>/<user_id>/<horizon>/neuralprophet/<version>/
        model.pkl
        metrics.json
        config.json
        figures/
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level

from src.config import MODELS_DIR
from src.models.evaluator import evaluate_and_plot
from src.models.population.trainer import _chronological_split
from src.utils import get_logger

log = get_logger(__name__)

set_log_level("ERROR")   # suppress NeuralProphet's verbose epoch output

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_FORECASTS  = 8    # 8 × 15 min = 2 h ahead
N_LAGS_AR    = 24   # 24 × 15 min = 6 h glucose AR history
N_LAGS_REG   = 4    # 4 × 15 min = 1 h history per external regressor
FREQ         = "15min"
EPOCHS       = 150

# Raw columns from preprocess_user() to use as lagged regressors.
# These are NOT pre-shifted — NeuralProphet handles temporal alignment.
NP_LAGGED_REGRESSORS = [
    "hr",                   # heart rate (autonomic / activity)
    "eda",                  # electrodermal activity (stress proxy)
    "ibi_mean",             # HRV mean inter-beat interval
    "temp",                 # skin temperature
    "acc_magnitude_mean",   # movement / steps proxy
    "total_carb",           # carbohydrate intake (impulse spikes)
    "calorie",              # caloric intake
    "gi_proxy",             # glycemic index proxy (sugar / total_carb)
]

MIN_ROWS = 200


class NeuralProphetTrainer:
    """
    Train and evaluate NeuralProphet for one user.

    Not a BaseModel subclass — NeuralProphet requires time-series format
    input and has fundamentally different fit/predict semantics.

    Args:
        dataset:  Dataset label ("nature_paper" or "cgmacros").
        user_id:  Zero-padded user ID string, e.g. "004".
        horizon:  "2h" (only 2h supported, matching N_FORECASTS=8).
        version:  Optional version tag appended as subdirectory
                  (e.g. "v2_np_20260529").
    """

    def __init__(
        self,
        dataset: str,
        user_id: str,
        horizon: str = "2h",
        version: str | None = None,
    ):
        self.dataset = dataset
        self.user_id = user_id
        self.horizon = horizon
        self.version = version

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        save: bool = True,
    ) -> dict:
        """
        Train, evaluate, and optionally save a NeuralProphet model.

        Args:
            df:   Preprocessed user DataFrame (output of preprocess_user()).
                  Must have a UTC DatetimeIndex and glucose_mg_dl column.
                  Do NOT pass the output of build_feature_matrix().
            save: Persist artefacts to disk if True.

        Returns:
            dict: val_rmse, test_rmse, clarke_a_pct, artefact_dir
        """
        if len(df) < MIN_ROWS:
            raise ValueError(
                f"User {self.user_id}: only {len(df)} rows — "
                f"NeuralProphet needs at least {MIN_ROWS}."
            )

        log.info(
            f"NeuralProphetTrainer: user={self.user_id} "
            f"dataset={self.dataset} horizon={self.horizon} rows={len(df)}"
        )

        df = df.copy()

        # ── 1. Add ground-truth future glucose columns ─────────────────────────
        # y_true_kk[t] = actual glucose at t + k×15 min
        # Must be added BEFORE splitting so slices inherit these columns.
        for k in range(1, N_FORECASTS + 1):
            df[f"_y{k:02d}"] = df["glucose_mg_dl"].shift(-k)

        # ── 2. Chronological 60 / 20 / 20 split ───────────────────────────────
        split = _chronological_split(df)
        if split is None:
            raise RuntimeError(f"User {self.user_id}: DataFrame too small to split.")

        df_train = split["train"]
        df_val   = split["val"]
        df_test  = split["test"]
        log.info(
            f"User {self.user_id} split — "
            f"train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}"
        )

        # ── 3. Identify available regressors ───────────────────────────────────
        regressors = [c for c in NP_LAGGED_REGRESSORS if c in df.columns]
        log.info(f"User {self.user_id}: using {len(regressors)} lagged regressors")

        # ── 4. Convert to NeuralProphet format (ds, y, regressors) ────────────
        def to_np(subset: pd.DataFrame) -> pd.DataFrame:
            d = pd.DataFrame({
                "ds": subset.index.tz_localize(None),   # NP needs naive datetime
                "y":  subset["glucose_mg_dl"].values,
            })
            for col in regressors:
                d[col] = subset[col].values
            return d.reset_index(drop=True)

        df_train_np    = to_np(df_train)
        df_val_np      = to_np(df_val)
        df_test_np     = to_np(df_test)
        df_trainval_np = pd.concat([df_train_np, df_val_np], ignore_index=True)

        # ── 5. Fit 1: train only → out-of-sample val predictions ──────────────
        m1 = self._build_model(regressors)
        m1.fit(df_train_np, freq=FREQ, validation_df=df_val_np, epochs=EPOCHS)

        forecast_tv = m1.predict(df_trainval_np)
        val_ds_set  = set(df_val_np["ds"])
        fc_val      = forecast_tv[forecast_tv["ds"].isin(val_ds_set)].copy()

        y_pred_val, y_true_val, ts_val = self._align(fc_val, df_val)

        # ── 6. Fit 2: train+val → out-of-sample test predictions ──────────────
        m2 = self._build_model(regressors)
        m2.fit(df_trainval_np, freq=FREQ, epochs=EPOCHS)

        df_all_np   = pd.concat([df_trainval_np, df_test_np], ignore_index=True)
        forecast_all = m2.predict(df_all_np)
        test_ds_set  = set(df_test_np["ds"])
        fc_test      = forecast_all[forecast_all["ds"].isin(test_ds_set)].copy()

        y_pred_test, y_true_test, ts_test = self._align(fc_test, df_test)

        if len(y_pred_val) == 0 or len(y_pred_test) == 0:
            raise RuntimeError(
                f"User {self.user_id}: NeuralProphet produced no valid predictions. "
                "Check data continuity and n_lags settings."
            )

        # ── 7. Evaluate + plot ─────────────────────────────────────────────────
        artefact_dir = self._get_artefact_dir()
        fig_dir      = artefact_dir / "figures"

        full_metrics = evaluate_and_plot(
            y_true_val=y_true_val,   y_pred_val=y_pred_val,
            y_true_test=y_true_test, y_pred_test=y_pred_test,
            out_dir=fig_dir,
            horizon=self.horizon,
            title_prefix=(
                f"NeuralProphet | {self.dataset} | "
                f"user {self.user_id} | {self.horizon}"
            ),
            timestamps_val=ts_val,
            timestamps_test=ts_test,
        )

        # ── 8. Save artefacts ──────────────────────────────────────────────────
        if save:
            self._save_artefacts(artefact_dir, m2, regressors, full_metrics)

        val_rmse  = full_metrics["val"]["rmse"]
        test_rmse = full_metrics["test"]["rmse"]
        clarke_a  = full_metrics["test"].get("clarke_a_pct", float("nan"))

        log.info(
            f"NeuralProphet done — user={self.user_id} "
            f"val_RMSE={val_rmse:.2f} test_RMSE={test_rmse:.2f} "
            f"ClarkeA={clarke_a:.1f}%"
        )
        return {
            "val_rmse":     val_rmse,
            "test_rmse":    test_rmse,
            "clarke_a_pct": clarke_a,
            "artefact_dir": artefact_dir,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_model(self, regressors: list[str]) -> NeuralProphet:
        m = NeuralProphet(
            n_forecasts=N_FORECASTS,
            n_lags=N_LAGS_AR,
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            learning_rate=0.001,
        )
        for col in regressors:
            m.add_lagged_regressor(col, n_lags=N_LAGS_REG)
        return m

    def _align(
        self,
        forecast: pd.DataFrame,
        df_split: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Join NeuralProphet forecast with actual future glucose values.

        Returns (y_pred, y_true, timestamps) — only rows where both are
        non-NaN (drops AR warmup rows and trailing rows beyond data end).
        """
        pred_cols = [f"yhat{k}" for k in range(1, N_FORECASTS + 1)]
        true_cols = [f"_y{k:02d}" for k in range(1, N_FORECASTS + 1)]

        # Build a lookup: naive ds → true future glucose values
        true_df = pd.DataFrame(
            df_split[true_cols].values,
            columns=true_cols,
        )
        true_df["ds"] = df_split.index.tz_localize(None)

        merged = pd.merge(
            forecast[["ds"] + [c for c in pred_cols if c in forecast.columns]],
            true_df,
            on="ds",
            how="inner",
        )

        present_preds = [c for c in pred_cols if c in merged.columns]
        preds = merged[present_preds].values.astype(float)
        trues = merged[true_cols].values.astype(float)

        valid = ~(np.isnan(preds).any(axis=1) | np.isnan(trues).any(axis=1))
        preds, trues = preds[valid], trues[valid]
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(merged["ds"].values[valid]).tz_localize("UTC")
        )
        return preds, trues, timestamps

    def _get_artefact_dir(self) -> Path:
        base = (MODELS_DIR / "individual" / self.dataset
                / self.user_id / self.horizon / "neuralprophet")
        d = base / self.version if self.version else base
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_artefacts(
        self,
        artefact_dir: Path,
        model: NeuralProphet,
        regressors: list[str],
        metrics: dict,
    ) -> None:
        with open(artefact_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(artefact_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        config = {
            "model_name":   "neuralprophet",
            "dataset":      self.dataset,
            "user_id":      self.user_id,
            "horizon":      self.horizon,
            "version":      self.version,
            "n_forecasts":  N_FORECASTS,
            "n_lags_ar":    N_LAGS_AR,
            "n_lags_reg":   N_LAGS_REG,
            "epochs":       EPOCHS,
            "regressors":   regressors,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
        }
        with open(artefact_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.info(f"NeuralProphet artefacts saved → {artefact_dir}")
