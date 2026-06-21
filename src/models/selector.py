"""
Model selector — runs all registered models, applies clinical filters,
picks the winner per slot, and updates models/registry.json.

Selection rules (from GlucoSenseAi.md §12.5.2):
  1. Clarke Zone A ≥ 70%           — clinical safety floor
  2. |val_rmse - test_rmse| / val_rmse ≤ 15%  — no overfitting to val set
  3. Sort remaining by val_rmse ascending       — lowest wins
  4. Tiebreaker (gap ≤ 0.5 mg/dL): prefer faster inference:
       LightGBM > XGBoost > RandomForest

Usage:
    selector = ModelSelector(dataset="cgmacros", horizon="2h")
    results  = selector.run(user_dfs, user_ids=user_ids)
    winner   = selector.best_result
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import MODELS_DIR
from src.models.base_model import BaseModel
from src.models.population.trainer import PopulationTrainer, TrainResult
from src.models.zoo import MODEL_REGISTRY, get_model
from src.utils import get_logger

log = get_logger(__name__)

# Models ranked by inference speed (fastest first) — used as tiebreaker
_SPEED_RANK = {"lightgbm": 0, "xgboost": 1, "random_forest": 2}

# Filters from blueprint §12.5.2
_MIN_CLARKE_A  = 70.0   # %
_MAX_VAL_TEST_GAP = 0.15  # 15 % relative gap


class ModelSelector:
    """
    Run all (or a subset of) registered models as population trainers,
    apply clinical selection filters, and return the best model.

    Args:
        dataset:    "cgmacros" or "nature_paper"
        horizon:    "2h" or "3h"
        model_names: Subset of MODEL_REGISTRY keys to evaluate.
                     If None, all registered models are run.
    """

    def __init__(
        self,
        dataset: str,
        horizon: str = "2h",
        model_names: Optional[list[str]] = None,
    ):
        self.dataset      = dataset
        self.horizon      = horizon
        self.model_names  = model_names or list(MODEL_REGISTRY.keys())
        self.all_results: list[tuple[str, TrainResult]] = []
        self.best_result: Optional[TrainResult]         = None
        self.best_name:   Optional[str]                 = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        user_dfs: list[pd.DataFrame],
        user_ids: Optional[list[str]] = None,
        save: bool = True,
        log_mlflow: bool = True,
    ) -> list[tuple[str, TrainResult]]:
        """
        Train and evaluate all configured models; return ranked results.

        Args:
            user_dfs:  Feature-matrix DataFrames, one per user.
            user_ids:  Matching user ID strings.
            save:      Persist artefacts for every model.
            log_mlflow: Log each run to MLflow.

        Returns:
            List of (model_name, TrainResult) sorted by val_rmse ascending.
        """
        self.all_results = []

        for name in self.model_names:
            log.info(f"Selector: training '{name}' on {self.dataset} / {self.horizon}")
            try:
                model   = get_model(name)
                trainer = PopulationTrainer(model, self.dataset, self.horizon)
                result  = trainer.run(user_dfs, user_ids=user_ids,
                                      save=save, log_mlflow=log_mlflow)
                self.all_results.append((name, result))
                log.info(
                    f"  {name}: val_RMSE={result.val_rmse:.2f} "
                    f"test_RMSE={result.test_rmse:.2f} "
                    f"ClarkeA={result.clarke_a_pct:.1f}%"
                )
            except Exception as exc:
                log.error(f"  {name} failed: {exc}")

        if not self.all_results:
            raise RuntimeError("All models failed — cannot select a winner.")

        self._select_winner()
        if save:
            self._update_registry()

        return self.all_results

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising all run results."""
        rows = []
        for name, r in self.all_results:
            rows.append({
                "model":        name,
                "dataset":      self.dataset,
                "horizon":      self.horizon,
                "val_rmse":     r.val_rmse,
                "test_rmse":    r.test_rmse,
                "mae":          r.test_metrics.get("mae"),
                "clarke_a_pct": r.clarke_a_pct,
                "tir":          r.test_metrics.get("tir"),
                "winner":       name == self.best_name,
            })
        df = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_winner(self) -> None:
        """Apply blueprint filters and set self.best_result / self.best_name."""
        candidates = list(self.all_results)

        # Filter 1: Clarke Zone A ≥ 70%
        passing = [(n, r) for n, r in candidates if r.clarke_a_pct >= _MIN_CLARKE_A]
        if not passing:
            log.warning(
                "No model passed Clarke A ≥ 70% filter. "
                "Falling back to best val_RMSE without clinical filter."
            )
            passing = candidates

        # Filter 2: val/test gap ≤ 15%
        stable = [
            (n, r) for n, r in passing
            if r.val_rmse > 0
            and abs(r.val_rmse - r.test_rmse) / r.val_rmse <= _MAX_VAL_TEST_GAP
        ]
        if not stable:
            log.warning(
                "No model passed val/test gap filter. "
                "Falling back to Clarke-filtered set."
            )
            stable = passing

        # Sort by val_rmse; tiebreak by inference speed rank
        stable.sort(key=lambda x: (
            round(x[1].val_rmse, 1),
            _SPEED_RANK.get(x[0], 99),
        ))

        self.best_name, self.best_result = stable[0]
        log.info(
            f"Selected winner: '{self.best_name}' — "
            f"val_RMSE={self.best_result.val_rmse:.2f} "
            f"test_RMSE={self.best_result.test_rmse:.2f}"
        )

    def _update_registry(self) -> None:
        """Upsert the winner into models/registry.json."""
        registry_path = MODELS_DIR / "registry.json"

        # Load existing or create fresh
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {"version": "1.0.0", "population": {}, "individual": {}}

        pop = registry.setdefault("population", {})
        ds  = pop.setdefault(self.dataset, {})
        r   = self.best_result

        ds[self.horizon] = {
            "best_model_type": self.best_name,
            "artefact_dir":    str(r.artefact_dir.relative_to(MODELS_DIR.parent)),
            "val_rmse":        r.val_rmse,
            "test_rmse":       r.test_rmse,
            "mae":             r.test_metrics.get("mae"),
            "clarke_a_pct":    r.clarke_a_pct,
            "tir":             r.test_metrics.get("tir"),
            "n_test":          r.test_metrics.get("n_samples"),
            "feature_cols":    r.feature_cols,
            "trained_at":      datetime.now(timezone.utc).isoformat(),
        }
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        log.info(f"Registry updated → {registry_path}")
