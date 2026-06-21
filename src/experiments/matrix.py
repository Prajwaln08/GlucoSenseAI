"""
Input-window × horizon experiment matrix for Stage-B virtual models.

Sweeps:
  input_window : [12, 24, 36] 15-min history steps used as RNN seq_len.
                 Not applicable to tabular models (recorded as None).
  horizon      : ["2h", "3h"]
  model_keys   : VIRTUAL_MODEL_KEYS or a caller-specified subset

For tabular models (virtual_lgbm, virtual_xgb) input_window has no effect on
the flat feature vector, so each tabular model gets one config per horizon.
For RNN models (virtual_gru, virtual_lstm) the full Cartesian product is run.

Artefacts
---------
Saved to:  reports/experiment_matrix/<dataset>/results.json
           reports/experiment_matrix/<dataset>/summary.csv

The results JSON is the authoritative input for Phase 5's model selector.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    EXPERIMENT_INPUT_WINDOWS,
    EXPERIMENT_HORIZONS,
    HORIZON_2H_STEPS,
    HORIZON_3H_STEPS,
    REPORTS_DIR,
)
from src.models.zoo import get_model, VIRTUAL_MODEL_KEYS
from src.models.virtual.trainer import VirtualTrainer
from src.utils import get_logger

log = get_logger(__name__)

# ── Model family classification ───────────────────────────────────────────────
# RNN models use input_window as seq_len; tabular models ignore it.
_RNN_KEYS     = frozenset({"virtual_gru", "virtual_lstm"})
_TABULAR_KEYS = frozenset({"virtual_lgbm", "virtual_xgb"})

_HORIZON_STEPS: dict[str, int] = {"2h": HORIZON_2H_STEPS, "3h": HORIZON_3H_STEPS}


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExperimentConfig:
    """One cell in the experiment matrix: (model, horizon, input_window)."""
    model_key:    str
    horizon:      str
    input_window: Optional[int]  # None for tabular models
    dataset:      str
    mode:         str = "post_cgm"

    @property
    def key(self) -> str:
        """Unique string identifier for this config — safe as a dict key or filename."""
        win = str(self.input_window) if self.input_window is not None else "na"
        return f"{self.model_key}__{self.horizon}__win{win}"

    @property
    def is_rnn(self) -> bool:
        return self.model_key in _RNN_KEYS

    @property
    def horizon_steps(self) -> int:
        return _HORIZON_STEPS[self.horizon]


@dataclass
class ExperimentResult:
    """Outcome of a single matrix cell."""
    config:       ExperimentConfig
    val_metrics:  dict
    test_metrics: dict
    elapsed_s:    float
    skipped:      bool = False
    error:        Optional[str] = None

    @property
    def val_rmse(self) -> float:
        return self.val_metrics.get("rmse", float("inf"))

    @property
    def test_rmse(self) -> float:
        return self.test_metrics.get("rmse", float("inf"))

    @property
    def clarke_a_pct(self) -> float:
        return self.test_metrics.get("clarke_a_pct", 0.0)

    def to_row(self) -> dict:
        """Flat dict suitable for a DataFrame row or JSON array element."""
        return {
            "config_key":   self.config.key,
            "model_key":    self.config.model_key,
            "horizon":      self.config.horizon,
            "input_window": self.config.input_window,
            "dataset":      self.config.dataset,
            "val_rmse":     round(self.val_rmse, 4),
            "test_rmse":    round(self.test_rmse, 4),
            "clarke_a_pct": round(self.clarke_a_pct, 2),
            "elapsed_s":    round(self.elapsed_s, 1),
            "skipped":      self.skipped,
            "error":        self.error or "",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Grid builder
# ══════════════════════════════════════════════════════════════════════════════

def build_experiment_grid(
    model_keys:    list[str]      = VIRTUAL_MODEL_KEYS,
    horizons:      list[str]      = EXPERIMENT_HORIZONS,
    input_windows: list[int]      = EXPERIMENT_INPUT_WINDOWS,
    dataset:       str            = "cgmacros",
    mode:          str            = "post_cgm",
) -> list[ExperimentConfig]:
    """
    Build the full Cartesian product of experiment configs.

    Tabular models (virtual_lgbm, virtual_xgb):
        One config per horizon; input_window=None (not applicable).
    RNN models (virtual_gru, virtual_lstm):
        One config per (horizon × input_window) combination.

    Returns a list of ExperimentConfig with unique .key values.
    """
    if not model_keys:
        raise ValueError("model_keys is empty.")
    if not horizons:
        raise ValueError("horizons is empty.")
    if not input_windows:
        raise ValueError("input_windows is empty.")

    configs: list[ExperimentConfig] = []
    for key in model_keys:
        if key in _TABULAR_KEYS:
            for h in horizons:
                configs.append(ExperimentConfig(
                    model_key=key, horizon=h,
                    input_window=None, dataset=dataset, mode=mode,
                ))
        elif key in _RNN_KEYS:
            for h in horizons:
                for win in input_windows:
                    configs.append(ExperimentConfig(
                        model_key=key, horizon=h,
                        input_window=win, dataset=dataset, mode=mode,
                    ))
        else:
            # Unknown model family — treat as tabular (no window sweep)
            log.warning(
                f"Model key {key!r} is not in _TABULAR_KEYS or _RNN_KEYS. "
                "Treating as tabular (input_window=None)."
            )
            for h in horizons:
                configs.append(ExperimentConfig(
                    model_key=key, horizon=h,
                    input_window=None, dataset=dataset, mode=mode,
                ))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Single-experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(
    config:   ExperimentConfig,
    user_dfs: list[pd.DataFrame],
    user_ids: list[str],
) -> ExperimentResult:
    """
    Train and evaluate one (model, horizon, input_window) cell.

    For RNN models, input_window is forwarded as the seq_len constructor kwarg.
    For tabular models, input_window is ignored (already None in the config).

    On failure the experiment is marked skipped=True with the error string
    stored, so one bad cell doesn't abort the full matrix run.
    """
    t0 = time.perf_counter()
    try:
        kwargs: dict = {}
        if config.input_window is not None and config.is_rnn:
            kwargs["seq_len"] = config.input_window

        model   = get_model(config.model_key, **kwargs)
        trainer = VirtualTrainer(model, dataset=config.dataset, horizon=config.horizon)
        result  = trainer.run(user_dfs, user_ids=user_ids, save=False, log_mlflow=False)

        # Stamp metadata onto the model for Phase 5's selector
        model.input_window       = config.input_window
        model.prediction_horizon = config.horizon_steps

        elapsed = time.perf_counter() - t0
        return ExperimentResult(
            config=config,
            val_metrics=result.val_metrics,
            test_metrics=result.test_metrics,
            elapsed_s=elapsed,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error(f"Experiment {config.key!r} failed: {exc}")
        return ExperimentResult(
            config=config,
            val_metrics={},
            test_metrics={},
            elapsed_s=elapsed,
            skipped=True,
            error=str(exc),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Full matrix runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_matrix(
    configs:  list[ExperimentConfig],
    user_dfs: list[pd.DataFrame],
    user_ids: list[str],
    out_dir:  Optional[Path] = None,
) -> list[ExperimentResult]:
    """
    Run all experiment configs sequentially, returning all results.

    Failed cells are marked skipped=True and included in the output — they
    do not abort the run.  If out_dir is given, results are persisted there.
    """
    if not configs:
        raise ValueError("configs is empty — build_experiment_grid first.")

    n       = len(configs)
    results: list[ExperimentResult] = []

    log.info(f"Experiment matrix: {n} configs to run.")
    for i, cfg in enumerate(configs, 1):
        log.info(
            f"[{i}/{n}] {cfg.model_key} | horizon={cfg.horizon} "
            f"| input_window={cfg.input_window}"
        )
        res = run_single_experiment(cfg, user_dfs, user_ids)
        results.append(res)

        if not res.skipped:
            log.info(
                f"  val_RMSE={res.val_rmse:.2f}  "
                f"test_RMSE={res.test_rmse:.2f}  "
                f"ClarkeA={res.clarke_a_pct:.1f}%  "
                f"({res.elapsed_s:.0f}s)"
            )
        else:
            log.warning(f"  SKIPPED — {res.error}")

    if out_dir is not None:
        save_matrix_results(results, Path(out_dir))

    succeeded = sum(1 for r in results if not r.skipped)
    log.info(f"Matrix done — {succeeded}/{n} succeeded.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_matrix_results(results: list[ExperimentResult], out_dir: Path) -> None:
    """
    Write results.json and summary.csv to out_dir.

    results.json — machine-readable; consumed by Phase 5's model selector.
    summary.csv  — human-readable ranking table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r.to_row() for r in results]

    dataset = results[0].config.dataset if results else "unknown"
    mode    = results[0].config.mode    if results else "post_cgm"

    payload = {
        "dataset": dataset,
        "mode":    mode,
        "n_total": len(results),
        "n_ok":    sum(1 for r in results if not r.skipped),
        "results": rows,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    df = pd.DataFrame(rows).sort_values("val_rmse")
    df.to_csv(out_dir / "summary.csv", index=False)

    log.info(f"Matrix results saved → {out_dir}")


def load_matrix_results(out_dir: Path) -> list[dict]:
    """Load previously saved results.json as a list of row dicts."""
    path = Path(out_dir) / "results.json"
    with open(path) as f:
        payload = json.load(f)
    return payload["results"]


# ══════════════════════════════════════════════════════════════════════════════
# Winner selection
# ══════════════════════════════════════════════════════════════════════════════

def find_best_per_horizon(
    results: list[ExperimentResult],
    metric:  str = "val_rmse",
) -> dict[str, ExperimentResult]:
    """
    For each horizon, return the ExperimentResult with the lowest metric.

    Failed / skipped results are excluded.  If no valid result exists for a
    horizon, that horizon is absent from the returned dict.

    Args:
        results: Output of run_experiment_matrix().
        metric:  Attribute name on ExperimentResult to minimise.
                 One of: "val_rmse", "test_rmse".
    """
    by_horizon: dict[str, list[ExperimentResult]] = {}
    for r in results:
        if r.skipped:
            continue
        by_horizon.setdefault(r.config.horizon, []).append(r)

    return {
        h: min(group, key=lambda r: getattr(r, metric))
        for h, group in by_horizon.items()
    }


def find_best_overall(
    results: list[ExperimentResult],
    metric:  str = "val_rmse",
) -> Optional[ExperimentResult]:
    """Return the single best ExperimentResult across all horizons and models."""
    valid = [r for r in results if not r.skipped]
    if not valid:
        return None
    return min(valid, key=lambda r: getattr(r, metric))


def print_matrix_summary(results: list[ExperimentResult]) -> None:
    """Pretty-print a ranked table of experiment results to stdout."""
    rows = sorted(
        [r for r in results if not r.skipped],
        key=lambda r: r.val_rmse,
    )
    failed = [r for r in results if r.skipped]

    print(f"\n{'='*75}")
    print(f"{'Config':<40} {'val_RMSE':>9} {'test_RMSE':>10} {'ClarkeA%':>9} {'t(s)':>6}")
    print(f"{'-'*75}")
    for r in rows:
        print(
            f"  {r.config.key:<38} "
            f"{r.val_rmse:>9.2f} "
            f"{r.test_rmse:>10.2f} "
            f"{r.clarke_a_pct:>9.1f} "
            f"{r.elapsed_s:>6.0f}"
        )
    if failed:
        print(f"\n  {len(failed)} config(s) failed:")
        for r in failed:
            print(f"    {r.config.key}: {r.error}")
    print(f"{'='*75}\n")
