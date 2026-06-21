"""
Phase 5 — Unified Clarke-Based Virtual Model Selector.

Consumes the results.json produced by Phase 4's experiment matrix and applies
two clinical selection filters per horizon:

  1. Clarke Zone A ≥ SELECTOR_MIN_CLARKE_A (default 70%)
     — clinical safety gate; a model that routinely mis-classifies glucose
       clinically should never serve predictions.

  2. |val_rmse − test_rmse| / val_rmse ≤ SELECTOR_MAX_VAL_TEST_GAP (default 15%)
     — overfitting guard; a model that fits val well but degrades badly on
       held-out test data is unreliable in production.

Selection cascade (per horizon):
  a. Keep non-skipped results only.
  b. Apply Clarke filter; if nothing passes, warn and skip this filter.
  c. Apply gap filter; if nothing passes, warn and skip this filter.
  d. Sort by (round(val_rmse, 1), speed_rank) — lowest wins; speed breaks ties.
  e. The head of the sorted list becomes the SelectedModel for that horizon.

Output
------
  reports/experiment_matrix/<dataset>/selected_models.json

The JSON is the authoritative input for Phase 6's stateful inference engine.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import (
    SELECTOR_MAX_VAL_TEST_GAP,
    SELECTOR_MIN_CLARKE_A,
)
from src.experiments.matrix import (
    ExperimentConfig,
    ExperimentResult,
    load_matrix_results,
)
from src.utils import get_logger

log = get_logger(__name__)

# Virtual model speed ranking for tiebreaking (fastest inference first)
_VIRTUAL_SPEED_RANK: dict[str, int] = {
    "virtual_lgbm": 0,
    "virtual_xgb":  1,
    "virtual_gru":  2,
    "virtual_lstm": 3,
}


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SelectedModel:
    """The winning virtual model for one (dataset, horizon) slot."""
    model_key:     str
    horizon:       str
    input_window:  Optional[int]   # None for tabular models
    dataset:       str
    config_key:    str             # unique key from ExperimentConfig.key
    val_rmse:      float
    test_rmse:     float
    clarke_a_pct:  float
    passed_clarke: bool            # did it pass the Clarke A% threshold?
    passed_gap:    bool            # did it pass the val/test RMSE gap check?

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Selector
# ══════════════════════════════════════════════════════════════════════════════

class VirtualModelSelector:
    """
    Select the best virtual model per horizon from Phase 4 experiment results.

    Args:
        min_clarke_a:    Minimum Clarke Zone A% (clinical safety gate).
        max_val_test_gap: Maximum allowed relative RMSE gap (overfitting guard).
    """

    def __init__(
        self,
        min_clarke_a:     float = SELECTOR_MIN_CLARKE_A,
        max_val_test_gap: float = SELECTOR_MAX_VAL_TEST_GAP,
    ):
        self.min_clarke_a     = min_clarke_a
        self.max_val_test_gap = max_val_test_gap

    # ── Public API ────────────────────────────────────────────────────────────

    def select(
        self,
        results: list[ExperimentResult],
    ) -> dict[str, "SelectedModel"]:
        """
        Select the best model per horizon.

        Args:
            results: Output of run_experiment_matrix() (ExperimentResult list).

        Returns:
            {horizon: SelectedModel} — one entry per horizon that has at least
            one non-skipped result.

        Raises:
            RuntimeError: if every result is skipped (no candidates at all).
        """
        valid = [r for r in results if not r.skipped]
        if not valid:
            raise RuntimeError(
                "VirtualModelSelector: all results are skipped — cannot select."
            )

        # Group by horizon
        by_horizon: dict[str, list[ExperimentResult]] = {}
        for r in valid:
            by_horizon.setdefault(r.config.horizon, []).append(r)

        selection: dict[str, SelectedModel] = {}
        for horizon, candidates in by_horizon.items():
            winner = self._select_one(horizon, candidates)
            selection[horizon] = winner
            log.info(
                f"[{horizon}] Selected: {winner.model_key}  "
                f"input_window={winner.input_window}  "
                f"val_RMSE={winner.val_rmse:.2f}  "
                f"ClarkeA={winner.clarke_a_pct:.1f}%  "
                f"passed_clarke={winner.passed_clarke}  "
                f"passed_gap={winner.passed_gap}"
            )

        return selection

    def select_from_file(self, results_dir: Path) -> dict[str, "SelectedModel"]:
        """
        Load results.json from results_dir and call select().

        Convenience wrapper so callers don't need to touch
        ExperimentResult / ExperimentConfig directly.
        """
        rows = load_matrix_results(Path(results_dir))
        results = _rows_to_experiment_results(rows)
        return self.select(results)

    @staticmethod
    def save_selection(
        selection: dict[str, "SelectedModel"],
        out_dir:   Path,
        dataset:   str  = "",
        selector_config: Optional[dict] = None,
    ) -> None:
        """
        Persist the selection to selected_models.json.

        The JSON is consumed by Phase 6's inference engine at startup.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset":         dataset,
            "selected_at":     datetime.now(timezone.utc).isoformat(),
            "selector_config": selector_config or {},
            "by_horizon":      {h: m.to_dict() for h, m in selection.items()},
        }
        path = out_dir / "selected_models.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        log.info(f"Selection saved → {path}")

    @staticmethod
    def load_selection(out_dir: Path) -> dict[str, dict]:
        """
        Load selected_models.json and return the by_horizon mapping.

        Returns:
            {horizon: dict} — raw dicts; callers may reconstruct SelectedModel
            via SelectedModel(**d) if needed.
        """
        path = Path(out_dir) / "selected_models.json"
        with open(path) as f:
            payload = json.load(f)
        return payload["by_horizon"]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_one(
        self,
        horizon:    str,
        candidates: list[ExperimentResult],
    ) -> "SelectedModel":
        """Apply filters and return the winning SelectedModel for one horizon."""

        # Annotate each candidate with filter flags
        annotated = [
            (r, self._passes_clarke(r), self._passes_gap(r))
            for r in candidates
        ]

        # Filter 1: Clarke gate
        clarke_passing = [(r, pc, pg) for r, pc, pg in annotated if pc]
        if not clarke_passing:
            log.warning(
                f"[{horizon}] No model passed Clarke A ≥ {self.min_clarke_a:.0f}% "
                f"filter — falling back to all {len(candidates)} candidates."
            )
            clarke_passing = annotated

        # Filter 2: overfitting guard
        gap_passing = [(r, pc, pg) for r, pc, pg in clarke_passing if pg]
        if not gap_passing:
            log.warning(
                f"[{horizon}] No model passed val/test gap ≤ "
                f"{self.max_val_test_gap*100:.0f}% filter — "
                f"falling back to Clarke-filtered set."
            )
            gap_passing = clarke_passing

        # Sort: primary = val_rmse (rounded to 0.1 to allow speed tiebreak),
        #       secondary = inference speed rank
        gap_passing.sort(key=lambda t: (
            round(t[0].val_rmse, 1),
            _VIRTUAL_SPEED_RANK.get(t[0].config.model_key, 99),
        ))

        best_r, best_pc, best_pg = gap_passing[0]
        return SelectedModel(
            model_key    = best_r.config.model_key,
            horizon      = horizon,
            input_window = best_r.config.input_window,
            dataset      = best_r.config.dataset,
            config_key   = best_r.config.key,
            val_rmse     = best_r.val_rmse,
            test_rmse    = best_r.test_rmse,
            clarke_a_pct = best_r.clarke_a_pct,
            passed_clarke= best_pc,
            passed_gap   = best_pg,
        )

    def _passes_clarke(self, r: ExperimentResult) -> bool:
        return r.clarke_a_pct >= self.min_clarke_a

    def _passes_gap(self, r: ExperimentResult) -> bool:
        if r.val_rmse <= 0:
            return False
        gap = abs(r.val_rmse - r.test_rmse) / r.val_rmse
        return gap <= self.max_val_test_gap


# ══════════════════════════════════════════════════════════════════════════════
# Internal helper
# ══════════════════════════════════════════════════════════════════════════════

def _rows_to_experiment_results(rows: list[dict]) -> list[ExperimentResult]:
    """Reconstruct ExperimentResult objects from load_matrix_results() rows."""
    results = []
    for row in rows:
        cfg = ExperimentConfig(
            model_key    = row["model_key"],
            horizon      = row["horizon"],
            input_window = row.get("input_window"),   # may be None
            dataset      = row["dataset"],
        )
        results.append(ExperimentResult(
            config      = cfg,
            val_metrics = {"rmse": row["val_rmse"]},
            test_metrics= {
                "rmse":         row["test_rmse"],
                "clarke_a_pct": row["clarke_a_pct"],
            },
            elapsed_s   = row.get("elapsed_s", 0.0),
            skipped     = bool(row.get("skipped", False)),
            error       = row.get("error") or None,
        ))
    return results
