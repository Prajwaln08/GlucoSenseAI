"""
CLI: select the best virtual model per horizon from Phase 4 experiment results.

Reads the results.json produced by run_experiment_matrix.py, applies Clarke
and overfitting-gap filters, and writes selected_models.json to the same dir.

Usage examples
--------------
    # Default thresholds (Clarke ≥ 70%, gap ≤ 15%)
    python scripts/select_virtual_model.py \\
        --results-dir reports/experiment_matrix/cgmacros

    # Tighter clinical gate
    python scripts/select_virtual_model.py \\
        --results-dir reports/experiment_matrix/cgmacros \\
        --min-clarke 80.0

    # Write selection to a different directory
    python scripts/select_virtual_model.py \\
        --results-dir reports/experiment_matrix/cgmacros \\
        --out-dir reports/selections/cgmacros

Selection rules
---------------
  1. Clarke Zone A ≥ --min-clarke (clinical safety gate)
  2. |val_rmse − test_rmse| / val_rmse ≤ --max-gap (overfitting guard)
  3. If no model passes a filter, that filter is skipped with a warning.
  4. Sort by (val_rmse rounded to 0.1, inference speed rank) — lowest wins.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    SELECTOR_MIN_CLARKE_A,
    SELECTOR_MAX_VAL_TEST_GAP,
)
from src.experiments.selector import VirtualModelSelector
from src.utils import get_logger

log = get_logger("select_virtual_model")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select the best virtual model per horizon from experiment results."
    )
    p.add_argument(
        "--results-dir", required=True, type=Path,
        dest="results_dir",
        help="Directory containing results.json (output of run_experiment_matrix.py).",
    )
    p.add_argument(
        "--min-clarke", type=float, default=SELECTOR_MIN_CLARKE_A,
        dest="min_clarke",
        help=f"Minimum Clarke Zone A%% (default: {SELECTOR_MIN_CLARKE_A}).",
    )
    p.add_argument(
        "--max-gap", type=float, default=SELECTOR_MAX_VAL_TEST_GAP,
        dest="max_gap",
        help=f"Max relative val/test RMSE gap (default: {SELECTOR_MAX_VAL_TEST_GAP}).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        dest="out_dir",
        help="Directory to write selected_models.json (default: --results-dir).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    out_dir     = args.out_dir or results_dir

    if not (results_dir / "results.json").exists():
        log.error(f"results.json not found in {results_dir}")
        sys.exit(1)

    selector = VirtualModelSelector(
        min_clarke_a     = args.min_clarke,
        max_val_test_gap = args.max_gap,
    )

    log.info(
        f"Selecting from {results_dir / 'results.json'}  "
        f"(Clarke ≥ {args.min_clarke:.0f}%,  gap ≤ {args.max_gap*100:.0f}%)"
    )

    selection = selector.select_from_file(results_dir)

    # Pretty-print the selection table
    print(f"\n{'='*70}")
    print(f"  Selected virtual models")
    print(f"  Clarke gate: ≥ {args.min_clarke:.0f}%   "
          f"Gap guard: ≤ {args.max_gap*100:.0f}%")
    print(f"{'-'*70}")
    print(f"  {'Horizon':<8} {'Model':<18} {'Win':>5} {'val':>8} {'test':>8} "
          f"{'ClarkeA':>8} {'Filters'}")
    print(f"{'-'*70}")
    for horizon, m in sorted(selection.items()):
        win = str(m.input_window) if m.input_window is not None else "N/A"
        filters = (
            ("Clarke" if m.passed_clarke else "fallback") + " / " +
            ("gap"    if m.passed_gap    else "fallback")
        )
        print(
            f"  {m.horizon:<8} {m.model_key:<18} {win:>5} "
            f"{m.val_rmse:>8.2f} {m.test_rmse:>8.2f} "
            f"{m.clarke_a_pct:>8.1f} {filters}"
        )
    print(f"{'='*70}\n")

    # Infer dataset from the first SelectedModel
    dataset = next(iter(selection.values())).dataset if selection else ""

    VirtualModelSelector.save_selection(
        selection,
        out_dir = out_dir,
        dataset = dataset,
        selector_config = {
            "min_clarke_a":     args.min_clarke,
            "max_val_test_gap": args.max_gap,
        },
    )
    print(f"selected_models.json → {out_dir / 'selected_models.json'}")


if __name__ == "__main__":
    main()
