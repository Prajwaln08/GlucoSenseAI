"""
CLI: run a single glucose prediction using the stateful inference engine.

Useful for smoke-testing Phase 6 artefacts without starting the API server.

Usage examples
--------------
    # Post-CGM prediction (virtual model)
    python scripts/run_stateful_inference.py \\
        --dataset cgmacros \\
        --horizon 2h \\
        --mode post_cgm \\
        --selection-dir reports/experiment_matrix/cgmacros \\
        --feature-csv data/processed/sample_virtual_features.csv

    # CGM-active prediction (population model, delta → absolute)
    python scripts/run_stateful_inference.py \\
        --dataset cgmacros \\
        --horizon 2h \\
        --mode cgm_active \\
        --feature-csv data/processed/sample_features.csv \\
        --current-glucose 112.5

Mode notes
----------
  cgm_active  Requires --current-glucose. Loads population model from
              models/registry.json (run 'make train-cgmacros' first).
              feature-csv must include CGM-derived features.

  post_cgm    Does NOT require --current-glucose. Loads the virtual model
              named in selected_models.json (run Phase 4 + 5 scripts first).
              feature-csv must be a post-CGM feature matrix (no glucose cols).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.serving.stateful_engine import InferenceMode, StatefulInferenceEngine
from src.utils import get_logger

log = get_logger("run_stateful_inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-shot inference using the Phase 6 stateful engine."
    )
    p.add_argument(
        "--dataset", required=True,
        choices=["cgmacros", "nature_paper"],
        help="Dataset the models were trained on.",
    )
    p.add_argument(
        "--horizon", required=True,
        choices=["2h", "3h"],
        help="Prediction horizon.",
    )
    p.add_argument(
        "--mode", required=True,
        choices=["cgm_active", "post_cgm"],
        help="Inference mode.",
    )
    p.add_argument(
        "--feature-csv", required=True, type=Path,
        dest="feature_csv",
        help="CSV file containing the feature matrix (at least one row).",
    )
    p.add_argument(
        "--selection-dir", type=Path, default=None,
        dest="selection_dir",
        help=(
            "Directory containing selected_models.json (Phase 5 output). "
            "Required for post_cgm mode."
        ),
    )
    p.add_argument(
        "--current-glucose", type=float, default=None,
        dest="current_glucose",
        help="Current CGM reading in mg/dL. Required for cgm_active mode.",
    )
    p.add_argument(
        "--artefact-base", type=Path, default=None,
        dest="artefact_base",
        help=(
            "Override for the virtual model artefact base directory "
            "(default: models/virtual). Rarely needed outside tests."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate arg combinations ─────────────────────────────────────────────
    mode = InferenceMode(args.mode)

    if mode == InferenceMode.POST_CGM and args.selection_dir is None:
        log.error("--selection-dir is required for post_cgm mode.")
        sys.exit(1)

    if mode == InferenceMode.CGM_ACTIVE and args.current_glucose is None:
        log.error("--current-glucose is required for cgm_active mode.")
        sys.exit(1)

    if not args.feature_csv.exists():
        log.error(f"feature-csv not found: {args.feature_csv}")
        sys.exit(1)

    # ── Load feature matrix ───────────────────────────────────────────────────
    feature_df = pd.read_csv(args.feature_csv, index_col=0, parse_dates=True)
    log.info(f"Loaded feature CSV: {len(feature_df)} rows × {len(feature_df.columns)} columns")

    # ── Build engine ──────────────────────────────────────────────────────────
    engine = StatefulInferenceEngine(
        dataset       = args.dataset,
        horizon       = args.horizon,
        selection_dir = args.selection_dir,
        mode          = mode,
        artefact_base = args.artefact_base,
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    result = engine.predict(feature_df, current_glucose=args.current_glucose)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stateful Inference Engine — Phase 6")
    print(f"  dataset={result.dataset}  horizon={result.horizon}  mode={result.mode.value}")
    print(f"  model={result.model_key}")
    if result.current_glucose is not None:
        print(f"  current_glucose={result.current_glucose:.1f} mg/dL")
    print(f"{'-'*60}")
    print(f"  {'Step':<6} {'Minutes':<10} {'Glucose (mg/dL)'}")
    print(f"{'-'*60}")
    for step in result.predictions:
        flag = ""
        if step.glucose_mg_dl < 70:
            flag = "  ⚠ HYPO RISK"
        elif step.glucose_mg_dl > 180:
            flag = "  ⚠ HYPER RISK"
        print(f"  {step.step:<6} +{step.minutes_ahead:<9} {step.glucose_mg_dl:.1f}{flag}")
    print(f"{'-'*60}")
    print(f"  Horizon glucose: {result.horizon_glucose:.1f} mg/dL")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
