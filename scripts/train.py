"""
Train glucose models on the unified tier pipeline.

Runs Steps 1–4 (load → clinical/10-min → impute → features), applies eligibility
+ availability cohorts, then trains/selects models per cohort (or per user for the
personalized post_cgm tier) and writes versioned artifacts + a leaderboard.

Examples
--------
  # while-on-CGM population models at all horizons (default models):
  python scripts/train.py --tier while_on_cgm

  # population virtual models, specific horizons + models:
  python scripts/train.py --tier without_cgm --horizons 60 120 --models lightgbm catboost ridge

  # personalized post-CGM models (one per eligible user):
  python scripts/train.py --tier post_cgm --horizons 60
"""

import argparse
from datetime import datetime, timezone

from src.config import HORIZONS_MIN
from src.data import eligibility as el
from src.data import prepare as prep
from src.models.glucose_models import available_models
from src.models.tier_trainer import TierTrainer
from src.utils import get_logger

log = get_logger("train")


# A population model needs more than a couple of users; smaller cohorts are skipped.
MIN_COHORT_USERS = 3


def _cohort_base(cohort_key: tuple[str, ...]) -> str:
    if "mets" in cohort_key or "calories_burned" in cohort_key:
        return "cgmacros"
    if "eda" in cohort_key or "acc_magnitude_mean" in cohort_key:
        return "nature_paper"
    return "_".join(cohort_key) or "none"


def _label_cohorts(groups: dict) -> list[tuple[str, list[str]]]:
    """
    Assign a UNIQUE, readable scope label to each cohort. The largest cohort keeps
    the clean base name ('cgmacros'); any later cohort sharing that base gets a
    numeric suffix ('cgmacros-2') so distinct cohorts never overwrite each other.
    """
    labelled, used = [], {}
    for key, uids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        base = _cohort_base(key)
        n = used.get(base, 0)
        used[base] = n + 1
        label = base if n == 0 else f"{base}-{n + 1}"
        labelled.append((label, uids))
    return labelled


def main() -> None:
    ap = argparse.ArgumentParser(description="Train glucose models (unified tier pipeline).")
    ap.add_argument("--tier", required=True, choices=list(el.TIER_CONFIG))
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS_MIN)
    ap.add_argument("--models", nargs="+", default=available_models())
    ap.add_argument("--datasets", nargs="+", default=["nature_paper", "cgmacros"])
    ap.add_argument("--tune", action="store_true",
                    help="grid-search each model's hyperparameters (≤27 combos) and "
                         "write a tuning leaderboard before selecting the winner")
    ap.add_argument("--version", default=None,
                    help="run version tag (default: one timestamp shared across every "
                         "tier/scope/horizon/model in this invocation)")
    ap.add_argument("--no-save", action="store_true", help="do not persist artifacts")
    args = ap.parse_args()

    run_version = args.version or "v" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    models = [m for m in args.models if m in available_models()]
    mode = el.TIER_CONFIG[args.tier]["mode"]
    log.info(f"Tier={args.tier} mode={mode} horizons={args.horizons} models={models} "
             f"version={run_version}")

    prepared = prep.prepare(mode=mode, datasets=tuple(args.datasets))
    profiles = prepared.profiles

    if args.tier == "post_cgm":
        # Personalized: one model set per eligible user.
        targets = [(f"user/{uid}", prepared.table[prepared.table["uid"] == uid])
                   for uid in el.eligible_users(profiles, args.tier)]
    else:
        # Population: one model set per availability cohort (unique label each).
        groups = el.cohorts(profiles, args.tier, min_users=MIN_COHORT_USERS)
        targets = [(f"population/{label}", prepared.table[prepared.table["uid"].isin(uids)])
                   for label, uids in _label_cohorts(groups)]

    if not targets:
        log.warning(f"No eligible users/cohorts for tier {args.tier!r}. Nothing to train.")
        return

    for scope, table in targets:
        for h in args.horizons:
            trainer = TierTrainer(args.tier, horizon_min=h, version=run_version)
            winner, _ = trainer.select_best(table, models, scope=scope,
                                            save=not args.no_save, tune=args.tune)
            if winner:
                log.info(f"[{args.tier}/{scope}/{h}min] winner={winner.model_name} "
                         f"test_RMSE={winner.test_rmse:.2f} ClarkeA={winner.clarke_a:.1f}%")


if __name__ == "__main__":
    main()
