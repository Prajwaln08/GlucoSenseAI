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

from src.config import HORIZONS_MIN
from src.data import eligibility as el
from src.data import prepare as prep
from src.models.glucose_models import available_models
from src.models.tier_trainer import TierTrainer
from src.utils import get_logger

log = get_logger("train")


def _cohort_label(cohort_key: tuple[str, ...]) -> str:
    if "mets" in cohort_key or "calories_burned" in cohort_key:
        return "cgmacros"
    if "eda" in cohort_key or "acc_magnitude_mean" in cohort_key:
        return "nature_paper"
    return "_".join(cohort_key) or "none"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train glucose models (unified tier pipeline).")
    ap.add_argument("--tier", required=True, choices=list(el.TIER_CONFIG))
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS_MIN)
    ap.add_argument("--models", nargs="+", default=available_models())
    ap.add_argument("--datasets", nargs="+", default=["nature_paper", "cgmacros"])
    ap.add_argument("--no-save", action="store_true", help="do not persist artifacts")
    args = ap.parse_args()

    models = [m for m in args.models if m in available_models()]
    mode = el.TIER_CONFIG[args.tier]["mode"]
    log.info(f"Tier={args.tier} mode={mode} horizons={args.horizons} models={models}")

    prepared = prep.prepare(mode=mode, datasets=tuple(args.datasets))
    profiles = prepared.profiles

    if args.tier == "post_cgm":
        # Personalized: one model set per eligible user.
        targets = [(f"user/{uid}", prepared.table[prepared.table["uid"] == uid])
                   for uid in el.eligible_users(profiles, args.tier)]
    else:
        # Population: one model set per availability cohort.
        targets = [(f"population/{_cohort_label(key)}",
                    prepared.table[prepared.table["uid"].isin(uids)])
                   for key, uids in el.cohorts(profiles, args.tier).items()]

    if not targets:
        log.warning(f"No eligible users/cohorts for tier {args.tier!r}. Nothing to train.")
        return

    for scope, table in targets:
        for h in args.horizons:
            trainer = TierTrainer(args.tier, horizon_min=h)
            winner, _ = trainer.select_best(table, models, scope=scope, save=not args.no_save)
            if winner:
                log.info(f"[{args.tier}/{scope}/{h}min] winner={winner.model_name} "
                         f"test_RMSE={winner.test_rmse:.2f} ClarkeA={winner.clarke_a:.1f}%")


if __name__ == "__main__":
    main()
