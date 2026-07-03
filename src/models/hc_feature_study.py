"""
Cumulative Health-Connect feature-engineering study (population/base models).

Trains the cgmacros population models restricted to the Health-Connect feature space,
sweeping the cumulative feature stages F1→F7 (see feature_groups.CUMULATIVE_HC) × tiers ×
horizons, with the FULL per-model hyperparameter grid (tune=True → all 147 combos/slot).

EVERY model's artifacts are saved (TierTrainer saves model.pkl + feature_cols.json + imputer/
scaler + metrics.json + config.json + importance.csv + figures, plus per-model tuning
leaderboards and a per-slot model leaderboard). On top of that this writes ONE consolidated
manifest (results.csv + results.json) spanning the whole study, so every
(tier × horizon × feature_set × model) row is reusable for analysis / performance review.

Nothing is trained on import — call ``run_hc_feature_study(...)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config import REPORTS_DIR, MODELS_DIR
from src.data.eligibility import TIER_CONFIG
from src.data.feature_groups import CUMULATIVE_HC
from src.models.glucose_models import GLUCOSE_MODELS
from src.models.tier_trainer import TierTrainer
from src.utils import get_logger

log = get_logger(__name__)

BASE_TIERS = ("while_on_cgm", "without_cgm")   # CGM-active base + no-CGM base (both @hc)
BASE_HORIZONS = (30, 60, 90, 120)


def run_hc_feature_study(
    *,
    tiers: tuple[str, ...] = BASE_TIERS,
    horizons: tuple[int, ...] = BASE_HORIZONS,
    models: Optional[list[str]] = None,
    tune: bool = True,
    users: Optional[dict] = None,
    base_scope: str = "population/cgmacros_hc",
    models_dir=None,
    reports_dir=None,
    version: Optional[str] = None,
    feature_sets: Optional[list] = None,    # default: full cumulative F1→F7; pass a subset for A
    study_name: Optional[str] = None,       # manifest subdir (keeps A/B manifests separate)
    table_by_mode: Optional[dict] = None,   # inject pre-built tables (testing); else prep.prepare
) -> dict:
    """Run the feature study. Returns a summary + the consolidated results rows.

    ``feature_sets`` defaults to the full cumulative F1→F7 (CUMULATIVE_HC). For the fast
    "scope A" run pass a single set, e.g. ``[("full_hc", HC_GROUPS)]``.
    """
    models = models or list(GLUCOSE_MODELS)
    sets = feature_sets or CUMULATIVE_HC
    rows: list[dict] = []

    for tier in tiers:
        mode = TIER_CONFIG[tier]["mode"]
        # Build the cgmacros population table for this mode ONCE (reused across feature sets).
        if table_by_mode is not None:
            table = table_by_mode[mode]
        else:
            from src.data import prepare as prep
            table = prep.prepare(mode=mode, datasets=("cgmacros",), users=users).table

        for fset_name, groups in sets:
            # post_cgm mode has no glucose features → drop that group; skip an empty stage.
            g = [x for x in groups if not (mode == "post_cgm" and x == "glucose")]
            if not g:
                continue
            scope = f"{base_scope}/{fset_name}"
            for h in horizons:
                kw: dict = {"feature_groups": g}
                if models_dir is not None: kw["models_dir"] = models_dir
                if reports_dir is not None: kw["reports_dir"] = reports_dir
                if version is not None: kw["version"] = version
                try:
                    trainer = TierTrainer(tier, h, **kw)
                    winner, allres = trainer.select_best(table, models, scope=scope,
                                                         save=True, tune=tune)
                except Exception as exc:                # noqa: BLE001 - log + continue the sweep
                    log.warning(f"[hc-study {tier}/{fset_name}/{h}min] failed: {exc}")
                    continue
                for r in allres:
                    rows.append({
                        "tier": tier, "scope": scope, "feature_set": fset_name, "horizon_min": h,
                        "model": r.model_name, "val_rmse": round(r.val_rmse, 3),
                        "test_rmse": round(r.test_rmse, 3), "clarke_a_pct": round(r.clarke_a, 2),
                        "n_features": r.n_features,
                        "winner": bool(winner is not None and r.model_name == winner.model_name),
                        "artefact_dir": str(r.artefact_dir),
                    })
                log.info(f"[hc-study] {tier}/{fset_name}/{h}min done "
                         f"(winner={winner.model_name if winner else None})")

    # ── Consolidated manifest (analysis / performance review) ─────────────────
    outdir = (Path(reports_dir) if reports_dir else REPORTS_DIR) / "hc_feature_study"
    if study_name:
        outdir = outdir / study_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    try:
        import pandas as pd
        pd.DataFrame(rows).sort_values(["tier", "horizon_min", "feature_set", "val_rmse"]) \
            .to_csv(outdir / "results.csv", index=False)
    except Exception as exc:                             # noqa: BLE001
        log.warning(f"results.csv not written: {exc}")

    log.info(f"HC feature study complete: {len(rows)} rows → {outdir/'results.csv'}")
    return {"n_results": len(rows), "manifest_dir": str(outdir), "rows": rows}


# ── Scope C: per-user individual showcase (full research features) ────────────

def _slot_done(tier: str, scope: str, horizon: int, version: str,
               models: list[str], models_dir=None) -> bool:
    """True if every requested model for this (scope, horizon, version) is on disk (resume)."""
    base = (Path(models_dir) if models_dir else MODELS_DIR) / tier / scope / f"{horizon}min"
    return all((base / m / version / "config.json").exists() for m in models)


def _collect_existing(rows: list, tier: str, uid: str, scope: str, h: int, reports_dir) -> None:
    """On resume, fold an already-trained slot's leaderboard into the manifest rows."""
    lb = (Path(reports_dir) if reports_dir else REPORTS_DIR) / "comparison" / tier / scope / f"{h}min" / "leaderboard.csv"
    if not lb.exists():
        return
    try:
        import pandas as pd
        for _, rr in pd.read_csv(lb).iterrows():
            rows.append({
                "tier": tier, "user": uid, "scope": scope, "horizon_min": h,
                "model": rr["model"], "val_rmse": rr["val_rmse"], "test_rmse": rr["test_rmse"],
                "clarke_a_pct": rr["clarke_a_pct"], "n_features": rr["n_features"],
                "winner": bool(rr["winner"]), "artefact_dir": "(resumed)",
            })
    except Exception:                                    # noqa: BLE001
        pass


def run_individual_showcase(
    *,
    tiers: tuple[str, ...] = ("while_on_cgm", "post_cgm"),
    horizons: tuple[int, ...] = BASE_HORIZONS,
    models: Optional[list[str]] = None,
    tune: bool = True,
    datasets: tuple[str, ...] = ("cgmacros",),
    models_dir=None,
    reports_dir=None,
    version: Optional[str] = None,
    study_name: str = "scopeC_individual",
    users: Optional[list] = None,   # override eligibility (testing / subset / resume)
) -> dict:
    """Train per-user (individual) models for every eligible cgmacros user, per tier ×
    horizon, using FULL research features (no @hc restriction). Saves under
    tiers[tier]["user/<uid>"]. Resume-safe: with a fixed ``version``, completed slots skip.
    """
    from src.data import prepare as prep
    from src.data import eligibility as el

    models = models or list(GLUCOSE_MODELS)
    rows: list[dict] = []

    for tier in tiers:
        mode = TIER_CONFIG[tier]["mode"]
        prepared = prep.prepare(mode=mode, datasets=tuple(datasets))
        elig = users if users is not None else el.eligible_users(prepared.profiles, tier)
        log.info(f"[scopeC] {tier}: {len(elig)} users")
        for uid in elig:
            table = prepared.table[prepared.table["uid"] == uid]
            scope = f"user/{uid}"
            for h in horizons:
                if version and _slot_done(tier, scope, h, version, models, models_dir):
                    _collect_existing(rows, tier, uid, scope, h, reports_dir)   # keep in manifest
                    continue
                kw: dict = {}
                if models_dir is not None: kw["models_dir"] = models_dir
                if reports_dir is not None: kw["reports_dir"] = reports_dir
                if version is not None: kw["version"] = version
                try:
                    trainer = TierTrainer(tier, h, **kw)   # full features (no feature_groups)
                    winner, allres = trainer.select_best(table, models, scope=scope,
                                                         save=True, tune=tune)
                except Exception as exc:                    # noqa: BLE001
                    log.warning(f"[scopeC {tier}/{scope}/{h}min] failed: {exc}")
                    continue
                for r in allres:
                    rows.append({
                        "tier": tier, "user": uid, "scope": scope, "horizon_min": h,
                        "model": r.model_name, "val_rmse": round(r.val_rmse, 3),
                        "test_rmse": round(r.test_rmse, 3), "clarke_a_pct": round(r.clarke_a, 2),
                        "n_features": r.n_features,
                        "winner": bool(winner is not None and r.model_name == winner.model_name),
                        "artefact_dir": str(r.artefact_dir),
                    })
            log.info(f"[scopeC] {tier}/{scope} done")

    outdir = (Path(reports_dir) if reports_dir else REPORTS_DIR) / "hc_feature_study" / study_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    try:
        import pandas as pd
        pd.DataFrame(rows).sort_values(["tier", "user", "horizon_min", "val_rmse"]) \
            .to_csv(outdir / "results.csv", index=False)
    except Exception as exc:                             # noqa: BLE001
        log.warning(f"results.csv not written: {exc}")
    log.info(f"scopeC complete: {len(rows)} rows → {outdir}")
    return {"n_results": len(rows), "manifest_dir": str(outdir), "rows": rows}
