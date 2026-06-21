"""
Populate models/registry.json with virtual (Stage B) and individual model entries.

Run once after training completes, and again any time new models are trained
outside the Celery retrain flow (e.g. bulk training scripts).

Usage:
    python scripts/populate_registry.py
    python scripts/populate_registry.py --dry-run   # print what would change, no write

What it does:
    1. Scans models/virtual/<dataset>/<horizon>/<model>/ for each slot.
       Picks the model with the lowest val RMSE as the slot winner.
    2. Scans models/individual/<dataset>/<user_id>/<horizon>/<model>/ for every
       trained user. Picks the best model per user/horizon by val RMSE.
    3. Writes registry["virtual"] and registry["individual"] sections.
       registry["population"] is preserved unchanged.

HR gate constants written into each slot:
    min_hr_readings    = 8   (2 h at 15-min intervals — minimum to start predicting)
    full_hr_readings   = 24  (6 h — all rolling windows fully saturated)
    hr_gap_tolerance_min = 30  (max gap between consecutive readings before blocked)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR    = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
VIRTUAL_DIR   = MODELS_DIR / "virtual"
INDIVIDUAL_DIR = MODELS_DIR / "individual"

DATASETS  = ["cgmacros", "nature_paper"]
HORIZONS  = ["2h", "3h"]

# HR gate constants — same for all slots (model-agnostic; deepest window is hr_roll_mean_24 = 6h)
HR_GATE = {
    "min_hr_readings":      8,   # 2h of watch data required before first prediction
    "full_hr_readings":     24,  # 6h for all rolling windows to saturate
    "hr_gap_tolerance_min": 30,  # consecutive reading gap above this = data_gap error
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"version": "1.0.0", "population": {}, "virtual": {}, "individual": {}}
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    reg.setdefault("virtual", {})
    reg.setdefault("individual", {})
    return reg


def _read_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    # Normalise: metrics may be flat (population trainer) or nested val/test (individual/virtual trainer)
    if "val" in m and "test" in m:
        return {
            "val_rmse":     m["val"]["rmse"],
            "test_rmse":    m["test"]["rmse"],
            "mae":          m["test"].get("mae", 0.0),
            "clarke_a_pct": m["test"].get("clarke_a_pct", 0.0),
            "n_test":       m["test"].get("n_samples", 0),
        }
    # Flat (population)
    return {
        "val_rmse":     m.get("val_rmse", float("inf")),
        "test_rmse":    m.get("test_rmse", 0.0),
        "mae":          m.get("mae", 0.0),
        "clarke_a_pct": m.get("clarke_a_pct", 0.0),
        "n_test":       m.get("n_test", 0),
    }


def _read_feature_cols(fc_path: Path) -> list:
    if not fc_path.exists():
        return []
    with open(fc_path) as f:
        return json.load(f)


def _read_trained_at(config_path: Path) -> str:
    if not config_path.exists():
        return datetime.now(timezone.utc).isoformat()
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("trained_at", datetime.now(timezone.utc).isoformat())


def _best_model_in_dir(horizon_dir: Path) -> tuple[str, dict] | None:
    """
    Scan all model subdirs in horizon_dir. Return (model_name, metrics) for
    the one with the lowest val_rmse. Returns None if no valid model found.
    """
    best_name, best_metrics = None, None
    for model_dir in sorted(horizon_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        m = _read_metrics(model_dir / "metrics.json")
        if m is None:
            continue
        if best_metrics is None or m["val_rmse"] < best_metrics["val_rmse"]:
            best_name    = model_dir.name
            best_metrics = m
            best_dir     = model_dir
    if best_name is None:
        return None
    return best_name, best_metrics, best_dir


# ── Virtual (Stage B) ─────────────────────────────────────────────────────────

def build_virtual_section(dry_run: bool) -> dict:
    virtual = {}
    for dataset in DATASETS:
        virtual[dataset] = {}
        for horizon in HORIZONS:
            h_dir = VIRTUAL_DIR / dataset / horizon
            if not h_dir.exists():
                print(f"  [skip] virtual/{dataset}/{horizon} — directory not found")
                continue

            result = _best_model_in_dir(h_dir)
            if result is None:
                print(f"  [skip] virtual/{dataset}/{horizon} — no valid metrics found")
                continue

            best_name, metrics, best_dir = result
            feature_cols = _read_feature_cols(best_dir / "feature_cols.json")
            trained_at   = _read_trained_at(best_dir / "config.json")

            slot = {
                "best_model_type": best_name,
                "artefact_dir":    str(best_dir.relative_to(ROOT)),
                "val_rmse":        round(metrics["val_rmse"], 4),
                "test_rmse":       round(metrics["test_rmse"], 4),
                "mae":             round(metrics["mae"], 4),
                "clarke_a_pct":    round(metrics["clarke_a_pct"], 2),
                "n_test":          metrics["n_test"],
                "feature_cols":    feature_cols,
                "trained_at":      trained_at,
                **HR_GATE,
            }
            virtual[dataset][horizon] = slot
            print(f"  [virtual] {dataset}/{horizon}: {best_name}  "
                  f"val={metrics['val_rmse']:.2f}  test={metrics['test_rmse']:.2f}  "
                  f"ClarkeA={metrics['clarke_a_pct']:.1f}%")

    return virtual


# ── Individual ────────────────────────────────────────────────────────────────

def build_individual_section(dry_run: bool) -> dict:
    individual = {}
    for dataset in DATASETS:
        ds_dir = INDIVIDUAL_DIR / dataset
        if not ds_dir.exists():
            continue

        individual[dataset] = {}
        user_dirs = sorted(d for d in ds_dir.iterdir() if d.is_dir())

        for user_dir in user_dirs:
            uid = user_dir.name
            individual[dataset][uid] = {}

            for horizon in HORIZONS:
                h_dir = user_dir / horizon
                if not h_dir.exists():
                    continue

                result = _best_model_in_dir(h_dir)
                if result is None:
                    continue

                best_name, metrics, best_dir = result
                feature_cols = _read_feature_cols(best_dir / "feature_cols.json")
                trained_at   = _read_trained_at(best_dir / "config.json")

                slot = {
                    "best_model_type": best_name,
                    "artefact_dir":    str(best_dir.relative_to(ROOT)),
                    "val_rmse":        round(metrics["val_rmse"], 4),
                    "test_rmse":       round(metrics["test_rmse"], 4),
                    "mae":             round(metrics["mae"], 4),
                    "clarke_a_pct":    round(metrics["clarke_a_pct"], 2),
                    "n_test":          metrics["n_test"],
                    "feature_cols":    feature_cols,
                    "trained_at":      trained_at,
                    **HR_GATE,
                }
                individual[dataset][uid][horizon] = slot
                print(f"  [individual] {dataset}/{uid}/{horizon}: {best_name}  "
                      f"val={metrics['val_rmse']:.2f}  test={metrics['test_rmse']:.2f}  "
                      f"ClarkeA={metrics['clarke_a_pct']:.1f}%")

    return individual


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Populate registry.json from trained artefacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying registry.json.")
    args = parser.parse_args()

    reg = _read_registry()
    print(f"\nRegistry: {REGISTRY_PATH}")
    print(f"Existing sections: {list(reg.keys())}\n")

    print("=== Virtual models (Stage B) ===")
    virtual = build_virtual_section(args.dry_run)

    print("\n=== Individual models ===")
    individual = build_individual_section(args.dry_run)

    # Count summary
    v_count = sum(len(v) for v in virtual.values())
    i_count = sum(
        sum(len(h) for h in u.values())
        for ds in individual.values()
        for u in [ds]
    )
    # Flatten individual count properly
    i_users  = sum(len(ds) for ds in individual.values())
    i_slots  = sum(
        len(horizons)
        for ds in individual.values()
        for horizons in ds.values()
    )

    print(f"\nSummary:")
    print(f"  Virtual slots:     {v_count} (across {len(virtual)} datasets × {len(HORIZONS)} horizons)")
    print(f"  Individual users:  {i_users} users, {i_slots} total slots")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    reg["virtual"]    = virtual
    reg["individual"] = individual
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

    print(f"\nRegistry written to {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
