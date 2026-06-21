"""
Thread-safe singleton model cache — 3-tier resolution.

Tier resolution order (first match wins):
    1. Individual  — registry["individual"][dataset][user_id][horizon]
                     Only served if the individual test_rmse < population test_rmse
                     (i.e. the personal model actually improved over the baseline).
    2. Virtual     — registry["virtual"][dataset][horizon]
                     Used when mode="post_cgm" (no live glucose signal).
    3. Population  — registry["population"][dataset][horizon]
                     Always available as final fallback.

Usage:
    # CGM-active user with an individual model
    loaded = load_model("cgmacros", "2h", user_id="019", mode="cgm_active")

    # Post-CGM user (no individual virtual model yet)
    loaded = load_model("nature_paper", "3h", mode="post_cgm")

    # Warm all population slots at startup
    warm_cache()

Each LoadedModel carries a model_tier field so callers know which tier was served.
"""

import json
import threading
from pathlib import Path
from typing import Literal, NamedTuple

from src.config import MODELS_DIR
from src.models.base_model import BaseModel
from src.models.zoo import MODEL_REGISTRY
from src.utils import get_logger

log = get_logger(__name__)

REGISTRY_PATH = MODELS_DIR / "registry.json"

ModelTier = Literal["individual", "virtual", "population"]


class LoadedModel(NamedTuple):
    model:        object        # BaseModel subclass instance
    scaler:       object        # sklearn StandardScaler
    feature_cols: list[str]
    model_type:   str
    dataset:      str
    horizon:      str
    model_tier:   ModelTier     # which tier was actually served


_cache: dict[tuple, LoadedModel] = {}
_lock = threading.Lock()


def load_model(
    dataset:  str,
    horizon:  str,
    user_id:  str | None = None,
    mode:     Literal["cgm_active", "post_cgm"] = "cgm_active",
) -> LoadedModel:
    """
    Return a cached LoadedModel, loading from disk on first call.

    Args:
        dataset:  "cgmacros" | "nature_paper"
        horizon:  "2h" | "3h"
        user_id:  Participant ID (e.g. "019"). When provided, tier 1 (individual)
                  is attempted first.
        mode:     "cgm_active"  → tiers: individual → population
                  "post_cgm"    → tiers: individual_virtual → virtual → population_virtual?
                  In post_cgm mode the virtual slot is preferred over population because
                  virtual models were trained without any glucose signal, making them the
                  correct choice when CGM is absent.
    """
    cache_key = (dataset, horizon, user_id, mode)
    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        registry = _read_registry()
        artefact_dir, model_type, tier = _resolve_slot(
            registry, dataset, horizon, user_id, mode
        )

        log.info(
            f"Loading {dataset}/{horizon} "
            f"(user={user_id}, mode={mode}, tier={tier}) "
            f"model_type={model_type} from {artefact_dir}"
        )

        model_cls = MODEL_REGISTRY.get(model_type)
        if model_cls is None:
            raise ValueError(
                f"Model type '{model_type}' not in MODEL_REGISTRY. "
                f"Available: {list(MODEL_REGISTRY)}"
            )

        model        = model_cls.load(artefact_dir / "model.pkl")
        scaler       = BaseModel._load_pickle(artefact_dir / "scaler.pkl")
        feature_cols = json.loads((artefact_dir / "feature_cols.json").read_text())

        loaded = LoadedModel(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            model_type=model_type,
            dataset=dataset,
            horizon=horizon,
            model_tier=tier,
        )
        _cache[cache_key] = loaded
        log.info(
            f"Cached {dataset}/{horizon} tier={tier} — {len(feature_cols)} features"
        )
        return loaded


def warm_cache() -> None:
    """Pre-load all 4 population slots at startup to eliminate first-request latency."""
    for dataset in ("nature_paper", "cgmacros"):
        for horizon in ("2h", "3h"):
            try:
                load_model(dataset, horizon)
            except Exception as exc:
                log.warning(f"Could not warm {dataset}/{horizon}: {exc}")


def invalidate_user_cache(dataset: str, horizon: str, user_id: str) -> None:
    """
    Drop cached entries for a specific user after their individual model is retrained.
    Called by the Celery retrain task after a successful registry update.
    """
    removed = 0
    for mode in ("cgm_active", "post_cgm"):
        key = (dataset, horizon, user_id, mode)
        if _cache.pop(key, None) is not None:
            removed += 1
    if removed:
        log.info(
            f"Invalidated {removed} cache entry/entries for "
            f"{dataset}/{horizon} user={user_id}"
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry not found at {REGISTRY_PATH}. "
            "Run 'python scripts/populate_registry.py' first."
        )
    return json.loads(REGISTRY_PATH.read_text())


def _resolve_slot(
    registry: dict,
    dataset:  str,
    horizon:  str,
    user_id:  str | None,
    mode:     str,
) -> tuple[Path, str, ModelTier]:
    """
    Walk the 3-tier hierarchy and return (artefact_dir, model_type, tier).

    Tier 1 — Individual
        Attempted when user_id is given.
        Only served if individual test_rmse < population test_rmse (proven improvement).
        If the user has no individual model yet, falls through silently.

    Tier 2 — Virtual  (post_cgm mode only)
        Used when no glucose signal is available.
        Always present once populate_registry.py has run.

    Tier 3 — Population
        Always the final fallback.
    """
    root = MODELS_DIR.parent  # project root

    pop_slot = registry.get("population", {}).get(dataset, {}).get(horizon)
    if not pop_slot:
        raise KeyError(
            f"No population model for dataset='{dataset}' horizon='{horizon}'. "
            "Run the population training scripts and populate_registry.py."
        )
    pop_rmse = pop_slot.get("test_rmse", float("inf"))

    # ── Tier 1: Individual (cgm_active only) ─────────────────────────────────
    # Individual models predict glucose deltas — they require a live CGM signal.
    # Never serve them in post_cgm mode; fall straight to Tier 2 (virtual).
    if mode == "cgm_active" and user_id:
        ind_slot = (
            registry.get("individual", {})
            .get(dataset, {})
            .get(user_id, {})
            .get(horizon)
        )
        if ind_slot:
            ind_rmse = ind_slot.get("test_rmse", float("inf"))
            if ind_rmse < pop_rmse:
                return (
                    root / ind_slot["artefact_dir"],
                    ind_slot.get("best_model_type") or ind_slot.get("model_type", "unknown"),
                    "individual",
                )
            else:
                log.debug(
                    f"Individual model for {dataset}/{user_id}/{horizon} exists but "
                    f"does not beat population (ind={ind_rmse:.2f} vs pop={pop_rmse:.2f}). "
                    "Falling through to next tier."
                )

    # ── Tier 2: Virtual (post_cgm only) ──────────────────────────────────────
    if mode == "post_cgm":
        virt_slot = registry.get("virtual", {}).get(dataset, {}).get(horizon)
        if virt_slot:
            return (
                root / virt_slot["artefact_dir"],
                virt_slot.get("best_model_type") or virt_slot.get("model_type", "unknown"),
                "virtual",
            )
        log.warning(
            f"No virtual model for {dataset}/{horizon} in registry — "
            "falling back to population. Run populate_registry.py."
        )

    # ── Tier 3: Population (always available) ────────────────────────────────
    return (
        root / pop_slot["artefact_dir"],
        pop_slot.get("best_model_type") or pop_slot.get("model_type", "unknown"),
        "population",
    )
