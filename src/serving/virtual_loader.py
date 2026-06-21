"""
Thread-safe loader and in-process cache for virtual (post-CGM) model artefacts.

Complements model_loader.py, which handles CGM-active population models.

Artefact layout expected on disk (written by VirtualTrainer):
    models/virtual/<dataset>/<horizon>/<model_key>/
        model.pkl
        scaler.pkl
        feature_cols.json
        config.json        (optional — written by VirtualTrainer)
        metrics.json       (optional)

Usage
-----
    # Direct load
    vm = load_virtual_model("virtual_lgbm", "2h", "cgmacros")

    # Load from Phase 5 selection file
    vm = load_virtual_model_from_selection(
        selection_dir=Path("reports/experiment_matrix/cgmacros"),
        horizon="2h",
    )

    # Evict cache between tests
    clear_cache()
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import NamedTuple, Optional

from src.config import MODELS_DIR
from src.models.base_model import BaseModel
from src.models.zoo import MODEL_REGISTRY
from src.utils import get_logger

log = get_logger(__name__)

_VIRTUAL_BASE: Path = MODELS_DIR / "virtual"

# ── Cache ─────────────────────────────────────────────────────────────────────
# Key: (model_key, horizon, dataset, resolved_artefact_dir_str)
_cache: dict[tuple, "LoadedVirtualModel"] = {}
_lock  = threading.Lock()


# ── Result type ───────────────────────────────────────────────────────────────

class LoadedVirtualModel(NamedTuple):
    """Loaded artefacts for one virtual model slot."""
    model:        object           # VirtualLGBM / VirtualXGB / etc.
    scaler:       object           # sklearn StandardScaler
    feature_cols: list[str]
    model_key:    str              # e.g. "virtual_lgbm"
    dataset:      str
    horizon:      str
    input_window: Optional[int]   # None for tabular models


# ── Public API ─────────────────────────────────────────────────────────────────

def load_virtual_model(
    model_key:    str,
    horizon:      str,
    dataset:      str,
    artefact_dir: Optional[Path] = None,
) -> LoadedVirtualModel:
    """
    Load and cache a virtual model artefact.

    Args:
        model_key:    Registry key, e.g. "virtual_lgbm".
        horizon:      "2h" or "3h".
        dataset:      "cgmacros" or "nature_paper".
        artefact_dir: Override for the artefact directory.
                      Default: MODELS_DIR/virtual/<dataset>/<horizon>/<model_key>

    Returns:
        LoadedVirtualModel — cached; subsequent calls return the same object.

    Raises:
        FileNotFoundError: if the artefact directory or required files are absent.
        KeyError: if model_key is not registered in MODEL_REGISTRY.
    """
    resolved = Path(artefact_dir) if artefact_dir else _VIRTUAL_BASE / dataset / horizon / model_key
    cache_key = (model_key, horizon, dataset, str(resolved))

    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        _check_artefacts(resolved, model_key)

        model_cls = MODEL_REGISTRY.get(model_key)
        if model_cls is None:
            raise KeyError(
                f"model_key {model_key!r} not in MODEL_REGISTRY. "
                f"Available keys: {list(MODEL_REGISTRY)}"
            )

        model        = model_cls.load(resolved / "model.pkl")
        scaler       = BaseModel._load_pickle(resolved / "scaler.pkl")
        feature_cols = json.loads((resolved / "feature_cols.json").read_text())

        config_path = resolved / "config.json"
        input_win: Optional[int] = None
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            input_win = cfg.get("input_window")

        loaded = LoadedVirtualModel(
            model        = model,
            scaler       = scaler,
            feature_cols = feature_cols,
            model_key    = model_key,
            dataset      = dataset,
            horizon      = horizon,
            input_window = input_win,
        )
        _cache[cache_key] = loaded
        log.info(
            f"Virtual model loaded: {model_key}/{dataset}/{horizon} "
            f"({len(feature_cols)} features) — {resolved}"
        )
        return loaded


def load_virtual_model_from_selection(
    selection_dir: Path,
    horizon:       str,
    artefact_base: Optional[Path] = None,
) -> LoadedVirtualModel:
    """
    Read selected_models.json and load the virtual model for the given horizon.

    Args:
        selection_dir: Directory containing selected_models.json (Phase 5 output).
        horizon:       "2h" or "3h".
        artefact_base: Override for the virtual artefact base directory.
                       Default: MODELS_DIR/virtual.
                       The artefact dir is resolved as:
                       <artefact_base>/<dataset>/<horizon>/<model_key>

    Returns:
        LoadedVirtualModel for the winning model at this horizon.

    Raises:
        FileNotFoundError: if selected_models.json is missing.
        KeyError: if horizon is not present in the selection file.
    """
    sel_path = Path(selection_dir) / "selected_models.json"
    if not sel_path.exists():
        raise FileNotFoundError(
            f"selected_models.json not found at {selection_dir}. "
            "Run 'python scripts/select_virtual_model.py' first."
        )

    payload    = json.loads(sel_path.read_text())
    by_horizon = payload.get("by_horizon", {})

    if horizon not in by_horizon:
        raise KeyError(
            f"Horizon {horizon!r} not found in selected_models.json. "
            f"Available: {list(by_horizon)}"
        )

    entry     = by_horizon[horizon]
    model_key = entry["model_key"]
    dataset   = entry["dataset"]

    artefact_dir: Optional[Path] = None
    if artefact_base is not None:
        artefact_dir = Path(artefact_base) / dataset / horizon / model_key

    return load_virtual_model(
        model_key    = model_key,
        horizon      = horizon,
        dataset      = dataset,
        artefact_dir = artefact_dir,
    )


def clear_cache() -> None:
    """Evict all cached virtual models. Use between tests or for hot-reload."""
    with _lock:
        _cache.clear()


# ── Private ───────────────────────────────────────────────────────────────────

def _check_artefacts(d: Path, model_key: str) -> None:
    """Raise FileNotFoundError if the artefact directory or required files are missing."""
    if not d.exists():
        raise FileNotFoundError(
            f"Virtual model artefact directory not found: {d}. "
            f"Train '{model_key}' first with 'python scripts/train_virtual.py'."
        )
    for fname in ("model.pkl", "scaler.pkl", "feature_cols.json"):
        p = d / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artefact '{fname}' for {model_key!r} at {d}."
            )
