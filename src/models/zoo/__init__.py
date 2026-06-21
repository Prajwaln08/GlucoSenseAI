"""
Model zoo — all registered model classes.

Adding a new model:
    1. Create src/models/zoo/<name>_model.py implementing BaseModel.
    2. Import it here and add it to MODEL_REGISTRY with a string key.
    3. The training pipeline and selector will pick it up automatically.
"""

from src.models.zoo.lgbm_model import LightGBMModel
from src.models.zoo.xgb_model import XGBoostModel
from src.models.zoo.rf_model import RandomForestModel
from src.models.zoo.rnn_models import GRUModel, LSTMModel
from src.models.zoo.virtual_lgbm import VirtualLGBM
from src.models.zoo.virtual_xgb import VirtualXGB
from src.models.zoo.virtual_rnn import VirtualGRU, VirtualLSTM

MODEL_REGISTRY: dict[str, type] = {
    # CGM-active models (delta targets)
    "lightgbm":      LightGBMModel,
    "xgboost":       XGBoostModel,
    "random_forest": RandomForestModel,
    "gru":           GRUModel,
    "lstm":          LSTMModel,
    # Virtual (post-CGM) models (absolute targets)
    "virtual_lgbm":  VirtualLGBM,
    "virtual_xgb":   VirtualXGB,
    "virtual_gru":   VirtualGRU,
    "virtual_lstm":  VirtualLSTM,
}

VIRTUAL_MODEL_KEYS: list[str] = [
    "virtual_lgbm", "virtual_xgb", "virtual_gru", "virtual_lstm",
]


def get_model(name: str, **kwargs):
    """
    Instantiate a model by registry key.

    Args:
        name:   Registry key (e.g. "lightgbm").
        **kwargs: Passed to the model constructor (hyperparameter overrides).

    Returns:
        Ready-to-fit BaseModel instance.

    Raises:
        KeyError: if *name* is not in MODEL_REGISTRY.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "MODEL_REGISTRY", "VIRTUAL_MODEL_KEYS", "get_model",
    "LightGBMModel", "XGBoostModel", "RandomForestModel",
    "GRUModel", "LSTMModel",
    "VirtualLGBM", "VirtualXGB", "VirtualGRU", "VirtualLSTM",
]
