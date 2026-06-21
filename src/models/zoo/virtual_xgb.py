"""
VirtualXGB — XGBoost variant for the post-CGM virtual glucose model.

See virtual_lgbm.py for design rationale.
"""

from src.models.zoo.xgb_model import XGBoostModel


class VirtualXGB(XGBoostModel):
    """XGBoost for Stage-B (post-CGM) absolute glucose prediction."""

    name                       = "virtual_xgb"
    requires_cgm               = False
    predicts_absolute_or_delta = "absolute"
    supported_feature_groups   = ["watch", "food", "medicine", "time", "interaction"]
