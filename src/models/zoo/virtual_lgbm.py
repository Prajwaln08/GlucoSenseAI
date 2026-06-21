"""
VirtualLGBM — LightGBM variant for the post-CGM virtual glucose model.

Identical to LightGBMModel in every way except:
  requires_cgm              = False
  predicts_absolute_or_delta = "absolute"
  supported_feature_groups  = non-CGM groups only

This subclass exists so that model artefacts are self-describing: loading a
VirtualLGBM tells the inference engine it must NOT receive any CGM-derived
column. The VirtualTrainer enforces this at training time; the inference engine
enforces it at serving time.
"""

from src.models.zoo.lgbm_model import LightGBMModel


class VirtualLGBM(LightGBMModel):
    """LightGBM for Stage-B (post-CGM) absolute glucose prediction."""

    name                       = "virtual_lgbm"
    requires_cgm               = False
    predicts_absolute_or_delta = "absolute"
    supported_feature_groups   = ["watch", "food", "medicine", "time", "interaction"]
