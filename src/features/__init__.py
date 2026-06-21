from .pipeline import (
    build_feature_matrix,
    build_virtual_feature_matrix,
    get_feature_cols,
    get_X_y,
    get_target_cols,
)
from .medicine_features import add_medicine_features, MEDICINE_OUTPUT_COLS
from .feature_groups import (
    CGM_FEATURES,
    WATCH_FEATURES,
    FOOD_FEATURES,
    MEDICINE_FEATURES,
    TIME_FEATURES,
    INTERACTION_FEATURES,
    NON_CGM_FEATURES,
    ALL_KNOWN_FEATURES,
    CGM_GROUP,
    WATCH_GROUP,
    FOOD_GROUP,
    MEDICINE_GROUP,
    TIME_GROUP,
    INTERACTION_GROUP,
    ALL_GROUPS,
    FeatureContract,
    FeatureGroupDef,
)
from .glucose_features import CGM_OUTPUT_COLS

__all__ = [
    # Pipeline entry points
    "build_feature_matrix",
    "build_virtual_feature_matrix",
    "get_feature_cols",
    "get_X_y",
    "get_target_cols",
    # Medicine features
    "add_medicine_features",
    "MEDICINE_OUTPUT_COLS",
    # Feature group sets
    "CGM_FEATURES",
    "WATCH_FEATURES",
    "FOOD_FEATURES",
    "MEDICINE_FEATURES",
    "TIME_FEATURES",
    "INTERACTION_FEATURES",
    "NON_CGM_FEATURES",
    "ALL_KNOWN_FEATURES",
    # Typed group definitions
    "CGM_GROUP",
    "WATCH_GROUP",
    "FOOD_GROUP",
    "MEDICINE_GROUP",
    "TIME_GROUP",
    "INTERACTION_GROUP",
    "ALL_GROUPS",
    # Classes
    "FeatureContract",
    "FeatureGroupDef",
    # Glucose module output manifest
    "CGM_OUTPUT_COLS",
]
