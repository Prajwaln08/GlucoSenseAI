"""
GlucoSense AI — Model layer.

Entry points (unified tier pipeline):
    from src.models.glucose_models import get_glucose_model, GLUCOSE_MODELS
    from src.models.tier_trainer import TierTrainer
    from src.models.evaluator import compute_metrics, evaluate_and_plot

Legacy serving path (kept until serving is migrated to the tier pipeline):
    from src.models.zoo import MODEL_REGISTRY, get_model
    from src.models.individual.trainer import IndividualTrainer
"""
