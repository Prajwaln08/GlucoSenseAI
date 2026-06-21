"""
VirtualGRU / VirtualLSTM — RNN variants for the post-CGM virtual glucose model.

Both use a 24-step (6 h) lookback window of non-CGM features.  The CGM-active
RNNs (GRUModel / LSTMModel) look back at glucose lags embedded in the sequence;
these virtual variants see only watch/food/medicine/time features — no glucose
history — which matches the post-CGM serving context.

See virtual_lgbm.py for design rationale.
"""

from src.models.zoo.rnn_models import GRUModel, LSTMModel


class VirtualGRU(GRUModel):
    """GRU for Stage-B (post-CGM) absolute glucose prediction."""

    name                       = "virtual_gru"
    requires_cgm               = False
    predicts_absolute_or_delta = "absolute"
    supported_feature_groups   = ["watch", "food", "medicine", "time", "interaction"]


class VirtualLSTM(LSTMModel):
    """LSTM for Stage-B (post-CGM) absolute glucose prediction."""

    name                       = "virtual_lstm"
    requires_cgm               = False
    predicts_absolute_or_delta = "absolute"
    supported_feature_groups   = ["watch", "food", "medicine", "time", "interaction"]
