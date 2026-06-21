"""
GRU and LSTM models for GlucoSense AI.

Both use a sliding-window approach: to predict at time t, the model sees
the last seq_len=24 consecutive feature vectors (6 h of context at 15-min
sampling) and outputs n_steps future glucose values (8 for 2h, 12 for 3h).

Architecture: 2-layer GRU/LSTM → LayerNorm on final hidden state → linear head.
Training: AdamW + MSE + gradient clipping + early stopping on val loss.

Why sliding window instead of single-row features?
  The tabular models already capture lag structure via engineered columns.
  RNNs learn temporal dependencies across the full sequence — they see HOW
  glucose evolved, not just snapshots — which complements the tabular approach.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils import get_logger

log = get_logger(__name__)


# ── PyTorch network ───────────────────────────────────────────────────────────
# Defined at module level so pickle/torch.save can resolve it by qualified name.

class _RNNNet:
    """Built on first torch import; replaced by the real nn.Module subclass."""
    pass


def _build_rnn_net(cell: str, n_features: int, hidden: int,
                   layers: int, n_steps: int, dropout: float) -> "_RNNNet":
    """Construct and return a GRU or LSTM nn.Module."""
    import torch.nn as nn

    class _RNNNet(nn.Module):  # shadows module-level placeholder after torch is loaded
        def __init__(self):
            super().__init__()
            RNN = nn.GRU if cell == "GRU" else nn.LSTM
            self.rnn  = RNN(n_features, hidden, layers,
                            dropout=dropout if layers > 1 else 0.0,
                            batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, n_steps)

        def forward(self, x):          # x: (B, T, n_features)
            out, _ = self.rnn(x)
            return self.head(self.norm(out[:, -1, :]))   # (B, n_steps)

    import src.models.zoo.rnn_models as _m
    _m._RNNNet = _RNNNet          # patch module-level name so torch.save can find it
    return _RNNNet()


# ── Shared base ───────────────────────────────────────────────────────────────

class _RNNBase(BaseModel):
    """
    Sliding-window RNN base shared by GRU and LSTM.

    For each row i in X, the model receives X[i-seq_len+1 : i+1] as its
    temporal context.  At inference time the last seq_len-1 rows of X_train
    are prepended so every row in X_val / X_test has a full context window.
    """

    _CELL: str = "GRU"   # overridden by subclasses

    _DEFAULTS: dict = {
        "hidden_size":  128,
        "num_layers":   2,
        "dropout":      0.2,
        "seq_len":      24,       # 24 × 15 min = 6 h lookback window
        "batch_size":   256,
        "max_epochs":   150,
        "patience":     15,
        "lr":           1e-3,
        "weight_decay": 1e-4,
    }

    def __init__(self, **kwargs):
        self._params     = {**self._DEFAULTS, **kwargs}
        self._net        = None
        self._ctx: Optional[pd.DataFrame] = None   # trailing X_train rows for context
        self._net_config: Optional[dict]  = None   # architecture kwargs for rebuild on load

    # ── Sequence construction ─────────────────────────────────────────────────

    def _windows(self, X: pd.DataFrame) -> np.ndarray:
        """
        Slide seq_len window over X.
        Returns shape (len(X) - seq_len + 1, seq_len, n_features).
        """
        seq = self._params["seq_len"]
        Xv  = X.values.astype(np.float32)
        return np.stack([Xv[i : i + seq] for i in range(len(Xv) - seq + 1)])

    def _windows_with_ctx(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepend training-tail context so every row in X gets a full window.
        Returns shape (len(X), seq_len, n_features).
        """
        full = pd.concat([self._ctx, X]) if self._ctx is not None else X
        return self._windows(full)

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "_RNNBase":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        p       = self._params
        seq_len = p["seq_len"]
        n_feat  = X_train.shape[1]
        y_df    = y_train if isinstance(y_train, pd.DataFrame) else pd.DataFrame(y_train)
        n_steps = y_df.shape[1]

        # Store architecture config so save/load can rebuild the network
        self._net_config = dict(
            cell=self._CELL, n_features=n_feat, hidden=p["hidden_size"],
            layers=p["num_layers"], n_steps=n_steps, dropout=p["dropout"],
        )

        # Save training tail — used to give val/test rows a full context window
        self._ctx = X_train.iloc[-(seq_len - 1):].copy()

        # Build training sequences; targets aligned to the last row of each window
        X_tr = self._windows(X_train)                           # (n-seq+1, T, F)
        y_tr = y_df.values.astype(np.float32)[seq_len - 1:]    # (n-seq+1, n_steps)

        self._net = _build_rnn_net(**self._net_config)
        opt     = torch.optim.AdamW(self._net.parameters(),
                                     lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.MSELoss()
        loader  = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=p["batch_size"], shuffle=True,
        )

        # Validation tensors (context-padded so len(X_val) predictions are produced)
        has_val = X_val is not None and y_val is not None
        if has_val:
            y_val_df = y_val if isinstance(y_val, pd.DataFrame) else pd.DataFrame(y_val)
            X_va_t   = torch.from_numpy(self._windows_with_ctx(X_val))
            y_va_t   = torch.from_numpy(y_val_df.values.astype(np.float32))

        best_val, patience_cnt, best_state = float("inf"), 0, None

        for epoch in range(p["max_epochs"]):
            self._net.train()
            for Xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self._net(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self._net(X_va_t), y_va_t).item()
                if val_loss < best_val:
                    best_val     = val_loss
                    best_state   = {k: v.cpu().clone()
                                    for k, v in self._net.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= p["patience"]:
                        log.debug(f"{self.name} early stop at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        log.debug(f"{self.name} trained — {epoch + 1} epochs, best_val_loss={best_val:.5f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self._net.eval()
        X_seq = self._windows_with_ctx(X)                           # (len(X), T, F)
        with torch.no_grad():
            return self._net(torch.from_numpy(X_seq)).numpy()       # (len(X), n_steps)

    def get_params(self) -> dict:
        return {**self._params, "cell": self._CELL}

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "params":     self._params,
            "cell":       self._CELL,
            "ctx":        self._ctx,
            "net_config": self._net_config,
            "net_state":  self._net.state_dict() if self._net is not None else None,
        }, path)
        log.debug(f"{self.name} saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "_RNNBase":
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**payload["params"])
        obj._ctx        = payload["ctx"]
        obj._net_config = payload["net_config"]
        if payload["net_state"] is not None:
            obj._net = _build_rnn_net(**payload["net_config"])
            obj._net.load_state_dict(payload["net_state"])
            obj._net.eval()
        log.debug(f"{cls.__name__} loaded ← {path}")
        return obj

    def get_search_space(self, trial) -> dict:
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
            "seq_len":     trial.suggest_categorical("seq_len", [12, 24, 36]),
            "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }

    @property
    def feature_importances_(self):
        return None   # RNNs have no scalar per-feature importance


# ── Concrete models ───────────────────────────────────────────────────────────

class GRUModel(_RNNBase):
    """Gated Recurrent Unit — faster to train than LSTM, comparable accuracy."""
    name  = "gru"
    _CELL = "GRU"


class LSTMModel(_RNNBase):
    """Long Short-Term Memory — classic deep sequence baseline."""
    name  = "lstm"
    _CELL = "LSTM"
