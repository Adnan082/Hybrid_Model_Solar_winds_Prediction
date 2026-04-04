"""
models/lstm_corrector.py
------------------------
BiLSTM Residual Corrector — Agent 3 architecture.

Predicts:
    residual = dst_actual - dst_burton

Final prediction:
    dst_final = dst_burton + residual_pred

Trained with:
    - SEQ_LEN=120 (2 hours of 1-min solar wind data)
    - 17 input features
    - Physics-constrained augmentation on intense/extreme storms
    - Storm-weighted MSE loss (extreme=200x)
    - Stratified random 80/20 split

Results (val set):
    overall  : 5.81 nT RMSE  (68.3% skill over Burton)
    intense  : 13.71 nT RMSE (82.6% skill)
    extreme  :  6.50 nT RMSE (92.6% skill)
"""

import torch
import torch.nn as nn

# ── Feature columns — must match training notebook Cell 4 ──
FEATURE_COLS = [
    "bz_gsm",
    "by_gsm",
    "bt",
    "density",
    "speed",
    "E_field",
    "Q",
    "dst_burton",
    "dDst_dt",
    "rolling_error_3h",
    "bz_gsm_mean_1h",
    "bz_gsm_std_3h",
    "speed_mean_3h",
    "E_field_mean_6h",
    "storm_phase",
    "source_encoded",
    "smoothed_ssn",
]

TARGET_COL = "residual"
SEQ_LEN    = 120   # 2 hours of 1-minute data


class LSTMCorrector(nn.Module):
    """
    Bidirectional LSTM that predicts the Burton residual correction.

    Input : (batch, seq_len, input_size)  — scaled feature window
    Output: (batch,)                      — residual prediction (nT)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.4,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size * 2, 1)   # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]       # last timestep: (batch, hidden*2)
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def load_model(model_path: str, config: dict, device: str = "cpu") -> LSTMCorrector:
    """Load a trained LSTMCorrector from a .pt checkpoint."""
    model = LSTMCorrector(
        input_size  = config["input_size"],
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        dropout     = config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
