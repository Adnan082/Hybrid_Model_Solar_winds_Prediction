"""
models/anomaly_autoencoder.py
------------------------------
Transformer Autoencoder — Agent 2 architecture.

Trained on quiet-time solar wind only (dst > -30 nT).
High reconstruction error = anomaly = storm onset signal.

Anomaly score (0–1):
    score = clip((recon_error - recon_threshold) / (recon_max - recon_threshold), 0, 1)

Alert levels:
    GREEN  : score < 0.40  — quiet, Burton reliable
    YELLOW : score 0.40–0.80 — onset likely, monitor
    RED    : score > 0.80  — storm confirmed

Trained with:
    - SEQ_LEN=60 (1 hour of 1-min solar wind data)
    - 8 input features
    - Quiet-time only training (dst > -30)
    - Random 80/20 split

Results (calibration set):
    quiet    reconstruction error : mean=0.029
    intense  reconstruction error : mean=0.448  (15x quiet)
    extreme  reconstruction error : mean=6.903  (237x quiet)
    extreme detection rate        : 100%
"""

import math
import torch
import torch.nn as nn

# ── Feature columns — must match training notebook Cell 4 ──
ANOMALY_FEATURES = [
    "bz_gsm",
    "by_gsm",
    "bt",
    "speed",
    "density",
    "E_field",
    "dDst_dt",
    "smoothed_ssn",
]

SEQ_LEN    = 60    # 1 hour of 1-minute data
LATENT_DIM = 32    # bottleneck size


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding injected before the Transformer layers.
    Input / output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder for solar wind anomaly detection.

    Encoder:
        input_proj  → PositionalEncoding → TransformerEncoder → to_latent
    Decoder:
        from_latent → TransformerDecoder (queries memory) → output_proj

    Input : (batch, seq_len, input_size)
    Output: (batch, seq_len, input_size)  — reconstructed sequence
    """

    def __init__(
        self,
        input_size: int,
        seq_len:    int   = 60,
        d_model:    int   = 64,
        latent_dim: int   = 32,
        nhead:      int   = 4,
        num_layers: int   = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.d_model    = d_model
        self.latent_dim = latent_dim

        # ── Input projection ──
        self.input_proj = nn.Linear(input_size, d_model)

        # ── Positional encoding ──
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        # ── Encoder ──
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Bottleneck ──
        self.to_latent   = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)

        # ── Decoder ──
        dec_layer    = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # ── Output projection ──
        self.output_proj = nn.Linear(d_model, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → latent: (batch, seq_len, latent_dim)"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.to_latent(x)          # (B, T, latent_dim)

    def decode(self, z: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """z: (B, T, latent_dim), memory: (B, T, d_model) → (B, T, input_size)"""
        z     = self.from_latent(z)       # (B, T, d_model)
        out   = self.decoder(z, memory)   # (B, T, d_model)
        return self.output_proj(out)      # (B, T, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass. Returns reconstructed sequence."""
        proj   = self.input_proj(x)
        memory = self.encoder(self.pos_enc(proj))   # (B, T, d_model)
        latent = self.to_latent(memory)             # (B, T, latent_dim)
        recon  = self.decode(latent, memory)        # (B, T, input_size)
        return recon


def load_model(
    model_path: str,
    config:     dict,
    device:     str = "cpu",
) -> TransformerAutoencoder:
    """Load a trained TransformerAutoencoder from a .pt checkpoint."""
    model = TransformerAutoencoder(
        input_size = config["input_size"],
        seq_len    = config["seq_len"],
        d_model    = config["d_model"],
        latent_dim = config["latent_dim"],
        nhead      = config["nhead"],
        num_layers = config["num_layers"],
        dropout    = config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
