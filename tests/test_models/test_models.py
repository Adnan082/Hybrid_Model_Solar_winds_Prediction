"""Tests for TransformerAutoencoder and LSTMCorrector model shapes."""

import pytest
import torch
import numpy as np
from models.anomaly_autoencoder import TransformerAutoencoder, ANOMALY_FEATURES, SEQ_LEN as A_SEQ
from models.lstm_corrector      import LSTMCorrector,          FEATURE_COLS,     SEQ_LEN as C_SEQ


# ── TransformerAutoencoder ────────────────────────────────────────────��───────

@pytest.fixture
def anomaly_model():
    return TransformerAutoencoder(
        input_size=8, seq_len=60, d_model=64,
        latent_dim=32, nhead=4, num_layers=2, dropout=0.0,
    )


def test_autoencoder_output_shape(anomaly_model):
    x   = torch.randn(2, 60, 8)
    out = anomaly_model(x)
    assert out.shape == x.shape


def test_autoencoder_output_is_finite(anomaly_model):
    x   = torch.randn(4, 60, 8)
    out = anomaly_model(x)
    assert torch.isfinite(out).all()


def test_anomaly_features_count():
    assert len(ANOMALY_FEATURES) == 8


def test_anomaly_seq_len():
    assert A_SEQ == 60


def test_reconstruction_error_range():
    """Score clamp logic produces values in [0, 1]."""
    recon_threshold = 0.151
    recon_max       = 5.13
    for error in [0.0, 0.05, 0.5, 2.0, 10.0, 100.0]:
        score = float(np.clip((error - recon_threshold) / (recon_max - recon_threshold), 0, 1))
        assert 0.0 <= score <= 1.0


# ── LSTMCorrector ─────────────────────────────────────────────────────────────

@pytest.fixture
def corrector_model():
    return LSTMCorrector(input_size=17, hidden_size=64, num_layers=2, dropout=0.0)


def test_corrector_output_shape(corrector_model):
    x   = torch.randn(4, 120, 17)
    out = corrector_model(x)
    assert out.shape[0] == 4   # batch dimension preserved


def test_corrector_output_is_finite(corrector_model):
    x   = torch.randn(4, 120, 17)
    out = corrector_model(x)
    assert torch.isfinite(out).all()


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 17


def test_corrector_seq_len():
    assert C_SEQ == 120


def test_corrector_single_sample(corrector_model):
    x   = torch.randn(1, 120, 17)
    out = corrector_model(x)
    assert out.numel() == 1
    assert isinstance(out.item(), float)
