"""
agents/anomaly_agent.py
-----------------------
Agent 2 — Transformer Autoencoder Anomaly Detector (real-time inference)

Subscribes to : burton.output
Publishes to  : anomaly.output

For each incoming 1-minute Burton output it:
  1. Appends the reading to a 60-step rolling window
  2. Once the window is full, runs inference with the Transformer Autoencoder
  3. Computes reconstruction error → normalized anomaly score (0–1)
  4. Determines alert level: GREEN / YELLOW / RED
  5. Publishes result every minute (None while window is filling)

Alert levels:
    GREEN  : score < 0.40  — quiet, Burton reliable
    YELLOW : score 0.40–0.80 — onset likely, monitor
    RED    : score > 0.80  — storm confirmed

The scaler, model, and config are loaded once at startup from models/.
"""

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from agents.base_agent import BaseAgent
from event_bus.bus import EventBus, TOPICS
from models.anomaly_autoencoder import TransformerAutoencoder, ANOMALY_FEATURES, SEQ_LEN


# ── Paths ──
MODELS_DIR   = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH   = MODELS_DIR / "anomaly_model.pt"
SCALER_PATH  = MODELS_DIR / "anomaly_scaler.pkl"
CONFIG_PATH  = MODELS_DIR / "anomaly_config.json"


class AnomalyAgent(BaseAgent):

    def __init__(self, bus: EventBus, device: str = "cpu"):
        super().__init__("AnomalyAgent", bus)

        # ── Load config ──
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)
        logger.info(f"[AnomalyAgent] Config loaded: {CONFIG_PATH}")

        # ── Load scaler ──
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"[AnomalyAgent] Scaler loaded: {SCALER_PATH}")

        # ── Load model ──
        self.device = device
        self.model  = TransformerAutoencoder(
            input_size = self.config["input_size"],
            seq_len    = self.config["seq_len"],
            d_model    = self.config["d_model"],
            latent_dim = self.config["latent_dim"],
            nhead      = self.config["nhead"],
            num_layers = self.config["num_layers"],
            dropout    = self.config["dropout"],
        ).to(device)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )
        self.model.eval()
        logger.info(f"[AnomalyAgent] Model loaded: {MODEL_PATH}  device={device}")

        # ── Calibration thresholds ──
        self._recon_threshold = float(self.config["recon_threshold"])  # quiet p99
        self._recon_max       = float(self.config["recon_max"])        # extreme p50
        self._alert_green     = float(self.config["alert_green"])      # 0.40
        self._alert_yellow    = float(self.config["alert_yellow"])     # 0.80

        # ── Rolling input window (SEQ_LEN steps × n_features) ──
        self._seq_len = self.config["seq_len"]
        self._window  = deque(maxlen=self._seq_len)

        # ── State ──
        self._step          = 0
        self._smoothed_ssn  = 50.0   # default if not provided

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def input_topics(self) -> list[str]:
        return [TOPICS["burton"]]

    @property
    def output_topic(self) -> str:
        return TOPICS["anomaly"]

    def process(self, channel: str, payload: dict) -> dict | None:
        """
        Receive Burton output, maintain rolling window, run Transformer inference.
        Returns None while the window is filling (first SEQ_LEN steps).
        """
        self._step += 1

        # ── Extract features matching ANOMALY_FEATURES order ──
        bz      = float(payload.get("bz_gsm",   0.0) or 0.0)
        by      = float(payload.get("by_gsm",   0.0) or 0.0)
        bt      = float(payload.get("bt",        5.0) or 5.0)
        speed   = float(payload.get("speed",   400.0) or 400.0)
        density = float(payload.get("density",   5.0) or 5.0)
        E_field = float(payload.get("E_field",   0.0) or 0.0)
        dDst_dt = float(payload.get("dDst_dt",   0.0) or 0.0)

        if "smoothed_ssn" in payload:
            self._smoothed_ssn = float(payload["smoothed_ssn"])

        feature_vec = [bz, by, bt, speed, density, E_field, dDst_dt, self._smoothed_ssn]

        # ── Scale features ──
        scaled = self.scaler.transform([feature_vec])[0]

        # ── Append to rolling window ──
        self._window.append(scaled)

        # ── Wait until window is full ──
        if len(self._window) < self._seq_len:
            logger.debug(
                f"[AnomalyAgent] Filling window: {len(self._window)}/{self._seq_len}"
            )
            return None

        # ── Run Transformer Autoencoder inference ──
        x = torch.tensor(
            np.array(self._window), dtype=torch.float32
        ).unsqueeze(0).to(self.device)   # (1, seq_len, n_features)

        with torch.no_grad():
            recon = self.model(x)
            recon_error = float(((x - recon) ** 2).mean().item())

        # ── Normalize to 0–1 anomaly score ──
        anomaly_score = float(np.clip(
            (recon_error - self._recon_threshold) / (self._recon_max - self._recon_threshold),
            0.0, 1.0
        ))

        # ── Determine alert level ──
        if anomaly_score < self._alert_green:
            alert_level = "GREEN"
        elif anomaly_score < self._alert_yellow:
            alert_level = "YELLOW"
        else:
            alert_level = "RED"

        if self._step % 60 == 0:
            logger.info(
                f"[AnomalyAgent] step={self._step}  "
                f"recon_error={recon_error:.4f}  "
                f"anomaly_score={anomaly_score:.3f}  "
                f"alert={alert_level}"
            )

        return {
            "period":        payload.get("period",    ""),
            "timedelta":     payload.get("timedelta", ""),
            "recon_error":   round(recon_error,   6),
            "anomaly_score": round(anomaly_score, 4),
            "alert_level":   alert_level,
            "dst_burton":    payload.get("dst_burton", 0.0),
            "bz_gsm":        bz,
            "speed":         speed,
            "E_field":       E_field,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all state (call between storm periods)."""
        self._window.clear()
        self._step = 0
        logger.info("[AnomalyAgent] State reset.")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os
    bus = EventBus(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    agent = AnomalyAgent(bus)
    agent.run()
