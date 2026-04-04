"""
agents/corrector_agent.py
--------------------------
Agent 3 — BiLSTM Residual Corrector (real-time inference)

Subscribes to : burton.output
Publishes to  : ml.output

For each incoming 1-minute Burton output it:
  1. Appends the reading to a 120-step rolling window
  2. Once the window is full, runs inference with the BiLSTM
  3. Computes dst_final = dst_burton + residual_pred
  4. Publishes result every minute (None while window is filling)

The scaler and model are loaded once at startup from models/.
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
from models.lstm_corrector import LSTMCorrector, FEATURE_COLS, SEQ_LEN


# ── Paths ──
MODELS_DIR   = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH   = MODELS_DIR / "corrector_model.pt"
SCALER_PATH  = MODELS_DIR / "corrector_scaler.pkl"
CONFIG_PATH  = MODELS_DIR / "corrector_config.json"

# ── Rolling feature windows for on-the-fly feature engineering ──
_BZ_WINDOW_1H  = 60    # 1-hour window for bz mean
_BZ_WINDOW_3H  = 180   # 3-hour window for bz std
_SP_WINDOW_3H  = 180   # 3-hour window for speed mean
_EF_WINDOW_6H  = 360   # 6-hour window for E_field mean
_ERR_WINDOW_3H = 3     # 3-step window for rolling Burton error (hourly)


class CorrectorAgent(BaseAgent):

    def __init__(self, bus: EventBus, device: str = "cpu"):
        super().__init__("CorrectorAgent", bus)

        # ── Load config ──
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)
        logger.info(f"[CorrectorAgent] Config loaded: {CONFIG_PATH}")

        # ── Load scaler ──
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"[CorrectorAgent] Scaler loaded: {SCALER_PATH}")

        # ── Load model ──
        self.device = device
        self.model  = LSTMCorrector(
            input_size  = self.config["input_size"],
            hidden_size = self.config["hidden_size"],
            num_layers  = self.config["num_layers"],
            dropout     = self.config["dropout"],
        ).to(device)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )
        self.model.eval()
        logger.info(f"[CorrectorAgent] Model loaded: {MODEL_PATH}  device={device}")

        # ── Rolling input window (SEQ_LEN steps × n_features) ──
        self._seq_len = self.config["seq_len"]
        self._window  = deque(maxlen=self._seq_len)

        # ── Rolling feature buffers ──
        self._bz_buf  = deque(maxlen=_BZ_WINDOW_3H)
        self._sp_buf  = deque(maxlen=_SP_WINDOW_3H)
        self._ef_buf  = deque(maxlen=_EF_WINDOW_6H)
        self._err_buf = deque(maxlen=_ERR_WINDOW_3H)  # hourly Burton errors

        # ── State ──
        self._step         = 0
        self._last_dst_act = None   # last known actual Dst (for rolling error)
        self._storm_phase  = 0      # 0=quiet 1=onset 2=main 3=recovery
        self._smoothed_ssn = 50.0   # default SSN if not provided
        self._source_enc   = 0      # 0=ACE, 1=DSCOVR

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def input_topics(self) -> list[str]:
        return [TOPICS["burton"]]

    @property
    def output_topic(self) -> str:
        return TOPICS["ml"]

    def process(self, channel: str, payload: dict) -> dict | None:
        """
        Receive Burton output, maintain rolling window, run BiLSTM inference.
        Returns None while the window is filling (first SEQ_LEN steps).
        """
        self._step += 1

        # ── Extract raw features from Burton payload ──
        bz       = float(payload.get("bz_gsm",    0.0) or 0.0)
        by       = float(payload.get("by_gsm",    0.0) or 0.0)
        bt       = float(payload.get("bt",         5.0) or 5.0)
        density  = float(payload.get("density",    5.0) or 5.0)
        speed    = float(payload.get("speed",    400.0) or 400.0)
        E_field  = float(payload.get("E_field",    0.0) or 0.0)
        Q        = float(payload.get("Q",          0.0) or 0.0)
        dst_b    = float(payload.get("dst_burton", 0.0) or 0.0)
        dDst_dt  = float(payload.get("dDst_dt",   0.0) or 0.0)
        source   = str(payload.get("source", "ace")).lower()

        # Update rolling buffers
        self._bz_buf.append(bz)
        self._sp_buf.append(speed)
        self._ef_buf.append(E_field)

        # Update source encoding
        self._source_enc = 1 if "dscovr" in source else 0

        # Update smoothed SSN if provided
        if "smoothed_ssn" in payload:
            self._smoothed_ssn = float(payload["smoothed_ssn"])

        # ── Compute rolling features ──
        bz_mean_1h  = np.mean(list(self._bz_buf)[-60:])  if len(self._bz_buf) >= 1  else bz
        bz_std_3h   = np.std(list(self._bz_buf))          if len(self._bz_buf) > 1   else 0.0
        sp_mean_3h  = np.mean(list(self._sp_buf))          if len(self._sp_buf) >= 1  else speed
        ef_mean_6h  = np.mean(list(self._ef_buf))          if len(self._ef_buf) >= 1  else E_field

        # Rolling Burton error (hourly — use last known actual Dst)
        rolling_error = 0.0
        if self._last_dst_act is not None:
            self._err_buf.append(self._last_dst_act - dst_b)
            rolling_error = float(np.mean(list(self._err_buf)))

        # Storm phase detection (simple heuristic)
        self._storm_phase = self._detect_storm_phase(dst_b, dDst_dt)

        # ── Build feature vector (must match FEATURE_COLS order) ──
        feature_vec = [
            bz,
            by,
            bt,
            density,
            speed,
            E_field,
            Q,
            dst_b,
            dDst_dt,
            rolling_error,
            bz_mean_1h,
            bz_std_3h,
            sp_mean_3h,
            ef_mean_6h,
            float(self._storm_phase),
            float(self._source_enc),
            self._smoothed_ssn,
        ]

        # ── Scale features ──
        scaled = self.scaler.transform([feature_vec])[0]

        # ── Append to rolling window ──
        self._window.append(scaled)

        # ── Wait until window is full ──
        if len(self._window) < self._seq_len:
            logger.debug(
                f"[CorrectorAgent] Filling window: {len(self._window)}/{self._seq_len}"
            )
            return None

        # ── Run BiLSTM inference ──
        x = torch.tensor(
            np.array(self._window), dtype=torch.float32
        ).unsqueeze(0).to(self.device)   # (1, seq_len, n_features)

        with torch.no_grad():
            residual_pred = self.model(x).item()

        dst_final = dst_b + residual_pred

        if self._step % 60 == 0:
            logger.info(
                f"[CorrectorAgent] step={self._step}  "
                f"dst_burton={dst_b:.2f}  residual={residual_pred:.2f}  "
                f"dst_final={dst_final:.2f} nT"
            )

        return {
            "period":        payload.get("period",    ""),
            "timedelta":     payload.get("timedelta", ""),
            "dst_burton":    round(dst_b,          4),
            "residual_pred": round(residual_pred,  4),
            "dst_final":     round(dst_final,      4),
            "storm_phase":   self._storm_phase,
            "bz_gsm":        bz,
            "speed":         speed,
            "E_field":       E_field,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_storm_phase(self, dst_burton: float, dDst_dt: float) -> int:
        """
        Simple heuristic storm phase detector.
        0=quiet, 1=onset, 2=main, 3=recovery
        """
        if dst_burton > -30:
            return 0   # quiet
        elif dDst_dt < -2.0:
            return 2   # main phase (rapid deepening)
        elif dst_burton < -30 and dDst_dt > 0.5:
            return 3   # recovery (Dst rising back)
        else:
            return 1   # onset

    def update_actual_dst(self, dst_actual: float):
        """
        Called when a real hourly Dst observation arrives.
        Updates the rolling error buffer for the next prediction.
        """
        self._last_dst_act = dst_actual

    def reset(self):
        """Reset all state (call between storm periods)."""
        self._window.clear()
        self._bz_buf.clear()
        self._sp_buf.clear()
        self._ef_buf.clear()
        self._err_buf.clear()
        self._step         = 0
        self._last_dst_act = None
        self._storm_phase  = 0
        logger.info("[CorrectorAgent] State reset.")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os
    bus = EventBus(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    agent = CorrectorAgent(bus)
    agent.run()
