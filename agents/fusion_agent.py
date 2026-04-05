"""
agents/fusion_agent.py
----------------------
Agent 4 — Context Fusion Agent

Subscribes to : burton.output   (Agent 1)
                anomaly.output  (Agent 2)
                ml.output       (Agent 3)

Publishes to  : fusion.output

Role
----
Waits for all three upstream agents to have published at least once,
then every time the corrector (slowest, 120-step warm-up) publishes,
merges the latest payload from each into a single unified context dict
and forwards it to fusion.output for the RL agent (Agent 5) to consume.

This agent contains NO model weights — it is pure data merging.
The RL blending decision is made downstream by RLBusAgent.
"""

from __future__ import annotations

import os
from loguru import logger

from agents.base_agent import BaseAgent
from event_bus.bus import EventBus, TOPICS


_ALL_UPSTREAM = [TOPICS["burton"], TOPICS["anomaly"], TOPICS["ml"]]

_SRC_BURTON  = TOPICS["burton"]
_SRC_ANOMALY = TOPICS["anomaly"]
_SRC_ML      = TOPICS["ml"]


class FusionAgent(BaseAgent):
    """
    Agent 4: merges outputs from Burton, Anomaly, and Corrector agents
    into a single context payload and publishes to fusion.output.
    """

    def __init__(self, bus: EventBus):
        super().__init__("FusionAgent", bus)

        self._cache: dict[str, dict | None] = {
            _SRC_BURTON:  None,
            _SRC_ANOMALY: None,
            _SRC_ML:      None,
        }
        self._step = 0
        logger.info("[FusionAgent] Initialised.")

    @property
    def input_topics(self) -> list[str]:
        return _ALL_UPSTREAM

    @property
    def output_topic(self) -> str:
        return TOPICS["fusion"]

    def process(self, channel: str, payload: dict) -> dict | None:
        # Update cache for whichever agent just published
        if channel in self._cache:
            self._cache[channel] = payload

        # Wait until all three caches have at least one reading
        if any(v is None for v in self._cache.values()):
            return None

        # Only fire when the corrector (slowest) publishes
        if channel != _SRC_ML:
            return None

        return self._merge()

    def _merge(self) -> dict:
        self._step += 1

        burton  = self._cache[_SRC_BURTON]
        anomaly = self._cache[_SRC_ANOMALY]
        ml      = self._cache[_SRC_ML]

        fused = {
            "period":        burton.get("period",    ""),
            "timedelta":     burton.get("timedelta", ""),
            # Agent 1 — physics
            "dst_burton":    burton.get("dst_burton",  0.0),
            "E_field":       burton.get("E_field",     0.0),
            "Q":             burton.get("Q",           0.0),
            "dDst_dt":       burton.get("dDst_dt",     0.0),
            "bz_gsm":        burton.get("bz_gsm",      0.0),
            "speed":         burton.get("speed",     400.0),
            # Agent 2 — anomaly
            "anomaly_score": anomaly.get("anomaly_score", 0.0),
            "alert_level":   anomaly.get("alert_level",   "GREEN"),
            "recon_error":   anomaly.get("recon_error",   0.0),
            # Agent 3 — ML corrector
            "dst_corrector": ml.get("dst_final",     burton.get("dst_burton", 0.0)),
            "residual_pred": ml.get("residual_pred", 0.0),
            "storm_phase":   ml.get("storm_phase",   0),
        }

        if self._step % 60 == 0:
            logger.info(
                f"[FusionAgent] step={self._step}  "
                f"dst_burton={fused['dst_burton']:.2f}  "
                f"dst_corrector={fused['dst_corrector']:.2f}  "
                f"anomaly={fused['anomaly_score']:.3f}  "
                f"alert={fused['alert_level']}"
            )

        return fused

    def reset(self) -> None:
        self._cache = {k: None for k in self._cache}
        self._step = 0
        logger.info("[FusionAgent] State reset.")


if __name__ == "__main__":
    bus = EventBus(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    FusionAgent(bus).run()
