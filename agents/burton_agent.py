"""
agents/burton_agent.py
----------------------
Agent 1 — Burton Physics Solver

Subscribes to : solar_wind.raw
Publishes to  : burton.output

For each incoming 1-minute solar wind reading it:
  1. Appends the reading to an internal state buffer
  2. Runs one step of the Burton ODE using the previous Dst estimate
  3. Publishes dst_burton + derived physics quantities

The ODE is solved incrementally (online), one step at a time,
so it works in real-time without needing the full history upfront.

Burton ODE:
    E        = -V * Bz * 1e-3          (mV/m)
    Q(E)     = -4.4*(E - 0.49)  if E > 0.49, else 0
    dDst/dt  = Q - Dst/tau             (nT/hr)
    Dst[t+1] = Dst[t] + dDst * dt      (Euler step)
"""

from agents.base_agent import BaseAgent
from event_bus.bus import EventBus, TOPICS
from loguru import logger
import numpy as np


TAU_HOURS   = 7.7    # ring current decay time constant
DT_MINUTES  = 1.0   # cadence of solar wind data


class BurtonAgent(BaseAgent):

    def __init__(self, bus: EventBus):
        super().__init__("BurtonAgent", bus)
        self._dst  = 0.0   # running Dst estimate (nT)
        self._step = 0     # number of messages processed

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def input_topics(self) -> list[str]:
        return [TOPICS["raw"]]

    @property
    def output_topic(self) -> str:
        return TOPICS["burton"]

    def process(self, channel: str, payload: dict) -> dict | None:
        """
        Receive one solar wind reading, advance Burton ODE by one step,
        return enriched physics dict.

        Expected payload keys:
            period     : str   (e.g. "train_a")
            timedelta  : str   (ISO-format timedelta string)
            bz_gsm     : float (nT)
            speed      : float (km/s)
            by_gsm     : float (optional)
            bt         : float (optional)
            density    : float (optional)
            temperature: float (optional)
            source     : str   (optional, "ace" or "dscovr")
        """
        bz    = float(payload.get("bz_gsm",  0.0) or 0.0)
        speed = float(payload.get("speed",  400.0) or 400.0)

        dt_hours = DT_MINUTES / 60.0

        # Burton physics
        E    = -speed * bz * 1e-3                             # mV/m
        Q    = -4.4 * (E - 0.49) if E > 0.49 else 0.0        # nT/hr
        dDst = Q - self._dst / TAU_HOURS                      # nT/hr

        # Euler integration
        dst_prev  = self._dst
        self._dst = self._dst + dDst * dt_hours

        self._step += 1
        if self._step % 1440 == 0:   # log once per simulated day
            logger.info(f"[BurtonAgent] step={self._step}  "
                        f"dst_burton={self._dst:.2f} nT  E={E:.3f} mV/m")

        result = {
            "period":     payload.get("period",    ""),
            "timedelta":  payload.get("timedelta", ""),
            "dst_burton": round(float(self._dst),   4),
            "dst_prev":   round(float(dst_prev),    4),
            "E_field":    round(float(E),            4),
            "Q":          round(float(Q),            4),
            "dDst_dt":    round(float(dDst),         4),
            # pass-through raw features for downstream agents
            "bz_gsm":     bz,
            "speed":      speed,
            "by_gsm":     payload.get("by_gsm",      0.0),
            "bt":         payload.get("bt",           0.0),
            "density":    payload.get("density",      5.0),
            "temperature":payload.get("temperature",  1e5),
            "source":     payload.get("source",       ""),
        }
        return result

    # ------------------------------------------------------------------
    # Reset (useful between periods / real-time restarts)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset Dst state to 0 (call between storm periods if needed)."""
        self._dst  = 0.0
        self._step = 0
        logger.info("[BurtonAgent] State reset.")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os
    bus = EventBus(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    agent = BurtonAgent(bus)
    agent.run()
