"""
event_bus/bus.py
----------------
Redis pub/sub wrapper used by all agents.

Topics (in order of pipeline flow):
  solar_wind.raw   : raw 1-min solar wind readings (ingested by data fetcher)
  burton.output    : dst_burton + E_field + Q + dDst_dt (published by Agent 1)
  anomaly.output   : anomaly_score (published by Agent 2)
  ml.output        : residual_pred (published by Agent 3)
  prediction.final : final Dst prediction (published by Agent 4)
"""

import json
import redis
from loguru import logger


TOPICS = {
    "raw":       "solar_wind.raw",
    "burton":    "burton.output",
    "anomaly":   "anomaly.output",
    "ml":        "ml.output",
    "final":     "prediction.final",
}


class EventBus:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._pubsub = self._r.pubsub()
        logger.info(f"EventBus connected to redis://{host}:{port}/{db}")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, topic: str, payload: dict) -> None:
        """Serialize payload to JSON and publish to topic."""
        self._r.publish(topic, json.dumps(payload))

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(self, *topics: str):
        """Subscribe to one or more topics."""
        self._pubsub.subscribe(*topics)
        logger.info(f"Subscribed to: {topics}")

    def listen(self):
        """
        Yield deserialized message dicts from subscribed topics.
        Skips control messages (subscribe confirmations).
        """
        for raw in self._pubsub.listen():
            if raw["type"] != "message":
                continue
            try:
                yield raw["channel"], json.loads(raw["data"])
            except json.JSONDecodeError as e:
                logger.warning(f"Bad JSON on {raw['channel']}: {e}")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_latest(self, topic: str) -> dict | None:
        """
        Read the most recent value stored under a Redis key (not pub/sub).
        Agents can cache their latest output here for the API to read.
        """
        raw = self._r.get(f"latest:{topic}")
        return json.loads(raw) if raw else None

    def set_latest(self, topic: str, payload: dict) -> None:
        """Store the latest output so REST endpoints can serve it."""
        self._r.set(f"latest:{topic}", json.dumps(payload))

    def close(self):
        self._pubsub.close()
        self._r.close()
