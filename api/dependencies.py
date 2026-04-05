"""api/dependencies.py — Shared dependencies (Redis, helpers)."""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Generator

import redis


def get_redis() -> Generator[redis.Redis, None, None]:
    """FastAPI dependency: yields a Redis client, closes on teardown."""
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )
    try:
        yield r
    finally:
        r.close()


def classify_storm(dst: float) -> tuple[str, str]:
    """Return (classification, description) based on Dst value."""
    if dst > -30:
        return "QUIET",    "Geomagnetic conditions quiet"
    elif dst > -50:
        return "MINOR",    "Minor geomagnetic storm (G1)"
    elif dst > -100:
        return "MODERATE", "Moderate geomagnetic storm (G2)"
    elif dst > -200:
        return "INTENSE",  "Intense geomagnetic storm (G3)"
    else:
        return "EXTREME",  "Extreme geomagnetic storm (G4+)"


def get_agent_status(r: redis.Redis, name: str, topic: str) -> dict:
    """
    Check an agent's health by reading its latest Redis key idle time.
    OBJECT IDLETIME returns seconds since the key was last written.
    """
    key = f"latest:{topic}"
    try:
        idle = r.object_idletime(key)
        if idle is None:
            return {"name": name, "topic": topic, "is_alive": False,
                    "idle_secs": -1, "status": "NO_DATA"}
        idle = int(idle)
        if idle < 10:
            status = "OK"
        elif idle < 30:
            status = "SLOW"
        else:
            status = "STALLED"
        return {"name": name, "topic": topic, "is_alive": idle < 30,
                "idle_secs": idle, "status": status}
    except Exception:
        return {"name": name, "topic": topic, "is_alive": False,
                "idle_secs": -1, "status": "NO_DATA"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
