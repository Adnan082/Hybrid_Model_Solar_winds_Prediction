"""api/routers/agents.py — Agent health/status endpoints."""

from __future__ import annotations
from fastapi import APIRouter, Depends
import redis

from api.dependencies import get_redis, get_agent_status
from api.schemas import AgentStatusResponse
from event_bus.bus import TOPICS

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

_AGENT_MAP = [
    ("BurtonAgent",    TOPICS["burton"]),
    ("AnomalyAgent",   TOPICS["anomaly"]),
    ("CorrectorAgent", TOPICS["ml"]),
    ("FusionAgent",    TOPICS["fusion"]),
    ("RLBusAgent",     TOPICS["final"]),
]


@router.get("/status", response_model=list[AgentStatusResponse])
def get_agents_status(r: redis.Redis = Depends(get_redis)):
    """Health status of all 5 pipeline agents."""
    return [get_agent_status(r, name, topic) for name, topic in _AGENT_MAP]
