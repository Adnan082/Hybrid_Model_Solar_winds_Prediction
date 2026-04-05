"""api/routers/predictions.py — Core prediction endpoints."""

from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException
import redis

from api.dependencies import get_redis, classify_storm, now_utc
from api.schemas import PredictionResponse, StormClassResponse, RewardRequest
from event_bus.bus import TOPICS

router = APIRouter(prefix="/api/v1/prediction", tags=["predictions"])

_FINAL_KEY  = f"latest:{TOPICS['final']}"
_FUSION_KEY = f"latest:{TOPICS['fusion']}"
_RL_KEY     = TOPICS["rl_reward"]


def _parse_prediction(raw: str | None) -> dict:
    if not raw:
        raise HTTPException(status_code=404, detail="No prediction available yet — pipeline may still be warming up.")
    return json.loads(raw)


@router.get("/latest", response_model=PredictionResponse)
def get_latest(r: redis.Redis = Depends(get_redis)):
    """Most recent blended Dst prediction from the RL agent."""
    data = _parse_prediction(r.get(_FINAL_KEY))
    storm_class, _ = classify_storm(data.get("dst_pred", 0.0))
    return {**data, "storm_class": storm_class, "timestamp": now_utc()}


@router.get("/storm-class", response_model=StormClassResponse)
def get_storm_class(r: redis.Redis = Depends(get_redis)):
    """Current storm classification based on latest Dst prediction."""
    data = _parse_prediction(r.get(_FINAL_KEY))
    dst  = data.get("dst_pred", 0.0)
    cls, desc = classify_storm(dst)
    return {
        "classification": cls,
        "dst_pred":       dst,
        "alert_level":    data.get("alert_level", "GREEN"),
        "description":    desc,
    }


@router.get("/fusion")
def get_fusion(r: redis.Redis = Depends(get_redis)):
    """Raw merged context from FusionAgent (pre-RL blend)."""
    raw = r.get(_FUSION_KEY)
    if not raw:
        raise HTTPException(status_code=404, detail="No fusion output yet.")
    return json.loads(raw)


@router.post("/reward", status_code=202)
def post_reward(body: RewardRequest, r: redis.Redis = Depends(get_redis)):
    """
    Submit an actual Dst observation so the RL agent can compute its reward
    and update its policy. Publishes to the rl.reward topic.
    """
    r.publish(_RL_KEY, json.dumps({
        "dst_actual": body.dst_actual,
        "period":     body.period,
        "timedelta":  body.timedelta,
    }))
    return {"accepted": True, "dst_actual": body.dst_actual}
