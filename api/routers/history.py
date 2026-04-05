"""api/routers/history.py — Historical prediction endpoints."""

from __future__ import annotations
import json
from fastapi import APIRouter, Depends, Query
import redis

from api.dependencies import get_redis
from api.schemas import HistoryResponse, HistoryPoint

router = APIRouter(prefix="/api/v1/history", tags=["history"])

_HIST_FINAL  = "history:prediction.final"
_HIST_FUSION = "history:fusion.output"


def _parse_history(r: redis.Redis, key: str, n: int) -> HistoryResponse:
    raw_list = r.lrange(key, 0, n - 1)
    points   = []
    for raw in raw_list:
        try:
            d = json.loads(raw)
            points.append(HistoryPoint(
                timestamp     = d.get("timestamp", ""),
                dst_pred      = float(d.get("dst_pred",      d.get("dst_burton", 0.0))),
                dst_burton    = float(d.get("dst_burton",    0.0)),
                dst_corrector = float(d.get("dst_corrector", 0.0)),
                anomaly_score = float(d.get("anomaly_score", 0.0)),
                alert_level   = d.get("alert_level", "GREEN"),
                w_burton      = float(d.get("w_burton",      0.5)),
                w_corrector   = float(d.get("w_corrector",   0.5)),
                bz_gsm        = float(d.get("bz_gsm",        0.0)),
                speed         = float(d.get("speed",       400.0)),
            ))
        except Exception:
            continue
    return HistoryResponse(count=len(points), points=points)


@router.get("/predictions", response_model=HistoryResponse)
def get_prediction_history(
    n: int = Query(default=240, ge=1, le=1440, description="Number of recent predictions (max 1440 = 24h)"),
    r: redis.Redis = Depends(get_redis),
):
    """Last N prediction outputs from the RL agent."""
    return _parse_history(r, _HIST_FINAL, n)


@router.get("/fusion", response_model=HistoryResponse)
def get_fusion_history(
    n: int = Query(default=240, ge=1, le=1440),
    r: redis.Redis = Depends(get_redis),
):
    """Last N fusion context outputs (pre-RL blend)."""
    return _parse_history(r, _HIST_FUSION, n)
