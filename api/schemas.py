"""api/schemas.py — Pydantic response models for all API endpoints."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    period:          str
    timedelta:       str
    dst_pred:        float
    dst_burton:      float
    dst_corrector:   float
    residual_pred:   float
    anomaly_score:   float
    alert_level:     str          # GREEN | YELLOW | RED
    recon_error:     float
    w_burton:        float
    w_corrector:     float
    blend_certainty: float
    confidence:      float
    storm_phase:     int
    E_field:         float
    bz_gsm:          float
    speed:           float
    dDst_dt:         float
    rl_reward_steps: int
    storm_class:     str          # QUIET | MINOR | MODERATE | INTENSE | EXTREME
    timestamp:       Optional[str] = None


class StormClassResponse(BaseModel):
    classification: str
    dst_pred:       float
    alert_level:    str
    description:    str


class AgentStatusResponse(BaseModel):
    name:       str
    topic:      str
    is_alive:   bool
    idle_secs:  int
    status:     str               # OK | SLOW | STALLED | NO_DATA


class HistoryPoint(BaseModel):
    timestamp:     str
    dst_pred:      float
    dst_burton:    float
    dst_corrector: float
    anomaly_score: float
    alert_level:   str
    w_burton:      float
    w_corrector:   float
    bz_gsm:        float
    speed:         float


class HistoryResponse(BaseModel):
    count:  int
    points: list[HistoryPoint]


class HealthResponse(BaseModel):
    status:        str
    redis:         bool
    pipeline_live: bool
    api_version:   str = "1.0.0"


class RewardRequest(BaseModel):
    dst_actual: float
    period:     str = ""
    timedelta:  str = ""
