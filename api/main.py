"""
api/main.py
-----------
FastAPI application for the Solar Wind Dst Prediction System.

Endpoints
---------
  GET  /health                          system health
  GET  /api/v1/prediction/latest        latest RL-blended prediction
  GET  /api/v1/prediction/storm-class   current storm classification
  GET  /api/v1/prediction/fusion        raw fusion context (pre-RL)
  POST /api/v1/prediction/reward        submit actual Dst for RL training
  GET  /api/v1/history/predictions?n=   last N predictions
  GET  /api/v1/history/fusion?n=        last N fusion outputs
  GET  /api/v1/agents/status            all 5 agent health statuses
  WS   /ws/live                         real-time prediction stream
  GET  /metrics                         Prometheus metrics

Run
---
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import redis

from api.dependencies import get_redis, now_utc
from api.routers import predictions, history, agents
from api.ws_manager import manager, broadcast_loop
from api.schemas import HealthResponse

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "solar_api_requests_total", "Total API requests", ["route"]
)
DST_GAUGE = Gauge(
    "solar_dst_prediction_nt", "Latest Dst prediction in nanoTesla"
)
ANOMALY_GAUGE = Gauge(
    "solar_anomaly_score", "Latest anomaly score (0-1)"
)
WS_CONNECTIONS = Gauge(
    "solar_ws_connections", "Active WebSocket connections"
)


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(broadcast_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Solar Wind Dst Prediction API",
    description="Real-time Dst prediction powered by a 5-agent ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(history.router)
app.include_router(agents.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(r: redis.Redis = Depends(get_redis)):
    REQUEST_COUNT.labels(route="/health").inc()
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    pipeline_live = r.exists("latest:prediction.final") if redis_ok else False

    return {
        "status":        "ok" if redis_ok else "degraded",
        "redis":         redis_ok,
        "pipeline_live": bool(pipeline_live),
        "api_version":   "1.0.0",
    }


# ── Prometheus metrics endpoint ───────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
def metrics(r: redis.Redis = Depends(get_redis)):
    try:
        raw = r.get("latest:prediction.final")
        if raw:
            data = json.loads(raw)
            DST_GAUGE.set(data.get("dst_pred", 0.0))
            ANOMALY_GAUGE.set(data.get("anomaly_score", 0.0))
    except Exception:
        pass
    WS_CONNECTIONS.set(manager.connection_count)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── WebSocket live stream ─────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await manager.connect(ws)
    WS_CONNECTIONS.set(manager.connection_count)
    try:
        while True:
            await ws.receive_text()   # keep-alive: client can send pings
    except WebSocketDisconnect:
        manager.disconnect(ws)
        WS_CONNECTIONS.set(manager.connection_count)
