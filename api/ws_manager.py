"""api/ws_manager.py — WebSocket connection manager for live streaming."""

from __future__ import annotations
import asyncio
import json
import os
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from fastapi import WebSocket
from loguru import logger

if TYPE_CHECKING:
    pass

_FINAL_KEY = "latest:prediction.final"


class ConnectionManager:
    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._active.append(ws)
        logger.info(f"[WS] Client connected. Total: {len(self._active)}")

    def disconnect(self, ws: WebSocket):
        self._active = [c for c in self._active if c is not ws]
        logger.info(f"[WS] Client disconnected. Total: {len(self._active)}")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)


manager = ConnectionManager()


async def broadcast_loop():
    """
    Background task: polls Redis for the latest prediction every second
    and broadcasts to all connected WebSocket clients.
    """
    r = aioredis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )
    logger.info("[WS] Broadcast loop started.")
    while True:
        try:
            raw = await r.get(_FINAL_KEY)
            if raw and manager.connection_count > 0:
                await manager.broadcast(json.loads(raw))
        except Exception as e:
            logger.warning(f"[WS] Broadcast error: {e}")
        await asyncio.sleep(1.0)
