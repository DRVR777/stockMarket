"""operator-dashboard entry point.

Usage:  uv run python -m operator_dashboard
        (or via Makefile: make operator-dashboard)

Serves FastAPI at localhost:8080 with:
  - WebSocket at /ws (live event stream from all Redis channels)
  - POST /action (copy-trade approve/dismiss)
  - GET/POST /params (read/write operator params)
  - Static frontend at /
  - Telegram relay for high-severity alerts
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from operator_dashboard.config import HOST, LOG_FORMAT, LOG_LEVEL, PORT, REDIS_URL
from operator_dashboard.redis_bridge import RedisBridge
from operator_dashboard.routes import router
from operator_dashboard.telegram_relay import TelegramRelay

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("operator_dashboard")

app = FastAPI(title="ORACLE Operator Dashboard", version="0.1.0")
app.include_router(router)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup() -> None:
    _safe_url = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    logger.info("operator-dashboard starting  redis=%s  port=%d", _safe_url, PORT)

    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.bridge = RedisBridge(app.state.redis)
    app.state.telegram = TelegramRelay(app.state.redis)

    # Start Redis bridge and Telegram relay as background tasks
    asyncio.create_task(app.state.bridge.start())
    asyncio.create_task(app.state.telegram.start())


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.bridge.stop()
    await app.state.telegram.stop()
    await app.state.redis.aclose()
    logger.info("operator-dashboard stopped")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    app.state.bridge.add_client(ws)
    try:
        while True:
            # Keep connection alive; client sends pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        app.state.bridge.remove_client(ws)


@app.get("/")
async def index() -> dict:
    """Redirect to static frontend or return API info."""
    return {
        "app": "ORACLE Operator Dashboard",
        "ws": "/ws",
        "endpoints": ["/action", "/params"],
        "static": "/static/index.html",
    }


def main() -> None:
    uvicorn.run(
        "operator_dashboard.__main__:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
