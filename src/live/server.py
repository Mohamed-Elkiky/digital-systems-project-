"""FastAPI live-data server — REST + WebSocket endpoints.

Launch
------
    python -m src.live.server          # default: 0.0.0.0:8001
    uvicorn src.live.server:app --port 8001

Endpoints
---------
REST
    GET  /prices              → all cached prices
    GET  /prices/{symbol}     → single symbol
    GET  /history/{symbol}    → rolling tick history (last N points)
    GET  /decisions           → recent strategy decisions
    GET  /status              → server health

    POST /strategy/models     → {"names": ["run_name", ...]}
    POST /strategy/start
    POST /strategy/stop

WebSocket
    /ws/prices                → multiplexed stream of tickers + decisions
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.live.price_cache import PriceCache
from src.live.exchange_ws import BybitWebSocket
from src.live.strategy_runner import StrategyRunner
from src.dashboard.trader import list_runs

# ── Shared singletons ────────────────────────────────────────────────────
cache = PriceCache(history_maxlen=7200)
exchange_ws = BybitWebSocket(cache)
strategy = StrategyRunner(cache)

_bg_tasks: list[asyncio.Task] = []


# ── Lifespan (startup / shutdown) ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-load all available models on startup
    runs = list_runs()
    if runs:
        names = [r["name"] for r in runs]
        strategy.load_models(names)

    # Spawn background tasks
    _bg_tasks.append(asyncio.create_task(exchange_ws.start()))
    _bg_tasks.append(asyncio.create_task(strategy.start()))

    yield

    # Shutdown
    exchange_ws.stop()
    strategy.stop()
    for t in _bg_tasks:
        t.cancel()
    _bg_tasks.clear()


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Trading Bot — Live Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST: Prices ─────────────────────────────────────────────────────────

@app.get("/prices")
def get_all_prices():
    return cache.get_all()


@app.get("/prices/{symbol}")
def get_price(symbol: str):
    data = cache.get(symbol.upper())
    if data is None:
        return {"error": f"No data for {symbol}"}
    return data


@app.get("/history/{symbol}")
def get_history(symbol: str, limit: int = 300):
    return cache.get_history(symbol.upper(), limit=limit)


# ── REST: Decisions ──────────────────────────────────────────────────────

@app.get("/decisions")
def get_decisions(limit: int = 50):
    items = list(strategy.decisions)
    return items[-limit:]


# ── REST: Status ─────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    return {
        "symbols": list(cache._latest.keys()),
        "models_loaded": len(strategy.models),
        "model_names": strategy.model_names,
        "total_decisions": len(strategy.decisions),
        "strategy_running": strategy._running,
        "ts": time.time(),
    }


# ── REST: Strategy control ───────────────────────────────────────────────

class ModelConfig(BaseModel):
    names: list[str]


@app.post("/strategy/models")
def set_strategy_models(cfg: ModelConfig):
    strategy.load_models(cfg.names)
    return {"ok": True, "loaded": len(strategy.models)}


@app.post("/strategy/start")
def start_strategy():
    if not strategy._running:
        _bg_tasks.append(asyncio.create_task(strategy.start()))
    return {"ok": True}


@app.post("/strategy/stop")
def stop_strategy():
    strategy.stop()
    return {"ok": True}


# ── WebSocket: multiplexed ticker + decision stream ──────────────────────

@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    await websocket.accept()

    price_q = cache.subscribe()
    decision_q = strategy.subscribe()
    merged: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def _pipe(src: asyncio.Queue) -> None:
        while True:
            msg = await src.get()
            await merged.put(msg)

    pipe_tasks = [
        asyncio.create_task(_pipe(price_q)),
        asyncio.create_task(_pipe(decision_q)),
    ]

    try:
        while True:
            msg = await merged.get()
            await websocket.send_json(msg)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        for t in pipe_tasks:
            t.cancel()
        cache.unsubscribe(price_q)
        strategy.unsubscribe(decision_q)


# ── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.live.server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
