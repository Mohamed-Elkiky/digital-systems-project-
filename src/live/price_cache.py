"""In-memory price cache — single source of truth for live market data.

Stores the latest tick per symbol and a rolling history buffer.
Supports async subscriber queues for push-based WebSocket broadcasting.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Tick:
    """A single price tick from the exchange."""

    symbol: str
    last: float
    bid: float
    ask: float
    high_24h: float = 0.0
    low_24h: float = 0.0
    volume_24h: float = 0.0
    ts: float = 0.0  # unix timestamp in seconds

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "last": self.last,
            "bid": self.bid,
            "ask": self.ask,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "volume_24h": self.volume_24h,
            "ts": self.ts,
        }


class PriceCache:
    """Thread-safe (via asyncio.Lock) in-memory cache for live prices.

    Features
    --------
    - Latest tick per symbol (dict keyed by symbol).
    - Rolling history per symbol (deque of dicts, default 7200 entries).
    - Pub/sub: call ``subscribe()`` to get an ``asyncio.Queue`` that
      receives every ticker update.
    """

    def __init__(self, history_maxlen: int = 7200):
        self._latest: dict[str, Tick] = {}
        self._history: dict[str, deque] = {}
        self._maxlen = history_maxlen
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def update(self, tick: Tick) -> None:
        """Store a new tick and notify all subscribers."""
        async with self._lock:
            self._latest[tick.symbol] = tick
            if tick.symbol not in self._history:
                self._history[tick.symbol] = deque(maxlen=self._maxlen)
            self._history[tick.symbol].append(
                {"ts": tick.ts, "price": tick.last, "bid": tick.bid, "ask": tick.ask}
            )

        # Push to subscriber queues (drop if full)
        msg = {"type": "ticker", **tick.to_dict()}
        dead: list[int] = []
        for i, q in enumerate(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(i)
        for i in reversed(dead):
            self._subscribers.pop(i)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> dict | None:
        tick = self._latest.get(symbol)
        return tick.to_dict() if tick else None

    def get_all(self) -> dict:
        return {sym: t.to_dict() for sym, t in self._latest.items()}

    def get_history(self, symbol: str, limit: int = 0) -> list[dict]:
        buf = self._history.get(symbol)
        if buf is None:
            return []
        data = list(buf)
        return data[-limit:] if limit > 0 else data

    # ------------------------------------------------------------------
    # Pub / sub
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass
