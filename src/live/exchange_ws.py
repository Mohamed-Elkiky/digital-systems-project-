"""Bybit v5 public WebSocket client with auto-reconnect and exponential backoff.

Connects to ``wss://stream.bybit.com/v5/public/linear`` and subscribes to
ticker topics for the configured symbols.  Each tick is pushed into the
shared :class:`PriceCache`.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time

import certifi
import websockets

from src.live.price_cache import PriceCache, Tick
from src.utils.logging import get_logger

logger = get_logger(__name__)

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


class BybitWebSocket:
    """Manages a persistent WebSocket connection to Bybit's public ticker feed.

    Features
    --------
    - Subscribes to ``tickers.<SYMBOL>`` for each symbol.
    - Handles application-level ping/pong (Bybit sends ``{"op":"ping"}``).
    - On disconnect, reconnects with exponential backoff (1 s → 60 s cap).
    - Uses certifi CA bundle for SSL (no verification disabled).
    """

    def __init__(
        self,
        cache: PriceCache,
        symbols: list[str] | None = None,
        max_backoff: float = 60.0,
    ):
        self.cache = cache
        self.symbols = symbols or list(DEFAULT_SYMBOLS)
        self._running = False
        self._backoff = 1.0
        self._max_backoff = max_backoff
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the connection loop forever (until ``stop()`` is called)."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"[BybitWS] Connection lost: {exc}")
            if self._running:
                logger.info(f"[BybitWS] Reconnecting in {self._backoff:.1f}s …")
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self._max_backoff)
        logger.info("[BybitWS] Stopped.")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _connect_and_listen(self) -> None:
        logger.info(f"[BybitWS] Connecting to {BYBIT_WS_URL} …")
        async with websockets.connect(
            BYBIT_WS_URL,
            ssl=self._ssl_ctx,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            self._backoff = 1.0  # reset on successful connect

            # Subscribe to ticker topics
            sub_msg = json.dumps({
                "op": "subscribe",
                "args": [f"tickers.{s}" for s in self.symbols],
            })
            await ws.send(sub_msg)
            logger.info(f"[BybitWS] Subscribed: {self.symbols}")

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Application-level ping from Bybit
                op = msg.get("op")
                if op == "ping":
                    await ws.send(json.dumps({"op": "pong"}))
                    continue
                if op is not None:
                    # subscription ack, pong response, etc.
                    continue

                topic = msg.get("topic", "")
                data = msg.get("data")
                if not topic.startswith("tickers.") or data is None:
                    continue

                tick = self._parse_tick(data, msg.get("ts", 0))
                if tick and tick.last > 0:
                    await self.cache.update(tick)

    @staticmethod
    def _parse_tick(data: dict, msg_ts: int) -> Tick | None:
        """Parse a Bybit v5 ticker data blob into a :class:`Tick`."""
        try:
            return Tick(
                symbol=data.get("symbol", ""),
                last=float(data.get("lastPrice", 0)),
                bid=float(data.get("bid1Price", 0)),
                ask=float(data.get("ask1Price", 0)),
                high_24h=float(data.get("highPrice24h", 0)),
                low_24h=float(data.get("lowPrice24h", 0)),
                volume_24h=float(data.get("volume24h", 0)),
                ts=msg_ts / 1000.0 if msg_ts > 1e12 else time.time(),
            )
        except (ValueError, TypeError):
            return None
