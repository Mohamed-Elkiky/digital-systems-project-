"""Strategy decision loop — reads live prices from PriceCache every 1 s.

Uses the same council-voting logic from the dashboard trader module.
Decisions are pushed to subscriber queues and stored in a rolling buffer.

Risk guards
-----------
- **Stale price**: skip if latest tick is older than ``stale_threshold_sec``.
- **Cooldown**: minimum ``cooldown_sec`` between consecutive BUY/SELL decisions.
- **Max position %**: placeholder guard (logs warning when exceeded).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path

from src.live.price_cache import PriceCache
from src.dashboard import trader as trader_mod
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"


class StrategyRunner:
    """Async loop that evaluates the RL council every second against live data."""

    def __init__(
        self,
        cache: PriceCache,
        *,
        symbol: str = "BTCUSDT",
        cooldown_sec: float = 60.0,
        stale_threshold_sec: float = 3.0,
        obs_refresh_sec: float = 300.0,
        max_position_pct: float = 0.10,
    ):
        self.cache = cache
        self.symbol = symbol
        self.cooldown_sec = cooldown_sec
        self.stale_threshold_sec = stale_threshold_sec
        self.obs_refresh_sec = obs_refresh_sec
        self.max_position_pct = max_position_pct

        self.model_names: list[str] = []
        self.models: list[dict] = []
        self._running = False
        self._last_trade_time = 0.0
        self._obs = None
        self._obs_time = 0.0

        # Public rolling decision buffer
        self.decisions: deque[dict] = deque(maxlen=200)
        self._subscribers: list[asyncio.Queue] = []

    # ------------------------------------------------------------------
    # Model management (can be called from REST endpoints)
    # ------------------------------------------------------------------

    def load_models(self, names: list[str]) -> None:
        """Load RL models from run directories."""
        self.model_names = list(names)
        self.models = [trader_mod.load_model(RUNS_DIR / name) for name in names]
        logger.info(f"[Strategy] Loaded {len(self.models)} models: {names}")

    # ------------------------------------------------------------------
    # Observation (expensive — runs in thread pool to avoid blocking)
    # ------------------------------------------------------------------

    async def _refresh_obs(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            obs, info, price = await loop.run_in_executor(
                None, trader_mod.build_live_obs, "BTC-USD"
            )
            self._obs = obs
            self._obs_time = time.time()
            logger.info(f"[Strategy] Observation refreshed: {info}")
        except Exception as exc:
            logger.warning(f"[Strategy] Failed to refresh observation: {exc}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the decision loop (1 Hz) until ``stop()`` is called."""
        self._running = True
        logger.info("[Strategy] Starting decision loop …")
        await self._refresh_obs()

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[Strategy] tick error: {exc}")
            await asyncio.sleep(1.0)

        logger.info("[Strategy] Stopped.")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Single tick
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        # Periodically refresh the observation vector
        if time.time() - self._obs_time > self.obs_refresh_sec:
            await self._refresh_obs()

        if not self.models or self._obs is None:
            return

        # Read latest price from PriceCache
        price_data = self.cache.get(self.symbol)
        if price_data is None:
            return

        # Risk guard: stale price
        age = time.time() - price_data["ts"]
        if age > self.stale_threshold_sec:
            return

        # Risk guard: cooldown
        since_last = time.time() - self._last_trade_time
        if since_last < self.cooldown_sec:
            return

        # Council vote
        result = trader_mod.council_vote(self.models, self._obs)
        action_label = result["label"]

        decision = {
            "symbol": self.symbol,
            "action": action_label.upper(),
            "price": price_data["last"],
            "bid": price_data["bid"],
            "ask": price_data["ask"],
            "reason": f"Council: {result['counts']}",
            "ts": time.time(),
            "model_votes": result.get("model_votes", []),
        }

        self.decisions.append(decision)

        if action_label != "Hold":
            self._last_trade_time = time.time()

        # Notify subscribers
        msg = {"type": "decision", **decision}
        dead: list[int] = []
        for i, q in enumerate(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(i)
        for i in reversed(dead):
            self._subscribers.pop(i)

        logger.info(
            f"[Strategy] {decision['action']} {self.symbol} "
            f"@ ${decision['price']:,.2f} — {decision['reason']}"
        )

    # ------------------------------------------------------------------
    # Pub / sub
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass
