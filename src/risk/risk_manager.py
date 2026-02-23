"""Risk management layer: stop-loss checks.

This module is used by the evaluation / backtesting loop to enforce
stop-loss rules *on top of* the agent's raw actions.

Position sizing and leverage-cap enforcement are handled inside
``TradingEnv._execute_buy()`` (see ``src/env/trading_env.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RiskManager:
    """Stateful risk manager that can override agent actions.

    Parameters
    ----------
    stop_loss_type : str
        ``"percent"`` or ``"atr"``.
    stop_loss_pct : float
        Percentage stop-loss (e.g. 0.05 = 5 %).
    stop_loss_atr_mult : float
        ATR multiplier for ATR-based stop.
    """

    stop_loss_type: str = "percent"
    stop_loss_pct: float = 0.05
    stop_loss_atr_mult: float = 2.0

    # --- internal tracking ---
    _entry_price: float = field(default=0.0, init=False, repr=False)
    _in_position: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_buy(self, price: float) -> None:
        """Record that we entered a position."""
        self._entry_price = price
        self._in_position = True

    def on_sell(self) -> None:
        """Record that we exited a position."""
        self._entry_price = 0.0
        self._in_position = False

    def check_stop_loss(
        self,
        current_price: float,
        atr: float | None = None,
    ) -> bool:
        """Return ``True`` if the stop-loss has been triggered.

        Parameters
        ----------
        current_price : float
            Current market price.
        atr : float | None
            Current ATR value (required when ``stop_loss_type == "atr"``).
        """
        if not self._in_position or self._entry_price <= 0:
            return False

        if self.stop_loss_type == "percent":
            loss_frac = 1.0 - current_price / self._entry_price
            return loss_frac >= self.stop_loss_pct

        elif self.stop_loss_type == "atr":
            if atr is None or atr <= 0:
                return False
            threshold = self._entry_price - self.stop_loss_atr_mult * atr
            return current_price <= threshold

        return False

    @classmethod
    def from_config(cls, risk_cfg: dict[str, Any]) -> "RiskManager":
        """Build from the ``risk`` section of the config dict."""
        return cls(
            stop_loss_type=risk_cfg.get("stop_loss_type", "percent"),
            stop_loss_pct=risk_cfg.get("stop_loss_pct", 0.05),
            stop_loss_atr_mult=risk_cfg.get("stop_loss_atr_mult", 2.0),
        )
