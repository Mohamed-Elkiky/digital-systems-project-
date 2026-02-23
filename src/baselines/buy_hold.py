"""Buy & Hold baseline strategy.

Invests all capital at the start and holds until the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BuyHoldResult:
    """Container for buy-and-hold backtest results."""

    equity_curve: pd.Series
    trades: list[dict[str, Any]]
    final_value: float
    total_return: float


def run_buy_hold(
    df: pd.DataFrame,
    initial_cash: float = 10_000.0,
    transaction_cost_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> BuyHoldResult:
    """Simulate buy-and-hold on the *Close* column of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Close`` column (and a DatetimeIndex).
    initial_cash : float
        Starting capital.

    Returns
    -------
    BuyHoldResult
    """
    close = df["Close"].values.astype(float)
    dates = df.index

    # Buy at first close
    entry_price = close[0] * (1.0 + slippage_pct)
    cost = initial_cash * transaction_cost_pct
    units = (initial_cash - cost) / entry_price

    equity = units * close  # value over time
    equity_series = pd.Series(equity, index=dates, name="equity")

    # Sell at last close
    exit_price = close[-1] * (1.0 - slippage_pct)
    final_value = units * exit_price - (units * exit_price * transaction_cost_pct)

    trades = [
        {"step": 0, "action": "buy", "price": entry_price, "units": units, "cost": cost},
        {
            "step": len(close) - 1,
            "action": "sell",
            "price": exit_price,
            "units": units,
            "cost": units * exit_price * transaction_cost_pct,
        },
    ]

    total_return = (final_value / initial_cash) - 1.0
    logger.info(f"[Buy&Hold] return={total_return:.4%}  final_value={final_value:.2f}")

    return BuyHoldResult(
        equity_curve=equity_series,
        trades=trades,
        final_value=final_value,
        total_return=total_return,
    )
