"""Moving-Average Crossover baseline strategy.

Generates buy/sell signals when the fast SMA crosses the slow SMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MACrossoverResult:
    """Container for MA crossover backtest results."""

    equity_curve: pd.Series
    trades: list[dict[str, Any]]
    final_value: float
    total_return: float


def run_ma_crossover(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    initial_cash: float = 10_000.0,
    transaction_cost_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> MACrossoverResult:
    """Simulate an SMA crossover strategy.

    Signal:
      - Buy  when fast SMA crosses *above* slow SMA (and not already holding).
      - Sell when fast SMA crosses *below* slow SMA (and holding).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Close`` column.
    fast_window / slow_window : int
        SMA look-back periods.

    Returns
    -------
    MACrossoverResult
    """
    close = df["Close"].values.astype(float)
    dates = df.index

    fast_sma = pd.Series(close).rolling(fast_window).mean().values
    slow_sma = pd.Series(close).rolling(slow_window).mean().values

    cash = initial_cash
    units = 0.0
    equity_values: list[float] = []
    trades: list[dict[str, Any]] = []
    in_position = False

    for i in range(len(close)):
        price = close[i]

        # Check for crossover signals (need at least slow_window bars)
        if i >= slow_window and not np.isnan(fast_sma[i]) and not np.isnan(slow_sma[i]):
            prev_fast = fast_sma[i - 1] if i > 0 else fast_sma[i]
            prev_slow = slow_sma[i - 1] if i > 0 else slow_sma[i]

            # Buy signal: fast crosses above slow
            if not in_position and prev_fast <= prev_slow and fast_sma[i] > slow_sma[i]:
                eff_price = price * (1.0 + slippage_pct)
                cost = cash * transaction_cost_pct
                units = (cash - cost) / eff_price
                cash = 0.0
                in_position = True
                trades.append({
                    "step": i, "action": "buy", "price": eff_price,
                    "units": units, "cost": cost,
                })

            # Sell signal: fast crosses below slow
            elif in_position and prev_fast >= prev_slow and fast_sma[i] < slow_sma[i]:
                eff_price = price * (1.0 - slippage_pct)
                proceeds = units * eff_price
                cost = proceeds * transaction_cost_pct
                cash = proceeds - cost
                trades.append({
                    "step": i, "action": "sell", "price": eff_price,
                    "units": units, "cost": cost,
                })
                units = 0.0
                in_position = False

        equity_values.append(cash + units * price)

    # Close any remaining position at the end
    if in_position:
        eff_price = close[-1] * (1.0 - slippage_pct)
        proceeds = units * eff_price
        cost = proceeds * transaction_cost_pct
        cash = proceeds - cost
        trades.append({
            "step": len(close) - 1, "action": "sell", "price": eff_price,
            "units": units, "cost": cost,
        })
        units = 0.0
        equity_values[-1] = cash

    equity_series = pd.Series(equity_values, index=dates, name="equity")
    final_value = equity_values[-1]
    total_return = (final_value / initial_cash) - 1.0

    logger.info(
        f"[MA Crossover] fast={fast_window} slow={slow_window}  "
        f"return={total_return:.4%}  trades={len(trades)}  final={final_value:.2f}"
    )

    return MACrossoverResult(
        equity_curve=equity_series,
        trades=trades,
        final_value=final_value,
        total_return=total_return,
    )
