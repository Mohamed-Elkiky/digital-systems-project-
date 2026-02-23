"""Performance metrics for strategy evaluation.

All functions accept either equity curves (pd.Series / np.ndarray) or
return series and output scalar metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cumulative_return(equity: np.ndarray | pd.Series) -> float:
    """Total percentage return."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] == 0:
        return 0.0
    return float((eq[-1] / eq[0]) - 1.0)


def annualised_return(equity: np.ndarray | pd.Series, periods_per_year: float = 252.0) -> float:
    """CAGR approximation."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    total = eq[-1] / eq[0]
    n_periods = len(eq) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def annualised_volatility(equity: np.ndarray | pd.Series, periods_per_year: float = 252.0) -> float:
    """Annualised volatility from daily returns."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 3:
        return 0.0
    rets = np.diff(eq) / eq[:-1]
    return float(np.std(rets) * np.sqrt(periods_per_year))


def sharpe_ratio(equity: np.ndarray | pd.Series, risk_free: float = 0.0, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio (risk-free assumed 0 by default)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 3:
        return 0.0
    rets = np.diff(eq) / eq[:-1]
    excess = rets - risk_free / periods_per_year
    std = np.std(excess)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def drawdown_series(equity: np.ndarray | pd.Series) -> np.ndarray:
    """Per-step drawdown as a positive fraction (e.g. 0.25 = 25 %)."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return (peak - eq) / np.where(peak > 0, peak, 1.0)


def max_drawdown(equity: np.ndarray | pd.Series) -> float:
    """Maximum drawdown (as a positive fraction, e.g. 0.25 = 25 %)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    return float(drawdown_series(eq).max())


def win_rate(trades: list[dict[str, Any]]) -> float:
    """Fraction of round-trip trades that were profitable."""
    buys: list[dict] = [t for t in trades if t.get("action") == "buy"]
    sells: list[dict] = [t for t in trades if t.get("action") == "sell"]
    n_round = min(len(buys), len(sells))
    if n_round == 0:
        return 0.0
    wins = 0
    for i in range(n_round):
        if sells[i]["price"] > buys[i]["price"]:
            wins += 1
    return wins / n_round


def number_of_trades(trades: list[dict[str, Any]]) -> int:
    return len(trades)


def exposure_time(positions: np.ndarray | list, total_steps: int) -> float:
    """Fraction of time the agent is in a position (position > 0)."""
    pos = np.asarray(positions, dtype=float)
    if total_steps <= 0:
        return 0.0
    return float(np.sum(pos > 0) / total_steps)


def compute_all_metrics(
    equity: np.ndarray | pd.Series,
    trades: list[dict[str, Any]],
    total_steps: int | None = None,
    positions: np.ndarray | list | None = None,
) -> dict[str, float]:
    """Compute the full metrics suite and return as a dict."""
    eq = np.asarray(equity, dtype=float)
    if total_steps is None:
        total_steps = len(eq)
    if positions is None:
        positions = np.zeros(total_steps)

    return {
        "cumulative_return": cumulative_return(eq),
        "annualised_return": annualised_return(eq),
        "annualised_volatility": annualised_volatility(eq),
        "sharpe_ratio": sharpe_ratio(eq),
        "max_drawdown": max_drawdown(eq),
        "win_rate": win_rate(trades),
        "num_trades": number_of_trades(trades),
        "exposure_time": exposure_time(positions, total_steps),
    }
