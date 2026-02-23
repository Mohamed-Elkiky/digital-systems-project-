"""Unified backtesting engine.

Runs any agent (QL / DQN / PPO) or baseline through the trading environment
on the test split and produces standardised outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from src.eval.metrics import compute_all_metrics
from src.env.trading_env import TradingEnv
from src.env.wrappers import make_env
from src.risk.risk_manager import RiskManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentProtocol(Protocol):
    """Minimal interface an agent must satisfy for backtesting."""

    def select_action(self, obs: np.ndarray, greedy: bool = ...) -> int: ...


def _unwrap_env(env) -> TradingEnv:
    """Fully unwrap a (possibly wrapped) env to the inner TradingEnv."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


def backtest_agent(
    agent: AgentProtocol,
    df: pd.DataFrame,
    feature_cols: list[str],
    env_cfg: dict[str, Any],
    risk_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run *agent* through one full episode on *df* and collect results.

    Returns
    -------
    dict with keys: equity_curve, trades, metrics, positions, dates
    """
    env = make_env(df, feature_cols, env_cfg)
    rm = RiskManager.from_config(risk_cfg) if risk_cfg else None

    obs, _ = env.reset()
    done = False
    equity_vals: list[float] = []
    positions: list[float] = []

    while not done:
        action = agent.select_action(obs, greedy=True)

        # Risk manager override
        if rm is not None:
            inner = _unwrap_env(env)
            price = inner._close[inner._current_step] if hasattr(inner, "_close") else 0.0
            atr_val = df["atr"].iloc[inner._current_step] if "atr" in df.columns and hasattr(inner, "_current_step") else None

            if rm.check_stop_loss(price, atr=atr_val):
                action = 2  # force sell

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        equity_vals.append(info.get("portfolio_value", 0.0))
        positions.append(info.get("position", 0.0))

        # Inform risk manager
        if rm is not None:
            if action == 1:
                rm.on_buy(info.get("price", 0.0))
            elif action == 2:
                rm.on_sell()

    inner = _unwrap_env(env)
    trades = inner.trades if hasattr(inner, "trades") else []

    dates = df.index[env_cfg.get("window_size", 50) + 1: env_cfg.get("window_size", 50) + 1 + len(equity_vals)]
    equity_series = pd.Series(equity_vals, index=dates[:len(equity_vals)], name="equity")

    metrics = compute_all_metrics(
        equity=np.array(equity_vals),
        trades=trades,
        total_steps=len(equity_vals),
        positions=np.array(positions),
    )

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
        "positions": positions,
        "dates": dates.tolist() if hasattr(dates, "tolist") else list(dates),
    }


def backtest_sb3(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    env_cfg: dict[str, Any],
    continuous: bool = False,
) -> dict[str, Any]:
    """Backtest a Stable-Baselines3 model (PPO)."""
    env = make_env(df, feature_cols, env_cfg, continuous=continuous)
    obs, _ = env.reset()
    done = False
    equity_vals: list[float] = []
    positions: list[float] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_vals.append(info.get("portfolio_value", 0.0))
        positions.append(info.get("position", 0.0))

    inner = _unwrap_env(env)
    trades = inner.trades if hasattr(inner, "trades") else []

    dates = df.index[env_cfg.get("window_size", 50) + 1: env_cfg.get("window_size", 50) + 1 + len(equity_vals)]
    equity_series = pd.Series(equity_vals, index=dates[:len(equity_vals)], name="equity")

    metrics = compute_all_metrics(
        equity=np.array(equity_vals),
        trades=trades,
        total_steps=len(equity_vals),
        positions=np.array(positions),
    )

    return {
        "equity_curve": equity_series,
        "trades": trades,
        "metrics": metrics,
        "positions": positions,
        "dates": dates.tolist() if hasattr(dates, "tolist") else list(dates),
    }


def save_backtest_results(results: dict[str, Any], run_dir: Path, tag: str = "") -> None:
    """Persist backtest results to the run directory."""
    prefix = f"{tag}_" if tag else ""

    # Metrics JSON
    with open(run_dir / f"{prefix}metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)

    # Trades CSV
    if results["trades"]:
        pd.DataFrame(results["trades"]).to_csv(run_dir / f"{prefix}trades.csv", index=False)

    # Equity curve CSV
    if results.get("equity_curve") is not None:
        results["equity_curve"].to_csv(run_dir / f"{prefix}equity_curve.csv", header=True)

    logger.info(f"Results saved to {run_dir} (prefix='{prefix}')")
