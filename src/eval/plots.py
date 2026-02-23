"""Plotting utilities using Plotly (with Matplotlib fallback for static export).

All plot functions accept data and a run_dir, saving PNG + interactive HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.eval.metrics import drawdown_series
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _save(fig: go.Figure, run_dir: Path, name: str) -> None:
    """Save a Plotly figure as both HTML and PNG."""
    fig.write_html(str(run_dir / f"{name}.html"))
    try:
        fig.write_image(str(run_dir / f"{name}.png"), width=1200, height=600)
    except Exception:
        logger.warning(f"Could not save PNG for {name} (kaleido may not be installed)")


# ======================================================================
# Equity curve
# ======================================================================

def plot_equity_curves(
    curves: dict[str, pd.Series],
    run_dir: Path,
    title: str = "Equity Curves",
) -> go.Figure:
    """Overlay multiple equity curves on one chart."""
    fig = go.Figure()
    for label, series in curves.items():
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=label,
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    _save(fig, run_dir, "equity_curves")
    return fig


# ======================================================================
# Drawdown
# ======================================================================

def plot_drawdown(
    equity: pd.Series,
    run_dir: Path,
    title: str = "Drawdown",
) -> go.Figure:
    eq = equity.values.astype(float)
    dd = drawdown_series(eq) * 100  # percent
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=-dd, fill="tozeroy", name="Drawdown %",
        line=dict(color="crimson"),
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown (%)")
    _save(fig, run_dir, "drawdown")
    return fig


# ======================================================================
# Price chart with trade markers
# ======================================================================

def plot_trades_on_price(
    price_series: pd.Series,
    trades: list[dict[str, Any]],
    run_dir: Path,
    title: str = "Trades on Price",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode="lines", name="Close Price", line=dict(color="steelblue"),
    ))

    buy_steps = [t["step"] for t in trades if t.get("action") == "buy"]
    sell_steps = [t["step"] for t in trades if t.get("action") == "sell"]

    # Map steps to dates/prices
    idx = price_series.index
    for label, steps, color, symbol in [
        ("Buy", buy_steps, "green", "triangle-up"),
        ("Sell", sell_steps, "red", "triangle-down"),
    ]:
        valid = [s for s in steps if s < len(idx)]
        if valid:
            fig.add_trace(go.Scatter(
                x=idx[valid],
                y=price_series.values[valid],
                mode="markers",
                name=label,
                marker=dict(color=color, size=10, symbol=symbol),
            ))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)")
    _save(fig, run_dir, "trades_on_price")
    return fig


# ======================================================================
# Positions over time
# ======================================================================

def plot_positions(
    positions: list[float] | np.ndarray,
    dates: list | pd.Index,
    run_dir: Path,
    title: str = "Position Over Time",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates)[:len(positions)],
        y=list(positions),
        mode="lines",
        name="Position",
        fill="tozeroy",
        line=dict(color="darkorange"),
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Units Held")
    _save(fig, run_dir, "positions")
    return fig


# ======================================================================
# Training curves
# ======================================================================

def plot_training_curve(
    rewards: list[float],
    run_dir: Path,
    title: str = "Training Reward Curve",
    window: int = 20,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode="lines", name="Episode Reward", opacity=0.4))
    if len(rewards) > window:
        smoothed = pd.Series(rewards).rolling(window).mean().values
        fig.add_trace(go.Scatter(y=smoothed, mode="lines", name=f"SMA-{window}"))
    fig.update_layout(title=title, xaxis_title="Episode", yaxis_title="Reward")
    _save(fig, run_dir, "training_curve")
    return fig


# ======================================================================
# Metrics table (static HTML)
# ======================================================================

def plot_metrics_table(
    metrics_dict: dict[str, dict[str, float]],
    run_dir: Path,
) -> go.Figure:
    """Create a comparison table of metrics across strategies."""
    strategies = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys()) if metrics_dict else []

    header = ["Metric"] + strategies
    rows = []
    for m in metric_names:
        row = [m]
        for s in strategies:
            val = metrics_dict[s].get(m, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        rows.append(row)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header, fill_color="paleturquoise", align="left"),
        cells=dict(
            values=list(zip(*rows)) if rows else [],
            fill_color="lavender",
            align="left",
        ),
    )])
    fig.update_layout(title="Strategy Comparison", height=400)
    _save(fig, run_dir, "metrics_table")
    return fig
