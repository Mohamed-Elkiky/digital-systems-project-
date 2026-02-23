"""CLI: Run baseline strategies on the test set.

Usage
-----
    python -m src.cli.baselines --symbol BTC-USD
    python -m src.cli.baselines --symbol BTC-USD --config configs/baselines.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baselines.buy_hold import run_buy_hold
from src.baselines.ma_crossover import run_ma_crossover
from src.eval.backtester import save_backtest_results
from src.eval.metrics import compute_all_metrics
from src.eval.plots import (
    plot_drawdown,
    plot_equity_curves,
    plot_metrics_table,
    plot_trades_on_price,
)
from src.utils.config import load_config, load_yaml, save_config_snapshot
from src.utils.logging import get_logger
from src.utils.paths import data_processed_dir, new_run_dir

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run baseline strategies")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    baselines_cfg = cfg.get("baselines", {})
    if args.config and "baselines" not in cfg:
        baselines_cfg = load_yaml(args.config).get("baselines", baselines_cfg)

    env_cfg = cfg.get("env", {})
    initial_cash = env_cfg.get("initial_cash", 10_000.0)
    tc = env_cfg.get("transaction_cost_pct", 0.001)
    sl = env_cfg.get("slippage_pct", 0.0005)

    proc = data_processed_dir(args.symbol)
    test_raw = pd.read_csv(proc / "test_raw.csv", parse_dates=True, index_col=0)

    run_dir = new_run_dir(tag=f"baselines_{args.symbol}")
    save_config_snapshot(cfg, run_dir)

    logger.info(f"[bold]Running baselines[/bold] on {args.symbol} test set ({len(test_raw)} rows)")

    all_curves: dict[str, pd.Series] = {}
    all_metrics: dict[str, dict] = {}

    # --- Buy & Hold ---
    if baselines_cfg.get("buy_hold", {}).get("enabled", True):
        bh = run_buy_hold(test_raw, initial_cash, tc, sl)
        all_curves["Buy&Hold"] = bh.equity_curve
        m = compute_all_metrics(bh.equity_curve.values, bh.trades, len(bh.equity_curve))
        all_metrics["Buy&Hold"] = m
        _save_one(run_dir, "buy_hold", bh.equity_curve, bh.trades, m)

    # --- MA Crossover ---
    ma_cfg = baselines_cfg.get("ma_crossover", {})
    if ma_cfg.get("enabled", True):
        mac = run_ma_crossover(
            test_raw,
            fast_window=ma_cfg.get("fast_window", 10),
            slow_window=ma_cfg.get("slow_window", 50),
            initial_cash=initial_cash,
            transaction_cost_pct=tc,
            slippage_pct=sl,
        )
        all_curves["MA Crossover"] = mac.equity_curve
        m = compute_all_metrics(mac.equity_curve.values, mac.trades, len(mac.equity_curve))
        all_metrics["MA Crossover"] = m
        _save_one(run_dir, "ma_crossover", mac.equity_curve, mac.trades, m)

    # --- Combined plots ---
    if all_curves:
        plot_equity_curves(all_curves, run_dir, title="Baseline Equity Curves")
    if all_metrics:
        plot_metrics_table(all_metrics, run_dir)

    # Summary markdown
    _write_summary(all_metrics, run_dir)
    logger.info(f"[bold green]Baselines complete.[/bold green]  See {run_dir}")


def _save_one(run_dir, tag, equity, trades, metrics):
    results = {"equity_curve": equity, "trades": trades, "metrics": metrics, "positions": []}
    save_backtest_results(results, run_dir, tag=tag)


def _write_summary(all_metrics, run_dir):
    lines = ["# Baseline Results Summary", ""]
    for name, m in all_metrics.items():
        lines.append(f"## {name}")
        lines.append("| Metric | Value |")
        lines.append("| ------ | ----- |")
        for k, v in m.items():
            lines.append(f"| {k} | {v:.6f}" if isinstance(v, float) else f"| {k} | {v}")
        lines.append("")
    (run_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
