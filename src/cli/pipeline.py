"""CLI: Full end-to-end pipeline.

Runs download → preprocess → train (all algos) → baselines → evaluate.

Usage
-----
    python -m src.cli.pipeline
    python -m src.cli.pipeline --symbol BTC-USD --algos qlearning dqn ppo
"""

from __future__ import annotations

import argparse
import sys

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline end-to-end")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["qlearning", "dqn", "ppo"],
        choices=["qlearning", "dqn", "ppo"],
    )
    parser.add_argument("--skip_download", action="store_true", help="Skip download if data exists")
    args = parser.parse_args(argv)

    # --- Step 1: Download ---
    if not args.skip_download:
        logger.info("[bold cyan]Step 1/5: Downloading data[/bold cyan]")
        from src.cli.download_data import main as dl_main

        dl_main(["--symbol", args.symbol, "--start", args.start, "--end", args.end])
    else:
        logger.info("[dim]Step 1/5: Skipping download (--skip_download)[/dim]")

    # --- Step 2: Preprocess ---
    logger.info("[bold cyan]Step 2/5: Preprocessing data[/bold cyan]")
    from src.cli.preprocess import main as pp_main

    pp_main(["--symbol", args.symbol])

    # --- Step 3: Train agents ---
    logger.info("[bold cyan]Step 3/5: Training agents[/bold cyan]")
    from src.cli.train import main as train_main

    for algo in args.algos:
        logger.info(f"  Training {algo} …")
        try:
            train_main(["--algo", algo, "--symbol", args.symbol])
        except Exception as exc:
            logger.error(f"  [red]Training {algo} failed: {exc}[/red]")

    # --- Step 4: Baselines ---
    logger.info("[bold cyan]Step 4/5: Running baselines[/bold cyan]")
    from src.cli.baselines import main as bl_main

    bl_main(["--symbol", args.symbol])

    # --- Step 5: Evaluate latest runs ---
    logger.info("[bold cyan]Step 5/5: Evaluating trained agents[/bold cyan]")
    from src.cli.evaluate import main as eval_main
    from src.utils.paths import runs_dir

    # Find the most recent run dirs for each algo
    all_runs = sorted(
        [d for d in runs_dir().iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for algo in args.algos:
        # Find most recent run for this algo
        for rd in all_runs:
            if algo in rd.name and args.symbol.replace("-", "") in rd.name.replace("-", ""):
                config_path = rd / "config.yaml"
                if config_path.exists():
                    logger.info(f"  Evaluating {algo} → {rd}")
                    try:
                        eval_main(["--run_dir", str(rd), "--symbol", args.symbol])
                    except Exception as exc:
                        logger.error(f"  [red]Eval {algo} failed: {exc}[/red]")
                    break

    logger.info("[bold green]Pipeline complete![/bold green]")
    logger.info(f"Check results in: {runs_dir()}")


if __name__ == "__main__":
    main()
