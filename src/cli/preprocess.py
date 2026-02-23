"""CLI: Preprocess downloaded data.

Usage
-----
    python -m src.cli.preprocess --symbol BTC-USD
    python -m src.cli.preprocess --symbol BTC-USD --config configs/base.yaml
"""

from __future__ import annotations

import argparse

from src.data.preprocess import run_preprocessing
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess OHLCV data")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger.info(f"[bold]Preprocessing {args.symbol}[/bold]")
    out = run_preprocessing(args.symbol, cfg)
    logger.info(f"[green]Done![/green] Processed data in {out}")


if __name__ == "__main__":
    main()
