"""CLI: Download OHLCV data.

Usage
-----
    python -m src.cli.download_data --symbol BTC-USD --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import sys

from src.data.fetch_yfinance import fetch_and_save
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data from Yahoo Finance")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args(argv)

    logger.info(f"[bold]Downloading {args.symbol}[/bold] ({args.start} → {args.end})")
    path = fetch_and_save(args.symbol, args.start, args.end)
    logger.info(f"[green]Done![/green] Saved to {path}")


if __name__ == "__main__":
    main()
