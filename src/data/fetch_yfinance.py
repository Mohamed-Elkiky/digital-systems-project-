"""Download OHLCV data from Yahoo Finance and save to data/raw/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger
from src.utils.paths import data_raw_dir

logger = get_logger(__name__)


def download_ohlcv(
    symbol: str = "BTC-USD",
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex named 'Date'.
    """
    logger.info(f"Downloading {symbol} from {start} to {end} (interval={interval}) …")
    ticker = yf.Ticker(symbol)
    df: pd.DataFrame = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} ({start}–{end}).")

    # Keep only standard OHLCV columns
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    logger.info(f"  → {len(df)} rows, date range {df.index[0].date()} … {df.index[-1].date()}")
    return df


def save_raw(df: pd.DataFrame, symbol: str) -> Path:
    """Persist raw OHLCV CSV to data/raw/<symbol>/ohlcv.csv."""
    out_dir = data_raw_dir(symbol)
    path = out_dir / "ohlcv.csv"
    df.to_csv(path)
    logger.info(f"  → Saved raw data to {path}")
    return path


def fetch_and_save(
    symbol: str = "BTC-USD",
    start: str = "2020-01-01",
    end: str = "2025-12-31",
) -> Path:
    """Convenience: download + save."""
    df = download_ohlcv(symbol, start, end)
    return save_raw(df, symbol)
