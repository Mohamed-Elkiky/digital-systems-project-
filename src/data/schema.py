"""Schema validation for OHLCV DataFrames (both yfinance & CSV import)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


class SchemaError(Exception):
    """Raised when data does not match the expected schema."""


def validate_ohlcv(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """Validate and normalise an OHLCV DataFrame.

    Checks:
    - Required columns present.
    - Index is DatetimeIndex (or column 'Date' convertible).
    - No all-NaN columns.
    - Sorted by date ascending.

    Returns the cleaned DataFrame.
    """
    # --- Column check ---
    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    # Try case-insensitive match
    rename_map: dict[str, str] = {}
    for col in df.columns:
        for req in REQUIRED_COLUMNS:
            if col.lower() == req.lower() and col != req:
                rename_map[col] = req
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaError(f"[{source}] Missing columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()

    # --- Index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            raise SchemaError(f"[{source}] Cannot parse index as datetime: {exc}") from exc
    df.index.name = "Date"
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

    # --- Sort ---
    df = df.sort_index()

    # --- NaN check ---
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        raise SchemaError(f"[{source}] All-NaN columns: {all_nan_cols}")

    logger.info(f"Schema OK for '{source}': {len(df)} rows, {df.index[0].date()}–{df.index[-1].date()}")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a user-supplied CSV and validate against OHLCV schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return validate_ohlcv(df, source=str(path.name))
