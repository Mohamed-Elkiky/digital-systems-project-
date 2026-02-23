"""Feature engineering for OHLCV data.

All features are computed from the raw OHLCV and appended as new columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple and log returns on Close."""
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_sma(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"sma_{p}"] = df["Close"].rolling(p).mean()
    return df


def add_ema(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"ema_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["volatility"] = df["log_return"].rolling(period).std()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def add_volume_features(df: pd.DataFrame, sma_period: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["volume_sma"] = df["Volume"].rolling(sma_period).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma"] + 1e-10)
    return df


def compute_all_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline driven by *cfg* (features section)."""
    feat = cfg.get("features", cfg)  # allow passing either full cfg or features sub-dict

    df = add_returns(df)
    df = add_sma(df, feat.get("sma_periods", [10, 30, 50]))
    df = add_ema(df, feat.get("ema_periods", [12, 26]))
    df = add_rsi(df, feat.get("rsi_period", 14))
    df = add_macd(df, feat.get("macd_fast", 12), feat.get("macd_slow", 26), feat.get("macd_signal", 9))
    df = add_volatility(df, feat.get("volatility_period", 20))
    df = add_atr(df, feat.get("atr_period", 14))
    df = add_volume_features(df, feat.get("volume_sma_period", 20))

    n_before = len(df)
    df = df.dropna()
    logger.info(f"Feature engineering complete: {n_before} → {len(df)} rows (dropped {n_before - len(df)} NaN lead-in)")
    return df
