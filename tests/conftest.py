"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame (200 rows)."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10.0)  # ensure positive
    df = pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.2,
            "High": close + np.abs(np.random.randn(n) * 0.5),
            "Low": close - np.abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1_000, 50_000, size=n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def feature_cfg() -> dict:
    """Default feature config dict."""
    return {
        "features": {
            "window_size": 10,
            "sma_periods": [5, 10],
            "ema_periods": [5, 10],
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
            "volatility_period": 10,
            "volume_sma_period": 10,
        },
        "split": {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15},
    }
