"""Data preprocessing pipeline: clean → features → split → normalise → save."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.features import compute_all_features
from src.data.schema import load_csv, validate_ohlcv
from src.utils.logging import get_logger
from src.utils.paths import data_processed_dir, data_raw_dir

logger = get_logger(__name__)

# Columns that will be normalised (exclude OHLCV prices + Date index)
EXCLUDE_FROM_SCALING = {"Open", "High", "Low", "Close", "Volume", "volume_sma"}


def _load_raw(symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Load raw OHLCV either from yfinance download or user CSV."""
    data_cfg = cfg.get("data", cfg)
    source = data_cfg.get("source", "yfinance")
    if source == "csv":
        csv_path = data_cfg.get("csv_path")
        if csv_path is None:
            raise ValueError("source=csv but csv_path is not set in config.")
        return load_csv(csv_path)
    raw_path = data_raw_dir(symbol) / "ohlcv.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run download_data first."
        )
    df = pd.read_csv(raw_path, parse_dates=True, index_col=0)
    return validate_ohlcv(df, source="raw_csv")


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted DataFrame into train / val / test (no leakage)."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    logger.info(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def fit_scaler(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> StandardScaler:
    """Fit a StandardScaler on the *training* set only."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that should be normalised."""
    return [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]


def run_preprocessing(symbol: str, cfg: dict[str, Any]) -> Path:
    """Full preprocessing pipeline.  Returns the output directory."""
    # 1. Load
    df = _load_raw(symbol, cfg)

    # 2. Feature engineering
    df = compute_all_features(df, cfg)

    # 3. Time split
    split_cfg = cfg.get("split", {})
    train, val, test = time_split(
        df,
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
    )

    # 4. Normalise (train-fitted scaler)
    feat_cols = _feature_cols(train)
    scaler = fit_scaler(train, feat_cols)

    train_norm = apply_scaler(train, scaler, feat_cols)
    val_norm = apply_scaler(val, scaler, feat_cols)
    test_norm = apply_scaler(test, scaler, feat_cols)

    # 5. Save
    out_dir = data_processed_dir(symbol)
    train_norm.to_csv(out_dir / "train.csv")
    val_norm.to_csv(out_dir / "val.csv")
    test_norm.to_csv(out_dir / "test.csv")

    # Also save un-normalised splits (handy for price-based eval)
    train.to_csv(out_dir / "train_raw.csv")
    val.to_csv(out_dir / "val_raw.csv")
    test.to_csv(out_dir / "test_raw.csv")

    # Save scaler
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature column list
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    logger.info(f"Preprocessing complete → {out_dir}")
    return out_dir
