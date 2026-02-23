"""Tests for the preprocessing / feature-engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import compute_all_features
from src.data.preprocess import EXCLUDE_FROM_SCALING, fit_scaler, apply_scaler, time_split


class TestFeatureEngineering:
    """Ensure features are computed without NaN leaks."""

    def test_no_nans_after_features(self, sample_ohlcv, feature_cfg):
        df = compute_all_features(sample_ohlcv, feature_cfg)
        assert not df.isna().any().any(), "Features should have no NaNs after dropna"

    def test_expected_columns_present(self, sample_ohlcv, feature_cfg):
        df = compute_all_features(sample_ohlcv, feature_cfg)
        expected = {"return", "log_return", "rsi", "macd", "macd_hist", "atr", "volatility", "volume_ratio"}
        assert expected.issubset(set(df.columns))

    def test_row_count_reduced(self, sample_ohlcv, feature_cfg):
        df = compute_all_features(sample_ohlcv, feature_cfg)
        # NaN lead-in should reduce rows
        assert len(df) < len(sample_ohlcv)
        assert len(df) > 0


class TestTimeSplit:
    """No data leakage between train/val/test."""

    def test_split_no_overlap(self, sample_ohlcv):
        train, val, test = time_split(sample_ohlcv, 0.7, 0.15)
        assert train.index[-1] < val.index[0], "Train must end before val starts"
        assert val.index[-1] < test.index[0], "Val must end before test starts"

    def test_split_covers_all(self, sample_ohlcv):
        train, val, test = time_split(sample_ohlcv, 0.7, 0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_ohlcv)


class TestScaler:
    """Scaler must be fitted on train only."""

    def test_scaler_fitted_on_train(self, sample_ohlcv, feature_cfg):
        df = compute_all_features(sample_ohlcv, feature_cfg)
        train, val, test = time_split(df, 0.7, 0.15)

        feat_cols = [c for c in train.columns if c not in EXCLUDE_FROM_SCALING]

        scaler = fit_scaler(train, feat_cols)
        train_sc = apply_scaler(train, scaler, feat_cols)
        val_sc = apply_scaler(val, scaler, feat_cols)

        # Train should be ~ zero mean / unit std
        means = train_sc[feat_cols].mean()
        assert (means.abs() < 0.1).all(), f"Train means should be near zero: {means}"

        # Scaler means should match train, not val
        # (val mean should generally NOT be ~0)
        val_means = val_sc[feat_cols].mean()
        # This is a soft check — at least one column should differ from 0
        # (unless data is perfectly stationary, which is unlikely)
        assert not np.allclose(val_means.values, 0, atol=0.05), (
            "Validation means shouldn't all be ~0 if scaler was train-fitted"
        )
