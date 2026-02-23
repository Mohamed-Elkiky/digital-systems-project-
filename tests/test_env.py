"""Tests for the TradingEnv gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.features import compute_all_features
from src.data.preprocess import EXCLUDE_FROM_SCALING
from src.env.trading_env import TradingEnv


@pytest.fixture
def trading_env(sample_ohlcv, feature_cfg):
    """Create a TradingEnv from synthetic data."""
    df = compute_all_features(sample_ohlcv, feature_cfg)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]
    env = TradingEnv(
        df=df,
        feature_cols=feat_cols,
        window_size=10,
        initial_cash=10_000.0,
    )
    return env


class TestEnvResetStep:
    """Basic env API tests."""

    def test_reset_returns_correct_shape(self, trading_env):
        obs, info = trading_env.reset(seed=42)
        expected_dim = trading_env.observation_space.shape[0]
        assert obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {obs.shape}"
        assert isinstance(info, dict)

    def test_obs_no_nans(self, trading_env):
        obs, _ = trading_env.reset(seed=42)
        assert not np.isnan(obs).any(), "Observation should not contain NaNs"

    def test_step_returns_five_tuple(self, trading_env):
        trading_env.reset(seed=42)
        result = trading_env.step(0)  # Hold
        assert len(result) == 5, "step() should return (obs, reward, terminated, truncated, info)"

    def test_step_obs_shape(self, trading_env):
        trading_env.reset(seed=42)
        obs, _, _, _, _ = trading_env.step(1)  # Buy
        expected_dim = trading_env.observation_space.shape[0]
        assert obs.shape == (expected_dim,)

    def test_action_space(self, trading_env):
        assert trading_env.action_space.n == 3

    def test_episode_terminates(self, trading_env):
        trading_env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = trading_env.step(0)
            done = terminated or truncated
            steps += 1
            if steps > 10_000:
                pytest.fail("Episode did not terminate within 10,000 steps")
        assert steps > 0

    def test_buy_then_sell(self, trading_env):
        trading_env.reset(seed=42)
        # Buy
        _, _, _, _, info1 = trading_env.step(1)
        assert info1["position"] > 0, "Position should be > 0 after buy"
        # Sell
        _, _, _, _, info2 = trading_env.step(2)
        assert info2["position"] == 0.0, "Position should be 0 after sell"


class TestRiskFeatures:
    """Tests for ATR volatility sizing and leverage cap."""

    def test_atr_volatility_sizing_scales_with_atr(self, sample_ohlcv, feature_cfg):
        """Higher ATR should produce a smaller position (fewer units bought)."""
        df = compute_all_features(sample_ohlcv, feature_cfg)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]

        common_kwargs = dict(
            feature_cols=feat_cols,
            window_size=10,
            initial_cash=10_000.0,
            position_sizing="atr_volatility",
            risk_per_trade=0.02,
            atr_mult=2.0,
            max_position_value=1.0,  # disable hard cap so ATR drives sizing
        )

        env_low = TradingEnv(df=df.copy(), **common_kwargs)
        env_high = TradingEnv(df=df.copy(), **common_kwargs)

        env_low.reset(seed=42)
        env_high.reset(seed=42)

        # Inject deterministic ATR values: low ATR → large position, high ATR → small position
        env_low._atr = np.full(len(df), 1.0)
        env_high._atr = np.full(len(df), 20.0)

        _, _, _, _, info_low = env_low.step(1)
        _, _, _, _, info_high = env_high.step(1)

        assert info_low["position"] > info_high["position"], (
            f"Low-ATR env should buy more units ({info_low['position']:.6f}) "
            f"than high-ATR env ({info_high['position']:.6f})"
        )

    def test_atr_volatility_sizing_max_position_value_cap(self, sample_ohlcv, feature_cfg):
        """max_position_value should cap spend regardless of ATR magnitude."""
        df = compute_all_features(sample_ohlcv, feature_cfg)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]

        max_pv = 0.05  # only allow 5% of equity per buy
        env = TradingEnv(
            df=df,
            feature_cols=feat_cols,
            window_size=10,
            initial_cash=10_000.0,
            position_sizing="atr_volatility",
            risk_per_trade=0.5,   # very aggressive — would buy a lot without cap
            atr_mult=0.1,         # very small ATR mult — units_from_risk is huge
            max_position_value=max_pv,
        )
        env.reset(seed=42)
        # Very low ATR: uncapped spend would far exceed max_position_value limit
        env._atr = np.full(len(df), 0.01)

        price = env._close[env._current_step]
        _, _, _, _, info = env.step(1)

        # Spend is capped at max_position_value × equity ≈ 0.05 × 10_000 = 500
        max_allowed_spend = max_pv * 10_000.0
        actual_spend = info["position"] * price  # approximate spend (ignores slippage rounding)
        assert actual_spend <= max_allowed_spend * 1.05, (  # 5% tolerance for slippage/cost
            f"Spend {actual_spend:.2f} exceeded cap {max_allowed_spend:.2f}"
        )

    def test_leverage_cap_limits_buy_exposure(self, sample_ohlcv, feature_cfg):
        """max_leverage should prevent exposure from exceeding max_leverage × equity."""
        df = compute_all_features(sample_ohlcv, feature_cfg)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]

        max_lev = 0.2  # allow at most 20% of portfolio in the asset
        env = TradingEnv(
            df=df,
            feature_cols=feat_cols,
            window_size=10,
            initial_cash=10_000.0,
            max_leverage=max_lev,
            position_size_frac=0.9,  # would spend 90% of cash without leverage cap
        )
        env.reset(seed=42)
        price = env._close[env._current_step]

        _, _, _, _, info = env.step(1)

        position = info["position"]
        portfolio_value = info["portfolio_value"]
        exposure = position * price  # exposure at raw close price

        assert exposure <= max_lev * portfolio_value + 1e-6, (
            f"Exposure {exposure:.4f} exceeded max_leverage × equity "
            f"({max_lev} × {portfolio_value:.4f} = {max_lev * portfolio_value:.4f})"
        )

    def test_leverage_cap_default_does_not_restrict_normal_buy(self, sample_ohlcv, feature_cfg):
        """With max_leverage=1.0 (default), a normal fixed_fraction buy should succeed."""
        df = compute_all_features(sample_ohlcv, feature_cfg)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]

        env = TradingEnv(
            df=df,
            feature_cols=feat_cols,
            window_size=10,
            initial_cash=10_000.0,
            max_leverage=1.0,
            position_size_frac=0.1,
        )
        env.reset(seed=42)
        _, _, _, _, info = env.step(1)

        assert info["position"] > 0, "Buy should succeed with default max_leverage=1.0"

    def test_atr_volatility_fallback_when_atr_unavailable(self, sample_ohlcv, feature_cfg):
        """ATR sizing should fall back to fixed_fraction when ATR is zero."""
        df = compute_all_features(sample_ohlcv, feature_cfg)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]

        env_atr = TradingEnv(
            df=df,
            feature_cols=feat_cols,
            window_size=10,
            initial_cash=10_000.0,
            position_sizing="atr_volatility",
            position_size_frac=0.1,
            risk_per_trade=0.01,
            atr_mult=2.0,
        )
        env_fixed = TradingEnv(
            df=df,
            feature_cols=feat_cols,
            window_size=10,
            initial_cash=10_000.0,
            position_sizing="fixed_fraction",
            position_size_frac=0.1,
        )

        env_atr.reset(seed=42)
        env_fixed.reset(seed=42)

        # Force ATR to zero → triggers fallback
        env_atr._atr = np.zeros(len(df))

        _, _, _, _, info_atr = env_atr.step(1)
        _, _, _, _, info_fixed = env_fixed.step(1)

        assert abs(info_atr["position"] - info_fixed["position"]) < 1e-9, (
            "ATR sizing with zero ATR should produce same result as fixed_fraction"
        )
