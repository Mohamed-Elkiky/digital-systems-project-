"""Tests for deterministic reproducibility with fixed seeds."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.features import compute_all_features
from src.data.preprocess import EXCLUDE_FROM_SCALING
from src.env.trading_env import TradingEnv
from src.utils.seeds import set_global_seed


@pytest.fixture
def env_and_cols(sample_ohlcv, feature_cfg):
    df = compute_all_features(sample_ohlcv, feature_cfg)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_SCALING]
    return df, feat_cols


def _run_episode(df, feat_cols, seed: int = 42) -> list[float]:
    """Run a deterministic episode returning a list of rewards."""
    set_global_seed(seed)
    env = TradingEnv(df=df, feature_cols=feat_cols, window_size=10, initial_cash=10_000.0)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards = []
    done = False
    while not done:
        action = int(rng.integers(0, 3))
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards


class TestDeterminism:
    """Same seed → same trajectory."""

    def test_same_seed_same_rewards(self, env_and_cols):
        df, feat_cols = env_and_cols
        r1 = _run_episode(df, feat_cols, seed=42)
        r2 = _run_episode(df, feat_cols, seed=42)
        assert len(r1) == len(r2), "Episode lengths should match"
        np.testing.assert_array_almost_equal(r1, r2, decimal=10)

    def test_different_seed_different_rewards(self, env_and_cols):
        df, feat_cols = env_and_cols
        r1 = _run_episode(df, feat_cols, seed=42)
        r2 = _run_episode(df, feat_cols, seed=99)
        # With different seeds, at least some rewards should differ
        if len(r1) == len(r2):
            assert not np.allclose(r1, r2), "Different seeds should yield different trajectories"
