"""Tabular Q-Learning agent with configurable state discretisation.

The continuous observation vector is mapped to a discrete state via
per-dimension binning.  Only a subset of feature indices are used
(configured via ``features_used``), keeping the table tractable.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

logger = get_logger(__name__)


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self,
        n_actions: int = 3,
        n_bins: int = 10,
        feature_indices: list[int] | None = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.feature_indices = feature_indices  # indices into obs vector
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.seed = seed

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # Bin edges — fitted on first episode's observations
        self._bin_edges: list[np.ndarray] | None = None
        self._rng = np.random.default_rng(seed)

        # Tracking
        self.training_rewards: list[float] = []
        self.training_epsilons: list[float] = []

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------

    def _select_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract a subset of the observation for state representation."""
        if self.feature_indices is not None:
            return obs[self.feature_indices]
        # Default: use the last 3 elements (position, cash_frac, value_frac)
        # plus a few from the window tail
        n = len(obs)
        idx = list(range(max(0, n - 8), n))
        return obs[idx]

    def fit_bins(self, observations: list[np.ndarray]) -> None:
        """Compute bin edges from a batch of observations (e.g. one episode)."""
        mat = np.array([self._select_features(o) for o in observations])
        self._bin_edges = []
        for col_idx in range(mat.shape[1]):
            col = mat[:, col_idx]
            edges = np.linspace(col.min() - 1e-6, col.max() + 1e-6, self.n_bins + 1)
            self._bin_edges.append(edges)

    def discretise(self, obs: np.ndarray) -> tuple:
        """Map a continuous observation to a discrete state tuple."""
        feats = self._select_features(obs)
        if self._bin_edges is None:
            # Fall back to uniform binning
            return tuple(np.clip(np.round(feats * self.n_bins).astype(int), 0, self.n_bins - 1))
        state = []
        for i, val in enumerate(feats):
            b = int(np.digitize(val, self._bin_edges[i])) - 1
            b = max(0, min(b, self.n_bins - 1))
            state.append(b)
        return tuple(state)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        state = self.discretise(obs)
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        q_vals = self.q_table[state]
        return int(np.argmax(q_vals))

    def update(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        state = self.discretise(obs)
        next_state = self.discretise(next_obs)
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 1000,
        log_interval: int = 50,
    ) -> list[float]:
        """Train for *n_episodes* and return per-episode rewards."""
        set_global_seed(self.seed)
        episode_rewards: list[float] = []

        # Collect one episode for bin fitting
        obs, _ = env.reset(seed=self.seed)
        obs_buffer = [obs.copy()]
        done = False
        while not done:
            action = int(self._rng.integers(0, self.n_actions))
            obs, _, terminated, truncated, _ = env.step(action)
            obs_buffer.append(obs.copy())
            done = terminated or truncated
        self.fit_bins(obs_buffer)

        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.update(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs
                total_reward += reward
                done = terminated or truncated

            self.decay_epsilon()
            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)
            self.training_epsilons.append(self.epsilon)

            if ep % log_interval == 0:
                avg = np.mean(episode_rewards[-log_interval:])
                logger.info(
                    f"[QL] Episode {ep}/{n_episodes}  avg_reward={avg:.4f}  "
                    f"eps={self.epsilon:.4f}  states={len(self.q_table)}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "q_table.pkl", "wb") as f:
            pickle.dump(dict(self.q_table), f)
        with open(path / "bin_edges.pkl", "wb") as f:
            pickle.dump(self._bin_edges, f)
        meta = {
            "n_actions": self.n_actions,
            "n_bins": self.n_bins,
            "feature_indices": self.feature_indices,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "n_states": len(self.q_table),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Q-table saved to {path}  ({len(self.q_table)} states)")

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        agent = cls(
            n_actions=meta["n_actions"],
            n_bins=meta["n_bins"],
            feature_indices=meta.get("feature_indices"),
            alpha=meta["alpha"],
            gamma=meta["gamma"],
        )
        agent.epsilon = meta.get("epsilon", 0.0)
        with open(path / "q_table.pkl", "rb") as f:
            raw = pickle.load(f)
            agent.q_table = defaultdict(lambda: np.zeros(agent.n_actions), raw)
        with open(path / "bin_edges.pkl", "rb") as f:
            agent._bin_edges = pickle.load(f)
        logger.info(f"Q-table loaded from {path}  ({len(agent.q_table)} states)")
        return agent
