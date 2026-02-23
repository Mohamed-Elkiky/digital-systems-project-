"""Deep Q-Network agent with experience replay and optional Double DQN.

Built on PyTorch.  Supports:
  - Configurable hidden layer sizes.
  - Target network with periodic hard updates.
  - Linear epsilon-greedy schedule.
  - Model checkpoint save / load.
"""

from __future__ import annotations

import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

logger = get_logger(__name__)


# ======================================================================
# Replay Buffer
# ======================================================================

class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# Q-Network
# ======================================================================

class QNetwork(nn.Module):
    """Multi-layer perceptron Q-function approximator."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int], activation: str = "relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(activation, nn.ReLU)
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# DQN Agent
# ======================================================================

class DQNAgent:
    """DQN (optionally Double DQN) for discrete-action trading."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50_000,
        double_dqn: bool = True,
        warmup_steps: int = 1000,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.double_dqn = double_dqn
        self.warmup_steps = warmup_steps
        self.seed = seed

        hidden_sizes = hidden_sizes or [128, 64]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(obs_dim, n_actions, hidden_sizes, activation).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_sizes, activation).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.total_steps = 0
        self.training_losses: list[float] = []
        self.training_rewards: list[float] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: select action with policy net, evaluate with target net
                best_actions = self.policy_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(dim=1).values

            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def _update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 500,
        max_steps: int = 0,
        log_interval: int = 10,
    ) -> list[float]:
        set_global_seed(self.seed)
        episode_rewards: list[float] = []

        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            ep_step = 0

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.buffer.push(obs, action, reward, next_obs, float(done))

                if self.total_steps >= self.warmup_steps:
                    loss = self._learn()
                    if loss > 0:
                        self.training_losses.append(loss)

                if self.total_steps % self.target_update_freq == 0:
                    self._update_target()

                obs = next_obs
                total_reward += reward
                self.total_steps += 1
                ep_step += 1
                if max_steps > 0 and ep_step >= max_steps:
                    break

            episode_rewards.append(total_reward)
            self.training_rewards.append(total_reward)

            if ep % log_interval == 0:
                avg = np.mean(episode_rewards[-log_interval:])
                avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0.0
                logger.info(
                    f"[DQN] Episode {ep}/{n_episodes}  avg_reward={avg:.4f}  "
                    f"eps={self.epsilon:.4f}  loss={avg_loss:.6f}  steps={self.total_steps}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path / "policy_net.pt")
        torch.save(self.target_net.state_dict(), path / "target_net.pt")
        meta = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "total_steps": self.total_steps,
            "double_dqn": self.double_dqn,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"DQN model saved to {path}")

    @classmethod
    def load(cls, path: Path, obs_dim: int, **kwargs) -> "DQNAgent":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        agent = cls(obs_dim=obs_dim, n_actions=meta["n_actions"], double_dqn=meta["double_dqn"], **kwargs)
        agent.policy_net.load_state_dict(torch.load(path / "policy_net.pt", map_location=agent.device))
        agent.target_net.load_state_dict(torch.load(path / "target_net.pt", map_location=agent.device))
        agent.total_steps = meta.get("total_steps", 0)
        logger.info(f"DQN model loaded from {path}")
        return agent
