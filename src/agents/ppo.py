"""PPO agent via Stable-Baselines3.

Supports both **discrete** and **continuous** action variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.wrappers import make_env
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

logger = get_logger(__name__)


# ======================================================================
# Custom callback for logging
# ======================================================================

class RewardLoggerCallback(BaseCallback):
    """Log episode rewards during PPO training."""

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        # SB3 auto-resets; check for episode end via info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        n = len(self.episode_rewards)
        if n > 0 and n % self.log_interval == 0:
            avg = np.mean(self.episode_rewards[-self.log_interval:])
            logger.info(f"[PPO] Episodes ~{n}  avg_reward={avg:.4f}")


# ======================================================================
# PPO training
# ======================================================================

def train_ppo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any],
    run_dir: Path,
) -> PPO:
    """Train a PPO model and save it to *run_dir*.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (already normalised).
    val_df : pd.DataFrame
        Validation data for eval callback.
    feature_cols : list[str]
        Feature column names.
    cfg : dict
        Full merged config dict.
    run_dir : Path
        Output directory for checkpoints.

    Returns
    -------
    PPO
        The trained SB3 model.
    """
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = {**cfg.get("env", {}), **cfg.get("features", {})}
    seed = cfg.get("training", {}).get("seed", 42)
    set_global_seed(seed)

    continuous = ppo_cfg.get("action_type", "discrete") == "continuous"

    # --- Build environments ---
    def _make_train_env():
        env = make_env(train_df, feature_cols, env_cfg, continuous=continuous)
        return Monitor(env)

    def _make_eval_env():
        env = make_env(val_df, feature_cols, env_cfg, continuous=continuous)
        return Monitor(env)

    train_vec = DummyVecEnv([_make_train_env])
    eval_vec = DummyVecEnv([_make_eval_env])

    # --- Build net arch ---
    net_arch_cfg = ppo_cfg.get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
    if isinstance(net_arch_cfg, dict):
        net_arch = [dict(pi=net_arch_cfg["pi"], vf=net_arch_cfg["vf"])]
    else:
        net_arch = net_arch_cfg

    # --- Create model ---
    model = PPO(
        policy="MlpPolicy",
        env=train_vec,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        policy_kwargs={"net_arch": net_arch},
        seed=seed,
        verbose=0,
    )

    # --- Callbacks ---
    reward_cb = RewardLoggerCallback(log_interval=5)
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=str(run_dir / "best_model"),
        eval_freq=ppo_cfg.get("eval_freq", 10_000),
        n_eval_episodes=ppo_cfg.get("n_eval_episodes", 5),
        deterministic=True,
        verbose=0,
    )

    total_timesteps = ppo_cfg.get("total_timesteps", 200_000)
    logger.info(f"[PPO] Training for {total_timesteps} timesteps  (continuous={continuous})")
    model.learn(total_timesteps=total_timesteps, callback=[reward_cb, eval_cb])

    # --- Save ---
    model.save(str(run_dir / "ppo_model"))
    meta = {
        "total_timesteps": total_timesteps,
        "action_type": "continuous" if continuous else "discrete",
        "episode_rewards": reward_cb.episode_rewards,
    }
    with open(run_dir / "ppo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[PPO] Model saved to {run_dir}")
    return model


def load_ppo(run_dir: Path) -> PPO:
    """Load a saved PPO model."""
    model_path = run_dir / "ppo_model.zip"
    if not model_path.exists():
        model_path = run_dir / "ppo_model"
    return PPO.load(str(model_path))
