"""CLI: Train an RL agent.

Usage
-----
    python -m src.cli.train --algo qlearning --symbol BTC-USD --config configs/qlearning.yaml
    python -m src.cli.train --algo dqn --symbol BTC-USD --config configs/dqn.yaml
    python -m src.cli.train --algo ppo --symbol BTC-USD --config configs/ppo.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import load_config, save_config_snapshot
from src.utils.logging import get_logger
from src.utils.paths import data_processed_dir, new_run_dir
from src.utils.seeds import set_global_seed

logger = get_logger(__name__)


def _load_data(symbol: str):
    """Load processed train/val DataFrames and feature columns."""
    d = data_processed_dir(symbol)
    train = pd.read_csv(d / "train.csv", parse_dates=True, index_col=0)
    val = pd.read_csv(d / "val.csv", parse_dates=True, index_col=0)
    with open(d / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return train, val, feature_cols


def _train_qlearning(train_df, feature_cols, cfg, run_dir):
    from src.agents.qlearning import QLearningAgent
    from src.env.wrappers import make_env
    from src.eval.plots import plot_training_curve

    ql_cfg = cfg.get("qlearning", {})
    env_cfg = {**cfg.get("env", {}), **cfg.get("features", {})}
    seed = cfg.get("training", {}).get("seed", 42)

    env = make_env(train_df, feature_cols, env_cfg)

    # Map feature names to observation indices (last N elements before portfolio stats)
    features_used = ql_cfg.get("features_used")
    feat_indices = None
    if features_used:
        # Use last 3 (position, cash, value) + requested features from window tail
        obs_dim = env.observation_space.shape[0]
        # The last 3 are portfolio stats; before that is the flattened window
        # Use a simple approach: include last-window-row features + portfolio stats
        n_feat = len(feature_cols)
        window = cfg.get("features", {}).get("window_size", 50)
        # Last row of window starts at index (window-1)*n_feat
        last_row_start = (window - 1) * n_feat
        selected = []
        for fname in features_used:
            if fname == "position":
                selected.append(obs_dim - 3)
            elif fname in feature_cols:
                selected.append(last_row_start + feature_cols.index(fname))
        # Always include portfolio stats
        selected.extend([obs_dim - 3, obs_dim - 2, obs_dim - 1])
        feat_indices = sorted(set(selected))

    agent = QLearningAgent(
        n_actions=3,
        n_bins=ql_cfg.get("n_bins", 10),
        feature_indices=feat_indices,
        alpha=ql_cfg.get("alpha", 0.1),
        gamma=ql_cfg.get("gamma", 0.99),
        epsilon_start=ql_cfg.get("epsilon_start", 1.0),
        epsilon_end=ql_cfg.get("epsilon_end", 0.01),
        epsilon_decay=ql_cfg.get("epsilon_decay", 0.995),
        seed=seed,
    )

    episodes = ql_cfg.get("episodes", cfg.get("training", {}).get("episodes", 500))
    rewards = agent.train(env, n_episodes=episodes, log_interval=cfg.get("training", {}).get("log_interval", 50))

    agent.save(run_dir / "model")
    plot_training_curve(rewards, run_dir, title="Q-Learning Training")
    logger.info(f"[green]Q-Learning training complete.[/green] {len(agent.q_table)} states learned.")
    return agent


def _train_dqn(train_df, feature_cols, cfg, run_dir):
    from src.agents.dqn import DQNAgent
    from src.env.wrappers import make_env
    from src.eval.plots import plot_training_curve

    dqn_cfg = cfg.get("dqn", {})
    env_cfg = {**cfg.get("env", {}), **cfg.get("features", {})}
    seed = cfg.get("training", {}).get("seed", 42)

    env = make_env(train_df, feature_cols, env_cfg)
    obs_dim = env.observation_space.shape[0]

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=3,
        hidden_sizes=dqn_cfg.get("hidden_sizes", [128, 64]),
        activation=dqn_cfg.get("activation", "relu"),
        lr=dqn_cfg.get("learning_rate", 3e-4),
        gamma=dqn_cfg.get("gamma", 0.99),
        batch_size=dqn_cfg.get("batch_size", 64),
        buffer_size=dqn_cfg.get("buffer_size", 50_000),
        target_update_freq=dqn_cfg.get("target_update_freq", 1000),
        epsilon_start=dqn_cfg.get("epsilon_start", 1.0),
        epsilon_end=dqn_cfg.get("epsilon_end", 0.01),
        epsilon_decay_steps=dqn_cfg.get("epsilon_decay_steps", 50_000),
        double_dqn=dqn_cfg.get("double_dqn", True),
        warmup_steps=dqn_cfg.get("warmup_steps", 1000),
        seed=seed,
    )

    episodes = dqn_cfg.get("episodes", cfg.get("training", {}).get("episodes", 500))
    max_steps = dqn_cfg.get("max_steps_per_episode", 0)
    rewards = agent.train(env, n_episodes=episodes, max_steps=max_steps,
                          log_interval=cfg.get("training", {}).get("log_interval", 10))

    agent.save(run_dir / "model")
    plot_training_curve(rewards, run_dir, title="DQN Training")
    logger.info("[green]DQN training complete.[/green]")
    return agent


def _train_ppo(train_df, val_df, feature_cols, cfg, run_dir):
    from src.agents.ppo import train_ppo
    from src.eval.plots import plot_training_curve

    model = train_ppo(train_df, val_df, feature_cols, cfg, run_dir)

    # Load episode rewards for plotting
    meta_path = run_dir / "ppo_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        rewards = meta.get("episode_rewards", [])
        if rewards:
            plot_training_curve(rewards, run_dir, title="PPO Training")

    logger.info("[green]PPO training complete.[/green]")
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--algo", type=str, required=True, choices=["qlearning", "dqn", "ppo"])
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--config", type=str, default=None, help="Algorithm-specific YAML config")
    args = parser.parse_args(argv)

    # Determine config file
    config_file = args.config
    if config_file is None:
        config_file = f"{args.algo}.yaml"

    cfg = load_config(config_file)
    seed = cfg.get("training", {}).get("seed", 42)
    set_global_seed(seed)

    train_df, val_df, feature_cols = _load_data(args.symbol)

    run_dir = new_run_dir(tag=f"{args.algo}_{args.symbol}")
    save_config_snapshot(cfg, run_dir)

    logger.info(f"[bold]Training {args.algo.upper()}[/bold] on {args.symbol}  →  {run_dir}")

    if args.algo == "qlearning":
        _train_qlearning(train_df, feature_cols, cfg, run_dir)
    elif args.algo == "dqn":
        _train_dqn(train_df, feature_cols, cfg, run_dir)
    elif args.algo == "ppo":
        _train_ppo(train_df, val_df, feature_cols, cfg, run_dir)

    logger.info(f"[bold green]Run directory:[/bold green] {run_dir}")


if __name__ == "__main__":
    main()
