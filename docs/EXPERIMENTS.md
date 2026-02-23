# Experiments Guide

## Reproducibility

All experiments are controlled by YAML configuration files under `configs/`.
Each training run saves a snapshot of the resolved config to
`runs/<run_dir>/config.yaml`, ensuring full reproducibility.

**Key reproducibility mechanisms:**
- Global seed set via `training.seed` in config (default: 42).
- Seeding covers Python `random`, NumPy, and PyTorch (CPU & CUDA).
- `torch.backends.cudnn.deterministic = True` when using GPU.
- Scaler fitted on training data only — saved and reloaded.

## Running Experiments

### 1. Full Pipeline (Recommended)

```bash
python -m src.cli.pipeline --symbol BTC-USD
```

This runs: download → preprocess → train (Q-Learning, DQN, PPO) → baselines → evaluate.

### 2. Individual Steps

```bash
# Download
python -m src.cli.download_data --symbol BTC-USD --start 2020-01-01 --end 2025-12-31

# Preprocess
python -m src.cli.preprocess --symbol BTC-USD

# Train (pick one)
python -m src.cli.train --algo qlearning --symbol BTC-USD --config configs/qlearning.yaml
python -m src.cli.train --algo dqn       --symbol BTC-USD --config configs/dqn.yaml
python -m src.cli.train --algo ppo       --symbol BTC-USD --config configs/ppo.yaml

# Baselines
python -m src.cli.baselines --symbol BTC-USD --config configs/baselines.yaml

# Evaluate a specific run
python -m src.cli.evaluate --run_dir runs/<run_dir> --symbol BTC-USD
```

### 3. Custom Configurations

Copy a config file and modify parameters:

```bash
cp configs/dqn.yaml configs/dqn_experiment2.yaml
# Edit the new file, then:
python -m src.cli.train --algo dqn --symbol BTC-USD --config configs/dqn_experiment2.yaml
```

## Hyperparameter Guide

### Q-Learning
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bins` | 10 | Discretisation bins per feature |
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_decay` | 0.995 | Per-episode multiplicative decay |
| `episodes` | 1000 | Training episodes |

### DQN
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_sizes` | [128, 64] | Network layer sizes |
| `learning_rate` | 0.0003 | Adam LR |
| `batch_size` | 64 | Replay batch size |
| `buffer_size` | 50000 | Replay buffer capacity |
| `target_update_freq` | 1000 | Steps between target net sync |
| `double_dqn` | true | Use Double DQN |
| `epsilon_decay_steps` | 50000 | Linear epsilon schedule steps |

### PPO
| Parameter | Default | Description |
|-----------|---------|-------------|
| `action_type` | discrete | discrete or continuous |
| `n_steps` | 2048 | Rollout length |
| `n_epochs` | 10 | PPO update epochs |
| `clip_range` | 0.2 | PPO clipping |
| `total_timesteps` | 200000 | Total training steps |

## Evaluation Metrics

All strategies are evaluated with the same metric suite:

- **Cumulative Return** — total percentage gain/loss
- **Annualised Return** — CAGR approximation
- **Annualised Volatility** — standard deviation of returns, annualised
- **Sharpe Ratio** — risk-adjusted return (risk-free = 0)
- **Max Drawdown** — worst peak-to-trough decline
- **Win Rate** — fraction of profitable round-trip trades
- **Number of Trades** — total trade actions
- **Exposure Time** — fraction of time holding a position

## Expected Outputs

Each run directory (`runs/<timestamp>/`) contains:

```
runs/<timestamp>/
├── config.yaml            # resolved config snapshot
├── <algo>_metrics.json    # performance metrics
├── <algo>_trades.csv      # individual trade log
├── <algo>_equity_curve.csv
├── equity_curves.png      # equity chart
├── drawdown.png
├── trades_on_price.png
├── positions.png
├── training_curve.png     # reward over episodes
├── RESULTS_SUMMARY.md     # auto-generated summary
└── model/                 # saved agent (Q-table / weights)
```

## Comparing Results

Use the Streamlit dashboard to compare multiple runs side-by-side:

```bash
streamlit run src/dashboard/app.py
```

Select multiple run directories from the sidebar to overlay equity curves and
compare metrics in a single table.
