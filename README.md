# Autonomous AI Trading Bot

**UWE Digital Systems Project (UFCFXK-30-3)**

An end-to-end reinforcement learning trading system that trains, evaluates,
and deploys three RL algorithms (Q-Learning, DQN, PPO) on live cryptocurrency
markets via the Bybit exchange.  Includes a real-time data pipeline, council-based
multi-model voting, and an interactive Streamlit dashboard with live price charts.

---

## Features

- **Data pipeline** — Yahoo Finance download, CSV import, feature engineering
  (SMA, EMA, RSI, MACD, ATR, volatility), train/val/test split with no
  leakage, train-only scaler normalisation.
- **Custom Gymnasium environment** — windowed observation, discrete and
  continuous action spaces, configurable reward (log-return / Sharpe),
  drawdown penalty, transaction costs & slippage.
- **Three RL agents**:
  - **Q-Learning** (tabular with configurable binning)
  - **DQN** (PyTorch, Double DQN, experience replay, target network)
  - **PPO** (Stable-Baselines3, discrete & continuous action support)
- **Baseline strategies** — Buy & Hold, Moving Average Crossover.
- **Risk management**:
  - *Stop-loss* — percent-based or ATR-based; enforced by `RiskManager`.
  - *Position sizing* — `fixed_fraction` or `atr_volatility`.
  - *Leverage limit* — `max_leverage` parameter (default 1.0).
  - *Live guards* — stale price detection (>3 s), cooldown between trades,
    max position size cap.
- **Live trading backend** — FastAPI server with:
  - Bybit v5 public WebSocket connection (`wss://stream.bybit.com/v5/public/linear`)
  - Auto-reconnect with exponential backoff (1 s → 60 s cap)
  - In-memory `PriceCache` (source of truth for BTCUSDT & ETHUSDT)
  - REST API: `GET /prices`, `GET /prices/{symbol}`, `GET /history/{symbol}`, `GET /decisions`
  - WebSocket endpoint: `/ws/prices` (multiplexed tickers + AI decisions)
  - 1-second strategy decision loop with council voting
- **Council mode** — select multiple trained models; majority-vote decides
  each trade (tie-breaking: Hold > Buy > Sell).
- **Bybit demo/testnet integration** — place real market orders on the Bybit
  demo account via `POST /v5/order/create` with HMAC-SHA256 signing.
- **Interactive dashboard** — Streamlit + Plotly with:
  - Real-time price chart (2 s auto-refresh via `@st.fragment`)
  - Live/Yahoo Finance fallback with connection status badge
  - Multi-symbol ticker strip (BTC & ETH bid/ask/last)
  - Entry point, take-profit, stop-loss visualisation
  - Configurable timeframes (1H, 1W, 1M, 1Y)
  - Auto-trading with configurable interval (30 s, 1 min, 5 min, 15 min)
  - Trade log with order IDs and AI decision stream
  - Experiment comparison (equity curves, drawdown, metrics)
  - User authentication with local password hashing (PBKDF2)
  - Dark / light theme toggle
- **Evaluation** — unified metrics suite (return, Sharpe, drawdown, win rate,
  trade count, exposure).
- **Reproducibility** — YAML configs, deterministic seeds, config snapshots.

---

## Quick Start

### 1. Environment Setup

```bash
cd "new project"

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e .
```

### 2. Download & Preprocess Data

```bash
python -m src.cli.download_data --symbol BTC-USD --start 2020-01-01 --end 2025-12-31
python -m src.cli.preprocess --symbol BTC-USD
```

### 3. Train Agents

```bash
python -m src.cli.train --algo qlearning --symbol BTC-USD --config configs/qlearning.yaml
python -m src.cli.train --algo dqn       --symbol BTC-USD --config configs/dqn.yaml
python -m src.cli.train --algo ppo       --symbol BTC-USD --config configs/ppo.yaml
```

### 4. Run Baselines & Evaluate

```bash
python -m src.cli.baselines --symbol BTC-USD --config configs/baselines.yaml
python -m src.cli.evaluate --run_dir runs/<run_dir> --symbol BTC-USD
```

### 5. Launch the Live Backend

```bash
python -m src.live.server
```

This starts the FastAPI server on `http://localhost:8001`:
- Connects to Bybit's public WebSocket for BTCUSDT & ETHUSDT live tickers
- Loads all trained models from `runs/`
- Runs the 1-second strategy decision loop
- Exposes REST + WebSocket endpoints for the dashboard

### 6. Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard automatically detects the live backend:
- **Backend running** — green "LIVE" badge, real-time Bybit prices, 2 s chart refresh
- **Backend not running** — yellow "Yahoo Finance" badge, falls back to historical data

### Full Pipeline (All Steps at Once)

```bash
python -m src.cli.pipeline --symbol BTC-USD
```

---

## Architecture

```
Bybit Exchange
    │
    │  WebSocket (wss://stream.bybit.com/v5/public/linear)
    │  tickers.BTCUSDT, tickers.ETHUSDT
    ▼
┌─────────────────────────────────┐
│  BybitWebSocket (exchange_ws)   │  Auto-reconnect + exponential backoff
│  Parses tickers → Tick objects  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  PriceCache (price_cache)       │  Single source of truth
│  {symbol: {last, bid, ask, ts}} │  Rolling history buffer (7200 ticks)
│  Pub/sub for WebSocket clients  │
└──────────┬──────────┬───────────┘
           │          │
           ▼          ▼
┌──────────────┐  ┌───────────────────────┐
│  FastAPI      │  │  StrategyRunner       │
│  REST + WS    │  │  1 Hz decision loop   │
│  /prices      │  │  Council vote         │
│  /history     │  │  Risk guards          │
│  /decisions   │  │  Decision buffer      │
│  /ws/prices   │  └───────────────────────┘
└──────┬───────┘
       │
       │  HTTP polling (every 2 s)
       ▼
┌─────────────────────────────────┐
│  Streamlit Dashboard (app.py)   │
│  Live chart + KPIs + trade log  │
│  Bybit order execution          │
│  Council config + auth          │
└─────────────────────────────────┘
```

---

## API Endpoints (Live Backend)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/prices` | All cached prices (BTCUSDT, ETHUSDT) |
| `GET` | `/prices/{symbol}` | Single symbol price |
| `GET` | `/history/{symbol}?limit=300` | Rolling tick history |
| `GET` | `/decisions?limit=50` | Recent strategy decisions |
| `GET` | `/status` | Server health & model info |
| `POST` | `/strategy/models` | Set active models `{"names": [...]}` |
| `POST` | `/strategy/start` | Start decision loop |
| `POST` | `/strategy/stop` | Stop decision loop |
| `WS` | `/ws/prices` | Multiplexed stream of tickers + decisions |

---

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml              # Shared defaults
│   ├── qlearning.yaml         # Q-Learning hyperparameters
│   ├── dqn.yaml               # DQN hyperparameters
│   ├── ppo.yaml               # PPO hyperparameters
│   └── baselines.yaml         # Baseline strategy config
├── data/
│   ├── raw/                   # Downloaded OHLCV
│   ├── processed/             # Features + splits
│   └── users.json             # Local user accounts (hashed passwords)
├── runs/                      # Experiment outputs (models, metrics, charts)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── RISK_MANAGEMENT.md
│   └── LITERATURE.md
├── src/
│   ├── cli/                   # CLI entry points (train, evaluate, pipeline)
│   ├── data/                  # Data loading, schema, feature engineering
│   ├── env/                   # Gymnasium trading environment
│   ├── agents/                # RL agents (Q-Learning, DQN, PPO)
│   ├── baselines/             # Buy & Hold, MA Crossover
│   ├── risk/                  # RiskManager (stop-loss, sizing)
│   ├── eval/                  # Backtester, metrics, plots
│   ├── live/                  # Live trading backend
│   │   ├── price_cache.py     #   In-memory price cache + pub/sub
│   │   ├── exchange_ws.py     #   Bybit v5 WebSocket client
│   │   ├── strategy_runner.py #   1 Hz decision loop with risk guards
│   │   └── server.py          #   FastAPI server (REST + WS)
│   ├── dashboard/             # Streamlit app
│   │   ├── app.py             #   Main dashboard (auth, trading, charts)
│   │   ├── auth.py            #   User auth + Bybit API helpers
│   │   └── trader.py          #   Model loading, council voting, obs builder
│   └── utils/                 # Config, logging, paths, seeds
└── tests/                     # pytest test suite
```

---

## Configuration

All parameters are controlled through YAML files in `configs/`.
Algorithm-specific configs inherit from `base.yaml` via deep merge.

| Parameter | Value |
|-----------|-------|
| Symbol | BTC-USD |
| Date range | 2020-01-01 to 2025-12-31 |
| Initial cash | $10,000 |
| Transaction cost | 0.1% |
| Slippage | 0.05% |
| Window size | 50 |
| Train/Val/Test split | 70/15/15% |
| Seed | 42 |
| Position sizing | fixed_fraction (10% of cash) |
| Max leverage | 1.0 (no borrowing) |
| Stop-loss | percent (5%) |

---

## Trading Modes

### Council Mode (Multi-Model Voting)
Select multiple trained models in the dashboard. Each model independently
evaluates the current market state and votes BUY, SELL, or HOLD. The
majority vote wins, with conservative tie-breaking (Hold > Buy > Sell).

### Bybit Integration
- **Demo account** — `https://api-demo.bybit.com` (recommended for testing)
- **Testnet** — `https://api-testnet.bybit.com`
- **Spot or Futures** — selectable per trading session
- Orders are placed via `POST /v5/order/create` with HMAC-SHA256 request signing
- API keys are stored locally only and never transmitted to third parties

### Live Data Pipeline
The backend connects to Bybit's public WebSocket for real-time ticker data.
No API keys are required for price data — only for placing orders.

---

## Testing

```bash
pytest tests/ -v
pytest tests/test_env.py -v
```

---

## Requirements

- Python 3.11+
- Dependencies managed via `pyproject.toml` (install with `pip install -e .`)
- Key libraries: PyTorch, Stable-Baselines3, Gymnasium, Pandas, Plotly,
  Streamlit, FastAPI, Uvicorn, WebSockets

---

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — system design with Mermaid diagram
- **[Experiments](docs/EXPERIMENTS.md)** — how to run, configure, and compare
- **[Risk Management](docs/RISK_MANAGEMENT.md)** — stop-loss, sizing, reward design
- **[Literature](docs/LITERATURE.md)** — academic references (UWE Harvard format)

---

## License

MIT — Academic project for UWE Bristol.
