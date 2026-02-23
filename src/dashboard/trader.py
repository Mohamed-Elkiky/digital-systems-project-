"""Council trading logic — model loading, live observation, majority vote.

Usage
-----
    from src.dashboard.trader import list_runs, load_model, build_live_obs, council_vote

    models   = [load_model(RUNS_DIR / name) for name in selected_names]
    obs, inf = build_live_obs("BTC-USD")
    result   = council_vote(models, obs)
    # result["action"]  → 0=Hold, 1=Buy, 2=Sell
    # result["model_votes"] → per-model breakdown
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR     = PROJECT_ROOT / "runs"

ACTION_NAMES  = {0: "Hold", 1: "Buy", 2: "Sell"}
ACTION_EMOJIS = {0: "⏸ Hold", 1: "📈 Buy", 2: "📉 Sell"}


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def list_runs() -> list[dict]:
    """Return metadata for every run directory found in runs/."""
    if not RUNS_DIR.exists():
        return []
    result = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name  = d.name
        parts = name.split("_")
        algo  = next((p for p in parts if p in ("dqn", "ppo", "qlearning")), "unknown")
        raw_date = parts[0] if parts else ""
        date_fmt = (
            f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            if len(raw_date) >= 8 else raw_date
        )
        result.append({
            "name":  name,
            "algo":  algo.upper(),
            "date":  date_fmt,
            "label": f"{algo.upper()} · {date_fmt}",
            "path":  d,
        })
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _detect_algo(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "qlearning" in name:
        return "qlearning"
    if "ppo" in name:
        return "ppo"
    return "dqn"


def load_model(run_dir: Path) -> dict[str, Any]:
    """Load any model type from *run_dir*.

    Returns a dict::

        {
          "algo":  "dqn" | "ppo" | "qlearning",
          "agent": <loaded model object>,
          "meta":  <meta.json contents>,
          "run":   <run directory name>,
        }
    """
    algo      = _detect_algo(run_dir)
    model_dir = run_dir / "model"

    if algo == "qlearning":
        from src.agents.qlearning import QLearningAgent
        agent = QLearningAgent.load(model_dir)
        meta  = json.loads((model_dir / "meta.json").read_text())
        return {"algo": "qlearning", "agent": agent, "meta": meta, "run": run_dir.name}

    if algo == "dqn":
        meta = json.loads((model_dir / "meta.json").read_text())
        from src.agents.dqn import DQNAgent
        agent = DQNAgent.load(model_dir, obs_dim=meta["obs_dim"])
        return {"algo": "dqn", "agent": agent, "meta": meta, "run": run_dir.name}

    # ppo
    from src.agents.ppo import load_ppo
    model    = load_ppo(run_dir)
    meta_p   = run_dir / "ppo_meta.json"
    meta     = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return {"algo": "ppo", "agent": model, "meta": meta, "run": run_dir.name}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_info: dict, obs: np.ndarray) -> int:
    """Return action int (0=Hold, 1=Buy, 2=Sell) from a loaded model."""
    algo  = model_info["algo"]
    agent = model_info["agent"]
    if algo in ("qlearning", "dqn"):
        return int(agent.select_action(obs, greedy=True))
    # ppo
    action, _ = agent.predict(obs, deterministic=True)
    return int(action)


def council_vote(models: list[dict], obs: np.ndarray) -> dict:
    """Run every model on *obs* and return a majority-vote result.

    Tie-breaking rule: **Hold (0) > Buy (1) > Sell (2)** so the council
    is conservative when models disagree equally.

    Returns::

        {
          "action":      int,               # winning action
          "label":       str,               # "Hold" | "Buy" | "Sell"
          "emoji":       str,               # e.g. "📈 Buy"
          "counts":      dict[int, int],    # {action: vote_count}
          "model_votes": list[dict],        # per-model detail
          "total":       int,               # number of models
        }
    """
    votes: list[int] = []
    model_votes: list[dict] = []

    for m in models:
        try:
            a = predict(m, obs)
        except Exception as exc:
            a = 0  # default to Hold on error
            print(f"[council] error in {m['run']}: {exc}")
        votes.append(a)
        model_votes.append({
            "run":    m["run"],
            "algo":   m["algo"].upper(),
            "action": a,
            "label":  ACTION_NAMES[a],
            "emoji":  ACTION_EMOJIS[a],
        })

    counts  = Counter(votes)
    # Sort by (-count, action_index) so ties favour lower index (Hold first)
    winning = sorted(counts.keys(), key=lambda a: (-counts[a], a))[0]

    return {
        "action":      winning,
        "label":       ACTION_NAMES[winning],
        "emoji":       ACTION_EMOJIS[winning],
        "counts":      dict(counts),
        "model_votes": model_votes,
        "total":       len(models),
    }


# ---------------------------------------------------------------------------
# Live observation builder
# ---------------------------------------------------------------------------

def build_live_obs(
    symbol: str = "BTC-USD",
    window_size: int = 50,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, str]:
    """Fetch recent OHLCV from Yahoo Finance and build the obs vector.

    The observation format matches what the trading env produces::

        obs = [window_rows.flatten(), pos_norm, cash_frac, value_frac]

    Portfolio stats are set to *neutral* (no position, 100 % cash) since
    we do not track live holdings between calls.

    Returns
    -------
    obs : np.ndarray, shape (window_size * n_features + 3,)
    info_str : str
        Human-readable last-price summary.
    """
    import yfinance as yf
    from src.data.features import compute_all_features

    ticker = yf.Ticker(symbol)
    df     = ticker.history(period="300d", interval="1d")
    if df.empty:
        raise RuntimeError(f"No market data returned for {symbol}.")

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    feat_df = compute_all_features(df, {})
    feat_df = feat_df.dropna()

    if len(feat_df) < window_size:
        raise RuntimeError(
            f"Only {len(feat_df)} rows after feature computation; need {window_size}."
        )

    # Resolve feature columns
    if feature_cols is None:
        cols_path = PROJECT_ROOT / "data" / "processed" / symbol / "feature_cols.json"
        if cols_path.exists():
            feature_cols = json.loads(cols_path.read_text())
        else:
            feature_cols = [
                c for c in feat_df.columns
                if c not in ("Open", "High", "Low", "Close", "Volume")
            ]
    feature_cols = [c for c in feature_cols if c in feat_df.columns]

    window = feat_df[feature_cols].tail(window_size).values.astype(np.float32).flatten()
    # Neutral portfolio state: no position, all cash, equity = starting capital
    obs = np.concatenate([window, [0.0, 1.0, 1.0]]).astype(np.float32)

    last_price = float(df["Close"].iloc[-1])
    last_date  = str(df.index[-1].date())
    info_str   = f"{last_date}  |  Close: ${last_price:,.2f}"
    return obs, info_str, last_price


# ---------------------------------------------------------------------------
# Price data for charting
# ---------------------------------------------------------------------------

# Timeframe presets: label → (yfinance period, yfinance interval)
TIMEFRAME_OPTIONS: dict[str, tuple[str, str]] = {
    "1H":  ("1d",   "1m"),     # last day at 1-min bars, sliced to ~60 rows
    "1W":  ("7d",   "15m"),    # last 7 days at 15-min bars
    "1M":  ("30d",  "1d"),     # last 30 days, daily bars
    "1Y":  ("365d", "1d"),     # last year, daily bars
}


def fetch_price_df(
    symbol: str = "BTC-USD",
    timeframe: str = "1M",
) -> pd.DataFrame:
    """Fetch recent OHLCV from Yahoo Finance for display in the live chart.

    *timeframe* must be one of ``"1H"``, ``"1W"``, ``"1M"``, ``"1Y"``.
    """
    import yfinance as yf

    period, interval = TIMEFRAME_OPTIONS.get(timeframe, ("30d", "1d"))
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError(f"No market data returned for {symbol}.")
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    # For 1H, slice to the last 60 rows (~1 hour of 1-min bars)
    if timeframe == "1H":
        df = df.tail(60)

    return df
