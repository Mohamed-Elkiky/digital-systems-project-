# Risk Management

## Overview

The risk management subsystem operates as a **protective layer** between
the agent's raw action decisions and the actual execution in the backtesting
engine.  It enforces:

1. **Stop-loss triggers** — automatically exit positions when losses exceed a threshold.
2. **Position sizing** — control how much capital is allocated per trade.
3. **Leverage limit** — prevent total exposure from exceeding a set multiple of equity.
4. **Transaction costs & slippage** — realistic friction modelling.

---

## Stop-Loss

Implemented in `src/risk/risk_manager.py` (`RiskManager`).  Two modes are
available, selected via `risk.stop_loss_type` in config.

### Percent-Based (default)

```yaml
risk:
  stop_loss_type: "percent"
  stop_loss_pct: 0.05  # exit if price drops 5% from entry
```

When the current price falls `stop_loss_pct` below the entry price, the
system overrides the agent's action and forces a sell.

### ATR-Based

```yaml
risk:
  stop_loss_type: "atr"
  stop_loss_atr_mult: 2.0  # exit if price drops 2× ATR from entry
```

Uses the Average True Range indicator to set a dynamic stop-loss level
that adapts to market volatility.  More volatile markets get wider stops.

**ATR stop threshold** = entry\_price − (ATR × `stop_loss_atr_mult`)

### Integration with the Backtester

During evaluation (`src/eval/backtester.py`), the `RiskManager` is
instantiated from config and checked at each step:

1. Agent proposes an action.
2. `RiskManager.check_stop_loss()` — if triggered, override action to Sell.
3. Execute through the environment.
4. `RiskManager.on_buy()` / `on_sell()` — update internal entry price tracker.

---

## Position Sizing

Implemented directly in `TradingEnv._execute_buy()` (`src/env/trading_env.py`).
Selected via `env.position_sizing` in config.

### Fixed Fraction (default)

```yaml
env:
  position_sizing: "fixed_fraction"
  position_size_frac: 0.10  # invest 10% of available cash per trade
```

Simple, predictable.  Good baseline for comparing agents.  At each buy
signal, the environment spends `position_size_frac × cash` on the asset.

### ATR Volatility Sizing

```yaml
env:
  position_sizing: "atr_volatility"
  risk_per_trade: 0.01    # risk at most 1% of equity per trade
  atr_mult: 2.0           # stop distance = atr_mult × ATR
  max_position_value: 0.5 # hard cap: spend ≤ 50% of equity per buy
```

Position size is derived from the ATR stop-loss distance so that if the
stop is triggered, the portfolio loss is bounded:

```
units  = (equity × risk_per_trade) / (atr_mult × ATR)
spend  = units × price
spend  = min(spend, max_position_value × equity)   # hard cap
```

This means the system takes **smaller positions in volatile markets**
(high ATR) and **larger positions in calm markets** (low ATR), targeting
a consistent worst-case loss per trade.  Falls back to `fixed_fraction`
when ATR is zero or unavailable.

---

## Leverage Limit

```yaml
env:
  max_leverage: 1.0   # 1.0 = no borrowing (exposure ≤ 100% of equity)
```

Enforced inside `TradingEnv._execute_buy()`.  Before any buy is executed,
the available budget is capped so that:

```
position × price  ≤  max_leverage × portfolio_value
```

With the default `max_leverage = 1.0`, total asset exposure cannot exceed
the current portfolio value (cash + existing position), which is equivalent
to a no-leverage, fully-funded constraint.  Setting `max_leverage = 0.5`
would restrict the asset allocation to at most 50 % of equity.

---

## Transaction Costs & Slippage

Modelled as percentage deductions on each trade:

```yaml
env:
  transaction_cost_pct: 0.001   # 0.1% per trade
  slippage_pct: 0.0005          # 0.05% adverse price movement
```

- **Transaction cost**: deducted from the trade value (both buys and sells).
- **Slippage**: the execution price is shifted against the trader
  (higher for buys, lower for sells).

These are applied both in the environment's `step()` method and in the
baseline backtests, ensuring fair comparison.

---

## Reward Design Rationale

The default reward function (`log_return`) includes:

1. **Portfolio log-return** — the core signal: how much value changed.
2. **Drawdown penalty** — penalises being in drawdown, encouraging
   the agent to protect gains.
3. **Transaction cost deduction** — discourages excessive trading.

```
reward = log(V_t / V_{t-1}) - λ × drawdown - cost / V_{t-1}
```

where λ = `env.drawdown_penalty` (default 0.5).

An alternative `sharpe` reward is available, which computes a rolling
Sharpe ratio from recent returns.  This encourages consistent risk-adjusted
performance rather than raw returns.

---

## Configuration Reference

| Parameter | Section | Default | Description |
|---|---|---|---|
| `stop_loss_type` | `risk` | `"percent"` | `"percent"` or `"atr"` |
| `stop_loss_pct` | `risk` | `0.05` | Percent stop threshold |
| `stop_loss_atr_mult` | `risk` | `2.0` | ATR multiplier for stop |
| `position_sizing` | `env` | `"fixed_fraction"` | Sizing mode |
| `position_size_frac` | `env` | `0.10` | Fraction of cash per buy (fixed) |
| `risk_per_trade` | `env` | `0.01` | Max equity fraction at risk (ATR) |
| `atr_mult` | `env` | `2.0` | ATR multiplier for sizing (ATR) |
| `max_position_value` | `env` | `0.50` | Hard spend cap as fraction of equity (ATR) |
| `max_leverage` | `env` | `1.0` | Max exposure / equity ratio |
| `transaction_cost_pct` | `env` | `0.001` | Round-trip cost fraction |
| `slippage_pct` | `env` | `0.0005` | Adverse price movement fraction |

See `configs/base.yaml` for the complete defaults with inline comments.
